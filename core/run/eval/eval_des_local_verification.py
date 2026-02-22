
import os
import json
import faiss
import argparse
from pathlib import Path
from typing import Optional, List
from loguru import logger
from tqdm import tqdm
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from core import distributed
from core.models import (
    build_local_verification_copy_descriptor_for_eval,
    build_sscd_resnet50_descriptor_for_eval,
    build_resnet50_descriptor_for_eval,
    build_local_verification_dino_for_eval,
)
from core.data.build import build_local_verification_data_for_descriptor_eval
from core.eval.setup import setup
from core.eval.utils import (
    extract_features_with_dataloader,
    des_postprocessor,
    default_des_preprocessor,
    des_trunk_postprocessor,
    all_gather_and_flatten,
)
from core.run.eval.measure_cls import measure_cls


def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = None,
        add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--mode",
        default="M:N",
        type=str,
        choices=["M:N", "1:N"],
        help="KNN mode, must be one of mode in [M:N, 1:N]",
    )
    parser.add_argument(
        "opts",
        help=""" 
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs.
    For python-based LazyConfig, use "path.key=value".
            """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@torch.inference_mode()
def inference(cfg, query_feats_list, reference_features):
    assert len(cfg.eval.local_verification) == len(query_feats_list)

    df_que = pd.read_csv(cfg.eval.que_csv_path)
    df_ref = pd.read_csv(cfg.eval.ref_csv_path)

    chunk_size = 8
    mode = cfg.mode

    n_ref, ref_crops, dim = reference_features.shape
    flattened_reference_features = reference_features.view(n_ref * ref_crops, dim)

    loc_ver_cosines, loc_ver_indices = [], []
    for query_features, _cfg in zip(query_feats_list, cfg.eval.local_verification):
        search_rank = _cfg.search_rank
        que_crops = query_features.shape[1]
        que_stride, ref_stride = mode.split(":")
        que_stride = 1 if que_stride == "1" else que_crops
        ref_stride = 1 if ref_stride == "1" else ref_crops

        # split query features into NGPU pieces to accelerate searching
        query_features = torch.chunk(query_features, distributed.get_global_size(), dim=0)
        query_features = query_features[distributed.get_global_rank()]
        # further split each piece into chunks to avoid OOM
        chunks = torch.split(query_features, chunk_size)
        sorted_cosines_chunks, indices_chunks = [], []
        for chunk_i, chunk in tqdm(enumerate(chunks), desc="Searching in chunks"):
            _chunk_size = chunk.size(0)
            flattened_chunk = chunk.view(_chunk_size * que_crops, dim)
            flattened_cosines = torch.mm(flattened_chunk, flattened_reference_features.t())
            cosines = F.max_pool2d(
                flattened_cosines.unsqueeze(dim=0),
                kernel_size=(que_stride, ref_stride),
                stride=(que_stride, ref_stride)
            ).squeeze(dim=0)
            sorted_cosines, indices = torch.sort(cosines, descending=True, dim=1)
            # sorted_cosines: [_chunk_size * que_crops / que_stride, n_ref * ref_crops / ref_stride]
            # indices: [_chunk_size * que_crops / que_stride, n_ref * ref_crops / ref_stride]
            sorted_cosines = sorted_cosines[:, : search_rank].reshape(_chunk_size, -1)
            indices = indices[:, : search_rank].reshape(_chunk_size, -1)

            sorted_cosines_chunks.append(sorted_cosines.clone())
            indices_chunks.append(indices.clone())

        sorted_cosines = torch.cat(sorted_cosines_chunks, dim=0)
        indices = torch.cat(indices_chunks, dim=0)
        loc_ver_cosines.append(sorted_cosines)
        loc_ver_indices.append(indices)

    # loc_ver_cosines.size(0) = n_que // NGPU, loc_ver_indices.size(0) = n_que // NGPU
    loc_ver_cosines = torch.cat(loc_ver_cosines, dim=1)
    loc_ver_indices = torch.cat(loc_ver_indices, dim=1)

    loc_ver_cosines = all_gather_and_flatten(loc_ver_cosines)
    loc_ver_indices = all_gather_and_flatten(loc_ver_indices)
    logger.info(f"=> loc_ver_cosines and loc_ver_indices gathered. Ready to build final scores.")
    if distributed.is_main_process():
        loc_ver_cosines = loc_ver_cosines.cpu()
        loc_ver_indices = loc_ver_indices.cpu()
        query_ids = df_que['uuid']
        reference_ids = np.array(df_ref['uuid'])
        scores = []
        for i in tqdm(range(loc_ver_cosines.shape[0])):
            _scores = [
                query_ids[i],
                [[uuid, float(s)] for uuid, s in zip(reference_ids[loc_ver_indices[i].tolist()], loc_ver_cosines[i])]
            ]
            scores.append(_scores)

        return scores


def eval_des(model, cfg, epoch=None):
    key = getattr(cfg.eval, "key", "cls_token")
    logger.info(f"Using key: {key}.")
    device = torch.device(distributed.get_global_rank())
    output_dir = Path(cfg.output_dir)
    que_path = output_dir / f'query_{key}_ep-{epoch}_loc-ver.pth'
    ref_path = output_dir / f'reference_{key}_ep-{epoch}_loc-ver.pth'

    # extract or load features
    if (not que_path.exists()) or (not ref_path.exists()):
        _, _, que_data_loaders, ref_data_loader = build_local_verification_data_for_descriptor_eval(cfg)
        outputs_que_list = [
            extract_features_with_dataloader(
                model,
                que_data_loader,
                preprocessor=default_des_preprocessor,
                postprocessor=partial(des_postprocessor, key=key),
            ) for que_data_loader in que_data_loaders
        ]
        outputs_ref = extract_features_with_dataloader(
            model,
            ref_data_loader,
            preprocessor=default_des_preprocessor,
            postprocessor=partial(des_postprocessor, key=key),
        )
        torch.cuda.synchronize()
        if distributed.is_main_process():
            query_feats_list = [outputs_que[key].to(device) for outputs_que in outputs_que_list]
            reference_feats = outputs_ref[key].to(device)
            torch.save(query_feats_list, que_path)
            torch.save(reference_feats, ref_path)
            logger.info(f"EVAL -- Feature dumped.")

    torch.distributed.barrier()
    logger.info(f"EVAL -- Found previously dumped feature.")
    query_feats_list = [feat.to(device) for feat in torch.load(que_path, map_location="cpu")]
    logger.info(f"EVAL -- Feature loaded from {que_path}.")
    reference_feats = torch.load(ref_path, map_location="cpu").to(device)
    logger.info(f"EVAL -- Feature loaded from {ref_path}.")

    query_feats_list = [F.normalize(query_feats, dim=2, p=2) for query_feats in query_feats_list]
    logger.info(f"=> Normalize query features done.")
    reference_feats = F.normalize(reference_feats, dim=2, p=2)
    logger.info(f"=> Normalize reference features done.")

    scores = inference(cfg, query_feats_list, reference_feats)

    if distributed.is_main_process():
        with open(Path(cfg.output_dir) / f'scores_{key}{f"_ep-{epoch}" if epoch else ""}.json', 'w') as file:
            json.dump(scores, file)
        measure_cls(scores, cfg, epoch)


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.arch = "descriptor"
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"output_dir={args.output_dir}", f"weights={args.weights}", f"mode={args.mode}"]
    cfg = setup(args)

    model, epoch = build_local_verification_copy_descriptor_for_eval(cfg)
    # model, epoch = build_local_verification_dino_for_eval(cfg)
    eval_des(model, cfg, epoch)


if __name__ == '__main__':
    main()
