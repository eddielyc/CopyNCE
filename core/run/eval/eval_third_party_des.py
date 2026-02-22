
import os
import json
import faiss
import argparse
from pathlib import Path
from typing import Optional, List
from loguru import logger
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from core import distributed
from core.models import build_sscd_resnet50_descriptor_for_eval
from core.data.build import build_data_for_descriptor_eval
from core.eval.setup import setup
from core.eval.utils import (
    extract_features_with_dataloader,
    default_third_party_des_postprocessor,
    default_des_preprocessor,
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


def inference(cfg, query_feats, reference_feats):
    df_que = pd.read_csv(cfg.eval.que_csv_path)
    df_ref = pd.read_csv(cfg.eval.ref_csv_path)

    # Image retrieve with Faiss
    dim = query_feats.shape[1]
    search_rank = cfg.eval.search_rank
    index = faiss.IndexFlatIP(dim)
    index.add(reference_feats)
    if cfg.eval.gpu_search:
        index = faiss.index_cpu_to_all_gpus(index)
    logger.info("INFERENCE -- Faiss index built...")
    topk_distances, topk_indices = index.search(query_feats, search_rank)
    logger.info("INFERENCE -- Search finished...")

    query_ids = df_que['uuid']
    reference_ids = np.array(df_ref['uuid'])
    scores = []
    for i in tqdm(range(query_feats.shape[0])):
        _scores = [
            query_ids[i],
            [[uuid, float(s)] for uuid, s in zip(reference_ids[topk_indices[i].tolist()], topk_distances[i])]
        ]
        scores.append(_scores)

    return scores


def eval_des(model, cfg, epoch=None):
    output_dir = Path(cfg.output_dir)
    que_path = output_dir / 'query_feats.pth'
    ref_path = output_dir / 'reference_feats.pth'

    # extract or load features
    if (not que_path.exists()) or (not ref_path.exists()):
        _, _, que_data_loader, ref_data_loader = build_data_for_descriptor_eval(cfg)
        outputs_query = extract_features_with_dataloader(
            model,
            que_data_loader,
            preprocessor=default_des_preprocessor,
            postprocessor=default_third_party_des_postprocessor,
        )
        outputs_ref = extract_features_with_dataloader(
            model,
            ref_data_loader,
            preprocessor=default_des_preprocessor,
            postprocessor=default_third_party_des_postprocessor,
        )
        torch.cuda.synchronize()
        if distributed.is_main_process():
            query_feats = outputs_query['feats']
            reference_feats = outputs_ref['feats']
            torch.save(query_feats, que_path)
            torch.save(reference_feats, ref_path)
            logger.info(f"EVAL -- Feature dumped.")
    else:
        if distributed.is_main_process():
            query_feats = torch.load(que_path)
            reference_feats = torch.load(ref_path)
            logger.info(f"EVAL -- Feature loaded from {que_path} and {ref_path}.")

    if distributed.is_main_process():
        query_feats = F.normalize(query_feats, dim=1, p=2)
        reference_feats = F.normalize(reference_feats, dim=1, p=2)
    
    if distributed.is_main_process():
        scores = inference(cfg, query_feats.numpy(), reference_feats.numpy())
        with open(Path(cfg.output_dir) / f'scores{f"_ep-{epoch}" if epoch else ""}.json', 'w') as file:
            json.dump(scores, file)
        measure_cls(scores, cfg, epoch)


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.arch = "descriptor"
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"output_dir={args.output_dir}", f"weights={args.weights}"]
    cfg = setup(args)

    model, epoch = build_sscd_resnet50_descriptor_for_eval(cfg)
    eval_des(model, cfg, epoch)


if __name__ == '__main__':
    main()
