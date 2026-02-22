# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
from collections import defaultdict
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from loguru import logger
from torch import nn
from tqdm import tqdm

from core import distributed


def default_seg_preprocessor(input):
    _input = {
        "queries": input["queries"].cuda(),
        "references": input["references"].cuda(),
        "indices": input["indices"].cuda(),
        "names": input["names"],
    }
    input.update(_input)
    return input


def default_seg_postprocessor(input, output):
    return {
        "indices": input["indices"].cuda(),
        "preds": F.softmax(output["preds"], dim=1)[:, 1].detach().to(torch.float16),
        "que_masks": F.softmax(output["que_masks"], dim=1)[:, 1].detach().to(torch.float16),
        "ref_masks": F.softmax(output["ref_masks"], dim=1)[:, 1].detach().to(torch.float16),
    }


def default_des_preprocessor(input):
    _input = {
        "images": input["images"].cuda(),
        "indices": input["indices"].cuda(),
    }
    input.update(_input)
    return input


@torch.inference_mode()
def extract_features_with_dataloader(
        model,
        data_loader,
        preprocessor=default_seg_preprocessor,
        postprocessor=default_seg_postprocessor,
        gather_to_cpu=True,
):
    gather_device = torch.device("cpu") if gather_to_cpu else torch.device("cuda")
    outputs = None
    for input in tqdm(data_loader, desc="Eval Progress", unit="mini-batch", unit_scale=True):
        _input = preprocessor(input) if preprocessor else input
        outputs_rank = model(_input)
        if postprocessor:
            outputs_rank = postprocessor(input, outputs_rank)
        outputs_ranks = all_gather_dict(outputs_rank)

        indices = outputs_ranks.pop("indices").cpu().long()

        if distributed.is_main_process():
            if outputs is None:
                outputs = dict()
                # init storage feature matrix
                for key, batch_tensor in outputs_ranks.items():
                    global_tensor = torch.zeros(
                        len(data_loader.dataset), *batch_tensor.shape[1:],
                        dtype=batch_tensor.dtype,
                        device=gather_device,
                    )
                    outputs[key] = global_tensor
                    logger.info(f"Initialize global tensor {key} with shape of {global_tensor.shape}")

            for key, batch_tensor in outputs_ranks.items():
                batch_tensor = batch_tensor.to(gather_device)
                outputs[key].index_copy_(0, indices, batch_tensor)

    return outputs


def all_gather_dict(dict_rank):
    dict_all_ranks = dict()
    for key, value in dict_rank.items():
        if isinstance(value, torch.Tensor):
            gathered_value = all_gather_and_flatten(value)
            dict_all_ranks[key] = gathered_value
        else:
            dict_all_ranks[key] = value
            logger.info(f"dict_rank[{key}] is not a tensor and is ignored in all_gather_dict op.")
    return dict_all_ranks


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def broadcast_tensor_to_all_ranks(tensor):
    device = torch.device(distributed.get_local_rank())
    # sync n dim
    if distributed.is_main_process():
        n_dim = torch.tensor(len(tensor.size()), dtype=torch.int, device=device)
        torch.distributed.broadcast(n_dim, src=0)
    else:
        n_dim = torch.tensor(0, dtype=torch.int, device=device)
        torch.distributed.broadcast(n_dim, src=0)

    # sync tensor size
    if distributed.is_main_process():
        size = torch.tensor(tensor.size(), dtype=torch.int, device=device)
        torch.distributed.broadcast(size, src=0)
    else:
        size = torch.empty(n_dim, dtype=torch.int, device=device)
        torch.distributed.broadcast(size, src=0)

    # sync tensor
    if distributed.is_main_process():
        torch.distributed.broadcast(tensor, src=0)
    else:
        tensor = torch.empty(size.tolist(), device=device)
        torch.distributed.broadcast(tensor, src=0)

    return tensor


def find_max_region(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(contour) for contour in contours]
    if len(areas) == 0:
        return np.zeros_like(mask)
    max_idx = np.argmax(areas)

    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [contours[max_idx]], 1)

    return mask


@torch.inference_mode()
def inference_from_cache(cache, que_th=0.5, ref_th=0.5, mask_size=224):
    results = {
        "preds": cache["preds"].cpu().numpy(),
        "que_bboxes": np.zeros((len(cache['que_masks']), 4), dtype=np.float32),
        "ref_bboxes": np.zeros((len(cache['ref_masks']), 4), dtype=np.float32),
    }

    def _inference(i, que_mask, ref_mask, results, que_th=0.5, ref_th=0.5, mask_size=224):
        que_mask = (que_mask >= que_th).cpu().numpy().astype(np.uint8)
        ref_mask = (ref_mask >= ref_th).cpu().numpy().astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        que_mask = cv2.morphologyEx(que_mask, cv2.MORPH_OPEN, kernel)
        ref_mask = cv2.morphologyEx(ref_mask, cv2.MORPH_OPEN, kernel)

        que_mask = find_max_region(que_mask)
        ref_mask = find_max_region(ref_mask)

        que_bbox_y, que_bbox_x = np.where(que_mask > 0)
        ref_bbox_y, ref_bbox_x = np.where(ref_mask > 0)

        if len(que_bbox_y) > 0 and len(que_bbox_x) > 0:
            # x,y,w,h
            que_bbox = np.array(
                [
                    que_bbox_x.min(),
                    que_bbox_y.min(),
                    que_bbox_x.max() - que_bbox_x.min(),
                    que_bbox_y.max() - que_bbox_y.min(),
                ]
            )
            que_bbox = que_bbox / mask_size
        else:
            que_bbox = np.zeros(4, dtype=results["que_bboxes"].dtype)

        if len(ref_bbox_y) > 0 and len(ref_bbox_x) > 0:
            ref_bbox = np.array(
                [
                    ref_bbox_x.min(),
                    ref_bbox_y.min(),
                    ref_bbox_x.max() - ref_bbox_x.min(),
                    ref_bbox_y.max() - ref_bbox_y.min(),
                ]
            )
            ref_bbox = ref_bbox / mask_size
        else:
            ref_bbox = np.zeros(4, dtype=results["ref_bboxes"].dtype)

        results["que_bboxes"][i] = que_bbox
        results["ref_bboxes"][i] = ref_bbox

    # transfer masks into bounding boxes
    for i, (que_mask, ref_mask) in tqdm(enumerate(zip(cache["que_masks"], cache["ref_masks"]))):
        _inference(i, que_mask, ref_mask, results, que_th, ref_th, mask_size)

    return results


def cls_postprocessor(input, output):
    return {
        "indices": input["indices"].cuda(),
        "preds": output["preds"],
    }


def des_trunk_postprocessor(input, output):
    que_features = output["que_cls_token"]
    features = F.normalize(que_features, dim=1, p=2)
    return {
        "indices": input["indices"].cuda(),
        "feats": features.cuda(),
    }


def des_postprocessor(input, output, key="cls_token"):
    features = output[key]
    features = F.normalize(features, dim=1, p=2)
    return {
        "indices": input["indices"].cuda(),
        key: features.cuda(),
    }


def default_third_party_des_postprocessor(input, output):
    features = output["feat"]
    features = F.normalize(features, dim=1, p=2)
    return {
        "indices": input["indices"].cuda(),
        "feats": features.cuda(),
    }


def encoder_preprocessor(input):
    queries = input["queries"].cuda()
    references = input["references"].cuda()
    x = torch.cat([queries, references], dim=0)

    return x


def encoder_postprocessor(input, output):
    B = output.size(0)
    que_features, ref_features = output.split(B // 2, dim=0)
    que_features = F.normalize(que_features, dim=1, p=2)
    ref_features = F.normalize(ref_features, dim=1, p=2)

    preds = (que_features * ref_features).sum(dim=1)

    return {
        "indices": input["indices"].cuda(),
        "preds": preds.detach(),
    }
