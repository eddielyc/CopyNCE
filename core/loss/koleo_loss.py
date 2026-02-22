# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.loss.base_loss import BaseLoss
from core import distributed
from core.loss.build import register_loss


def pairwise_inner_nearest_neighbors(x):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
    # max inner prod -> min distance
    _, indices = torch.max(dots, dim=1)  # noqa: E741
    return indices


def koleo_loss(features, labels, gather=False):
    features_all = features
    if gather:
        features_all_ranks = torch.empty(
            distributed.get_global_size(),
            *features.shape,
            dtype=features.dtype,
            device=features.device,
        )
        features_list = list(features_all_ranks.unbind(0))
        torch.distributed.all_gather(features_list, features.contiguous())
        features_list.pop(distributed.get_local_rank())
        features_all = torch.cat([features, *features_list], dim=0)

    features = F.normalize(features, dim=1, p=2)
    features_all = F.normalize(features_all, dim=1, p=2)
    cosines = features.matmul(features_all.transpose(0, 1))

    cosines = cosines.fill_diagonal_(-1)
    cosines = cosines.scatter_(dim=1, index=labels.unsqueeze(dim=1), value=-1)

    max_non_match_cosines, _ = cosines.max(dim=1, keepdim=True)
    closest_distance = (2 - 2 * max_non_match_cosines).clamp(min=1e-6).sqrt()
    entropy_loss = -closest_distance.log().mean()

    return entropy_loss


@register_loss("keleo_loss")
class KoLeoLoss(BaseLoss):
    """
        Kozachenko-Leonenko entropic loss regularizer
        from Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    """
    def __init__(self, cfg, model):
        super(KoLeoLoss, self).__init__(cfg, model)
        self.gather = getattr(cfg, "gather", False)
        self.key = getattr(cfg, "key", "feat")
        logger.info(self)

    def __repr__(self):
        return f"Build KoLeo loss with " \
               f"weight={self.weight}, gather={self.gather}, key={self.key}."

    def calculate(self, outputs, inputs):
        features, labels = self.input_preprocess(outputs, inputs)
        return koleo_loss(features, labels, gather=self.gather)

    def input_preprocess(self, outputs, inputs):
        que_cls_tokens = outputs[f"que_{self.key}"]
        ref_cls_tokens = outputs[f"ref_{self.key}"]

        targets = inputs["cls_gt"]

        positive_que_cls_tokens = que_cls_tokens[targets.to(torch.bool)]
        positive_ref_cls_tokens = ref_cls_tokens[targets.to(torch.bool)]
        negative_que_cls_tokens = que_cls_tokens[(1 - targets).to(torch.bool)]
        negative_ref_cls_tokens = ref_cls_tokens[(1 - targets).to(torch.bool)]
        n_pos = positive_que_cls_tokens.size(0)

        features = torch.cat(
            [
                positive_que_cls_tokens, positive_ref_cls_tokens,
                negative_que_cls_tokens, negative_ref_cls_tokens,
            ],
            dim=0
        )

        positive_indices = torch.arange(
            n_pos,
            dtype=torch.long,
            device=que_cls_tokens.device
        ).tile(2)
        positive_indices[:n_pos] += n_pos

        return features, positive_indices


def reverse_koleo_loss(features, labels):
    features = F.normalize(features, dim=1, p=2)
    cosines = features.matmul(features.transpose(0, 1))

    positive_cosines = torch.gather(cosines, dim=1, index=labels.unsqueeze(dim=1))
    positive_distance = (2 - 2 * positive_cosines).clamp(min=1e-6).sqrt()
    loss = positive_distance.log().mean()

    return loss


@register_loss("rkeleo_loss")
class ReverseKoLeoLoss(KoLeoLoss):
    def __init__(self, cfg, model):
        super(ReverseKoLeoLoss, self).__init__(cfg, model)

    def __repr__(self):
        return f"Build ReverseKoLeoLoss loss with " \
               f"weight={self.weight}, key={self.key}."

    def calculate(self, outputs, inputs):
        features, labels = self.input_preprocess(outputs, inputs)
        return reverse_koleo_loss(features, labels)
