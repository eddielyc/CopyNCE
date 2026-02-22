from collections import OrderedDict
from copy import deepcopy

from loguru import logger

import torch
from torch import nn
from torch.nn import functional as F

from core import distributed
from core.loss.base_loss import BaseLoss
from core.loss.build import register_loss


@register_loss("sce_loss")
class SegCrossEntropyLoss(BaseLoss):
    def __init__(self, cfg, model):
        super(SegCrossEntropyLoss, self).__init__(cfg, model)

        if hasattr(cfg, "eos_coef"):
            self.eos_coef = cfg.eos_coef
            self.weight = torch.ones(getattr(cfg, "num_classes", 1) + 1)
            self.weight[0] = self.eos_coef
            self.weight = self.weight.to(distributed.get_local_rank())
        else:
            self.weight = None
        self.reduction = getattr(cfg, "reduction", "mean")

        logger.info(self)

    def __repr__(self):
        return f"Build CrossEntropyLoss with " \
               f"weight: {self.weight}, " \
               f"reduction: {self.reduction}."

    def calculate(self, outputs, inputs):
        que_masks = outputs["que_masks"]
        que_masks_gt = inputs["que_masks_gt"]

        que_ce_loss = F.cross_entropy(
            que_masks, que_masks_gt,
            self.weight,
            reduction=self.reduction,
        )

        ref_masks = outputs["ref_masks"]
        ref_masks_gt = inputs["ref_masks_gt"]

        ref_ce_loss = F.cross_entropy(
            ref_masks, ref_masks_gt,
            self.weight,
            reduction=self.reduction,
        )

        return OrderedDict(
            [
                ("que", que_ce_loss),
                ("ref", ref_ce_loss),
            ]
        )


@register_loss("bce_loss")
class BinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, cfg, model):
        super(BinaryCrossEntropyLoss, self).__init__(cfg, model)
        self.reduction = getattr(cfg, "reduction", "mean")

        logger.info(self)

    def __repr__(self):
        return f"Build BinaryCrossEntropyLoss with weight: {self.weight}, reduction: {self.reduction}."

    def calculate(self, outputs, inputs):
        preds = outputs["preds"].squeeze()
        cls_gt = inputs["cls_gt"].float()

        return F.binary_cross_entropy_with_logits(preds, cls_gt, reduction=self.reduction)
