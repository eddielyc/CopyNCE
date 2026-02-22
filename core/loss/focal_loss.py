from collections import OrderedDict

from loguru import logger

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.loss.base_loss import BaseLoss
from core.loss.build import register_loss


def focal_loss(inputs, targets, alpha: float = 0.5, gamma: float = 2, reduction="mean"):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.to(torch.float), reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss


@register_loss("focal_loss")
class SegFocalLoss(BaseLoss):
    def __init__(self, cfg, model):
        super(SegFocalLoss, self).__init__(cfg, model)

        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.reduction = getattr(cfg, "reduction", "mean")

        logger.info(self)

    def __repr__(self):
        return f"Build FocalLoss with " \
               f"weight: {self.weights}, " \
               f"alpha: {self.alpha}, " \
               f"gamma: {self.gamma}, " \
               f"reduction: {self.reduction}."

    def calculate(self, outputs, inputs):
        que_masks = outputs["que_masks"][:, 1]
        que_masks_gt = inputs["que_masks_gt"]

        ref_masks = outputs["ref_masks"][:, 1]
        ref_masks_gt = inputs["ref_masks_gt"]

        return OrderedDict(
            [
                ("que", focal_loss(que_masks, que_masks_gt, self.reduction)),
                ("ref", focal_loss(ref_masks, ref_masks_gt, self.reduction)),
            ]
        )
