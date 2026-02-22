from collections import OrderedDict

from loguru import logger

import torch.nn as nn

from core.loss.base_loss import BaseLoss
from core.loss.build import register_loss


def dice_loss(masks, masks_gt, reduction='mean'):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        masks: A float tensor of arbitrary shape.
                The predictions for each example.
        masks_gt: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    masks = masks.sigmoid()
    masks = masks.flatten(1)
    masks_gt = masks_gt.flatten(1)
    numerator = 2 * (masks * masks_gt).sum(-1)
    denominator = masks.sum(-1) + masks_gt.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss


@register_loss("dice_loss")
class DiceLoss(BaseLoss):
    def __init__(self, cfg, model):
        super(DiceLoss, self).__init__(cfg, model)
        self.reduction = getattr(cfg, "reduction", "mean")
        logger.info(self)

    def __repr__(self):
        return f"Build DiceLoss with weight: {self.weight}, reduction: {self.reduction}."

    def calculate(self, outputs, inputs):
        que_masks = outputs["que_masks"][:, 1]
        que_masks_gt = inputs["que_masks_gt"]

        ref_masks = outputs["ref_masks"][:, 1]
        ref_masks_gt = inputs["ref_masks_gt"]

        return OrderedDict(
            [
                ("que", dice_loss(que_masks, que_masks_gt, self.reduction)),
                ("ref", dice_loss(ref_masks, ref_masks_gt, self.reduction)),
            ]
        )
