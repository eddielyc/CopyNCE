import torch
from torch import nn
from torch.nn import functional as F
from loguru import logger
from core import distributed

from core.loss.base_loss import BaseLoss
from core.loss.build import register_loss


def intra_cosine_loss(features):
    assert len(features.size()) == 3
    # feaures: [B, L, C]
    features = F.normalize(features, dim=2, p=2)
    # batch_cosines: [B, L, L]
    batch_cosines = features.bmm(features.permute((0, 2, 1)))
    intra_cosines_loss = 1. - batch_cosines.mean(dim=(1, 2))
    intra_cosines_loss = intra_cosines_loss.mean()
    return intra_cosines_loss


@register_loss("intra_cosine")
class IntraCosineLoss(BaseLoss):
    def __init__(self, cfg, model):
        super(IntraCosineLoss, self).__init__(cfg, model)
        logger.info(self)

    def __repr__(self):
        return f"Build IntraCosine loss with " \
               f"weight: {self.weight}."

    def calculate(self, outputs, inputs):
        que_patch_tokens = outputs["que_patch_tokens"]
        ref_patch_tokens = outputs["ref_patch_tokens"]
        loss = 0.5 * intra_cosine_loss(que_patch_tokens) + 0.5 * intra_cosine_loss(ref_patch_tokens)
        return loss
