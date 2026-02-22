import torch
from torch import nn
from torch.nn import functional as F
from loguru import logger
from core import distributed

from core.loss.base_loss import BaseLoss
from core.loss.build import register_loss


def contrastive_loss(features, positive_indices, margin=0.0, gather=False):
    assert features.size(0) == positive_indices.size(0)
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
    distances_pos = 1. - torch.gather(cosines, dim=1, index=positive_indices.unsqueeze(dim=1))
    _cosines = cosines.detach()
    _cosines = torch.scatter(_cosines, dim=1, index=positive_indices.unsqueeze(dim=1), value=-1)
    _cosines = _cosines.fill_diagonal_(-1)
    hnm_indices = _cosines.argmax(dim=1)
    distances_neg = 1. - torch.gather(cosines, dim=1, index=hnm_indices.unsqueeze(dim=1))

    _loss = (distances_pos - distances_neg + margin).exp() + 1
    loss = _loss.log().mean()

    return loss


@register_loss("contrastive")
class ContrastiveLoss(BaseLoss):
    def __init__(self, cfg, model):
        super(ContrastiveLoss, self).__init__(cfg, model)
        self.margin = cfg.margin
        self.key = cfg.key
        self.gather = cfg.gather
        logger.info(self)

    def __repr__(self):
        return f"Build Contrastive loss with " \
               f"weight: {self.weight}, margin: {self.margin}, key: {self.key}, gather: {self.gather}."

    def calculate(self, outputs, inputs):
        features, positive_indices = self.input_preprocess(outputs, inputs)
        return contrastive_loss(features, positive_indices, margin=self.margin, gather=self.gather)

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
            ], dim=0
        )

        positive_indices = torch.arange(
            n_pos,
            dtype=torch.long,
            device=que_cls_tokens.device
        ).tile(2)
        positive_indices[:n_pos] += n_pos

        return features, positive_indices
