from copy import deepcopy

from loguru import logger

import torch
from torch import nn
from torch.nn import functional as F

from core import distributed
from core.eval.utils import all_gather_and_flatten
from core.layers import Block
from core.loss.base_loss import BaseLoss
from core.loss.build import register_loss


def infonce_loss(features, positive_indices, scale, gather=False):
    """
        Compute the infonce_loss loss between two feature groups.
        :param features: [N, dim]
        :param positive_indices: [n]
        :param scale: scaling factor for the similarity matrix
        :param gather: whether to gather features from other process
        :return: infonce_loss loss
    """

    n = positive_indices.size(0)
    positive_indices = positive_indices.unsqueeze(1)

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
        features = torch.cat([features, *features_list], dim=0)

    N = features.size(0)
    features = F.normalize(features, dim=1, p=2)

    cosines_similarity = torch.mm(features, features.t())
    cosines_similarity.view(-1)[:: (N + 1)].fill_(-1)  # Trick to fill diagonal with -1
    cosines_similarity = cosines_similarity[: n]

    logits = scale * cosines_similarity

    probs = F.softmax(logits, dim=-1)
    loss = -torch.log(torch.gather(probs, dim=1, index=positive_indices.long()))
    return loss.mean()


@register_loss("infonce")
class InfoNCELoss(BaseLoss):
    def __init__(self, cfg, model):
        super(InfoNCELoss, self).__init__(cfg, model)
        self.log_scale_init = cfg.log_scale_init
        self.log_scale = nn.Parameter(torch.tensor(self.log_scale_init), requires_grad=True)
        # self.log_scale = nn.Parameter(torch.tensor(self.log_scale_init), requires_grad=False)
        model.register_parameter("infonce_log_scale", self.log_scale)
        self.gather = cfg.gather
        self.preprocessor = self.paired_input_preprocess

        logger.info(self)

    def __repr__(self):
        return f"Build InfoNCE loss with " \
               f"weight={self.weight}, " \
               f"log_scale_init={self.log_scale_init}, " \
               f"gather={self.gather}."

    def calculate(self, outputs, inputs):
        features, positive_indices = self.preprocessor(outputs, inputs)
        return infonce_loss(features, positive_indices, self.log_scale.exp(), self.gather)

    def paired_input_preprocess(self, outputs, inputs):
        que_cls_tokens = outputs["que_feat"]
        ref_cls_tokens = outputs["ref_feat"]

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

