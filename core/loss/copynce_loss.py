from collections import OrderedDict
from copy import deepcopy

from loguru import logger

from typing import Union, Iterable, List
import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain
from core.loss.base_loss import BaseLoss
from timm.layers import to_2tuple
from core.data.augmentation.mask_mapper import MaskMapper
from core.loss.build import register_loss


def is_1d(tokens: torch.Tensor) -> bool:
    return len(tokens.size()) == 3


def is_2d(tokens: torch.Tensor) -> bool:
    return len(tokens.size()) == 4


def to_2d(tokens: torch.Tensor, h_t, w_t) -> torch.Tensor:
    if is_2d(tokens):
        return tokens
    B, L, C = tokens.size()
    assert L == h_t * w_t
    return tokens.view((B, h_t, w_t, C))


def to_1d(tokens: torch.Tensor) -> torch.Tensor:
    if is_1d(tokens):
        return tokens
    B, h_t, w_t, C = tokens.size()
    L = h_t * w_t
    return tokens.view((B, L, C))


class PatchManager(object):
    def __init__(self, patch_size=(16, 16)):
        self.patch_size = patch_size
        self.w_p, self.h_p = self.patch_size

    def patchify(self, tensor: torch.Tensor):
        *_, h, w = tensor.size()
        assert h % self.h_p == 0 and w % self.w_p == 0
        strips = tensor.split(self.h_p, dim=-2)
        patches = [strip.split(self.w_p, dim=-1) for strip in strips]
        return patches

    def get_positive_tokens(
            self,
            reference_tokens: torch.Tensor,
            query_tokens: torch.Tensor,
            mapping: torch.Tensor,
    ):
        """
        DEPRECATED!!!!! TOO SLOW!!!!!
        """
        assert len(reference_tokens.size()) == 2, len(query_tokens.size()) == 2
        mapping = mapping.permute((2, 0, 1))
        h_m, w_m = mapping.size(-2), mapping.size(-1)
        # patchify mapping
        patchified_mapping = self.patchify(mapping)
        flattened_mapping = list(chain(*patchified_mapping))
        # flatten mapping
        positives, weights = [], []
        for patch_mapping in flattened_mapping:
            y, x = patch_mapping[0], patch_mapping[1]
            # scale image-level indices to patch-level indices
            patch_y = torch.div(y, self.h_p, rounding_mode="floor")
            patch_x = torch.div(x, self.w_p, rounding_mode="floor")
            indices = patch_y * (w_m // self.w_p) + patch_x
            # counting patch-level indices
            indices[indices < 0] = -1
            token_indices, counts = indices.unique(sorted=True, return_counts=True)
            token_indices = token_indices.long()
            # assign _weights
            if token_indices[0] < 0:
                positives.append(reference_tokens[token_indices[1:]])
                weights.append(counts[1:].div(self.h_p * self.w_p))
            else:
                positives.append(reference_tokens[token_indices])
                weights.append(counts.div(self.h_p * self.w_p))

        return positives, weights

    def convert_batch_mapping_to_patch_indices(self, batch_mapping: torch.Tensor):
        assert len(batch_mapping.size()) == 4
        h_m, w_m = batch_mapping.size(-2), batch_mapping.size(-1)
        # scale image-level indices to patch-level indices
        y, x = batch_mapping[:, 0], batch_mapping[:, 1]
        patch_y = torch.div(y, self.h_p, rounding_mode="floor")
        patch_x = torch.div(x, self.w_p, rounding_mode="floor")
        indices = patch_y * (w_m // self.w_p) + patch_x
        # patchify indices
        patchified = self.patchify(indices)
        indices = list(chain(*patchified))
        # indices: [L, B, patch_h=16, patch_w=16]
        indices = torch.stack(indices)
        indices = indices.flatten(start_dim=2).permute((1, 0, 2))

        return indices


@register_loss("copynce")
class CopyNCELoss(BaseLoss):
    def __init__(self, cfg, model):
        super(CopyNCELoss, self).__init__(cfg, model)
        self.cfg = cfg
        self.scale = cfg.scale
        self.input_mode = cfg.input_mode
        self.copynce_heads, self.heads = list(map(int, cfg.heads.strip().split(":")))

        self.patch_size = to_2tuple(cfg.patch_size)
        self.patch_manager = PatchManager(self.patch_size)

        self.gamma = getattr(self.cfg, "gamma", 1.)

        logger.info(self)

    def __repr__(self):
        return f"Build CopyNCELoss with " \
               f"scale={self.scale}, " \
               f"weight={self.weight}, " \
               f"patch_size={self.patch_size}, " \
               f"copynce_heads:heads={self.copynce_heads}:{self.heads}, " \
               f"gamma={self.gamma}, " \
               f"cfg={self.cfg}."

    def calculate(self, outputs, inputs):
        queries, references = self.fetch_tokens(outputs)
        que_to_ori_mappings = inputs["que_to_ori_mappings"]
        ref_to_ori_mappings = inputs["ref_to_ori_mappings"]
        copynce_flags = inputs["copynce"]

        ori_to_ref_mappings = MaskMapper.generate_batch_inverse_mappings(ref_to_ori_mappings)
        que_to_ref_mappings = MaskMapper.generate_batch_transferred_mapping(que_to_ori_mappings, ori_to_ref_mappings)
        ref_to_que_mappings = MaskMapper.generate_batch_inverse_mappings(que_to_ref_mappings)

        copynce_qr = self.copynce(
            queries,
            references,
            que_to_ref_mappings,
        )
        copynce_rq = self.copynce(
            references,
            queries,
            ref_to_que_mappings,
        )

        copynce = self.cfg.lamb * copynce_qr + (1. - self.cfg.lamb) * copynce_rq
        if torch.any(copynce_flags):
            copynce = (copynce_flags * copynce).sum() / copynce_flags.sum()
        else:
            # calculate dummy loss with result of forward graph to avoid the case that not all parameters
            # that used in forward process are back-propagated.
            copynce = 0. * (copynce_flags * copynce).sum()

        return copynce

    def fetch_tokens(self, outputs):
        if self.input_mode == "fusion_head":
            queries = outputs["fusion_head_que_patch_tokens"]
            references = outputs["fusion_head_ref_patch_tokens"]
        elif self.input_mode == "descriptor":
            layer = getattr(self.cfg, "layer", 12)
            queries = outputs["que_encoder_output"]["block_outputs"][layer][:, 1:]
            references = outputs["ref_encoder_output"]["block_outputs"][layer][:, 1:]
        elif self.input_mode == "fusion":
            queries = outputs["fused_que_patch_tokens"]
            references = outputs["fused_ref_patch_tokens"]
        elif self.input_mode == "projected_descriptor":
            queries = outputs["projected_que_patch_tokens"]
            references = outputs["projected_ref_patch_tokens"]
        elif self.input_mode == "cnn":
            queries = outputs["que_patch_tokens"]
            references = outputs["ref_patch_tokens"]
        else:
            raise ValueError(f"Invalid input mode: {self.input_mode}.")

        dims = queries.size(-1)
        copynce_dims = self.copynce_heads * (dims // self.heads)
        queries, references = queries[..., : copynce_dims], references[..., : copynce_dims]

        return queries, references

    def copynce(self, queries, references, que_to_ref_mappings):
        # gamma-based CopyNCE
        queries, references = to_1d(queries), to_1d(references)

        # batch_scaled_cosines: [B, L, L]
        batch_scaled_cosines = self.scale * self.batch_cosine_similarity(queries, references)
        B, L, _ = batch_scaled_cosines.size()
        # denominators: [B, L, 1]
        denominators = batch_scaled_cosines.exp().sum(dim=-1, keepdim=True)
        # batch_logits: [B, L, L]
        batch_logits = batch_scaled_cosines.exp()
        # dummy_cosines are prepared for invalid patch index, i.e., -1
        dummy_cosines = self.scale * torch.ones(
            (B, L, 1),
            dtype=batch_scaled_cosines.dtype,
            device=batch_scaled_cosines.device,
        )
        # batch_scaled_cosines: [B, L, L + 1]
        batch_scaled_cosines = torch.cat([batch_scaled_cosines, dummy_cosines], dim=2)

        # batch_patch_indices: [B, L, hxw=256]
        batch_patch_indices = self.patch_manager.convert_batch_mapping_to_patch_indices(
            que_to_ref_mappings
        ).to(torch.long)
        hw = batch_patch_indices.size(-1)
        batch_patch_indices[batch_patch_indices < 0] = batch_scaled_cosines.size(-1) - 1

        # bincount
        bias = torch.arange(B * L, device=batch_logits.device) * batch_scaled_cosines.size(-1)
        bias = bias.view(B, L, 1).repeat(1, 1, hw)
        biased_batch_patch_indices = batch_patch_indices + bias
        biased_batch_patch_indices = biased_batch_patch_indices.contiguous().view(-1)
        # the last is dummy number to make sure the max number is B * L * (L + 1)
        biased_batch_patch_indices = torch.cat(
            [
                biased_batch_patch_indices,
                torch.tensor(
                    [B * L * batch_scaled_cosines.size(-1)],
                    dtype=biased_batch_patch_indices.dtype,
                    device=biased_batch_patch_indices.device,
                )
            ],
            dim=0,
        )
        indices_counts = torch.bincount(biased_batch_patch_indices)[: -1]
        # indices_counts: [B, L, L + 1]
        indices_counts = indices_counts.contiguous().view(B, L, -1)
        assert indices_counts.size(-1) == batch_scaled_cosines.size(-1)

        # weights: [B, L, L + 1]
        weights = indices_counts / hw
        weights = torch.pow(weights, self.gamma)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights[..., :-1].detach()

        loss = weights * -torch.log(batch_logits / denominators)
        loss = loss.sum(dim=-1).mean(dim=-1)
        return loss

    @staticmethod
    def batch_cosine_similarity(a, b):
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
        return torch.bmm(a, b.permute(dims=(0, 2, 1)))

    @staticmethod
    def batch_opt_kl_div(a, b):
        """
        Args:
            a: [B, L, C]
            b: [B, L, C]
        Returns:
            dino_loss: [B, L, L]
        """
        a = F.softmax(a, dim=-1)
        b = F.log_softmax(b, dim=-1)
        opt_kl_div = -torch.bmm(a, b.permute(dims=(0, 2, 1)))
        return opt_kl_div

    @staticmethod
    def cross_image_tokens_cosine_similarity(ques, refs):
        B, L, C = ques.size()
        ques = F.normalize(ques, p=2, dim=-1)
        refs = F.normalize(refs, p=2, dim=-1)
        # ref: [L x C ... x B]
        refs = list(refs.unbind(dim=0))
        global_refs = []
        for i in range(B):
            refs[0], refs[i] = refs[i], refs[0]
            # global_ref: [BL x C]
            global_ref = torch.cat(refs, dim=0)
            global_refs.append(global_ref)
        # global_refs: [B, BL x C]
        global_refs = torch.stack(global_refs)
        return torch.bmm(ques, global_refs.permute(dims=(0, 2, 1)))

    def update_center(self, tokens):
        tokens = tokens.detach()
        B, N, C = tokens.size()
        batch_token_center = tokens.view(B * N, C)
        batch_token_center = torch.sum(batch_token_center, dim=0, keepdim=True)
        dist.all_reduce(batch_token_center)
        batch_token_center = batch_token_center / (B * N * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_token_center * (1 - self.center_momentum)
