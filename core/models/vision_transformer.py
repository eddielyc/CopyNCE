# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import math
# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)

from loguru import logger

import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F

from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models import register_model

from core.layers import (
    Attention,
    Block, BlockParalx2, LayerScaleInitBlockParalx2, LayerScaleInitBlock,
    ConvStem, hMLPStem,
)
from core.models.model_utils import init_weights


class VisionTransformer(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            block_layers=Block,
            patch_layer=PatchEmbed,
            act_layer=nn.GELU,
            attention_block=Attention,
            mlp_block=Mlp,
            init_scale=1e-4,
            last_norm=False,
            **kwargs
    ):
        super().__init__()

        self.dropout_rate = drop_rate

        self.num_features = self.embed_dim = embed_dim
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.last_norm = last_norm

        self.patch_embed = patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.n_cls_tokens = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, attention_block=attention_block, mlp_block=mlp_block, init_values=init_scale)
            for i in range(depth)
        )

        if self.last_norm:
            self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.num_tokens = self.num_patches + self.n_cls_tokens

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(init_weights)

        logger.info(f"Build Vision Transformer.")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def split_tokens(self, tokens):
        cls_tokens, patch_tokens = tokens[:, :self.n_cls_tokens], tokens[:, self.n_cls_tokens:]
        if self.n_cls_tokens == 1:
            cls_tokens = cls_tokens.squeeze(dim=1)
        return cls_tokens, patch_tokens

    def get_num_layers(self):
        return len(self.blocks)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):

        x = self.prepare_tokens(x)
        block_outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            block_outputs.append(x)

        if self.last_norm:
            x = self.norm(x)

        cls_token, patch_tokens = self.split_tokens(x)

        return {
            "block_outputs": block_outputs,
            "tokens": x,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }


if __name__ == '__main__':
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    ckpt = torch.load("dino_deitsmall16_pretrain.pth")
    ckpt.pop("mask_token")
    model.load_state_dict(ckpt)

    input = torch.rand(8, 3, 224, 224, dtype=torch.float)
    output = model(input)

