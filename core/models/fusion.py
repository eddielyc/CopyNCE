
from loguru import logger

import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F

from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import to_2tuple, trunc_normal_
from core.models.model_utils import init_weights

from core.layers import (
    Attention,
    Block, BlockParalx2, LayerScaleInitBlockParalx2, LayerScaleInitBlock,
    ConvStem, hMLPStem,
)


class Fusion(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(
            self,
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
            act_layer=nn.GELU,
            attention_block=Attention,
            mlp_block=Mlp,
            init_scale=1e-4,
            cls_token=True,
            **kwargs
    ):
        super().__init__()
        self.dropout_rate = drop_rate
        self.num_features = self.embed_dim = embed_dim

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.n_cls_tokens = 1
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None
            self.n_cls_tokens = 0

        dpr = [drop_path_rate for _ in range(depth)]
        self.blocks = nn.ModuleList(
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, attention_block=attention_block, mlp_block=mlp_block, init_values=init_scale)
            for i in range(depth)
        )
        self.norm = norm_layer(embed_dim)

        self.apply(init_weights)

        logger.info(f"Build Fusion with{' cls_token' if cls_token else 'out cls_token'}.")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def split_tokens(self, tokens):
        if self.cls_token is not None:
            cls_tokens, patch_tokens = tokens[:, :self.n_cls_tokens], tokens[:, self.n_cls_tokens:]
            if self.n_cls_tokens == 1:
                cls_tokens = cls_tokens.squeeze(dim=1)
        else:
            cls_tokens, patch_tokens = None, tokens
        return cls_tokens, patch_tokens

    def prepare_input(self, x):
        if self.cls_token is not None:
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        return x

    def forward(self, x):
        x = self.prepare_input(x)

        block_outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            block_outputs.append(x)
        x = self.norm(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)

        cls_token, patch_tokens = self.split_tokens(x)

        return {
            "block_outputs": block_outputs,
            "tokens": x,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }
