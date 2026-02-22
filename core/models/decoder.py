from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.vision_transformer import Mlp

from core.layers.block import Block
from core.layers.attention import Attention
from core.models.model_utils import init_weights


class Linear(nn.Module):
    def __init__(self, num_cls, img_size, patch_size, encoder_dim):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.num_cls = num_cls

        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])

        self.head = nn.Linear(self.encoder_dim, num_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x):
        masks = self.head(x)
        masks = rearrange(masks, "b (h w) c -> b c h w", h=self.grid_size[0])
        masks = F.interpolate(masks, size=self.img_size, mode="bilinear")

        return {
            "masks": masks,
        }


class MaskTransformer(nn.Module):
    def __init__(
            self,
            num_cls,
            img_size,
            patch_size,
            encoder_dim,
            # block args
            block_dim,
            num_layers,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            attention_block=Attention,
            mlp_block=Mlp,
            scale=1e-4,
            **kwargs,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.encoder_dim = encoder_dim
        self.scale = scale

        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])

        self.block_dim = block_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_layers)]
        self.blocks = nn.ModuleList(
            Block(
                dim=block_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                attention_block=attention_block,
                mlp_block=mlp_block,
            )
            for i in range(self.num_layers)
        )

        self.cls_emb = nn.Parameter(torch.randn(1, self.num_cls, self.block_dim))
        self.proj_dec = nn.Linear(self.encoder_dim, self.block_dim)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(self.block_dim, self.block_dim))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(self.block_dim, self.block_dim))

        self.decoder_norm = nn.LayerNorm(self.block_dim, eps=1e-6)
        self.mask_norm = nn.LayerNorm(self.num_cls, eps=1e-6)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

        logger.info(f"Build MaskTransformer as the segmentation head.")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float)
    def forward(self, x):
        x = self.proj_dec(x)
        # [bs, num_patch, block_dim]
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((cls_emb, x), 1)

        block_outputs = []
        for blk in self.blocks:
            x = blk(x)
            block_outputs.append(x)

        x = self.decoder_norm(x)

        cls_tokens, patch_tokens = x[:, :self.num_cls], x[:, self.num_cls:]
        patch_tokens = patch_tokens @ self.proj_patch
        cls_tokens = cls_tokens @ self.proj_classes

        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
        cls_tokens = F.normalize(cls_tokens, p=2, dim=-1)

        masks = patch_tokens @ cls_tokens.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=self.grid_size[0])
        masks = F.interpolate(masks, size=self.img_size, mode="bilinear")

        return {
            "masks": masks,
            "block_outputs": block_outputs,
            "tokens": x,
            "cls_tokens": cls_tokens,
            "patch_tokens": patch_tokens,
        }
