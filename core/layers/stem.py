from typing import Union, Tuple, Optional, Callable

from loguru import logger

import torch
import torch.nn as nn

from timm.models.layers import to_2tuple
from torch import Tensor


class ConvStem(nn.Module):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        logger.info(self)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    def __repr__(self):
        return f"Build ConvStem: in_chans: {self.in_chans}, " \
               f"embed_dim: {self.embed_dim}, " \
               f"norm_layer: {self.norm_layer}, " \
               f"flatten: {self.flatten}"


class hMLPStem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=nn.SyncBatchNorm
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=4, stride=4),
            norm_layer(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
            norm_layer(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
            norm_layer(embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = to_2tuple(img_size)
        patch_HW = to_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
