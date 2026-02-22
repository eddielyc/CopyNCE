
from pathlib import Path

import torch
from torch import nn


def main():
    grid_h0, grid_w0 = 14, 14
    grid_h, grid_w = 21, 21

    ckpt_path = Path("weights/dino_vits16_enc-8_fus-4.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pos_embed = ckpt["model"]["encoder.pos_embed"]

    assert grid_w0 * grid_h0 + 1 == pos_embed.shape[1], \
        f"Wrong grid size of original model ({grid_w0} x {grid_h0} + 1 != {pos_embed.shape[1]})"

    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = pos_embed.shape[-1]

    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, grid_h0, grid_w0, dim).permute(0, 3, 1, 2),
        size=(grid_h, grid_w),
        mode='bicubic',
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    patch_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    ckpt["model"]["encoder.pos_embed"] = patch_pos_embed
    torch.save(ckpt, ckpt_path.parent / f"{ckpt_path.stem.split('.')[0]}_pos-emb-{grid_h}x{grid_w}.pth.tar")


if __name__ == '__main__':
    main()
