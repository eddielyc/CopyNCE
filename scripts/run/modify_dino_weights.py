from pathlib import Path
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Extract encoder and fusion layers from DINO ViT checkpoint."
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=12,
        help="Total number of transformer blocks (default: 12)"
    )
    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=8,
        help="Number of blocks to assign to encoder (default: 8)"
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="weights/dino_deitsmall16_pretrain.pth",
        help="Path to the original DINO checkpoint"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for the new checkpoint. If not provided, auto-generated based on layer config."
    )
    args = parser.parse_args()

    layers = args.layers
    encoder_layers = args.encoder_layers
    if encoder_layers > layers:
        raise ValueError(f"--encoder-layers ({encoder_layers}) cannot be greater than --layers ({layers})")

    fusion_layers = layers - encoder_layers

    dino_weights_path = Path(args.weights_path)
    if not dino_weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {dino_weights_path}")

    ckpt = torch.load(dino_weights_path, map_location="cpu")  # 加载到 CPU 避免 GPU 内存问题

    encoder_layers_indices = [str(i) for i in range(encoder_layers)]
    fusion_layers_indices = [str(i + encoder_layers) for i in range(fusion_layers)]
    _ckpt = {}

    for key, tensor in ckpt.items():
        if (
            any(f"blocks.{index}." in key for index in encoder_layers_indices)
            or "pos_embed" in key
            or "cls_token" in key
            or "patch_embed" in key
        ):
            _ckpt[f"encoder.{key}"] = tensor
        elif (
            any(f"blocks.{index}." in key for index in fusion_layers_indices)
            or "norm." in key
        ):
            if "norm." in key:
                _key = f"fusion.{key}"
            else:
                parts = key.split('.')
                parts[1] = str(int(parts[1]) - encoder_layers)
                _key = f"fusion.{'.'.join(parts)}"
            _ckpt[_key] = tensor

    _ckpt = {"model": _ckpt}

    output_path = args.output_path
    if output_path is None:
        output_path = f"dino_vits16_enc-{encoder_layers}_fus-{fusion_layers}.pth"

    torch.save(_ckpt, output_path)
    print(f"Saved processed checkpoint to: {output_path}")


if __name__ == '__main__':
    main()