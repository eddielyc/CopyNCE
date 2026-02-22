
import argparse
import os
from pathlib import Path
from typing import Optional, List
from functools import partial

import torch
from loguru import logger

from core import distributed
from core.models import build as model_build
from core.data import build as data_build
from core.eval.setup import setup
from core.eval.utils import extract_features_with_dataloader


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="isc",
        help="The dataset which acts as input. Reference to 'core/data/data_paths.py' for dataset name.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="features.pth",
        help="Path of output features",
    )
    parser.add_argument(
        "opts",
        help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs.
    For python-based LazyConfig, use "path.key=value".
            """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def descriptor_preprocessor(input):
    input["images"] = input["images"].cuda()
    return input


def descriptor_postprocessor(input, output, key="feat"):
    return {
        "indices": input["indices"].cuda(),
        "features": output[key].detach(),
    }


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.arch = "descriptor"
    output_path = Path(args.output_path)
    args.opts += [
        f"output_path={output_path}",
        f"output_dir={output_path.parent}",
        f"weights={args.weights}",
        f"dataset={args.dataset}",
    ]
    cfg = setup(args, verbose=False)

    model, epoch = model_build.build_dino_model_for_eval(cfg)
    dataset, data_loader = data_build.build_unpaired_data_for_eval(cfg)
    outputs = extract_features_with_dataloader(
        model,
        data_loader,
        descriptor_preprocessor,
        partial(descriptor_postprocessor, key="cls_token")
    )

    if distributed.is_main_process():
        cache_path = Path(cfg.output_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving features into {cache_path}.")
        torch.save(outputs, cache_path)


if __name__ == '__main__':
    main()
