
import os
import json
import argparse
from pathlib import Path
from typing import Optional, List
from loguru import logger

import torch
from torch.nn import functional as F

from core import distributed
from core.models import build_paired_model_for_eval
from core.data.build import build_data_for_matching_eval
from core.eval.setup import setup
from core.eval.utils import extract_features_with_dataloader, cls_postprocessor
from core.run.eval.inference_score_from_cache import inference
from core.run.eval.measure_cls import measure_cls


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
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
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


def eval_cls(model, cfg, epoch=None):
    dataset, data_loader = build_data_for_matching_eval(cfg)
    outputs = extract_features_with_dataloader(model, data_loader, postprocessor=cls_postprocessor)
    if distributed.is_main_process():
        scores = inference(dataset, outputs)
        with open(Path(cfg.output_dir) / f'scores{f"_ep-{epoch}" if epoch else ""}.json', 'w') as file:
            json.dump(scores, file)
        measure_cls(scores, cfg, epoch)


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.arch = "matching"
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"output_dir={args.output_dir}", f"weights={args.weights}"]
    cfg = setup(args)

    model, epoch = build_paired_model_for_eval(cfg)
    eval_cls(model, cfg, epoch)


if __name__ == '__main__':
    main()
