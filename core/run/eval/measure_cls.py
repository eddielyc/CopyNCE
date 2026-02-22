import os
import argparse
import json
from pathlib import Path
from typing import Optional, List
from loguru import logger
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np

from core.eval.setup import setup
from core.logging import setup_logging
from core.metrics.build import build_metric


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
        "--result-file",
        type=str,
        help="Prediction json file",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--fars",
        default=[0.01, 0.001, 0.0001, 0.00001],
        nargs='+',
        type=float,
        help="False accept rate",
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


def measure_cls(scores, cfg, epoch=None):
    gt = dict()
    for anno_file in cfg.eval.anno_file:
        logger.info(f"=> Load GT from: {anno_file}")
        with open(anno_file, 'r') as file:
            gt.update(json.load(file))

    for metric_cfg in cfg.eval.metrics:
        metric_type = metric_cfg.pop("type")
        metric = build_metric(metric_type, cfg.output_dir, epoch)
        metric(scores, gt, **metric_cfg)


def main():
    parser = get_args_parser(description="Evaluate the performance of copy detection from cls view.")
    args = parser.parse_args()
    # args.arch = "matching"
    args.arch = "descriptor"
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"output_dir={args.output_dir}", f"eval.fars={args.fars}", f"result_file={args.result_file}"]
    cfg = setup(args, verbose=False)

    with open(cfg.result_file, 'r') as file:
        scores = json.load(file)
    measure_cls(scores, cfg)


if __name__ == '__main__':
    main()
