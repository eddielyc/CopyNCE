import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from loguru import logger

from core.eval.setup import setup
from core.data.build import build_data_for_matching_eval


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
        "--cache",
        type=str,
        help="Path of cache file",
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


def inference(dataset, outputs):
    logger.info(f"Launch inference scores with cache.")

    que_to_candidate_ref = defaultdict(list)
    que_to_candidate_preds = defaultdict(list)

    preds = outputs["preds"].cpu().numpy()

    for pred, (que, ref) in zip(preds, dataset.uuid_pairs):
        que_to_candidate_ref[que].append(ref)
        que_to_candidate_preds[que].append(pred.item())

    scores = []
    for que, preds in que_to_candidate_preds.items():
        sorted_indices = np.argsort(-np.array(preds))
        scores.append([que, [[que_to_candidate_ref[que][index], preds[index]] for index in sorted_indices]])

    return scores


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.arch = "matching"
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"output_dir={args.output_dir}", f"cache={args.cache}"]
    cfg = setup(args)

    dataset, _ = build_data_for_matching_eval(cfg)
    logger.info(f"Loading cache from {cfg.cache}.")
    cache = torch.load(cfg.cache)
    scores = inference(dataset, cache)

    logger.info(f"Dump results into JSON file...")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.output_dir) / f"score_results.json", 'w') as f:
        json.dump(scores, f)


if __name__ == '__main__':
    main()
