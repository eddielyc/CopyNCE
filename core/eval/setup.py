# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
from loguru import logger

from omegaconf import OmegaConf

from core.configs import __default_config_factory__
import core.distributed as distributed
from core.logging import setup_logging


def parse_args(args):
    default_cfg = OmegaConf.create(__default_config_factory__[getattr(args, "arch", "default")])
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))

    return cfg


def setup(args, verbose=True):
    cfg = parse_args(args)
    setup_logging(cfg.output_dir if verbose else None)
    logger.info(OmegaConf.to_yaml(cfg))
    if verbose:
        saved_cfg_path = os.path.join(cfg.output_dir, "config.yaml")
        with open(saved_cfg_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)

    cudnn.benchmark = True
    distributed.enable(overwrite=True)

    return cfg
