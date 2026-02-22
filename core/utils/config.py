# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
from loguru import logger
import os

from omegaconf import OmegaConf

import core.distributed as distributed
from core.utils import utils
from core.configs import __default_config_factory__


def apply_scaling_rules_to_cfg(cfg):  # to fix
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"output_dir={args.output_dir}", f"arch={getattr(args, 'arch', 'default')}"]
    default_cfg = OmegaConf.create(__default_config_factory__[args.arch])
    logger.info(f"Load default {args.arch} config.")
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def default_setup(args):
    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    default_setup(args)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg
