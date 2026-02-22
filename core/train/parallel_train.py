from copy import deepcopy
from pathlib import Path

from loguru import logger
import os

import core.distributed as distributed
from core.logging import setup_logging
from core.utils.config import setup
from core.utils.serialization import Checkpointer
from core.train.copy_meta_archs import (
    PairedCopyMetaArch,
    CopyDescriptorMetaArch,
    CopyCNNDescriptorMetaArch,
    ParallelCopyMetaArch,
)
from core.utils.zipper import Zipper


__arch_factory__ = {
    "matching": PairedCopyMetaArch,
    "descriptor": CopyDescriptorMetaArch,
    "cnn_descriptor": CopyCNNDescriptorMetaArch,
}


def do_train(cfgs, models):
    checkpointers = [
        Checkpointer(
            distributed.get_local_rank(),
            cfg.output_dir,
            milestone_every=cfg.train.save_freq,
            model=model.model_wo_ddp,
        ) for cfg, model in zip(cfgs, models)
    ]

    # Note that: Parameters must be loaded after model and criterion are created,
    # because criterion may register parameters to models,
    # and pretrained checkpoint may contain these parameters.
    last_epoch = 0
    for i, (cfg, checkpointer) in enumerate(zip(cfgs, checkpointers)):
        if cfg.train.pretrain and Path(cfg.train.pretrain).is_file():
            checkpointer.load(cfg.train.pretrain)
            logger.info(f"=> MetaArch #{i} Load pretrained parameters from {cfg.train.pretrain}.")

        if cfg.train.resume:
            last_epoch = checkpointer.load()
            logger.warning(
                f"=> MetaArch #{i} detects resume flag."
                f"However, parallel_training MUST go in parallel, so make sure that all tasks share the same progress."
                f"The checkpoint will be loaded (if there is), the train process will start at last_epoch loaded from 'Task # {len(cfgs) - 1}'."
            )

    for epoch in range(last_epoch + 1, cfgs[0].optim.epochs + 1):
        models.train_epoch(epoch)
        for checkpointer in checkpointers:
            checkpointer.step(epoch)


def main(args):
    cfgs = before_train(args)
    models = [__arch_factory__[cfg.arch](cfg) for cfg in cfgs]
    models = ParallelCopyMetaArch(*models)
    logger.info("Parallel Meta arch loaded...")
    do_train(cfgs, models)


def before_train(args):
    if (
        "LOCAL_RANK" not in os.environ
        or "MASTER_PORT" not in os.environ
        or "RANK" not in os.environ
        or "WORLD_SIZE" not in os.environ
        or "LOCAL_WORLD_SIZE" not in os.environ
    ):
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_PORT"] = "12345"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"

    assert isinstance(args.config_file, list) and len(args.config_file) > 1,\
        "config_file must be a list and has at least 2 elements, otherwise please use run/train/train.py."
    assert len(args.config_file) == len(args.output_dir), \
        "Each config_file should have its output_dir correspondingly."

    for i, output_dir in enumerate(args.output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if int(os.environ["LOCAL_RANK"]) == 0:
            zipper = Zipper(zippath=Path(output_dir) / "snapshot.zip")
            zipper.zip()
        setup_logging(
            output_dir,
            incremental=True if i > 0 else False,
            cmd=True if i == 0 else False,
        )

    cfgs = []
    for i, (config_file, output_dir) in enumerate(zip(args.config_file, args.output_dir)):
        _args = deepcopy(args)
        _args.opts += [f"meta_arch_number={i}"]
        _args.config_file = config_file
        _args.output_dir = output_dir
        cfgs.append(setup(_args))

    return cfgs
