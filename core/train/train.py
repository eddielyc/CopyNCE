from pathlib import Path

from loguru import logger
import os

import torch

import core.distributed as distributed
from core.logging import setup_logging
from core.utils.config import setup
from core.utils.serialization import Checkpointer
from core.train.copy_meta_archs import PairedCopyMetaArch, CopyDescriptorMetaArch, CopyCNNDescriptorMetaArch
from core.utils.zipper import Zipper
from core.run.eval.eval_cls import eval_cls


__arch_factory__ = {
    "matching": PairedCopyMetaArch,
    "descriptor": CopyDescriptorMetaArch,
    "cnn_descriptor": CopyCNNDescriptorMetaArch,
}


@torch.inference_mode()
def do_cls_test(cfg, model, epoch=None):
    model = model.model
    model.eval()
    eval_cls(model, cfg, epoch)


def do_train(cfg, model):
    checkpointer = Checkpointer(
        distributed.get_local_rank(),
        cfg.output_dir,
        milestone_every=cfg.train.save_freq,
        model=model.model_wo_ddp,
    )

    # Note that: Parameters must be loaded after model and criterion are created,
    # because criterion may register parameters to models,
    # and pretrained checkpoint may contain these parameters.
    if cfg.train.pretrain and Path(cfg.train.pretrain).is_file():
        checkpointer.load(cfg.train.pretrain)
        logger.info(f"=> Load pretrained parameters from {cfg.train.pretrain}.")

    last_epoch = 0
    if cfg.train.resume:
        last_epoch = checkpointer.load()

    for epoch in range(last_epoch + 1, cfg.optim.epochs + 1):
        model.train_epoch(epoch)
        checkpointer.step(epoch)
        if cfg.train.eval_freq > 0 and epoch % cfg.train.eval_freq == 0:
            do_cls_test(cfg, model, epoch)
            torch.cuda.synchronize()


def main(args):
    cfg = before_train(args)
    model = __arch_factory__[cfg.arch](cfg)
    logger.info("Meta arch loaded...")
    do_train(cfg, model)


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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if int(os.environ["LOCAL_RANK"]) == 0:
        zipper = Zipper(zippath=Path(args.output_dir) / "snapshot.zip")
        zipper.zip()

    setup_logging(args.output_dir)
    cfg = setup(args)

    return cfg
