import datetime
import time
from copy import deepcopy
from functools import partial

import math
from loguru import logger
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler

import core.distributed as distributed
from core.data.samplers import EpochSampler
from core.logging import MetricLogger
from core.loss import build as loss_build
from core.models import build as model_build
from core.data import build as data_build
from core.utils.utils import has_batchnorms


class CopyMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16 = cfg.train.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.fp16)
        self.device = torch.device(distributed.get_local_rank())
        self.model = self.model_wo_ddp = self.build_model(cfg)
        self.data_loader = self.build_data_loader(cfg)
        self.criterion = self.build_criterion(cfg, self.model_wo_ddp)

        self.prepare_for_distributed_training()

        self.optimizer = create_optimizer_v2(
            self.model_wo_ddp,
            opt="adamw",
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
            betas=self.cfg.optim.betas,
        )
        logger.info(f"TRAIN -- Build AdamW optimizer.")

        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.cfg.optim.epochs * len(self.data_loader),
            lr_min=self.cfg.optim.min_lr,
            cycle_mul=1,
            cycle_decay=1.,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_t=self.cfg.optim.warmup_epochs * len(self.data_loader),
            warmup_lr_init=0.01 * self.cfg.optim.lr,
            warmup_prefix=True,
        )
        logger.info(f"TRAIN -- Build CosineLRScheduler.")

        metrics_file = Path(cfg.output_dir) / "training_metrics.json"
        self.metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
        self.n_steps = 0

        logger.info(f"TRAIN -- Build {self.__class__.__name__}.")

    def build_model(self, cfg):
        raise NotImplementedError

    def build_data_loader(self, cfg):
        raise NotImplementedError

    def build_criterion(self, cfg, model):
        raise NotImplementedError

    def forward(self, data):
        data = self.parse_data(data)
        with torch.cuda.amp.autocast(enabled=self.fp16):
            output = self.model(data)
            output = self.post_process(output, data)
            loss_dict = self.criterion(output, data)
        return loss_dict

    def backward(self, loss_dict):
        loss = sum(loss_dict.values())
        if math.isnan(loss):
            logger.info(f"NaN detected: {loss_dict}")
            for name, param in self.model_wo_ddp.named_parameters():
                if torch.isnan(param).any():
                    logger.info(f"NaN detected in {name}")

        self.scheduler.step_update(self.n_steps)
        self.n_steps += 1

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        if self.cfg.optim.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.optim.clip_grad)

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED -- preparing model for distributed training")
        if has_batchnorms(self.model):
            self.model_wo_ddp = nn.SyncBatchNorm.convert_sync_batchnorm(self.model_wo_ddp)
            logger.info("DISTRIBUTED -- convert BatchNorm to SyncBatchNorm")
        self.model_wo_ddp = self.model_wo_ddp.to(self.device)
        self.model = DistributedDataParallel(
            self.model_wo_ddp,
            device_ids=[distributed.get_local_rank()],
            broadcast_buffers=False,
        )

    def train_epoch(self, epoch):
        self.before_epoch(epoch)
        try:
            self.train_in_epoch(epoch)
        except Exception:
            raise
        finally:
            self.after_epoch(epoch)

    def before_epoch(self, epoch):
        self.n_steps = (epoch - 1) * len(self.data_loader)

        if isinstance(self.data_loader.sampler, EpochSampler):
            self.data_loader.sampler.set_epoch(epoch)

        self.train()

    def train_in_epoch(self, epoch):
        logger.info(f"Start training epoch {epoch}.")
        start_timestamp = time.time()
        for data in self.metric_logger.log_every(
            self.data_loader,
            self.cfg.train.log_freq,
            header=f"Training Epoch: {epoch}",
        ):

            loss_dict = self.forward(data)
            self.backward(loss_dict)

            # logging
            if distributed.get_global_size() > 1:
                for v in loss_dict.values():
                    torch.distributed.all_reduce(v)
            loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

            if math.isnan(sum(loss_dict_reduced.values())):
                logger.info(f"NaN detected: {loss_dict_reduced}")
                raise AssertionError
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            lr = self.scheduler._get_values(self.n_steps, on_epoch=False)
            lr = lr[0] if isinstance(lr, (tuple, list)) else lr
            self.metric_logger.update(lr=lr)
            self.metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        finish_timestamp = time.time()
        logger.info(f"Finish training epoch {epoch} "
                    f"in {datetime.timedelta(seconds=finish_timestamp - start_timestamp)}")

    def after_epoch(self, epoch):
        self.eval()

    def parse_data(self, data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
        return data

    def post_process(self, output, input):
        # output["que_masks"] = output["que_masks"][:, 1]
        # output["ref_masks"] = output["ref_masks"][:, 1]
        return output


class PairedCopyMetaArch(CopyMetaArch):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"Build PairCopyMetaArch.")

    def build_model(self, cfg):
        return model_build.build_paired_copy_detector(cfg)

    def build_data_loader(self, cfg):
        return data_build.build_data_loader(cfg)

    def build_criterion(self, cfg, model):
        return loss_build.build_loss(cfg, model)


class NPCopyMetaArch(CopyMetaArch):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"Build NPCopyMetaArch.")

    def build_model(self, cfg):
        return model_build.build_paired_copy_detector(cfg)

    def build_data_loader(self, cfg):
        return data_build.build_data_loader(cfg)

    def build_criterion(self, cfg, model):
        return loss_build.build_loss(cfg, model)


class CopyDescriptorMetaArch(CopyMetaArch):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"Build CopyDescriptor.")

    def build_model(self, cfg):
        if getattr(cfg.model, "type", "vit") == "vit":
            return model_build.build_copy_descriptor(cfg)
        elif getattr(cfg.model, "type", "vit") == "cnn":
            return model_build.build_copy_cnn_descriptor(cfg)

    def build_data_loader(self, cfg):
        return data_build.build_data_loader(cfg)

    def build_criterion(self, cfg, model):
        return loss_build.build_loss(cfg, model)


class CopyCNNDescriptorMetaArch(CopyDescriptorMetaArch):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"Build CopyCNNDescriptor.")

    def build_model(self, cfg):
        return model_build.build_copy_cnn_descriptor(cfg)


class ParallelCopyMetaArch(nn.Module):
    def __init__(self, *meta_archs):
        super().__init__()
        self.meta_archs = meta_archs
        self.main_meta_arch = self.meta_archs[0]
        self.data_loader = self.main_meta_arch.data_loader
        for meta_arch in self.meta_archs[1:]:
            meta_arch.data_loader = self.data_loader
        logger.info(f"TRAIN -- Using MetaArch: {self.main_meta_arch.__class__.__name__}'s data_loader as shared "
                    f"data_loader.")

        logger.info(f"TRAIN -- Build {self.__class__.__name__} with {len(self.meta_archs)} MetaArchs.")

    def train_epoch(self, epoch):
        # set data_loader sampler epoch will be done in MetaArch.before_epoch function.
        self.before_epoch(epoch)
        try:
            self.train_in_epoch(epoch)
        except Exception:
            raise
        finally:
            self.after_epoch(epoch)

    def train_in_epoch(self, epoch):
        logger.info(f"Start training epoch {epoch}.")
        start_timestamp = time.time()

        # build dummy metric_logger.log_every for other meta_archs
        dummy_log_every_iters = [
            meta_arch.metric_logger.log_every(
                range(len(self.data_loader)),
                meta_arch.cfg.train.log_freq,
                header=f"MetaArch #{i} Training Epoch: {epoch}",
            ) for i, meta_arch in enumerate(self.meta_archs[1:], 1)
        ]

        for data in self.main_meta_arch.metric_logger.log_every(
            self.data_loader,
            self.main_meta_arch.cfg.train.log_freq,
            header=f"MetaArch #0 Training Epoch: {epoch}",
        ):
            # iter through dummy_log_every_iters to maintain log in other meta_archs
            for dummy_log_every in dummy_log_every_iters:
                next(dummy_log_every)

            for meta_arch in self.meta_archs:
                _data = deepcopy(data)
                loss_dict = meta_arch.forward(data)
                meta_arch.backward(loss_dict)

                # logging
                if distributed.get_global_size() > 1:
                    for v in loss_dict.values():
                        torch.distributed.all_reduce(v)
                loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

                if math.isnan(sum(loss_dict_reduced.values())):
                    logger.info(f"NaN detected: {loss_dict_reduced}")
                    raise AssertionError
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                lr = meta_arch.scheduler._get_values(meta_arch.n_steps, on_epoch=False)
                lr = lr[0] if isinstance(lr, (tuple, list)) else lr
                meta_arch.metric_logger.update(lr=lr)
                meta_arch.metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        finish_timestamp = time.time()
        logger.info(f"Finish training epoch {epoch} "
                    f"in {datetime.timedelta(seconds=finish_timestamp - start_timestamp)}")

    def before_epoch(self, epoch):
        for meta_arch in self.meta_archs:
            meta_arch.before_epoch(epoch)

    def after_epoch(self, epoch):
        for meta_arch in self.meta_archs:
            meta_arch.after_epoch(epoch)

    def __iter__(self):
        for meta_arch in self.meta_archs:
            yield meta_arch
