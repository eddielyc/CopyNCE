from typing import TypeVar

import pandas as pd
from loguru import logger
from pathlib import Path

import torch
from torch.utils.data import Sampler

from .datasets import (
    CopyDataset,
    UnpairedCopyDataset,
    CopyDatasetMatchingEval,
    NPCopyDataset,
    JointDataset,
    LocalVerificationCopyDatasetMatchingEval,
    LocalVerificationCopyDatasetDescriptorEval,
    CopyDatasetDescriptorEval,
    OxfordParisDataset,
    PairedCopyDataset,
    SupervisedCopyDataset,
)
from .augmentation import build as aug_build

from .samplers import (
    InfiniteSampler,
    EpochSampler,
    ShardedInfiniteSampler,
    SamplerType,
)
from .collate_funcs import (
    np_collate_func,
    force_hard_negative_mining_collate_func,
    default_collate,
)
from .. import distributed


__supervised_datasets__ = {
    "isc_val",
}

__paths_factory__ = {}


def register_data_paths(data_name=""):
    def wrapper(fn):
        if data_name in __paths_factory__:
            logger.warning(
                f'Overwriting {data_name} in registry with {fn.__name__}. This is because the name being '
                'registered conflicts with an existing name. Please check if this is not expected.'
            )
        __paths_factory__[data_name] = fn
        return fn

    return wrapper


def build_data_paths(data_name, cfg):
    return __paths_factory__[data_name](cfg)


def build_collate_function(func_type):
    __factory__ = {
        "default": default_collate,
        "np": np_collate_func,
        "force_hard_negative_mining": force_hard_negative_mining_collate_func,
    }
    return __factory__[func_type]


def build_dataset(cfg):
    def _build_self_supervised_dataest(_cfg):
        paths = build_data_paths(_cfg.data_src, _cfg)
        basic_transform, train_transform = aug_build.build_transforms(_cfg, background_paths=paths)
        return CopyDataset(paths, basic_transform, train_transform, _cfg)

    def _build_supervised_dataset(_cfg):
        que_paths, ref_paths = build_data_paths(_cfg.data_src, _cfg)
        basic_transform, train_transform = aug_build.build_transforms(_cfg, background_paths=ref_paths)
        return SupervisedCopyDataset(que_paths, ref_paths, basic_transform, train_transform, _cfg)

    datasets = []
    for _cfg in cfg.input.dataset:
        if _cfg is None:
            continue
        if not hasattr(_cfg, "augmentation"):
            _cfg.augmentation = cfg.input.augmentation
        _cfg.img_size = cfg.input.img_size

        if _cfg.data_src in __supervised_datasets__:
            dataset = _build_supervised_dataset(_cfg)
        else:
            dataset = _build_self_supervised_dataest(_cfg)
        datasets.append(dataset)
    dataset = JointDataset(*datasets)

    logger.info(f"Build dataset with {len(dataset):,d} samples.")

    return dataset


def build_sampler(dataset, cfg):

    type = SamplerType[cfg.train.sampler.upper()]
    seed = cfg.train.seed
    size = -1
    advance = 0

    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=True,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=True,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=True,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def build_data_loader(cfg):
    dataset = build_dataset(cfg)
    sampler = build_sampler(dataset, cfg)
    collate_func = build_collate_function(getattr(cfg.train, "collate_func", "default"))

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_func,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader


def build_data_for_matching_eval(cfg):
    dataset = CopyDatasetMatchingEval(cfg)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=False,
        rank=distributed.get_local_rank(),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.eval.batch_size_per_gpu,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, data_loader


def build_data_for_descriptor_eval(cfg):
    que_dataset = CopyDatasetDescriptorEval(cfg, split="query")

    sampler = torch.utils.data.distributed.DistributedSampler(
        que_dataset,
        shuffle=False,
        rank=distributed.get_local_rank(),
    )
    que_data_loader = torch.utils.data.DataLoader(
        que_dataset,
        sampler=sampler,
        batch_size=cfg.eval.batch_size_per_gpu,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )

    ref_dataset = CopyDatasetDescriptorEval(cfg, split="reference")

    sampler = torch.utils.data.distributed.DistributedSampler(
        ref_dataset,
        shuffle=False,
        rank=distributed.get_local_rank(),
    )
    ref_data_loader = torch.utils.data.DataLoader(
        ref_dataset,
        sampler=sampler,
        batch_size=cfg.eval.batch_size_per_gpu,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )

    return que_dataset, ref_dataset, que_data_loader, ref_data_loader


def build_local_verification_data_for_matching_eval(cfg):
    dataset = LocalVerificationCopyDatasetMatchingEval(cfg)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=False,
        rank=distributed.get_local_rank(),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.eval.batch_size_per_gpu,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, data_loader


def build_local_verification_data_for_descriptor_eval(cfg):
    loc_ver_funcs_for_query = {
        "one_of_one": LocalVerificationCopyDatasetDescriptorEval.local_verification_for_query_one_of_one,
        "one_of_two": LocalVerificationCopyDatasetDescriptorEval.local_verification_for_query_one_of_two,
        "one_of_four": LocalVerificationCopyDatasetDescriptorEval.local_verification_for_query_one_of_four,
        "one_of_nine": LocalVerificationCopyDatasetDescriptorEval.local_verification_for_query_one_of_nine,
    }

    que_datasets, que_data_loaders = [], []
    for loc_ver_cfg in cfg.eval.local_verification:
        loc_ver_func = loc_ver_funcs_for_query[loc_ver_cfg["func"]]
        que_dataset = LocalVerificationCopyDatasetDescriptorEval(
            cfg,
            split="query",
            local_verification_func=loc_ver_func,
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            que_dataset,
            shuffle=False,
            rank=distributed.get_local_rank(),
        )
        que_data_loader = torch.utils.data.DataLoader(
            que_dataset,
            sampler=sampler,
            batch_size=cfg.eval.batch_size_per_gpu,
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )
        que_datasets.append(que_dataset)
        que_data_loaders.append(que_data_loader)

    ref_dataset = LocalVerificationCopyDatasetDescriptorEval(
        cfg,
        split="reference",
        local_verification_func=LocalVerificationCopyDatasetDescriptorEval.local_verification_for_reference,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        ref_dataset,
        shuffle=False,
        rank=distributed.get_local_rank(),
    )
    ref_data_loader = torch.utils.data.DataLoader(
        ref_dataset,
        sampler=sampler,
        batch_size=cfg.eval.batch_size_per_gpu,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )

    return que_datasets, ref_dataset, que_data_loaders, ref_data_loader


def build_unpaired_data_for_eval(cfg):
    paths = build_data_paths(cfg.dataset, None)
    basic_transform = aug_build.build_eval_transforms(cfg.input)
    dataset = UnpairedCopyDataset(paths, basic_transform, cfg)
    logger.info(f"Len of dataset: {len(paths)}")

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=False,
        rank=distributed.get_local_rank(),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.eval.batch_size_per_gpu,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, data_loader


def build_oxford_paris_data_for_eval(cfg):
    basic_transform = aug_build.build_eval_transforms(cfg.input)

    dataset = OxfordParisDataset(
        "datasets/roxford5k_rparis6k",
        "roxford5k",
        split="query",
        transform=basic_transform,
        imsize=None,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=False,
        rank=distributed.get_local_rank(),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.eval.batch_size_per_gpu,
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, data_loader


def build_np_dataset(cfg):
    datasets = []
    for name, _cfg in cfg.input.dataset.items():
        if _cfg is None:
            continue
        paths = build_data_paths(name, _cfg)
        if hasattr(_cfg, "augmentation"):
            # use specific augmentations settings for this dataset
            _cfg.augmentation = cfg.input.augmentation
            _cfg.img_size = cfg.input.img_size
        basic_transform, train_transform = aug_build.build_transforms(_cfg, background_paths=paths)
        datasets.append(NPCopyDataset(paths, basic_transform, train_transform, _cfg))
    dataset = JointDataset(*datasets)

    logger.info(f"Build dataset with {len(dataset):,d} samples.")

    return dataset


def build_np_data_loader(cfg):
    dataset = build_np_dataset(cfg)
    sampler = build_sampler(dataset, cfg)

    logger.info("using PyTorch data loader.")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.train.batch_size_per_gpu,
        collate_fn=np_collate_func,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
