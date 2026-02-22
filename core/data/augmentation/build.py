from loguru import logger

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from timm.models.layers import to_2tuple

from . import (
    transforms_pt as _transforms,
    custom_transforms as cus_transforms,
)

__augmentation_factory__ = {}


def register_augmentation(aug_name=""):
    def wrapper(cls):
        if aug_name in __augmentation_factory__:
            logger.warning(
                f'Overwriting {aug_name} in registry with {cls.__name__}. This is because the name being '
                'registered conflicts with an existing name. Please check if this is not expected.'
            )
        __augmentation_factory__[aug_name] = cls
        return cls

    return wrapper


def build_eval_transforms(cfg):
    img_size = to_2tuple(cfg.img_size)
    basic_transform = cus_transforms.Compose(
        [
            _transforms.Resize(img_size, InterpolationMode[cfg.augmentation.interpolation.upper()]),
            transforms.ToTensor(),
            transforms.Normalize(
                cfg.augmentation.post_process.normalize.mean,
                cfg.augmentation.post_process.normalize.std,
            )
        ]
    )

    return basic_transform


def build_basic_transforms(cfg):
    img_size = to_2tuple(cfg.img_size)
    basic_transform = cus_transforms.Compose(
        [
            _transforms.Resize(img_size, InterpolationMode[cfg.augmentation.interpolation.upper()]),
            # CHECKME
            # _transforms.RandomGrayscale(p=0.2),
            # _transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                cfg.augmentation.post_process.normalize.mean,
                cfg.augmentation.post_process.normalize.std,
            ),
        ]
    )
    return basic_transform


def build_transforms(cfg, background_paths):
    basic_transform = build_basic_transforms(cfg)
    train_transform = __augmentation_factory__[cfg.augmentation.type](cfg, background_paths)

    return basic_transform, train_transform
