from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.data.augmentation import (
    transforms_pt as _transforms,
    custom_transforms as cus_transforms,
    build,
)

from timm.models.layers import to_2tuple
from loguru import logger


@build.register_augmentation("mild")
class MildAugmentation(object):
    def __init__(self, cfg, *args, **kwargs):
        self.size_hw = to_2tuple(cfg.img_size)
        self.cfg = cfg
        self.interpolation = InterpolationMode[self.cfg.augmentation.interpolation.upper()]

        mild_transforms = self.build_mild_transforms(self.size_hw, interpolation=self.interpolation)
        post_process = self.build_post_process()

        self.transform = cus_transforms.Compose(
            [
                mild_transforms,
                post_process,
            ]
        )

        logger.info(f"Build MildAugmentation.")

    def __call__(self, image, *args, **kwargs):
        image, configs = self.transform(image, *args, **kwargs)
        return image, configs

    def build_post_process(self):
        post_process = cus_transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    self.cfg.augmentation.post_process.normalize.mean,
                    self.cfg.augmentation.post_process.normalize.std,
                ),
            ]
        )

        return post_process

    @staticmethod
    def build_mild_transforms(size, interpolation=InterpolationMode.BICUBIC):
        mild_transforms = cus_transforms.Compose(
            [
                cus_transforms.RandomApply(
                    transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0), interpolation=interpolation),
                    p=0.5,
                ),
                cus_transforms.RandomApply(
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    p=0.5,
                ),
                cus_transforms.RandomApply(
                    cus_transforms.RandomPadding(ratio=0.2, fill=0),
                    p=0.5,
                ),
                _transforms.RandomGrayscale(p=0.2),
                _transforms.Resize(size, interpolation=interpolation),
            ]
        )

        return mild_transforms
