from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.data.augmentation import (
    overlay,
    transforms_pt as _transforms,
    custom_transforms as cus_transforms,
    build,
)

from timm.models.layers import to_2tuple
from loguru import logger
from augly import image as Au


@build.register_augmentation("sscd")
class SSCDAugmentation(object):
    def __init__(self, cfg):
        self.size_hw = to_2tuple(cfg.img_size)
        self.cfg = cfg
        self.interpolation = InterpolationMode[self.cfg.augmentation.interpolation.upper()]

        augmentations = self.build_augmentations()
        post_process = self.build_post_process()

        self.transform = cus_transforms.Compose(
            [
                augmentations,
                post_process,
            ]
        )

        logger.info(f"Build SSCDAugmentation.")

    def __call__(self, image, *args, **kwargs):
        image, configs = self.transform(image, *args, **kwargs)
        return image, configs

    def build_augmentations(self):
        augmentations = cus_transforms.Compose(
            [
                _transforms.RandomHorizontalFlip(p=0.5),
                cus_transforms.RandomApply(
                    cus_transforms.RandomAffine(
                        degrees=(0, 360),
                        interpolation=self.interpolation,
                        rotation_affinity=45.,
                    ),
                    p=0.05,
                ),
                overlay.RandomOverlayText(p=0.1),
                overlay.RandomOverlayEmoji(p=0.2),
                cus_transforms.RandomApply(
                    cus_transforms.RandomAffine(
                        degrees=(0, 359),
                        interpolation=self.interpolation,
                        rotation_affinity=0.
                    ),
                    p=0.05,
                ),
                _transforms.RandomResizedCrop(
                    self.size_hw,
                    interpolation=self.interpolation,
                ),
                cus_transforms.RandomApply(
                    _transforms.ColorJitter(
                        brightness=0.8,
                        contrast=0.8,
                        saturation=0.8,
                        hue=0.2,
                    ),
                    p=0.8,
                ),
                _transforms.RandomGrayscale(p=0.2),
                Au.RandomBlur(min_radius=1, max_radius=5, p=0.5),
                Au.OneOf(
                    [
                        Au.EncodingQuality(quality=q)
                        for q in range(0, 100)
                    ], p=0.2
                ),
            ]
        )

        return augmentations

    def build_post_process(self):
        post_process = cus_transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        return post_process
