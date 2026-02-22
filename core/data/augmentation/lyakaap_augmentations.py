from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.data.augmentation import (
    overlay,
    transforms_pt as _transforms,
    custom_transforms as cus_transforms,
    build,
)

from PIL import Image, ImageFilter
import albumentations as A
from timm.models.layers import to_2tuple
from loguru import logger
from augly import image as Au


@build.register_augmentation("lyakaap")
class LyakaapAugmentation(object):
    def __init__(self, cfg, background_paths=None):
        self.size_hw = to_2tuple(cfg.img_size)
        self.cfg = cfg
        self.interpolation = InterpolationMode[self.cfg.augmentation.interpolation.upper()]
        self.background_paths = background_paths

        crop_transform = self.build_crop_transform()
        geo_transform = self.build_geo_transform()
        aug_transform = self.build_aug_transform()
        additional_transform = self.build_additional_transform()
        # sam_crop_transform = self.build_sam_crop_transform()

        post_process = self.build_post_process()

        self.traditional_transforms = cus_transforms.Compose(
            [
                cus_transforms.Shuffle(
                    [
                        crop_transform,
                        geo_transform,
                        aug_transform,
                        additional_transform,
                    ]
                ),
                post_process,
            ]
        )

        # self.sam_transforms = cus_transforms.Compose(
        #     [
        #         cus_transforms.Shuffle(
        #             [
        #                 sam_crop_transform,
        #                 aug_transform,
        #                 additional_transform,
        #             ]
        #         ),
        #         post_process,
        #     ]
        # )

        self.transform = cus_transforms.RandomChoice(
            [
                self.traditional_transforms,
                # self.sam_transforms,
            ],
        )

        logger.info(f"Build LyakaapAugmentation.")

    def __call__(self, image, *args, **kwargs):
        image, configs = self.transform(image, *args, **kwargs)
        return image, configs

    def build_sam_crop_transform(self):
        sam_crop_transform = overlay.OverlayObjectOntoTheOther(
            mask_dir=self.cfg.augmentation.crop_ops.overlay_object_onto_the_other.mask_dir,
            background_images_paths=self.background_paths,
            size=self.size_hw,
            scale=self.cfg.augmentation.crop_ops.overlay_object_onto_the_other.scale,
            ratio=self.cfg.augmentation.crop_ops.overlay_object_onto_the_other.ratio,
            # foreground object opacity
            opacity=self.cfg.augmentation.crop_ops.overlay_object_onto_the_other.opacity,
            interpolation=self.interpolation,
        )

        return cus_transforms.RandomApply(
            sam_crop_transform,
            p=self.cfg.augmentation.crop_ops.overlay_object_onto_the_other.p,
        )

    def build_additional_transform(self):
        additional_ops = [
            A.CLAHE(),
            A.ChannelDropout(),
            A.ChannelShuffle(),
            # A.Emboss(),
            A.Equalize(),
            A.FancyPCA(),
            A.GaussNoise(),
            A.HueSaturationValue(),
            A.ISONoise(),
            A.ImageCompression(),
            A.InvertImg(),
            A.MedianBlur(),
            A.MotionBlur(),
            A.MultiplicativeNoise(),
            A.Posterize(),
            A.RGBShift(),
            A.RandomBrightnessContrast(),
            A.RandomFog(),
            A.RandomGamma(),
            A.RandomRain(),
            A.RandomSnow(),
            A.RandomToneCurve(),
            A.Sharpen(),
            A.Solarize(),
            A.ToSepia(),
        ]
        additional_transform = cus_transforms.AlbumentationsTransform(
            A.OneOf(additional_ops, p=self.cfg.augmentation.additional_ops.p)
        )
        return additional_transform

    def build_aug_transform(self):
        aug_ops = [
            _transforms.ColorJitter(
                brightness=self.cfg.augmentation.aug_ops.color_jitter.brightness,
                contrast=self.cfg.augmentation.aug_ops.color_jitter.contrast,
                saturation=self.cfg.augmentation.aug_ops.color_jitter.saturation,
                hue=self.cfg.augmentation.aug_ops.color_jitter.hue,
            ),
            Au.RandomPixelization(p=self.cfg.augmentation.aug_ops.random_pixelization.p),
            Au.ShufflePixels(
                factor=self.cfg.augmentation.aug_ops.shuffle_pixels.factor,
                p=self.cfg.augmentation.aug_ops.shuffle_pixels.p,
            ),
            Au.OneOf(
                [
                    Au.EncodingQuality(quality=q)
                    for q in self.cfg.augmentation.aug_ops.encoding_quality.qs
                ], p=self.cfg.augmentation.aug_ops.encoding_quality.p
            ),
            _transforms.RandomGrayscale(p=self.cfg.augmentation.aug_ops.random_grayscale.p),
            Au.RandomBlur(p=self.cfg.augmentation.aug_ops.random_blur.p),
            overlay.RandomOverlayText(p=self.cfg.augmentation.aug_ops.random_overlay_text.p),
            overlay.RandomOverlayEmoji(p=self.cfg.augmentation.aug_ops.random_overlay_emoji.p),
            Au.OneOf(
                [
                    cus_transforms.RandomEdgeEnhance(mode=ImageFilter.EDGE_ENHANCE),
                    cus_transforms.RandomEdgeEnhance(mode=ImageFilter.EDGE_ENHANCE_MORE)
                ],
                p=self.cfg.augmentation.aug_ops.random_edge_enhance.p,
            ),
            overlay.RandomSmearing(
                points=self.cfg.augmentation.aug_ops.random_smearing.points,
                width=self.cfg.augmentation.aug_ops.random_smearing.width,
                scale=self.cfg.augmentation.aug_ops.random_smearing.scale,
                ratio=self.cfg.augmentation.aug_ops.random_smearing.ratio,
                p=self.cfg.augmentation.aug_ops.random_smearing.p,
            ),
        ]

        if self.cfg.augmentation.aug_ops.shuffle:
            aug_transform = cus_transforms.Shuffle(aug_ops)
        else:
            aug_transform = cus_transforms.Compose(aug_ops)
        return aug_transform

    def build_geo_transform(self):
        linear_ops = [
            _transforms.RandomPerspective(interpolation=self.interpolation, p=1.),
            cus_transforms.RandomAffine(
                degrees=self.cfg.augmentation.geo_ops.random_affine.degrees,
                shear=self.cfg.augmentation.geo_ops.random_affine.shear,
                interpolation=self.interpolation,
                rotation_affinity=self.cfg.augmentation.geo_ops.random_affine.rotation_affinity,
            ),
        ]

        geo_transform = cus_transforms.Compose(
            [
                _transforms.RandomHorizontalFlip(p=self.cfg.augmentation.geo_ops.random_horizontal_flip.p),
                _transforms.RandomVerticalFlip(p=self.cfg.augmentation.geo_ops.random_vertical_flip.p),
                cus_transforms.RandomApply(
                    cus_transforms.RandomChoice(linear_ops),
                    p=self.cfg.augmentation.geo_ops.p,
                )
            ]
        )

        return geo_transform

    def build_crop_transform(self):
        crop_ops = [
            _transforms.RandomResizedCrop(
                self.size_hw,
                scale=self.cfg.augmentation.crop_ops.random_resized_crop.scale,
                ratio=self.cfg.augmentation.crop_ops.random_resized_crop.ratio,
                interpolation=self.interpolation,
            ),
            overlay.OverlayImageOntoTheOther(
                background_images_paths=self.background_paths,
                size=self.size_hw,
                scale=self.cfg.augmentation.crop_ops.overlay_image_onto_the_other.scale,
                ratio=self.cfg.augmentation.crop_ops.overlay_image_onto_the_other.ratio,
                # foreground opacity
                opacity=self.cfg.augmentation.crop_ops.overlay_image_onto_the_other.opacity,
                p_scale=self.cfg.augmentation.crop_ops.overlay_image_onto_the_other.p_scale,
                p_ratio=self.cfg.augmentation.crop_ops.overlay_image_onto_the_other.p_ratio,
                interpolation=self.interpolation,
            ),
            overlay.OverlayTheOtherOntoImage(
                background_images_paths=self.background_paths,
                size=self.size_hw,
                scale=self.cfg.augmentation.crop_ops.overlay_the_other_onto_image.scale,
                ratio=self.cfg.augmentation.crop_ops.overlay_the_other_onto_image.ratio,
                # foreground opacity
                opacity=self.cfg.augmentation.crop_ops.overlay_the_other_onto_image.opacity,
                p_scale=self.cfg.augmentation.crop_ops.overlay_the_other_onto_image.p_scale,
                p_ratio=self.cfg.augmentation.crop_ops.overlay_the_other_onto_image.p_ratio,
                interpolation=self.interpolation,
            ),
        ]
        crop_transform = cus_transforms.RandomApply(
            cus_transforms.RandomChoice(crop_ops),
            p=self.cfg.augmentation.crop_ops.p,
        )

        return crop_transform

    def build_post_process(self):
        post_process = cus_transforms.Compose(
            [
                cus_transforms.RandomApply(
                    cus_transforms.RandomPadding(
                        ratio=self.cfg.augmentation.post_process.random_padding.ratio,
                        fill=self.cfg.augmentation.post_process.random_padding.fill,
                    ), p=self.cfg.augmentation.post_process.random_padding.p,
                ),

                _transforms.Resize(self.size_hw, interpolation=self.interpolation),
                transforms.ToTensor(),
                # FIXME: RandomErasing should be put after Normalization. But for some reason,
                # it is here and do not move, or it will take a long period of time for model to fit the domain change.
                transforms.RandomErasing(
                    value=self.cfg.augmentation.post_process.random_erasing.value,
                    p=self.cfg.augmentation.post_process.random_erasing.p,
                ),
                transforms.Normalize(
                    self.cfg.augmentation.post_process.normalize.mean,
                    self.cfg.augmentation.post_process.normalize.std,
                ),
            ]
        )

        return post_process
