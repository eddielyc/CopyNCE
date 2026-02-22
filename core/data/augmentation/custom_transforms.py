import random
from typing import List, Optional, Tuple, Sequence
import math
import numpy as np
import torch
from PIL import Image
from matplotlib import colormaps as cm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from core.data.augmentation import transforms_pt, Augmentation, overlay
from augly.image.transforms import BaseTransform


def should_pass_in_extra_args(transform):
    return isinstance(
        transform,
        (
            overlay.OverlayObjectOntoTheOther,
            overlay.OverlayObjectOntoTheOtherWithTransforms,
            RandomApply,
            RandomChoice,
            Shuffle,
            Compose,
        ),
    )


class RandomAffine(transforms_pt.RandomAffine):
    def __init__(
            self, degrees, translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=0,
            fillcolor=None, resample=None, rotation_affinity=22.5
    ):
        super().__init__(degrees, translate, scale, shear, interpolation, fill, fillcolor, resample)
        self.rotation_affinity = rotation_affinity

    def get_params(
            self,
            degrees: List[float],
            translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int]
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        angle -= math.floor(angle / 360) * 360
        if 0 <= angle < self.rotation_affinity or 360 - self.rotation_affinity <= angle <= 360:
            angle = 0
        elif 90 - self.rotation_affinity <= angle <= 90 + self.rotation_affinity:
            angle = 90
        elif 180 - self.rotation_affinity <= angle <= 180 + self.rotation_affinity:
            angle = 180
        elif 270 - self.rotation_affinity <= angle <= 270 + self.rotation_affinity:
            angle = 270

        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear


class RandomPadding(torch.nn.Module):
    def __init__(self, ratio=0.2, fill=0):
        super().__init__()
        self.ratio = ratio
        self.fill = fill

    @staticmethod
    def get_params(
            ratio: float,
            size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        w, h = size
        left, top, right, bottom = round(w * ratio), round(h * ratio), round(w * ratio), round(h * ratio)
        return left, top, right, bottom

    def forward(self, img, *args, **kwargs):
        size = F.get_image_size(img)
        left, top, right, bottom = self.get_params(self.ratio, size)
        img = F.pad(img, padding=[left, top, right, bottom], padding_mode='constant', fill=self.fill)

        config = {
            "type": Augmentation.PADDING,
            "params": {"left": left, "right": right, "top": top, "bottom": bottom}
        }

        return img, [config]

    def __repr__(self):
        s = f"{self.__class__.__name__}(ratio={self.ratio}, fill={self.fill})"
        return s


class RandomEdgeEnhance(BaseTransform):
    def __init__(
        self,
        mode,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.mode = mode

    def apply_transform(self, image: Image.Image, *args, **kwargs) -> Image.Image:
        return image.filter(self.mode)


class AlbumentationsTransform(object):
    def __init__(self, a_transform):
        self.transform = a_transform

    def __call__(self, img, *args, **kwargs):
        return Image.fromarray(self.transform(image=np.array(img), *args, **kwargs)['image'])


class Shuffle:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        # without replacement
        shuffled_transforms = random.sample(self.transforms, len(self.transforms))
        configs = []
        for transform in shuffled_transforms:
            if should_pass_in_extra_args(transform):
                ret = transform(img, *args, **kwargs)
            else:
                ret = transform(img)
            if isinstance(ret, (tuple, list)):
                img, _configs = ret
                configs.extend(_configs)
            else:
                img = ret
        return img, configs


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        configs = []
        for t in self.transforms:
            if should_pass_in_extra_args(t):
                ret = t(img, *args, **kwargs)
            else:
                ret = t(img)
            if isinstance(ret, tuple):
                img, _configs = ret
                configs.extend(_configs)
            else:
                img = ret
        if len(configs) == 0:
            configs = [{"type": Augmentation.NO_CHANGE, "params": None}]

        # import pdb; pdb.set_trace()

        return img, configs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply:
    """Apply randomly a list of transformations with a given probability.
    Args:
        transforms (sequence or transform): list of transformations
        p (float): probability
    """

    def __init__(self, *transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def __call__(self, img, *args, **kwargs):
        configs = []
        if torch.rand(1) < self.p:
            for t in self.transforms:
                if should_pass_in_extra_args(t):
                    ret = t(img, *args, **kwargs)
                else:
                    ret = t(img)
                if isinstance(ret, (tuple, list)):
                    img, _configs = ret
                    configs.extend(_configs)
        if len(configs) == 0:
            configs.append({"type": Augmentation.NO_CHANGE, "params": None})
        return img, configs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomTransforms:
    """Base class for a list of transformations with randomness

    Args:
        transforms (sequence): list of transformations
    """

    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list. This transform does not support torchscript.
    """
    def __call__(self, img, *args, **kwargs):
        t = random.choice(self.transforms)
        if should_pass_in_extra_args(t):
            return t(img, *args, **kwargs)
        else:
            return t(img)


class Repeat(object):
    def __init__(self, base_transform, mild_transform, n=2):
        self.n = n
        self.base_transform = base_transform
        self.mild_transform = mild_transform

    def __call__(self, img, *args, **kwargs):
        repeateds = [self.mild_transform(img)[0] for _ in range(self.n - 1)]
        repeateds.append(self.base_transform(img)[0])
        random.shuffle(repeateds)

        if np.random.random() < 0.5:
            # horizontal
            repeateds = np.concatenate([np.array(img, dtype=np.uint8) for img in repeateds], axis=1)
            catted = Image.fromarray(repeateds)
        else:
            # vertical
            repeateds = np.concatenate([np.array(img, dtype=np.uint8) for img in repeateds], axis=0)
            catted = Image.fromarray(repeateds)
        return catted, [{"type": Augmentation.NO_CHANGE, "params": None}]


class ColorMapping(object):
    def __init__(self):
        self.color_maps = list(cm.values())

    def __call__(self, img, *args, **kwargs):
        img_l = img.convert("L")
        img_np = np.array(img_l)
        img_np = img_np / 255
        img_mapped = random.choice(self.color_maps)(img_np)
        img_mapped = (np.clip(img_mapped, 0, 1) * 255).astype(np.uint8)
        img_mapped = Image.fromarray(img_mapped).convert("RGB")

        return img_mapped, [{"type": Augmentation.COLOR_MAPPING, "params": None}]
