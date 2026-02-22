import os
import io
import pickle
import functools
import glob
import json
import dataclasses

import cv2
import math
import random
import string
from pathlib import Path
from loguru import logger
from typing import List, Tuple, Iterable, Optional, Dict, Any
from fontTools.ttLib import TTFont

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from timm.models.layers import to_2tuple

import albumentations as A
from augly.image.transforms import BaseTransform
from augly.utils.constants import FONT_LIST_PATH, SMILEY_EMOJI_DIR
from augly.utils.base_paths import MODULE_BASE_DIR
from augly.utils import pathmgr, EMOJI_DIR, FONTS_DIR
from augly.image.functional import overlay_emoji, overlay_text

import torch
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode
from . import (
    transforms_pt as _transforms,
    custom_transforms as cus_transforms,
    data_utils,
)

from .mask_mapper import Augmentation


class Overlay(object):
    @staticmethod
    def get_params(img: Image.Image, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = data_utils.get_wh(img)
        area = height * width
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, image, *args, **kwargs):
        raise NotImplementedError


class DeprecatedRandomOverlayEmoji(BaseTransform):
    # DEPRECATED
    def __init__(
        self,
        emoji_directory: str = SMILEY_EMOJI_DIR,
        opacity: float = 1.0,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.emoji_directory = emoji_directory
        self.emoji_paths = pathmgr.ls(emoji_directory)
        self.opacity = opacity

    def apply_transform(self, image: Image.Image, *args, **kwargs) -> Image.Image:
        emoji_path = random.choice(self.emoji_paths)
        return overlay_emoji(
            image,
            emoji_path=os.path.join(self.emoji_directory, emoji_path),
            opacity=self.opacity,
            emoji_size=random.uniform(0.1, 0.3),
            x_pos=random.uniform(0.0, 1.0),
            y_pos=random.uniform(0.0, 1.0),
        )


class DeprecatedRandomOverlayText(BaseTransform):
    # DEPRECATED
    def __init__(
        self,
        opacity: float = 1.0,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.opacity = opacity

        with open(Path(FONTS_DIR) / FONT_LIST_PATH) as f:
            font_list = [s.strip() for s in f.readlines()]
            blacklist = [
                'TypeMyMusic',
                'PainttheSky-Regular',
            ]
            self.font_list = [
                f for f in font_list
                if all(_ not in f for _ in blacklist)
            ]

        self.font_lens = []
        for ff in self.font_list:
            font_file = Path(MODULE_BASE_DIR) / ff.replace('.ttf', '.pkl')
            with open(font_file, 'rb') as f:
                self.font_lens.append(len(pickle.load(f)))

    def apply_transform(
        self, image: Image.Image, metadata: Optional[List[Dict[str, Any]]] = None, *args, **kwargs
    ) -> Image.Image:
        i = random.randrange(0, len(self.font_list))
        kwargs = dict(
            font_file=Path(MODULE_BASE_DIR) / self.font_list[i],
            font_size=random.uniform(0.1, 0.3),
            color=[random.randrange(0, 256) for _ in range(3)],
            x_pos=random.uniform(0.0, 0.5),
            metadata=metadata,
            opacity=self.opacity,
        )
        try:
            for j in range(random.randrange(1, 3)):
                if j == 0:
                    y_pos = random.uniform(0.0, 0.5)
                else:
                    y_pos += kwargs['font_size'] # noqa
                image = overlay_text(
                    image,
                    text=[random.randrange(0, self.font_lens[i]) for _ in range(random.randrange(5, 10))],
                    y_pos=y_pos,
                    **kwargs,
                )
            return image
        except OSError:
            return image


class OverlayImageOntoTheOther(Overlay):
    def __init__(
            self,
            background_images_paths,
            size=(224, 224),
            scale=(0.08, 0.5),
            ratio=(0.5, 2.0),
            opacity=(0.8, 1.0),
            p_resized_crop=0.5,
            p_circle_mask=0,
            p_affine=0,
            p_scale=(0.25, 1.0),
            p_ratio=(0.5, 2.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
    ):
        super().__init__()
        self.background_images_paths = background_images_paths

        self.size = to_2tuple(size)
        self.scale = scale
        self.ratio = ratio
        self.opacity = opacity
        self.interpolation = interpolation
        self.basic_spatial = _transforms.RandomAffine(
            degrees=360,
            shear=45,
            interpolation=self.interpolation,
        )

        self.p_resized_crop = p_resized_crop
        self.p_circle_mask = p_circle_mask
        self.p_affine = p_affine

        self.resized_crop = _transforms.RandomResizedCrop(
            self.size,
            scale=p_scale,
            ratio=p_ratio,
            interpolation=self.interpolation,
        )

        self.circle_mask = data_utils.get_circle_mask(self.size)

    def __call__(self, image: Image, batch_indices: Optional[Iterable] = None, *args, **kwargs):
        configs = []
        image = image.convert('RGBA')

        background = self.get_the_other_image(batch_indices)

        i, j, h, w = self.get_params(background, self.scale, self.ratio)
        if random.random() < self.p_resized_crop:
            image, _configs = self.resized_crop(image)
            configs.extend(_configs)
        image = image.resize((w, h))
        configs.append({"type": Augmentation.RESIZE, "params": {"h": h, "w": w}})

        if random.random() < self.p_circle_mask:
            circle_mask = self.circle_mask.resize((w, h))
            image_np = np.array(image)
            image_np[:, :, 3] = np.array(circle_mask)
            image = Image.fromarray(image_np)
            is_circle_mask = True
        else:
            is_circle_mask = False

        if random.random() < self.p_affine:
            image, _configs = self.basic_spatial(image)
            configs.extend(_configs)

        opacity = random.uniform(*self.opacity)
        mask = image.getchannel('A')
        mask = Image.fromarray((np.array(mask) * opacity).astype(np.uint8)) # noqa
        _config_mask = {"type": Augmentation.ALPHA_CHANGE, "params": {"type": int(is_circle_mask), "opacity": opacity}}

        background.paste(im=image, box=(j, i), mask=mask)
        configs.append(
            {
                "type": Augmentation.PASTE_IMAGE_ONTO,
                "params": {
                    "i": i, "j": j, "h": h, "w": w,
                    "mask": _config_mask,
                    "size": background.size,
                }
            }
        )
        return background.convert('RGB'), configs

    def get_the_other_image(self, batch_indices: Iterable = None):
        i = random.randrange(0, len(self.background_images_paths))
        if batch_indices:
            while i in batch_indices:
                i = random.randrange(0, len(self.background_images_paths))
        background_path = self.background_images_paths[i]

        background = Image.open(background_path).convert('RGBA')
        background = background.resize(self.size)
        return background


class OverlayTheOtherOntoImage(OverlayImageOntoTheOther):
    def __init__(
            self,
            background_images_paths,
            size=(224, 224),
            scale=(0.08, 0.5),
            ratio=(0.5, 2.0),
            opacity=(0.8, 1.0),
            p_resized_crop=0.5,
            p_circle_mask=0,
            p_affine=0,
            p_scale=(0.25, 1.0),
            p_ratio=(0.5, 2.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
    ):
        super().__init__(
            background_images_paths,
            size=size,
            scale=scale,
            ratio=ratio,
            opacity=opacity,
            p_resized_crop=p_resized_crop,
            p_circle_mask=p_circle_mask,
            p_affine=p_affine,
            p_scale=p_scale,
            p_ratio=p_ratio,
            interpolation=interpolation,
        )

    def __call__(self, image: Image, batch_indices: Optional[Iterable] = None, *args, **kwargs):
        configs = []
        image = image.convert('RGBA')

        foreground = self.get_the_other_image(batch_indices)
        if random.random() < self.p_resized_crop:
            image, _config = self.resized_crop(image)
            configs.extend(_config)

        i, j, h, w = self.get_params(image, self.scale, self.ratio)

        foreground = foreground.resize((w, h))

        if random.random() < self.p_affine:
            image, _configs = self.basic_spatial(image)
            configs.extend(_configs)

        opacity = random.uniform(*self.opacity)
        mask = foreground.getchannel('A')
        mask = Image.fromarray((np.array(mask) * opacity).astype(np.uint8)) # noqa

        image.paste(im=foreground, box=(j, i), mask=mask)
        configs.append(
            {
                "type": Augmentation.PASTE_IMAGE_BENEATH,
                "params": {
                    "i": i, "j": j, "h": h, "w": w, "opacity": opacity
                }
            }
        )
        return image.convert('RGB'), configs


class OverlayObjectOntoTheOther(Overlay):
    """
    This version was developed on FastSAM in 'Segment Everything' mode. The map is
    an uint8 ndarray with the same width and height with the original image. It contains
    all maps of objects that are present in the image. Each object has it own label
    number in this map.
    """
    def __init__(
            self,
            mask_dir,
            background_images_paths,
            size=(224, 224),
            scale=(0.08, 0.5),
            ratio=(0.5, 2.0),
            opacity=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
            p_edge_gradient=0.5,
            gradient_radius=(0.05, 0.2),
    ):
        super().__init__()
        self.mask_dir = Path(mask_dir)
        self.background_images_paths = background_images_paths

        self.size = to_2tuple(size)
        self.scale = scale
        self.ratio = ratio
        self.opacity = opacity
        self.interpolation = interpolation
        self.p_edge_gradient = p_edge_gradient
        self.gradient_radius = gradient_radius

        self.resized_crop = _transforms.RandomResizedCrop(
            self.size,
            scale=(0.2, 1.0),
            ratio=(0.33, 3.0),
            interpolation=self.interpolation,
        )

    def get_the_other_image(self, batch_indices: Iterable = None):
        i = random.randrange(0, len(self.background_images_paths))
        if batch_indices:
            while i in batch_indices:
                i = random.randrange(0, len(self.background_images_paths))
        background_path = self.background_images_paths[i]

        background = Image.open(background_path).convert('RGBA')
        background = background.resize(self.size)
        return background

    @staticmethod
    def generate_mask(obj_mask, left, top, right, bottom):
        cropped_mask = obj_mask[top: bottom, left: right]
        obj_indices = np.unique(cropped_mask)
        if len(obj_indices) and obj_indices[0] == 0:
            obj_indices = obj_indices[1:]
        if len(obj_indices) == 0:
            return None
        selected_mask = np.isin(obj_mask, obj_indices).astype(np.uint8)
        return selected_mask

    @staticmethod
    def crop_image_according_to_mask(image, mask):
        rows, cols = np.where(mask == 1)

        top, bottom = np.min(rows), np.max(rows)
        left, right = np.min(cols), np.max(cols)

        mask = mask[top: bottom + 1, left: right + 1]
        cropped_image = image.crop((left, top, right + 1, bottom + 1))
        return cropped_image, mask, (left, top)

    def __call__(self, image: Image, path: [str, Path], batch_indices: Optional[Iterable] = None, *args, **kwargs): # noqa
        configs = []
        image_rgba = image.convert('RGBA')
        background = self.get_the_other_image(batch_indices)
        mask_path = self.mask_dir / f"{Path(path).stem}.pth"
        if not mask_path.exists():
            return self.resized_crop(image.convert("RGB"))

        mask = torch.load(mask_path).numpy()
        for attempt in range(10):
            i, j, h, w = self.get_params(background, self.scale, self.ratio)
            selected_mask = self.generate_mask(mask, j, i, j + w, i + h)
            if selected_mask is None:
                continue
            cropped_image, cropped_mask, (obj_j, obj_i) = self.crop_image_according_to_mask(image_rgba, selected_mask)
            cropped_image = cropped_image.resize((w, h))

            opacity = random.uniform(*self.opacity)
            mask = (255 * opacity * cropped_mask).astype(np.uint8)
            mask = Image.fromarray(mask, mode='L')
            mask = mask.resize((w, h), resample=Image.NEAREST)

            if random.random() < self.p_edge_gradient:
                mask = self.apply_edge_gradient_transition(mask)

            background.paste(
                cropped_image, box=(j, i), mask=mask
            )
            configs.append(
                {
                    "type": Augmentation.PASTE_OBJECT_ONTO,
                    "params": {
                        "i": i, "j": j, "h": h, "w": w,
                        "obj_i": obj_i, "obj_j": obj_j,
                        "cropped_mask": cropped_mask, "mask": np.array(mask), # noqa
                        "image_size": background.size,
                    }
                }
            )
            return background.convert('RGB'), configs

        return self.resized_crop(image.convert("RGB"))

    def apply_edge_gradient_transition(self, mask):
        assert isinstance(mask, Image.Image)
        radius = min(mask.width, mask.height) * random.uniform(*self.gradient_radius)
        radius = math.ceil(radius)

        # padding mask with zeroes around the edges in order to erode the borders
        padded_mask = F.pad(mask, padding=[radius for _ in range(4)], fill=0, padding_mode='constant') # noqa

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
        eroded_mask = cv2.erode(np.array(padded_mask), kernel)
        blurred_mask = cv2.blur(eroded_mask, (2 * radius, 2 * radius))
        # unpadding
        blurred_mask = blurred_mask[radius:-radius, radius:-radius]

        return Image.fromarray(blurred_mask)


class RandomSmearing(Overlay):
    def __init__(
            self,
            points=(5, 50),
            width=(1, 5),
            num_segments=500,
            scale=(0.08, 0.8),
            ratio=(0.5, 2.0),
            p=1.,
    ):
        if isinstance(points, int):
            points = (1, points)
        assert len(points) == 2
        self.points = points

        if isinstance(width, int):
            width = (1, width)
        assert len(width) == 2
        self.width = width

        self.num_segments = num_segments
        self.scale = scale
        self.ratio = ratio
        self.p = p

    def __call__(self, image: Image.Image, *args, **kwargs):
        if np.random.random() > self.p:
            return image, []

        draw = ImageDraw.Draw(image)

        # generate random points on canvas
        num_points = np.random.randint(self.points[0], self.points[1] + 1)
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        points = [(np.random.randint(j, j + w), np.random.randint(i, i + h)) for _ in range(num_points)]

        # generate the Bézier curve according to the random points on canvas
        curve_points = self.get_bezier_curve(points, self.num_segments)

        # plot the curve on canvas
        width = np.random.randint(self.width[0], self.width[1] + 1)
        color = np.random.randint(0, 256, size=3).tolist()
        draw.line(curve_points, fill=tuple(color), width=width) # noqa

        return image, []

    def get_bezier_curve(self, points, num_segments=100):
        result = []
        n = len(points) - 1
        for i in range(num_segments + 1):
            t = i / num_segments
            x = sum([points[j][0] * self.bernstein_poly(n, j, t) for j in range(n + 1)])
            y = sum([points[j][1] * self.bernstein_poly(n, j, t) for j in range(n + 1)])
            result.append((x, y))
        return result

    @staticmethod
    def bernstein_poly(n, i, t):
        return math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


class EmojiRepository:
    def __init__(self, path):
        self._emoji_fpaths = glob.glob(os.path.join(path, "*/*.png"))
        self._emoji_images = {}

    def map_path(self, path, local_path):
        path = path.strip()
        if local_path:
            local_mapped = os.path.join(local_path, os.path.basename(path))
            if os.path.isfile(local_mapped):
                return local_mapped
        return path

    def random_emoji(self) -> Image.Image:
        emoji_fpath = random.choice(self._emoji_fpaths)
        return self.get_emoji(emoji_fpath)

    @functools.lru_cache(maxsize=None)
    def get_emoji(self, emoji_fpath: str) -> Image.Image:
        return Image.open(open(emoji_fpath, "rb"))

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get(cls, path) -> "EmojiRepository":
        return cls(path)

    def size(self):
        return len(self._emoji_fpaths)


class RandomOverlayEmoji(Overlay):
    """
    Overlays (random) emoji on image
    """

    def __init__(self, p):
        self.p = p
        self._emojis = EmojiRepository.get(EMOJI_DIR)
        assert self._emojis.size() > 0
        logger.info(f"Build SSCD RandomOverlayEmoji with p: {self.p}.")

    def __call__(self, image: Image.Image, *args, **kwargs):
        if random.random() > self.p:
            return image

        emoji: Image.Image = self._emojis.random_emoji()
        emoji_w, emoji_h = emoji.size
        image_w, image_h = image.size
        max_scale = min(image_w / emoji_w, image_h / emoji_h)
        emoji_size_frac = 0.1 + random.random() * 0.4
        emoji_scale = max_scale * emoji_size_frac
        emoji = emoji.resize(
            (int(emoji_w * emoji_scale), int(emoji_h * emoji_scale)),
            resample=Image.BILINEAR,
        )
        fx = random.random()
        fy = random.random()
        topleft_x = int(fx * (image_w - emoji.width))
        topleft_y = int(fy * (image_h - emoji.height))
        opacity = 0.7 + 0.3 * random.random()
        # perform overlay
        image_rgba = image.copy().convert("RGBA")
        # Get the mask of the emoji if it has one, otherwise create it
        try:
            mask = emoji.getchannel("A")
            mask = Image.fromarray((np.array(mask) * opacity).astype(np.uint8)) # noqa
        except ValueError:
            mask = Image.new(mode="L", size=emoji.size, color=int(opacity * 255))
        image_rgba.paste(emoji, box=(topleft_x, topleft_y), mask=mask)
        return image_rgba.convert("RGB")


@dataclasses.dataclass
class Font:
    name: str
    path: str
    ttf_bytes: bytes
    charset: Any  # numpy array

    def ttf(self):
        return io.BytesIO(self.ttf_bytes)

    def image_font(self, size) -> ImageFont:
        return ImageFont.truetype(self.ttf(), size)

    @classmethod
    def load(cls, path) -> "Font":
        prefix, ext = os.path.splitext(path)
        assert ext in [".ttf", ".pkl"]
        ttf_path = f"{prefix}.ttf"
        name = os.path.basename(ttf_path)
        with open(ttf_path, "rb") as f:
            ttf_bytes = f.read()
        with open(f"{prefix}.pkl", "rb") as f:
            charset = np.array(pickle.load(f), dtype=np.int64)
        return cls(name=name, path=ttf_path, ttf_bytes=ttf_bytes, charset=charset)

    def sample_chars(self, length) -> List[int]:
        return random.choices(self.charset, k=length)

    def sample_string(self, length) -> str:
        characters = self.sample_chars(length)
        return "".join(chr(x) for x in characters)


class FontRepository:

    fonts = List[Font]

    def __init__(self, path):
        filenames = [
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith(".ttf")
        ]
        logger.info(f"Loading {len(filenames)} fonts from {path}.")
        self.fonts = [Font.load(filename) for filename in filenames]
        logger.info(f"Finished loading {len(filenames)} fonts.")

    def random_font(self) -> Font:
        return random.choice(self.fonts)

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get(cls, path) -> "FontRepository":
        return cls(path)

    def size(self):
        return len(self.fonts)


class RandomOverlayText(Overlay):
    """
    Overlays text on image
    """

    def __init__(self, p):
        self.p = p
        self._fonts = FontRepository.get(FONTS_DIR)
        assert self._fonts.size() > 0
        logger.info(f"Build SSCD RandomOverlayText with p: {self.p}.")

    def __call__(self, image: Image.Image, *args, **kwargs):
        if random.random() < self.p:
            return image

        # instantiate font
        font: Font = self._fonts.random_font()
        font_size_frac = 0.1 + 0.2 * random.random()
        font_size = int(min(image.width, image.height) * font_size_frac)
        image_font = font.image_font(font_size)
        # sample a string of fixed length from charset
        _SAMPLE_STR_LEN = 100
        text_str = font.sample_string(_SAMPLE_STR_LEN)
        # compute maximum length that fits into image
        # TODO: binary search over a lazy list of fixed length
        # (tw and th are monotonically increasing)
        maxlen = 0
        for i in range(1, len(text_str)):
            substr = text_str[:i]
            try:
                tw, th = image_font.getsize(substr)
            except OSError as e:
                # Safeguard against invalid chars in charset
                # that produce "invalid composite glyph" error
                logger.warning(f"Error, font={font.path}, char_i={ord(substr[-1])}")
                logger.warning(e)
                # don't overlay text in case of invalid glyphs
                return image
            if (tw > image.width) or (th > image.height):
                maxlen = i - 1
                break
        if maxlen == 0:
            return image
        # sample text length and get definitive text size
        text_len = random.randint(1, maxlen)
        text_str = text_str[:text_len]
        text_width, text_height = image_font.getsize(text_str)
        assert (text_width <= image.width) and (text_height <= image.height), (
            f"Text has size (H={text_height}, W={text_width}) which does "
            f"not fit into image of size (H={image.height}, W={image.width})"
        )
        # sample text location
        fx = random.random()
        fy = random.random()
        topleft_x = fx * (image.width - text_width)
        topleft_y = fy * (image.height - text_height)
        opacity = 0.1 + 0.9 * random.random()
        alpha = int(opacity * 255 + 0.5)
        color = (random.randrange(256), random.randrange(256), random.randrange(256))
        color_w_opacity = color + (alpha,)
        # create output image
        image_base = image.convert("RGBA")
        image_txt = Image.new("RGBA", image_base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image_txt)
        draw.text(
            xy=(topleft_x, topleft_y),
            text=text_str,
            fill=color_w_opacity,
            font=image_font,
        )
        image_out = Image.alpha_composite(image_base, image_txt).convert("RGB")
        return image_out


class OverlayObjectOntoTheOtherWithTransforms(Overlay):
    """
    This version was developed on SEEM in 'Text' mode. The text prompt is the name of
    original image. The map is a bool ndarray with the same width and height with the
    original image. It only contains map of the main object.

    Steps:
    1. Load image and its bool map
    2. Resize the map into the same with image
    3. Apply affine (rotation, translation, scaling and shear) and perspective transformations on the image and its map
    4. Select background image randomly (pure color or random image from the training set)
    5. Overlay the main object onto background
    """

    def __init__(
            self,
            mask_dir,
            background_images_paths,
            interpolation=transforms.InterpolationMode.BICUBIC,
    ):
        super().__init__()
        self.mask_dir = Path(mask_dir)
        self.background_images_paths = background_images_paths
        self.interpolation = interpolation

        # ip surface transforms
        self.surface_transforms = cus_transforms.AlbumentationsTransform(
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=(0.8, 1),  # Union[float, Tuple[float, float]]
                        contrast=(0.8, 1),  # Union[float, Tuple[float, float]]
                        saturation=(0.8, 1),  # Union[float, Tuple[float, float]]
                        hue=(-0.2, 0.2),  # Union[float, Tuple[float, float]]
                        always_apply=True,  # Optional[bool]
                        p=1.0,  # float
                    ),
                    A.CLAHE(),
                    A.Defocus(
                        radius=(2, 5),  # Union[int, Tuple[int, int]]
                        alias_blur=(0.1, 0.5),  # Union[float, Tuple[float, float]]
                        always_apply=True,  # Optional[bool]
                        p=1.0,  # float
                    ),
                    A.GaussNoise(),
                    A.HueSaturationValue(
                        hue_shift_limit=20,  # Union[int, Tuple[int, int]]
                        sat_shift_limit=30,  # Union[int, Tuple[int, int]]
                        val_shift_limit=20,  # Union[int, Tuple[int, int]]
                        always_apply=True,  # Optional[bool]
                        p=1.0,  # float
                    ),
                    A.InvertImg(),
                    A.RandomRain(),
                    A.Sharpen(),
                    A.Spatter(
                        mean=(0.65, 0.65),  # Union[float, Tuple[float, float]]
                        std=(0.3, 0.3),  # Union[float, Tuple[float, float]]
                        gauss_sigma=(2, 2),  # Union[float, Tuple[float, float]]
                        cutout_threshold=(0.68, 0.68),  # Union[float, Tuple[float, float]]
                        intensity=(0.6, 0.6),  # Union[float, Tuple[float, float]]
                        mode="rain",  # Union[Literal['rain', 'mud'], Sequence[Literal['rain', 'mud']]]
                        color=None,  # Union[Sequence[int], Dict[str, Sequence[int]], NoneType]
                        always_apply=True,  # Optional[bool]
                        p=1.0,  # float
                    ),
                    A.ToGray(
                        always_apply=True,  # Optional[bool]
                        p=1.0,  # float
                    ),
                ],
                p=0.5,
            )
        )

        # extra transforms
        self.extra_transforms = cus_transforms.AlbumentationsTransform(
            A.OneOf(
                [
                    # A.ChromaticAberration(
                    #     primary_distortion_limit=(-0.5, 0.5),  # Union[float, Tuple[float, float]]
                    #     secondary_distortion_limit=(-0.1, 0.1),  # Union[float, Tuple[float, float]]
                    #     mode="red_blue",  # Literal['green_purple', 'red_blue', 'random']
                    #     interpolation=1,  # int
                    #     always_apply=True,  # Optional[bool]
                    #     p=1.0,  # float
                    # ),
                    A.GlassBlur(
                        sigma=0.7,  # float
                        max_delta=2,  # int
                        iterations=2,  # int
                        mode="fast",  # Literal['fast', 'exact']
                        always_apply=True,  # Optional[bool]
                        p=1.0,  # float
                    ),
                    A.PiecewiseAffine(
                        scale=(0.01, 0.03),  # Union[float, Tuple[float, float]]
                        nb_rows=4,  # Union[int, Tuple[int, int]]
                        nb_cols=4,  # Union[int, Tuple[int, int]]
                        interpolation=1,  # int
                        mask_interpolation=0,  # int
                        cval=0,  # int
                        cval_mask=0,  # int
                        mode="constant",  # Literal['constant', 'edge', 'symmetric', 'reflect', 'wrap']
                        absolute_scale=False,  # bool
                        always_apply=True,  # Optional[bool]
                        keypoints_threshold=0.01,  # float
                        p=1.0,  # float
                    ),
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1),  # Tuple[float, float, float, float]
                        num_shadows_lower=1,  # Optional[int]
                        num_shadows_upper=2,  # Optional[int]
                        shadow_dimension=5,  # int
                        always_apply=True,  # Optional[bool]
                        p=1.0,  # float
                    ),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),  # Tuple[float, float, float, float]
                        angle_lower=0,  # float
                        angle_upper=1,  # float
                        num_flare_circles_lower=6,  # int
                        num_flare_circles_upper=10,  # int
                        src_radius=400,  # int
                        src_color=(255, 255, 255),  # Tuple[int, int, int]
                        always_apply=True,  # Optional[bool]
                        p=1.0,  # float
                    ),
                    A.ZoomBlur(
                        max_factor=(1, 1.1),  # Union[float, Tuple[float, float]]
                        step_factor=(0.01, 0.03),  # Union[float, Tuple[float, float]]
                        always_apply=True,  # Optional[bool]
                        p=1.0,  # float
                    ),
                ],
                p=0.5,
            )
        )

        # affine transform params
        self.degrees = [0., 360.]
        self.translate = [0.25, 0.25]
        self.scale = [0.33, 1.2]
        self.shear = [45, 45]

        # shrink params
        self.shrink_factor = (0.75, 1.0)

        # perspective params
        self.distortion_scale = 0.6

    def get_the_other_image(self, image: Image, batch_indices: Iterable = None):
        if np.random.random() < 0.6:
            i = random.randrange(0, len(self.background_images_paths))
            if batch_indices:
                while i in batch_indices:
                    i = random.randrange(0, len(self.background_images_paths))
            background_path = self.background_images_paths[i]

            background = Image.open(background_path).convert('RGB')
            background = background.resize(image.size)
            return background
        else:
            # use pure color background
            if np.random.random() < 0.5:
                color = tuple(random.randint(0, 255) for _ in range(3))
            else:
                color = (255, 255, 255)
            return Image.new('RGB', image.size, color)

    def __call__(self, image: Image, path: [str, Path], batch_indices: Optional[Iterable] = None, *args, **kwargs):  # noqa
        image = image.convert('RGB')
        background = self.get_the_other_image(image, batch_indices)
        mask_path = self.mask_dir / f"{Path(path).stem}.npy"
        if not mask_path.exists() or np.load(mask_path).sum() == 0:
            image = self.surface_transforms(image)
            image = self.extra_transforms(image)
            return image

        mask = np.load(str(mask_path)).astype(np.uint8) * 255
        mask = Image.fromarray(mask, mode='L')
        if mask.size != image.size:
            mask = mask.resize(image.size, resample=Image.NEAREST)

        # surface transforms
        image = self.surface_transforms(image)

        # affine transform
        success = False
        while not success:
            img_size = F.get_image_size(image)
            angle, translations, scale, shear = _transforms.RandomAffine.get_params(
                self.degrees,
                self.translate,
                self.scale,
                self.shear,
                img_size,
            )
            angle = angle if np.random.random() < 0.5 else 0.
            shear = shear if np.random.random() < 0.5 else (0., 0.)

            center = [img_size[0] / 2, img_size[1] / 2]
            affine_matrix = data_utils.generate_affine_matrix(
                scale=scale,
                angle=-angle,
                shear=[-shear[0], -shear[1]],
                translations=translations,
                center=center,
            )
            mask = F.affine(
                mask, angle, list(translations), scale, list(shear),  # noqa
                interpolation=InterpolationMode.NEAREST
            )
            if np.any(mask):
                # It means that affine transform moves all valid pixels out of image, when there is not 255 in mask,
                # And if there are 255s in mask, then there is no need to do affine transform again.
                success = True

        image = F.affine(
            image, angle, list(translations), scale, list(shear),
            interpolation=self.interpolation, fill=[255, 255, 255]
        )

        success = False
        while not success:
            # perspective transform
            startpoints, endpoints = _transforms.RandomPerspective.get_params(img_size[0], img_size[1],
                                                                              self.distortion_scale)
            perspective_matrix = cv2.getPerspectiveTransform(
                np.array(startpoints, dtype=np.float32),
                np.array(endpoints, dtype=np.float32)
            )
            mask = F.perspective(mask, startpoints, endpoints, InterpolationMode.NEAREST)
            if np.any(mask):
                # It means that affine transform moves all valid pixels out of image, when there is not 255 in mask,
                # And if there are 255s in mask, then there is no need to do affine transform again.
                success = True

        image = F.perspective(image, startpoints, endpoints, self.interpolation)

        # shrink
        shrink_scale = float(torch.empty(1).uniform_(self.shrink_factor[0], self.shrink_factor[1]).item())
        center = [img_size[0] / 2, img_size[1] / 2]
        angle, shear, translations = 0, (0, 0), (0, 0)
        shrink_matrix = data_utils.generate_affine_matrix(
            scale=shrink_scale,
            angle=angle,
            shear=shear,
            translations=translations,
            center=center,
        )
        mask = F.affine(
            mask, angle, list(translations), scale, list(shear),  # noqa
            interpolation=InterpolationMode.NEAREST
        )

        image = F.affine(
            image, angle, list(translations), scale, list(shear),
            interpolation=self.interpolation, fill=[255, 255, 255]
        )

        mask = (np.array(mask) > 0.).astype(np.uint8)
        image = np.array(image, dtype=np.uint8) * mask[..., np.newaxis] + np.array(background, dtype=np.uint8) * (
                    1 - mask[..., np.newaxis])
        image = Image.fromarray(image)

        # extra transforms
        image = self.extra_transforms(image)

        config = {
            "type": Augmentation.PASTE_OBJECT_ONTO_WITH_TRANSFORMS,
            "params": {
                "affine_matrix": affine_matrix,
                "perspective_matrix": perspective_matrix,
                "shrink_matrix": shrink_matrix,
                "mask": mask,
            }
        }

        return image, [config]
