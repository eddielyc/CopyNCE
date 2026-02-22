from typing import Union, List, Tuple
import math
from loguru import logger

import cv2
import numpy as np
from PIL import Image

import torch
from torch import Tensor


def get_shape(tensor: Union[Tensor, np.ndarray]):
    if isinstance(tensor, Tensor):
        return tensor.size()
    elif isinstance(tensor, np.ndarray):
        return tensor.shape
    else:
        raise TypeError("Only torch.Tensor and numpy.ndarray are expected.")


def get_wh(image: Union[Image.Image, Tensor, np.ndarray]):
    if isinstance(image, Image.Image):
        return image.size
    elif isinstance(image, (Tensor, np.ndarray)):
        shape = get_shape(image)
        if len(shape) > 3:
            logger.warning(f"Expect single image input not a batch of images with shape: {shape}.")
        if shape[0] in (3, 4):
            logger.warning(f"Expect image input with shape: HxWxC, not tensor after ToTensor() function "
                           f"with shape: {shape}.")
        h, w, *_ = shape
        return w, h
    else:
        raise TypeError("Only Pillow Image, torch.Tensor and numpy.ndarray are expected.")


def get_circle_mask(size: Union[int, List, Tuple] = 224, center=None, radius=None):
    size = (size, size) if isinstance(size, (int, float)) else size
    width, height = size

    heights = np.arange(height)[:, np.newaxis].repeat(width, axis=1)
    widths = np.arange(width)[np.newaxis, :].repeat(height, axis=0)

    center = (width // 2, height // 2) if center is None else center
    center_w, center_h = center
    radius = min(center_w, center_h, width - center_w, height - center_h) - 1 if radius is None else radius

    distance_to_center = np.sqrt((heights - center_h) ** 2 + (widths - center_w) ** 2)
    mask = distance_to_center <= radius
    mask = mask.astype(np.uint8) * 255
    return Image.fromarray(mask)


def generate_affine_matrix(scale, angle, shear, translations, center=(0, 0)):
    center_x, center_y = center

    # rotation
    rotation_scale_matrix = np.eye(3)
    _rotation_scale_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotation_scale_matrix[0: 2] = _rotation_scale_matrix

    # shear
    shear_matrix = np.eye(3)
    shear_x, shear_y = math.radians(shear[0]), math.radians(shear[1])
    u, v = math.tan(shear_x), math.tan(shear_y)
    shear_matrix[0, 1], shear_matrix[1, 0] = u, v
    shear_matrix[0, 0] = 1 + u * v

    shear_translate_x = -u * v * center_x - u * center_y
    shear_translate_y = -v * center_x
    shear_matrix[0, 2], shear_matrix[1, 2] = shear_translate_x, shear_translate_y

    # translation
    translation_matrix = np.eye(3)
    translation_matrix[0, 2], translation_matrix[1, 2] = translations

    # compose transforms
    h = np.matmul(rotation_scale_matrix, shear_matrix)
    h = np.matmul(translation_matrix, h)
    return h


if __name__ == '__main__':
    from PIL import ImageTransform
    image = Image.open("figs/1.jpg")
    image = image.convert("RGBA")
    image.transform(image.size, ImageTransform.AffineTransform)





