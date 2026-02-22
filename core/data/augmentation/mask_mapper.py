from typing import Union, List, Tuple, Dict
import copy
from enum import Enum

import numpy as np
import torch
from kornia.utils import create_meshgrid
import cv2

from torch.nn import functional as F


class Augmentation(Enum):
    """
    {
        "type": Augmentation.xxx,
        "params": {
            ...
        }
    }

    Affine params includes: "matrix", "scale", "angle", "shear", "translation", "center"

    Crop params includes: "i", "j", "h", "w"

    Resize params includes: "h", "w"

    Paste image onto params includes:
        "i", "j", "w", "h",
        # "image_path": "black" / path,
            # "black": use black as background
            # path: path of the other image
        "mask",
        "size": [w, h]
            # size of augmented image

    Alpha change params includes:
        "type": 0 / 1 / 2
            # 0: no circle mask
            # 1: circle mask
            # 2: normal alpha change
        "opacity"

    No change params includes:
        "matrix": np.eye(3)

    Crop resize params includes:
        "i", "j", "h", "w",
        "size": [resized_w, resized_h]

    Flip params includes:
        "matrix" # flip could be expressed as a specific affine transform

    Posterize params includes: None

    Solarize params includes: None

    Color jitter params includes:

        "fn_idx", "brightness_factor", "contrast_factor", "saturation_factor", "hue_factor"
    Gaussian blur params includes: "sigma"

    Channel shuffled params includes: "channels_shuffled"
    Shift RGB params includes: "r_shift", "g_shift", "b_shift"

    Paste image beneath params includes:
        "type": "background" / "foreground",
            # "background": paste other images onto this image (as background)
            # "foreground": paste this image onto other images (as foreground)
        "i", "j", "w", "h",
        # "image_path": "black" / path,
            # "black": use black as background
            # path: path of the other image
        "mask",
        "size": [w, h]
            # size of augmented image

    Padding params includes: "left", "right", "top", "bottom"

    paste object onto with transforms params includes:
        "affine_matrix": affine transform matrix,
        "perspective_matrix": perspective transform matrix,
        "mask": mask of object

    """

    AFFINE = 0
    CROP = 1
    RESIZE = 2
    PASTE_IMAGE_ONTO = 3
    ALPHA_CHANGE = 4
    NO_CHANGE = 5
    CROP_RESIZE = 6
    FLIP = 7
    POSTERIZE = 8
    SOLARIZE = 9
    COLOR_JITTER = 10
    GAUSSIAN_BLUR = 11
    CHANNELS_SHUFFLED = 12
    # SHIFT_RGB = 13
    # EMOJI_IMAGE = 14
    # CenterCrop = 15
    OVERLAY_TEXT = 16
    PASTE_IMAGE_BENEATH = 17
    PADDING = 18
    PERSPECTIVE = 19
    PASTE_OBJECT_ONTO = 20
    PASTE_OBJECT_BENEATH = 21
    PASTE_OBJECT_ONTO_WITH_TRANSFORMS = 22
    COLOR_MAPPING = 24


def affine_warp_pts(pts: np.ndarray, matrix: np.ndarray):
    """
    apply affine transform matrix to pts in shape: [2, ...]
    :param pts: points to be transformed
    :param matrix: linear transform matrix
    :return: transformed points
    """
    assert pts.shape[0] == 2 and matrix.shape == (3, 3)
    shape = pts.shape
    pts = pts.reshape([2, -1])
    ones = np.ones((1, pts.shape[1]), dtype=float)
    pts = np.vstack([pts, ones])
    pts = matrix @ pts

    return pts[: 2].reshape(shape)


def perspective_warp_pts(pts: np.ndarray, matrix: np.ndarray):
    """
    apply perspective transform matrix to pts in shape: [2, ...]
    :param pts: points to be transformed
    :param matrix: linear transform matrix
    :return: transformed points
    """
    assert pts.shape[0] == 2 and matrix.shape == (3, 3)
    pts = pts.reshape([2, -1])
    ones = np.ones((1, pts.shape[1]), dtype=float)
    pts = np.vstack([pts, ones])
    pts = matrix @ pts

    t = pts[2]
    invalids = np.isclose(t, 0)
    pts[:, invalids] = np.array([-1, -1, 1])[:, np.newaxis]
    pts[:2] = pts[:2] / pts[2]

    return pts[:2]


class MaskMapper(object):
    def __init__(self, imsize: Union[List, Tuple]):
        # [width, height]
        self.imsize = imsize
        self.width, self.height = self.imsize
        # aug_to_src: [2, self.height, self.width]
        self.aug_to_src = create_meshgrid(
            self.height, self.width,
            normalized_coordinates=False
        ).numpy()[0][..., [1, 0]].transpose(2, 0, 1)

    @property
    def factory(self):
        return {
            Augmentation.AFFINE: self.affine,
            Augmentation.CROP: self.crop,
            Augmentation.RESIZE: self.resize,
            Augmentation.CROP_RESIZE: self.crop_resize,
            Augmentation.FLIP: self.flip,
            Augmentation.PASTE_IMAGE_ONTO: self.paste_image_onto,
            Augmentation.PASTE_IMAGE_BENEATH: self.paste_image_beneath,
            Augmentation.PADDING: self.padding,
            Augmentation.PERSPECTIVE: self.perspective,
            Augmentation.PASTE_OBJECT_ONTO: self.paste_object_onto,
            Augmentation.PASTE_OBJECT_ONTO_WITH_TRANSFORMS: self.paste_object_onto_with_transforms,
        }

    def __call__(self, configs):
        for config in configs:
            mapping_func = self.factory.get(config["type"], None)
            if mapping_func is not None:
                mapping_func(config["params"])
        return self.aug_to_src

    def crop(self, params: Dict, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src

        i, j, h, w = params["i"], params["j"], params["h"], params["w"]
        aug_to_src = aug_to_src[:, i: i + h, j: j + w]
        self.aug_to_src = aug_to_src
        return aug_to_src

    def resize(self, params: Dict, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src

        _, h, w = aug_to_src.shape
        resized_h, resized_w = params["h"], params["w"]

        # Use pytorch to interpolate position index
        aug_to_src_pt = torch.tensor(aug_to_src).unsqueeze(dim=0)
        aug_to_src_pt = F.interpolate(aug_to_src_pt, (resized_h, resized_w), mode="nearest")
        aug_to_src = aug_to_src_pt.squeeze(dim=0).round().numpy()

        self.aug_to_src = aug_to_src
        return aug_to_src

    def crop_resize(self, params: Dict, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src

        i, j, h, w, (resized_w, resized_h) = params["i"], params["j"], params["h"], params["w"], params["size"]
        aug_to_src = self.crop(params, aug_to_src)
        aug_to_src = self.resize({"h": resized_h, "w": resized_w}, aug_to_src)

        self.aug_to_src = aug_to_src
        return aug_to_src

    def flip(self, params: Dict, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src

        aug_to_src = self.aug_to_src = self.affine(params, aug_to_src)
        return aug_to_src

    def affine(self, params: Dict, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src
        matrix = params["matrix"]
        _, h, w = aug_to_src.shape

        grid = create_meshgrid(
            h, w,
            normalized_coordinates=False
        ).numpy()[0].transpose(2, 1, 0).reshape(2, -1)
        _aug_to_src = affine_warp_pts(grid, np.linalg.inv(matrix))
        _aug_to_src = _aug_to_src[[1, 0], :].astype(int)

        valids = (0 <= _aug_to_src[0, :]) & (_aug_to_src[0, :] < h) & (0 <= _aug_to_src[1, :]) & (_aug_to_src[1, :] < w)
        _aug_to_src = _aug_to_src[:, valids]

        _new_aug_to_src = -1 * np.ones((2, h * w))
        _new_aug_to_src[:, valids] = aug_to_src.reshape(2, -1)[:, (_aug_to_src[0]) * w + (_aug_to_src[1])]
        _new_aug_to_src = _new_aug_to_src.reshape(2, w, h).transpose(0, 2, 1)
        aug_to_src = self.aug_to_src = _new_aug_to_src
        return aug_to_src

    def perspective(self, params: Dict, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src
        matrix = params["matrix"]

        _, h, w = aug_to_src.shape

        grid = create_meshgrid(
            h, w,
            normalized_coordinates=False
        ).numpy()[0].transpose(2, 1, 0).reshape(2, -1)
        _aug_to_src = perspective_warp_pts(grid, np.linalg.inv(matrix))
        _aug_to_src = _aug_to_src[[1, 0], :].astype(int)

        valids = (0 <= _aug_to_src[0, :]) & (_aug_to_src[0, :] < h) & (0 <= _aug_to_src[1, :]) & (_aug_to_src[1, :] < w)
        _aug_to_src = _aug_to_src[:, valids]

        _new_aug_to_src = -1 * np.ones((2, h * w))
        _new_aug_to_src[:, valids] = aug_to_src.reshape(2, -1)[:, (_aug_to_src[0]) * w + (_aug_to_src[1])]
        _new_aug_to_src = _new_aug_to_src.reshape(2, w, h).transpose(0, 2, 1)

        aug_to_src = self.aug_to_src = _new_aug_to_src
        return aug_to_src

    def paste_image_onto(self, params: Dict, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src

        i, j, h, w = params["i"], params["j"], params["h"], params["w"]
        image_w, image_h = params["size"]

        _aug_to_src = -1 * np.ones((2, image_h, image_w))
        _, _h, _w = aug_to_src.shape
        assert _h == h, _w == w
        _aug_to_src[:, i: i + h, j: j + w] = aug_to_src
        aug_to_src = self.aug_to_src = _aug_to_src
        return aug_to_src

        # FIXME: Circle Mask branches are missing.

    def paste_image_beneath(self, params: Dict, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src
        i, j, h, w = params["i"], params["j"], params["h"], params["w"]
        # opacity = params["opacity"]
        # aug_to_src[:, i: i + h, j: j + w] = -1
        self.aug_to_src = aug_to_src
        return aug_to_src

    def paste_object_onto(self, params: Dict, aug_to_src=None):
        # NOTE THAT: paste_object_onto augmentations should be placed at the first position,
        #            which means that aug_to_src is the original meshgrid matrix.
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src

        i, j, h, w = params["i"], params["j"], params["h"], params["w"]
        obj_i, obj_j = params["obj_i"], params["obj_j"]
        cropped_mask, mask = params["cropped_mask"], params["mask"]
        [image_h, image_w] = params["image_size"]
        obj_h, obj_w = cropped_mask.shape[0], cropped_mask.shape[1]

        _aug_to_src = -1 * np.ones((2, image_h, image_w))
        obj_mapping = aug_to_src[:, obj_i: obj_i + obj_h, obj_j: obj_j + obj_w]
        obj_mapping = self.interpolate_mapping(obj_mapping, (h, w))
        valids = (obj_mapping[0] >= 0) & (obj_mapping[1] >= 0) & (mask > 0)
        obj_mapping[:, np.logical_not(valids)] = -1

        _aug_to_src[:, i: i + h, j: j + w] = obj_mapping
        aug_to_src = self.aug_to_src = _aug_to_src
        return aug_to_src
    
    def paste_object_onto_with_transforms(self, params: Dict, aug_to_src=None):
        # NOTE THAT: paste_object_onto augmentations should be placed at the first position,
        #            which means that aug_to_src is the original meshgrid matrix.
        _aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src

        if "affine_matrix" in params:
            _aug_to_src = self.affine({"matrix": params["affine_matrix"]}, _aug_to_src)
        if "perspective_matrix" in params:
            _aug_to_src = self.perspective({"matrix": params["perspective_matrix"]}, _aug_to_src)
        if "shrink_matrix" in params:
            _aug_to_src = self.affine({"matrix": params["shrink_matrix"]}, _aug_to_src)
        mask = params["mask"][np.newaxis, ...]
        _aug_to_src = _aug_to_src * mask + -1 * np.ones_like(_aug_to_src) * (1 - mask)
        aug_to_src = self.aug_to_src = _aug_to_src
        return aug_to_src

    @staticmethod
    def interpolate_mapping(mapping, size):
        mapping = torch.from_numpy(mapping).unsqueeze(dim=0)
        mapping = F.interpolate(mapping, size)
        return mapping.squeeze(dim=0).numpy()

    def padding(self, params: Dict, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src
        left, right, top, bottom = params["left"], params["right"], params["top"], params["bottom"]
        aug_to_src = np.pad(aug_to_src, ((0, 0), (top, bottom), (left, right)), mode="constant", constant_values=-1)
        self.aug_to_src = aug_to_src
        return aug_to_src

    @property
    def mapping(self):
        return self.aug_to_src

    def get_bounding_box(self, aug_to_src=None):
        aug_to_src = self.aug_to_src if aug_to_src is None else aug_to_src

        _, h, w = aug_to_src.shape

        src_y, src_x = aug_to_src[0].flatten(), aug_to_src[1].flatten()
        src_y, src_x = src_y[src_y >= 0], src_x[src_x >= 0]

        aug_y, aug_x = np.where(aug_to_src[0] >= 0)

        if len(src_y) == 0 or len(aug_x) == 0:
            return [(0, 0), (0, 0), (0, 0), (0, 0)]
        src_l, src_r, src_t, src_b = int(min(src_x)), int(max(src_x)), int(min(src_y)), int(max(src_y))
        aug_l, aug_r, aug_t, aug_b = int(min(aug_x)), int(max(aug_x)), int(min(aug_y)), int(max(aug_y))

        return [(src_l, src_t), (src_r, src_b), (aug_l, aug_t), (aug_r, aug_b)]

    def generate_mask(
            self,
            image_src: np.ndarray = None,
            image_aug: np.ndarray = None,
            src_size_wh: Union[List, Tuple] = (480, 480),
            visualize: bool = False,
            convex_hull: bool = False,
    ):
        _, h, w = self.aug_to_src.shape
        # self.aug_to_src = self.aug_to_src.astype(int)

        src_y, src_x = self.aug_to_src[0].flatten(), self.aug_to_src[1].flatten()
        src_y, src_x = src_y[src_y >= 0], src_x[src_x >= 0]
        aug_y, aug_x = np.where(self.aug_to_src[0] >= 0)

        if image_aug is None or image_src is None:
            mask_src = np.zeros(src_size_wh[::-1], dtype=np.int32)
            mask_aug = np.zeros((h, w), dtype=np.int32)
        else:
            mask_src = np.zeros(image_src.shape[:2], dtype=np.int32)
            mask_aug = np.zeros(image_aug.shape[:2], dtype=np.int32)

        if convex_hull:
            points_aug = np.vstack([aug_x, aug_y]).T.astype(int)
            points_src = np.vstack([src_x, src_y]).T.astype(int)
            hull_src = cv2.convexHull(points_src, clockwise=True)
            hull_aug = cv2.convexHull(points_aug, clockwise=True)

            cv2.fillPoly(mask_src, [hull_src], 1)
            cv2.fillPoly(mask_aug, [hull_aug], 1)
        else:
            mask_src[src_y.astype(int), src_x.astype(int)] = 1
            mask_aug[aug_y.astype(int), aug_x.astype(int)] = 1

        if visualize:
            image_src = self.draw_mask(image_src, mask_src)
            image_aug = self.draw_mask(image_aug, mask_aug)
            return image_src, image_aug
        else:
            return mask_src, mask_aug

    def draw_mask(self, image, mask, mask_color=(0, 50, 0), mask_tk=1):
        assert image.shape[:2] == mask.shape[:2]

        h, w = mask.shape[:2]
        image = copy.deepcopy(image)
        color_mask = copy.deepcopy(image)
        color_mask[mask > 0] = mask_color

        # 合并mask
        image = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)
        # 绘制边界，边界不需要透视效果
        mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), mask_tk)
        return image

    def get_similarity_map(self, stride=16, src_size_wh=(240, 480)):
        src_h, src_w = src_size_wh[::-1]
        n_patch_h_src, n_patch_w_src = src_h // stride, src_w // stride
        aug_h, aug_w = self.aug_to_src.shape[-2:]
        n_patch_h_aug, n_patch_w_aug = aug_h // stride, aug_w // stride

        # row: augmented image
        # col: source image
        sim_map = np.zeros((n_patch_h_aug * n_patch_w_aug, n_patch_h_src * n_patch_w_src), dtype=int)
        for i in range(n_patch_h_aug):
            for j in range(n_patch_w_aug):
                src_pts = self.aug_to_src[:, i * stride: (i + 1) * stride, j * stride: (j + 1) * stride]
                # 如果没有对应区域就pass
                if src_pts.max() < 0:
                    continue
                valid_src_pts_y, valid_src_pts_x = src_pts[0][src_pts[0, :, :] > 0], src_pts[1][src_pts[1, :, :] > 0]
                if len(valid_src_pts_y) == 0 or len(valid_src_pts_x) == 0:
                    continue
                b_src_patch_i = valid_src_pts_y.max() // stride
                r_src_patch_i = valid_src_pts_x.max() // stride
                t_src_patch_i = valid_src_pts_y.min() // stride
                l_src_patch_i = valid_src_pts_x.min() // stride

                for m in np.arange(t_src_patch_i, b_src_patch_i + 1):
                    for n in np.arange(l_src_patch_i, r_src_patch_i + 1):
                        sim_map[i * n_patch_w_aug + j, m * n_patch_w_src + n] = 1

        return sim_map

    def adapt_mapping_with_resized_src(self, ratio_hw: Union[Tuple, List]):
        """
        :param ratio_hw: [ratio_h, ratio_w], resize ratio of source image.
        :return: None
        """
        self.aug_to_src[0, :, :] = self.aug_to_src[0, :, :] * ratio_hw[0]
        self.aug_to_src[1, :, :] = self.aug_to_src[1, :, :] * ratio_hw[1]

    @staticmethod
    def generate_inverse_mapping(mapping, inverse_hw=None):
        assert mapping.shape[0] == 2
        _, H, W = mapping.shape
        inverse_h, inverse_w = (H, W) if inverse_hw is None else inverse_hw
        valids = (mapping[0] >= 0) & (mapping[1] >= 0) & (mapping[0] < inverse_h) & (mapping[1] < inverse_w)
        mapping = np.round(mapping).astype(int)
        valids &= (mapping[0] >= 0) & (mapping[1] >= 0) & (mapping[0] < inverse_h) & (mapping[1] < inverse_w)

        h_indices, w_indices = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        h_indices, w_indices = h_indices.reshape(-1), w_indices.reshape(-1)
        mapped_h, mapped_w = mapping[0].reshape(-1), mapping[1].reshape(-1)
        valids = valids.reshape(-1)
        inversed_mapping = -np.ones((2, inverse_h, inverse_w), dtype=float)
        h_indices, w_indices, mapped_h, mapped_w = h_indices[valids], w_indices[valids], mapped_h[valids], mapped_w[valids]
        inversed_mapping[:, mapped_h, mapped_w] = np.stack([h_indices, w_indices], axis=0)
        return inversed_mapping

    @staticmethod
    def generate_batch_inverse_mappings(mapping, inverse_hw=None):
        # if inverse_hw is not set, the default inverse mapping has the same shape as mapping
        # Must be an instance of torch.cuda.Tensor
        mapping = torch.tensor(mapping).cuda() if not isinstance(mapping, torch.Tensor) else mapping.cuda()

        assert len(mapping.size()) == 4 and mapping.size(1) == 2
        B, _, H, W = mapping.size()
        inverse_h, inverse_w = (H, W) if inverse_hw is None else inverse_hw

        valids = (mapping[:, 0] >= 0) & (mapping[:, 1] >= 0) & (mapping[:, 0] < inverse_h) & (mapping[:, 1] < inverse_w)
        if mapping.dtype not in (torch.long, torch.int):
            mapping = torch.round(mapping).to(torch.int)
        valids &= (mapping[:, 0] >= 0) & (mapping[:, 1] >= 0) & (mapping[:, 0] < inverse_h) & (mapping[:, 1] < inverse_w)
        # valids: [B, H, W]
        batch_flattened_valids = valids.flatten(1)
        # batch_flattened_valids: [B, HxW]

        h_indices, w_indices = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        h_indices, w_indices = h_indices.repeat(B, 1, 1).flatten(1).cuda(), w_indices.repeat(B, 1, 1).flatten(1).cuda()
        # h_indices: [B, HxW], w_indices: [B, HxW]

        mapped_h, mapped_w = mapping[:, 0].flatten(1), mapping[:, 1].flatten(1)
        # mapped_h: [B, HxW], mapped_w: [B, HxW]

        inversed_mapping = -torch.ones(B, 2, inverse_h * inverse_w, dtype=torch.long, device=mapping.device)
        # inversed_mapping: [B, 2, inverse_h x inverse_w]
        batch_indices = mapped_h * inverse_w + mapped_w
        # batch_indices: [B, HxW]

        # AUGLY IMPLEMENT: flatten batch wise, and index the flattened array and set value in flattened array
        # The reason for doing this is that modifying torch.gather function cannot in-place modify values
        # of the source tensor.
        flattened_inversed_mapping_h = inversed_mapping[:, 0].flatten()
        flattened_inversed_mapping_w = inversed_mapping[:, 1].flatten()

        bias = torch.arange(B, device=mapping.device, dtype=torch.long).unsqueeze(dim=1) * (inverse_h * inverse_w)
        # bias: [B, 1]
        flattened_batch_indices = (batch_indices + bias).flatten()
        # flattened_batch_indices: [B, HxW]
        flattened_valids = batch_flattened_valids.flatten()
        flattened_batch_indices = flattened_batch_indices[flattened_valids]

        flattened_inversed_mapping_h[flattened_batch_indices] = h_indices.flatten()[flattened_valids]
        flattened_inversed_mapping_w[flattened_batch_indices] = w_indices.flatten()[flattened_valids]

        inversed_mapping_h = flattened_inversed_mapping_h.view(B, 1, inverse_h, inverse_w)
        inversed_mapping_w = flattened_inversed_mapping_w.view(B, 1, inverse_h, inverse_w)
        inversed_mapping = torch.cat([inversed_mapping_h, inversed_mapping_w], dim=1)

        return inversed_mapping

    @staticmethod
    def generate_transferred_mapping(mapping_ab, mapping_bc):
        # the transferred mapping has the same shape as mapping_ab
        assert mapping_ab.shape[0] == 2 and mapping_bc.shape[0] == 2
        _, mapping_ab_H, mapping_ab_W = mapping_ab.shape
        _, mapping_bc_H, mapping_bc_W = mapping_bc.shape
        valids = (mapping_ab[0] >= 0) & (mapping_ab[1] >= 0) & (mapping_ab[0] < mapping_bc_H) & (mapping_ab[1] < mapping_bc_W)
        mapping_ab = np.round(mapping_ab).astype(int)
        valids &= (mapping_ab[0] >= 0) & (mapping_ab[1] >= 0) & (mapping_ab[0] < mapping_bc_H) & (mapping_ab[1] < mapping_bc_W)

        transferred_mapping = -np.ones((2, mapping_ab_H, mapping_ab_W), dtype=float)
        valids_h, valids_w = np.nonzero(valids)
        # coordinates of the valid pixels in mapping_bc
        mapped_bc_h, mapped_bc_w = mapping_ab[:, valids_h, valids_w]

        transferred_mapping[:, valids_h, valids_w] = mapping_bc[:, mapped_bc_h, mapped_bc_w]
        return transferred_mapping

    @staticmethod
    def generate_batch_transferred_mapping(mapping_ab, mapping_bc):
        # the transferred mapping has the same shape as mapping_ab
        mapping_ab = torch.tensor(mapping_ab).cuda() if not isinstance(mapping_ab, torch.Tensor) else mapping_ab.cuda()
        mapping_bc = torch.tensor(mapping_bc).cuda() if not isinstance(mapping_bc, torch.Tensor) else mapping_bc.cuda()

        assert len(mapping_ab.size()) == 4 and mapping_ab.size(1) == 2
        assert len(mapping_bc.size()) == 4 and mapping_bc.size(1) == 2
        assert mapping_ab.size(0) == mapping_bc.size(0)

        B, _, mapping_ab_H, mapping_ab_W = mapping_ab.size()
        _, _, mapping_bc_H, mapping_bc_W = mapping_bc.size()

        valids = (mapping_ab[:, 0] >= 0) & (mapping_ab[:, 1] >= 0) & (mapping_ab[:, 0] < mapping_bc_H) & (mapping_ab[:, 1] < mapping_bc_W)
        if mapping_ab.dtype not in (torch.long, torch.int):
            mapping_ab = torch.round(mapping_ab).to(torch.long)
        if mapping_bc.dtype not in (torch.long, torch.int):
            mapping_bc = torch.round(mapping_bc).to(torch.long)
        valids &= (mapping_ab[:, 0] >= 0) & (mapping_ab[:, 1] >= 0) & (mapping_ab[:, 0] < mapping_bc_H) & (mapping_ab[:, 1] < mapping_bc_W)
        # valids: [B, mapping_ab_H, mapping_ab_W]

        mapping_ab_h, mapping_ab_w = mapping_ab[:, 0].flatten(1), mapping_ab[:, 1].flatten(1)
        batch_indices = mapping_ab_h * mapping_bc_W + mapping_ab_w
        # batch_indices: [B, mapping_ab_H x mapping_ab_W]
        bias = torch.arange(B, device=mapping_bc.device, dtype=torch.long).unsqueeze(dim=1) * (mapping_bc_H * mapping_bc_W)
        # bias: [B, 1]
        flattened_batch_indices = (batch_indices + bias).flatten()

        flattened_mapping_bc_h, flattened_mapping_bc_w = mapping_bc[:, 0].flatten(), mapping_bc[:, 1].flatten()
        flattened_valids = valids.flatten()
        # flattened_valids: [B x mapping_ab_H x mapping_ab_W]

        flattened_batch_indices = flattened_batch_indices[flattened_valids]

        flattened_transferred_mapping_h = -torch.ones(B * mapping_ab_H * mapping_ab_W, dtype=torch.long, device=mapping_ab.device)
        flattened_transferred_mapping_w = -torch.ones(B * mapping_ab_H * mapping_ab_W, dtype=torch.long, device=mapping_ab.device)
        flattened_transferred_mapping_h[flattened_valids] = flattened_mapping_bc_h[flattened_batch_indices]
        flattened_transferred_mapping_w[flattened_valids] = flattened_mapping_bc_w[flattened_batch_indices]

        transferred_mapping = torch.cat(
            [
                flattened_transferred_mapping_h.view(B, 1, mapping_ab_H, mapping_ab_W),
                flattened_transferred_mapping_w.view(B, 1, mapping_ab_H, mapping_ab_W)
            ], dim=1
        )
        return transferred_mapping
