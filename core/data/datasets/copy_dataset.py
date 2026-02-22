import json

import math
import os.path
import random
from pathlib import Path
import func_timeout
from func_timeout import func_set_timeout

import pandas as pd
from loguru import logger
from PIL import Image
import numpy as np

import torch
from timm.layers import to_2tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from core.data.augmentation.mask_mapper import MaskMapper


class CopyDataset(torch.utils.data.Dataset):
    def __init__(self, paths, basic_transform, copy_transform, cfg):
        self.paths = paths
        self.basic_transform = basic_transform
        self.copy_transform = copy_transform
        self.cfg = cfg
        self.return_mapping = getattr(self.cfg, 'return_mapping', True)
        self.return_mask = getattr(self.cfg, 'return_mask', False)
        self.sampling_rate = getattr(self.cfg, 'sampling_rate', 1.)
        logger.info(f"Set sampling rate = {self.sampling_rate}.")

        self.positive_rate = self.cfg.positive_rate
        logger.info(f"Set positive rate = {self.positive_rate}.")

        self.positive_ref_aug_p = getattr(self.cfg, 'positive_ref_aug_p', 0.)
        self.negative_ref_aug_p = getattr(self.cfg, 'negative_ref_aug_p', 0.)
        logger.info(f"Set prob of positive reference augmentation {self.positive_ref_aug_p}.")
        logger.info(f"Set prob of negative reference augmentation {self.negative_ref_aug_p}.")

        self.hard_negative_mining = getattr(self.cfg.hard_negative_mining, 'enable', False)
        self.force_hard_negative_mining = getattr(self.cfg, 'force_hard_negative_mining', False)
        if self.force_hard_negative_mining and not math.isclose(self.positive_rate, 1.):
            logger.warning(f"When 'force_hard_negative_mining' is on, please make sure that 'positive_rate' is 1,"
                           f"or it may cause noise when calculating loss.")

        if self.hard_negative_mining or self.force_hard_negative_mining:
            knns_path = getattr(self.cfg.hard_negative_mining, 'knns_path', "weights/dino_vits_isc_knn.pth")
            knn_l = getattr(self.cfg.hard_negative_mining, "knn_l", 1)
            knn_r = getattr(self.cfg.hard_negative_mining, "knn_r", 128)
            # CHECKME: If the first knn is not itself, modify 1 to 0.
            self.knns = torch.load(knns_path, map_location="cpu")[:, knn_l: knn_r]
            self.hnm_p = getattr(self.cfg.hard_negative_mining, "p", 0.5)
            if self.force_hard_negative_mining:
                logger.warning(f"'force_hard_negative_mining' is set to be 'True' "
                               f"and hnm will be enabled whether or not. Besides, each sample will contain two pairs: "
                               f"['que': random sample A, 'ref': Augmentation of A] and "
                               f"['que': hard negative sample of A, 'ref': Augmentation of the former one]")
                self.hnm_p = 1.
            logger.info(f"Hard negative mining is enabled with p={self.hnm_p}.")
            logger.info(f"The knn graph is loaded from {knns_path}, shape of which is {self.knns.shape}.")

        self.force_abandon_copynce = getattr(self.cfg, 'force_abandon_copynce', False)
        if self.force_abandon_copynce:
            logger.warning(f"'force_abandon_copynce' is set to be 'True' "
                           f"and copynce will be abandoned in this dataset whether or not.")

    def __len__(self):
        return int(len(self.paths) * self.sampling_rate)

    def get_negative_paired_image_path(self, index):
        if self.hard_negative_mining and random.random() < self.hnm_p:
            cand_indices = self.knns[index]
            cand_index = np.random.choice(cand_indices)
            cand_path = self.paths[cand_index]
        else:
            cand_path = random.choice(self.paths)
        return cand_path

    def __getitem__(self, index):
        index_after_sampling = int(index / self.sampling_rate) + random.randrange(max(1, int(1. / self.sampling_rate)))
        index_after_sampling = min(index_after_sampling, len(self.paths) - 1)

        sample = self.fetch_sample_with_retry(index_after_sampling)
        if self.force_hard_negative_mining:
            hard_negative_index = np.random.choice(self.knns[index_after_sampling])
            hard_negative = self.fetch_sample_with_retry(hard_negative_index)
            return [sample, hard_negative]
        else:
            return sample

    def fetch_sample_with_retry(self, index, fetch_func=None):
        if fetch_func is None:
            fetch_func = self.fetch_sample
        sample, cnt = None, 0
        while sample is None:
            cnt += 1
            try:
                sample = fetch_func(index)
            except func_timeout.exceptions.FunctionTimedOut:
                logger.warning(f"Failed to load image #{index} {cnt} time(s), retry...")
                if cnt >= 2:
                    raise RuntimeError(f"Failed to load image #{index} after retrying for 2 times.")
                continue
            else:
                return sample
        return sample

    @staticmethod
    def read_image(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if img.mode not in ["RGB", "RGBA"]:
                img = img.convert('RGBA')
                img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
            img = img.convert('RGB')
        return img

    @func_set_timeout(5)
    def fetch_sample(self, index):
        path = self.paths[index]
        img_origin = self.read_image(path)
        w_origin, h_origin = img_origin.size
        is_positive = random.random() < self.positive_rate

        ref_aug_p = self.positive_ref_aug_p if is_positive else self.negative_ref_aug_p
        if random.random() < ref_aug_p:
            reference, ref_configs = self.copy_transform(img_origin, path=path)
        else:
            reference, ref_configs = self.basic_transform(img_origin)
        h_ref, w_ref = reference.size(1), reference.size(2)

        if is_positive:
            query, que_configs = self.copy_transform(img_origin, path=path)

            if self.return_mapping or self.return_mask:
                mask_mapper = MaskMapper(img_origin.size)
                mask_mapper(que_configs)
                # we assume that original size is [w_ref, h_ref]
                mask_mapper.adapt_mapping_with_resized_src([h_ref / h_origin, w_ref / w_origin])
                que_to_ori_mapping = mask_mapper.mapping

                mask_mapper = MaskMapper(img_origin.size)
                mask_mapper(ref_configs)
                # we assume that original size is [w_ref, h_ref]
                mask_mapper.adapt_mapping_with_resized_src([h_ref / h_origin, w_ref / w_origin])
                ref_to_ori_mapping = mask_mapper.mapping

            if self.return_mask:
                # FIXME: the mask here is not right now.
                ref_mask, que_mask = mask_mapper.generate_mask(
                    src_size_wh=[w_ref, h_ref],
                    visualize=False
                )

        else:
            query_path = self.get_negative_paired_image_path(index)
            query = self.read_image(query_path)

            if random.random() < 0.5:
                query, _ = self.basic_transform(query)
            else:
                query, _ = self.copy_transform(query, path=path)

            h_que, w_que = query.size(1), query.size(2)

            if self.return_mapping or self.return_mask:
                # NOTE THAT: It is not necessary to calculate correct mapping here since there is no
                # pixel level correspondence between the two images in a negative pair.
                que_to_ori_mapping = -1 * np.ones((2, h_que, w_que), dtype=np.float64)
                ref_to_ori_mapping = -1 * np.ones((2, h_que, w_que), dtype=np.float64)

            if self.return_mask:
                ref_mask = np.zeros((h_ref, w_ref), dtype=np.int32)
                que_mask = np.zeros((h_que, w_que), dtype=np.int32)

        input = {
            "references": reference,
            "queries": query,
            "cls_gt": torch.tensor(int(is_positive), dtype=torch.long),
        }

        if self.return_mapping:
            input["que_to_ori_mappings"] = que_to_ori_mapping  # noqa
            input["ref_to_ori_mappings"] = ref_to_ori_mapping  # noqa

        if self.return_mask:
            input["ref_masks_gt"] = torch.tensor(ref_mask, dtype=torch.long)  # noqa
            input["que_masks_gt"] = torch.tensor(que_mask, dtype=torch.long)  # noqa

        input["copynce"] = True if not self.force_abandon_copynce else False

        return input


class UnpairedCopyDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform, cfg):
        self.paths = paths
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with open(path, 'rb') as f:
            img_origin = Image.open(f)
            img_origin = img_origin.convert('RGB')
        img, configs = self.transform(img_origin)

        return {
            "images": img,
            "indices": index,
        }


class PairedCopyDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform, cfg):
        self.paths = paths
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        q_path, r_path = self.paths[index]
        with open(q_path, 'rb') as f:
            query = Image.open(f)
            query = query.convert('RGB')
        query, _ = self.transform(query)

        with open(r_path, 'rb') as f:
            reference = Image.open(f)
            reference = reference.convert('RGB')
        reference, _ = self.transform(reference)

        return {
            "references": reference,
            "queries": query,
            "indices": index,
            "names": f"{q_path.stem}-{r_path.stem}",
        }


class CopyDatasetMatchingEval(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.data_root = Path(cfg.eval.data_root)
        self.pair_json_path = cfg.eval.pair_json_path
        self.interpolation = InterpolationMode[cfg.input.augmentation.interpolation.upper()]

        ref_csv = pd.read_csv(cfg.eval.ref_csv_path)
        self.ref_ids = {uuid: path.strip() for uuid, path in ref_csv[["uuid", "path"]].values}
        que_csv = pd.read_csv(cfg.eval.que_csv_path)
        self.que_ids = {uuid: path.strip() for uuid, path in que_csv[["uuid", "path"]].values}

        logger.info(f"Loading pairs from: {self.pair_json_path}")
        with open(self.pair_json_path, 'r') as fd:
            pairs = json.load(fd)
        self.uuid_pairs, self.paths_pairs = [], []
        for que, ref in tqdm(pairs):
            for _ref, _ in ref:
                self.uuid_pairs.append([que, _ref])
                self.paths_pairs.append(
                    [
                        self.data_root / self.que_ids[que],
                        self.data_root / self.ref_ids[_ref],
                    ]
                )

        img_size = to_2tuple(cfg.input.img_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=self.interpolation),
                # transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    cfg.input.augmentation.post_process.normalize.mean,
                    cfg.input.augmentation.post_process.normalize.std,
                ),
            ]
        )

    def __len__(self):
        return len(self.uuid_pairs)

    def __getitem__(self, index):
        que_uuid, ref_uuid = self.uuid_pairs[index]
        name = f"{que_uuid} - {ref_uuid}"

        que_path, ref_path = self.paths_pairs[index]
        img_que_ori = Image.open(que_path).convert('RGB')
        img_ref_ori = Image.open(ref_path).convert('RGB')
        img_que = self.transform(img_que_ori)
        img_ref = self.transform(img_ref_ori)

        return {
            "queries": img_que,
            "queries_ori": np.array(img_que_ori.resize((1024, 1024), resample=Image.LANCZOS)),  # noqa
            "references": img_ref,
            "references_ori": np.array(img_ref_ori.resize((1024, 1024), resample=Image.LANCZOS)),  # noqa
            "indices": index,
            "names": name,
        }


class CopyDatasetDescriptorEval(torch.utils.data.Dataset):
    def __init__(self, cfg, split="reference"):
        if split == "reference":
            df = pd.read_csv(cfg.eval.ref_csv_path)
        elif split == "query":
            df = pd.read_csv(cfg.eval.que_csv_path)
        else:
            raise ValueError(f'Unknown split type: {split}.')

        self.data_root = Path(cfg.eval.data_root)
        self.paths = [os.path.join(self.data_root, p) for p in df['path']]
        logger.info(f'Length of dataset: {len(self.paths)}')
        self.interpolation = InterpolationMode[cfg.input.augmentation.interpolation.upper()]
        img_size = to_2tuple(cfg.input.img_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size, self.interpolation),
                transforms.ToTensor(),
                transforms.Normalize(
                    cfg.input.augmentation.post_process.normalize.mean,
                    cfg.input.augmentation.post_process.normalize.std,
                ),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            image = self.fetch_sample(i)
        except func_timeout.exceptions.FunctionTimedOut:
            logger.warning(f"Failed to load image #{i}.")
            raise RuntimeError
        return {
            "indices": i,
            "images": image
        }

    @func_set_timeout(5)
    def fetch_sample(self, index):
        image = Image.open(self.paths[index])
        image = image.convert("RGB")
        image = self.transform(image)
        return image


class LocalVerificationCopyDatasetMatchingEval(CopyDatasetMatchingEval):
    def __getitem__(self, index):
        que_uuid, ref_uuid = self.uuid_pairs[index]
        name = f"{que_uuid} - {ref_uuid}"

        que_path, ref_path = self.paths_pairs[index]
        img_que_ori = Image.open(que_path).convert('RGB')
        img_ref_ori = Image.open(ref_path).convert('RGB')
        img_ques = self.local_verification_for_query(img_que_ori)
        img_refs = self.local_verification_for_reference(img_ref_ori)

        img_ques = [self.transform(que) for que in img_ques]
        img_refs = [self.transform(ref) for ref in img_refs]

        return {
            "queries": torch.stack(img_ques, dim=0),
            "references": torch.stack(img_refs, dim=0),
            "indices": index,
            "names": name,
        }

    @staticmethod
    def local_verification_for_query(query: Image.Image):
        width, height = query.width, query.height
        rotations = [
            query.rotate(45, expand=True, resample=Image.BICUBIC),
            query.rotate(90, expand=True, resample=Image.BICUBIC),
            query.rotate(135, expand=True, resample=Image.BICUBIC),
            query.rotate(180, expand=True, resample=Image.BICUBIC),
            query.rotate(225, expand=True, resample=Image.BICUBIC),
            query.rotate(270, expand=True, resample=Image.BICUBIC),
            query.rotate(315, expand=True, resample=Image.BICUBIC),
        ]
        crops = [
            # center crop
            query.crop((width // 3, height // 3, 2 * width // 3, 2 * height // 3)),
            # top left crop
            query.crop((width // 6, height // 6, width // 2, height // 2)),
            # top right crop
            query.crop((width // 2, height // 6, 5 * width // 6, height // 2)),
            # bottom left crop
            query.crop((width // 6, height // 2, width // 2, 5 * height // 6)),
            # bottom right crop
            query.crop((width // 2, height // 2, 5 * width // 6, 5 * height // 6)),
        ]

        four_even_crops = [
            # top left crop
            query.crop((0, 0, width // 2, height // 2)),
            # top right crop
            query.crop((width // 2, 0, width, height // 2)),
            # bottom left crop
            query.crop((0, height // 2, width // 2, height)),
            # bottom right crop
            query.crop((width // 2, height // 2, width, height)),
        ]

        nine_even_crops = [
            query.crop((0, 0, width // 3, height // 3)),
            query.crop((width // 3, 0, 2 * width // 3, height // 3)),
            query.crop((2 * width // 3, 0, width, height // 3)),
            query.crop((0, height // 3, width // 3, 2 * height // 3)),
            query.crop((width // 3, height // 3, 2 * width // 3, 2 * height // 3)),
            query.crop((2 * width // 3, height // 3, width, 2 * height // 3)),
            query.crop((0, 2 * height // 3, width // 3, height)),
            query.crop((width // 3, 2 * height // 3, 2 * width // 3, height)),
            query.crop((2 * width // 3, 2 * height // 3, width, height)),
        ]

        return [
            query,
            *crops,
            *four_even_crops,
            *nine_even_crops,
            *rotations,
        ]

    @staticmethod
    def local_verification_for_reference(reference: Image.Image):
        width, height = reference.width, reference.height

        center_crops = [
            # center crop
            reference.crop((width // 3, height // 3, 2 * width // 3, 2 * height // 3)),
            # top left crop
            reference.crop((width // 6, height // 6, width // 2, height // 2)),
            # top right crop
            reference.crop((width // 2, height // 6, 5 * width // 6, height // 2)),
            # bottom left crop
            reference.crop((width // 6, height // 2, width // 2, 5 * height // 6)),
            # bottom right crop
            reference.crop((width // 2, height // 2, 5 * width // 6, 5 * height // 6)),
        ]

        four_even_crops = [
            # top left crop
            reference.crop((0, 0, width // 2, height // 2)),
            # top right crop
            reference.crop((width // 2, 0, width, height // 2)),
            # bottom left crop
            reference.crop((0, height // 2, width // 2, height)),
            # bottom right crop
            reference.crop((width // 2, height // 2, width, height)),
        ]

        # nine_even_crops = [
        #     reference.crop((0, 0, width // 3, height // 3)),
        #     reference.crop((width // 3, 0, 2 * width // 3, height // 3)),
        #     reference.crop((2 * width // 3, 0, width, height // 3)),
        #     reference.crop((0, height // 3, width // 3, 2 * height // 3)),
        #     reference.crop((width // 3, height // 3, 2 * width // 3, 2 * height // 3)),
        #     reference.crop((2 * width // 3, height // 3, width, 2 * height // 3)),
        #     reference.crop((0, 2 * height // 3, width // 3, height)),
        #     reference.crop((width // 3, 2 * height // 3, 2 * width // 3, height)),
        #     reference.crop((2 * width // 3, 2 * height // 3, width, height)),
        # ]

        return [
            reference,
            *center_crops,
            *four_even_crops,
            # *nine_even_crops,
        ]


class LocalVerificationCopyDatasetDescriptorEval(CopyDatasetDescriptorEval):
    def __init__(self, cfg, split, local_verification_func):
        super(LocalVerificationCopyDatasetDescriptorEval, self).__init__(cfg, split)
        self.local_verification = local_verification_func
        logger.info(
            f"Build {self.__class__.__name__} for {split} with loc_ver_fun: {local_verification_func.__name__}.")

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        images = self.local_verification(image)
        images = [self.transform(image) for image in images]

        return {
            "images": torch.stack(images, dim=0),
            "indices": index,
        }

    @staticmethod
    def local_verification_for_query_one_of_one(image: Image.Image):
        return [image]

    @staticmethod
    def local_verification_for_query_one_of_two(image: Image.Image):
        width, height = image.width, image.height

        crops = [
            # left vertical crop
            image.crop((0, 0, width // 2, height)),
            # center vertical crop
            image.crop((width // 4, 0, 3 * width // 2, height)),
            # right vertical crop
            image.crop((width // 2, 0, width, height)),

            # top horizontal crop
            image.crop((0, 0, width, height // 2)),
            # center horizontal crop
            image.crop((0, height // 4, width, 3 * height // 4)),
            # bottom horizontal crop
            image.crop((0, height // 2, width, height)),

            # top left crop
            image.crop((0, 0, int(0.707 * width), int(0.707 * height))),
            # top right crop
            image.crop((int(0.293 * width), 0, width, int(0.707 * height))),
            # bottom left crop
            image.crop((0, int(0.293 * height), int(0.707 * width), height)),
            # bottom right crop
            image.crop((int(0.293 * width), int(0.293 * height), width, height)),
        ]

        return crops

    @staticmethod
    def local_verification_for_query_one_of_four(image: Image.Image):
        width, height = image.width, image.height

        crops = [
            # top left crop
            image.crop((0, 0, width // 2, height // 2)),
            # top center crop
            image.crop((width // 4, 0, 3 * width // 4, height // 2)),
            # top right crop
            image.crop((width // 2, 0, width, height // 2)),

            # center left crop
            image.crop((0, height // 4, width // 2, 3 * height // 4)),
            # center center crop
            image.crop((width // 4, height // 4, 3 * width // 4, 3 * height // 4)),
            # center right crop
            image.crop((width // 2, height // 4, width, 3 * height // 4)),

            # bottom left crop
            image.crop((0, height // 2, width // 2, height)),
            # bottom center crop
            image.crop((width // 4, height // 2, 3 * width // 4, height)),
            # bottom right crop
            image.crop((width // 2, height // 2, width, height)),
        ]

        return crops

    @staticmethod
    def local_verification_for_query_one_of_nine(image: Image.Image):
        width, height = image.width, image.height
        folders, folder_range = 6, 2
        crops = []
        for top in range(folders - folder_range + 1):
            for left in range(folders - folder_range + 1):
                crops.append(image.crop(
                    (
                        left * width // folders,
                        top * height // folders,
                        (left + folder_range) * width // folders,
                        (top + folder_range) * height // folders,
                    )
                )
                )

        return crops

    @staticmethod
    def local_verification_for_reference(image: Image.Image):
        width, height = image.width, image.height

        crops = [
            # center crop
            image.crop((width // 6, height // 6, 5 * width // 6, 5 * height // 6)),
            # top left crop
            image.crop((width // 8, height // 8, 5 * width // 8, 5 * height // 8)),
            # top right crop
            image.crop((3 * width // 8, height // 8, 7 * width // 8, 5 * height // 8)),
            # bottom left crop
            image.crop((width // 8, 3 * height // 8, 5 * width // 8, 7 * height // 8)),
            # bottom right crop
            image.crop((3 * width // 8, 3 * height // 8, 7 * width // 8, 7 * height // 8)),
        ]

        rotations = [
            # 90 degrees
            image.rotate(90, expand=True, resample=Image.BICUBIC),
            # 180 degrees
            image.rotate(180, expand=True, resample=Image.BICUBIC),
            # 270 degrees
            image.rotate(270, expand=True, resample=Image.BICUBIC),
        ]

        return [
            image,
            *crops,
            *rotations,
        ]


class NPCopyDataset(torch.utils.data.Dataset):
    def __init__(self, paths, basic_transform, copy_transform, cfg):
        self.paths = paths
        self.basic_transform = basic_transform
        self.copy_transform = copy_transform
        self.n_positive = cfg.n_positive

        self.cfg = cfg

        logger.info(f"Build NPCopyDataset with n_positive = {self.n_positive}.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        anchor_img, *_ = self.basic_transform(img)
        imgs = [anchor_img]

        for _ in range(self.n_positive - 1):
            img, _ = self.copy_transform(img)
            imgs.append(img)

        return {
            "images": torch.stack(imgs, dim=0),
            "indices": torch.tensor([index for _ in range(self.n_positive)], dtype=torch.long),
        }


class SupervisedCopyDataset(CopyDataset):
    def __init__(self, que_paths, ref_paths, basic_transform, copy_transform, cfg):
        super().__init__(ref_paths, basic_transform, copy_transform, cfg)
        # self.path contains reference sets
        self.root = Path(cfg.root)
        self.ref_paths = ref_paths
        self.que_paths = que_paths

        self.que_df = pd.read_csv(cfg.que_csv)
        self.ref_df = pd.read_csv(cfg.ref_csv)
        self.que_uuid_to_index = {uuid: i for i, uuid in enumerate(self.que_df['uuid'])}
        self.ref_uuid_to_index = {uuid: i for i, uuid in enumerate(self.ref_df['uuid'])}
        self.que_uuid_to_path = {uuid: self.root / path for uuid, path in self.que_df[['uuid', 'path']].values.tolist()}
        self.ref_uuid_to_path = {uuid: self.root / path for uuid, path in self.ref_df[['uuid', 'path']].values.tolist()}

        with open(cfg.label_json, 'r') as file:
            self.labels = json.load(file)
        self.positive_que_uuids = [que_uuid for que_uuid in self.que_df['uuid'] if self.labels.get(que_uuid, [])]
        self.negative_que_uuids = [que_uuid for que_uuid in self.que_df['uuid'] if not self.labels.get(que_uuid, [])]

        if self.force_abandon_copynce:
            logger.warning(f"force_abandon_copynce is True. This will abandon copynce in descriptor "
                           f"training when force_hard_negative_mining is on, which copynce could be "
                           f"calculated.")

        logger.info(f"Build SupervisedCopyDataset with {len(self.positive_que_uuids)} positive queries.")

    def __len__(self):
        return int(len(self.positive_que_uuids) * self.sampling_rate / self.positive_rate)

    def __getitem__(self, index):
        index_after_sampling = int(index / self.sampling_rate) + random.randrange(max(1, int(1. / self.sampling_rate)))
        index_after_sampling = min(index_after_sampling, len(self.positive_que_uuids) - 1)

        if index_after_sampling < len(self.positive_que_uuids):
            # positive
            sample = self.fetch_sample_with_retry(index_after_sampling, self.fetch_positive_sample_que_vs_ref)
        else:
            # negative
            sample = self.fetch_sample_with_retry(index_after_sampling, self.fetch_negative_sample_que_vs_ref)

        return sample

    @func_set_timeout(5)
    def fetch_positive_sample_que_vs_ref(self, index):
        que_uuid = self.positive_que_uuids[index]
        que_path = self.que_uuid_to_path[que_uuid]
        query = self.read_image(que_path)
        query, _ = self.basic_transform(query)
        h_que, w_que = query.shape[-2:]

        # NOTE THAT: In default setting, only first gt will be used, despite the fact that
        # there may be more than one gt.
        ref_uuid = self.labels[que_uuid][0]
        ref_path = self.ref_uuid_to_path[ref_uuid]
        reference = self.read_image(ref_path)
        reference, _ = self.basic_transform(reference)

        input = {
            "references": reference,
            "queries": query,
            # NOTE THAT: image pair must be positive
            "cls_gt": torch.tensor(1., dtype=torch.long),
        }

        if self.return_mapping:
            # NOTE THAT: In SupervisedCopyDataset, there is no mapping between query and reference.
            # Because supervised data is built manually. To train model with mixed data with
            # self-supervised data (with actual mapping), we have to generate dummy mapping and
            # set force_abandon_copynce True to avoid involving dummy loss into back-propagation.
            que_to_ori_mapping = -1 * np.ones((2, h_que, w_que), dtype=np.float64)
            ref_to_ori_mapping = -1 * np.ones((2, h_que, w_que), dtype=np.float64)

            input["que_to_ori_mappings"] = que_to_ori_mapping  # noqa
            input["ref_to_ori_mappings"] = ref_to_ori_mapping  # noqa

        input["copynce"] = False

        return input
    
    @func_set_timeout(5)
    def fetch_negative_sample_que_vs_ref(self, index=None):
        # index will not be used
        que_uuid = random.choice(self.negative_que_uuids)
        que_path = self.que_uuid_to_path[que_uuid]
        query = self.read_image(que_path)
        query, _ = self.basic_transform(query)
        h_que, w_que = query.shape[-2:]
        que_index = self.que_uuid_to_index[que_uuid]

        ref_path = self.get_negative_paired_image_path(que_index)
        reference = self.read_image(ref_path)
        reference, _ = self.basic_transform(reference)

        input = {
            "references": reference,
            "queries": query,
            # NOTE THAT: image pair must be negative
            "cls_gt": torch.tensor(0., dtype=torch.long),
        }

        if self.return_mapping:
            # NOTE THAT: In SupervisedCopyDataset, there is no mapping between query and reference.
            # Because supervised data is built manually. To train model with mixed data with
            # self-supervised data (with actual mapping), we have to generate dummy mapping and
            # set force_abandon_copynce True to avoid involving dummy loss into back-propagation.
            que_to_ori_mapping = -1 * np.ones((2, h_que, w_que), dtype=np.float64)
            ref_to_ori_mapping = -1 * np.ones((2, h_que, w_que), dtype=np.float64)

            input["que_to_ori_mappings"] = que_to_ori_mapping  # noqa
            input["ref_to_ori_mappings"] = ref_to_ori_mapping  # noqa

        input["copynce"] = False

        return input
