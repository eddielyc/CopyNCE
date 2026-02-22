import json
from itertools import chain
from pathlib import Path

import pandas as pd
from loguru import logger

from core.data.build import register_data_paths


def build_image_paths_from_folder(path):
    image_paths = list(
        chain(
            Path(path).glob('*.jpg'),
            Path(path).glob('*.png'),
            Path(path).glob('*.jpeg'),
            Path(path).glob('*.JPG'),
            Path(path).glob('*.PNG'),
            Path(path).glob('*.JPEG'),
        )
    )
    image_paths = sorted(image_paths)
    return image_paths


@register_data_paths("folder")
def build_folder_paths(_cfg):
    return build_image_paths_from_folder(_cfg.root)


# for fast reference
@register_data_paths("isc_query_val")
def build_isc_query_val_paths(_cfg):
    query_paths = build_image_paths_from_folder("datasets/DISC21/query_images")
    return [path for path in query_paths if int(path.stem[1:]) < 25_000]


# for fast reference
@register_data_paths("isc_query_dev")
def build_isc_query_dev_paths(_cfg):
    query_paths = build_image_paths_from_folder("datasets/DISC21/query_images")
    return [path for path in query_paths if 25_000 <= int(path.stem[1:]) < 50_000]


# for fast reference
@register_data_paths("isc_query_test")
def build_isc_query_test_paths(_cfg):
    query_paths = build_image_paths_from_folder("datasets/DISC21/query_images")
    return [path for path in query_paths if 50_000 <= int(path.stem[1:])]


# for fast reference
@register_data_paths("isc_reference")
def build_isc_reference_paths(_cfg):
    return build_image_paths_from_folder("datasets/DISC21/reference_images")


# for SSL training
@register_data_paths("isc")
def build_isc_paths(_cfg):
    from core import project_root
    data_root = getattr(_cfg, "root", "datasets/DISC21/training_images")
    data_root = Path(data_root)
    json_path = getattr(_cfg, "json_path", project_root / "data" / "DISC21_train.json")
    with open(json_path, 'r') as file:
        logger.info(f"Loading DISC training data, this might take a while...")
        paths = json.load(file)
        paths = [data_root / path for path in paths]
    return paths


# for supervised training
@register_data_paths("isc_val")
def build_isc_val_paths(_cfg):
    data_root = getattr(_cfg, "root", "datasets/DISC21/training_images")
    data_root = Path(data_root)
    que_df = pd.read_csv(_cfg.que_csv)
    ref_df = pd.read_csv(_cfg.ref_csv)
    que_paths = [data_root / path for path in que_df["path"].tolist()]
    ref_paths = [data_root / path for path in ref_df["path"].tolist()]
    return que_paths, ref_paths
