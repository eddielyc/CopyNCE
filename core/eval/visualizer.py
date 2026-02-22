
from pathlib import Path
from typing import Sequence, Dict

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import io


def plt_fig_to_pil_image(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi)
    buf.seek(0)
    image = Image.open(buf)
    return image


def fetch_image(data, size=(384, 384)):
    if isinstance(data, Image.Image):
        return data.resize(size, resample=Image.LANCZOS)
    elif isinstance(data, (str, Path)):
        return Image.open(data).resize(size, resample=Image.LANCZOS)
    elif isinstance(data, np.ndarray):
        return Image.fromarray(data).resize(size, resample=Image.LANCZOS)
    else:
        raise ValueError(f"Unsupported image type: {type(data)}")


def false_match_processor(data):
    que = fetch_image(data["query"])

    n_images = len(data["references"]) + 1
    fig, axs = plt.subplots(nrows=1, ncols=n_images, figsize=(4.8 * n_images, 5))

    axs[0].imshow(que)
    axs[0].set_title('query')

    for i, (ref, score, match) in enumerate(zip(data["references"], data["scores"], data["matches"]), start=1):
        ref = fetch_image(ref)
        axs[i].imshow(ref)
        axs[i].set_title(f'ref #{i} score: {score:.4f} match: {match}')

    plt.tight_layout()
    visualized = plt_fig_to_pil_image(fig)
    return visualized


class Visualizer(object):
    def __init__(self, vis_root="visualization", exp_name="default"):
        self.vis_root = Path(vis_root)
        self.vis_root.mkdir(exist_ok=True, parents=True)

        self.vis_dir = self.vis_root / exp_name
        self.vis_dir.mkdir(exist_ok=True, parents=True)

    def visualize(self, data: Sequence, processor=fetch_image, sub_dir=None):
        data = data if isinstance(data, Sequence) else [data]
        vis_dir = self.vis_dir / sub_dir if sub_dir is not None else self.vis_dir
        vis_dir.mkdir(exist_ok=True, parents=True)

        for i, _data in tqdm(enumerate(data)):
            visualized = processor(_data)
            visualized.save(vis_dir / f"{i:05}.png")
