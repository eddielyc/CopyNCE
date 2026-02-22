import argparse
import json
from tqdm import tqdm
from pathlib import Path
from loguru import logger

import torch
from torch.nn import functional as F


EXCLUSIVE = False
if EXCLUSIVE:
    exclusive_mapping_path = "data.json"
    with open(exclusive_mapping_path, 'r') as file:
        index_to_class = json.load(file)
        index_to_class = {int(k): v for k, v in index_to_class.items()}
    logger.warning(f"=> EXCLUSIVE mode is on. Load exclusive mapping from {exclusive_mapping_path}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Extract encoder and fusion layers from DINO ViT checkpoint."
    )
    parser.add_argument(
        "--query-feature-path",
        type=str,
        help="Path of query features."
    )
    parser.add_argument(
        "--reference-feature-path",
        type=str,
        help="Path of reference features."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=128,
        help="Number of the nearest neighbors."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for the k-NN matrix."
    )
    args = parser.parse_args()

    query_feature_path = Path(args.query_feature_path)
    reference_feature_path = Path(args.reference_feature_path)
    chunk_size = 256
    knn = 128

    query_features = torch.load(query_feature_path, map_location='cuda')["features"]
    logger.info(f"=> Load query features with shape {query_features.shape}")
    reference_features = torch.load(reference_feature_path, map_location='cuda')["features"]
    logger.info(f"=> Load reference features with shape {reference_features.shape}")

    query_features = F.normalize(query_features, dim=1).to(torch.float16)
    logger.info(f"=> Normalize query features done.")
    reference_features = F.normalize(reference_features, dim=1).to(torch.float16)
    logger.info(f"=> Normalize reference features done.")

    chunks = torch.split(query_features, chunk_size)
    knn_chunks = []
    for chunk_i, chunk in tqdm(enumerate(chunks)):
        cosines = torch.mm(chunk, reference_features.t())
        # CHECKME
        sorted_cosines, indices = torch.sort(cosines, descending=True, dim=1)
        if EXCLUSIVE:
            knn_chunk = []
            for i, _indices in enumerate(indices):
                index = chunk_i * chunk_size + i
                # knns is initialized with the index itself
                knns = [index]
                candidate_i = 1
                while len(knns) < 1 + knn and candidate_i < _indices.size(0):
                    _knn_i = _indices[candidate_i].item()
                    if index_to_class[index] != index_to_class[_knn_i]:
                        knns.append(_knn_i)
                    candidate_i += 1
                if len(knns) < 1 + knn:
                    logger.error(f"=> Index {index} has only {len(knns)} exclusive knns.")
                    raise RuntimeError
                knn_chunk.append(knns)
            knn_chunks.append(torch.tensor(knn_chunk, dtype=torch.int))
        else:
            knn_chunks.append(indices[:, : knn + 1].cpu().clone())
    knn_indices = torch.cat(knn_chunks, dim=0)
    torch.save(knn_indices, args.output_path)


if __name__ == '__main__':
    main()
