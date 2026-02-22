import argparse
from typing import Dict

import torch
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from loguru import logger

from core.loss import DiceLoss


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
        "--auxiliary-feature-path",
        type=str,
        help="Path of auxiliary features."
    )
    parser.add_argument(
        "--query-csv",
        type=str,
        default='data/isc_25k_matching_dev_set_query.csv',
        help="Path of auxiliary features.",
    )
    parser.add_argument(
        "--reference-csv",
        type=str,
        default='data/isc_matching_test_set_reference.csv',
        help="Path of auxiliary features."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.,
        help="Alpha weight of score normalization (Default: 1.)."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Start index of sorted auxiliary similarities (Default: 1, INCLUSIVE)."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=5,
        help="End index of sorted auxiliary similarities (Default: 5, EXCLUSIVE)."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for normalized rankings."
    )
    args = parser.parse_args()

    logger.info(f"Loading auxiliary features...")
    auxiliary_feats = torch.load(args.auxiliary_feature_path)
    if isinstance(auxiliary_feats, Dict) and "features" in auxiliary_feats:
        auxiliary_feats = auxiliary_feats["features"]
    elif not isinstance(auxiliary_feats, torch.Tensor):
        raise TypeError(
            f'Invalid type of auxiliary_feats, expect torch.Tensor or dict that contains key "features", '
            f'got {type(auxiliary_feats)} instead.'
        )

    logger.info(f"Loading query features...")
    query_feats = torch.load(args.query_feature_path)
    if isinstance(query_feats, Dict) and "features" in query_feats:
        query_feats = query_feats["features"]
    elif not isinstance(query_feats, torch.Tensor):
        raise TypeError(
            f'Invalid type of query_feats, expect torch.Tensor or dict that contains key "features", '
            f'got {type(query_feats)} instead.'
        )

    logger.info(f"Loading reference features...")
    reference_feats = torch.load(args.reference_feature_path)
    if isinstance(reference_feats, Dict) and "features" in reference_feats:
        reference_feats = reference_feats["features"]
    elif not isinstance(reference_feats, torch.Tensor):
        raise TypeError(
            f'Invalid type of reference_feats, expect torch.Tensor or dict that contains key "features", '
            f'got {type(reference_feats)} instead.'
        )

    auxiliary_feats = F.normalize(auxiliary_feats, dim=1, p=2).cuda()
    query_feats = F.normalize(query_feats, dim=1, p=2).cuda()
    reference_feats = F.normalize(reference_feats, dim=1, p=2).cuda()
    
    df_que = pd.read_csv(args.query_csv)
    df_ref = pd.read_csv(args.reference_csv)
    query_ids = np.array(df_que['uuid'])
    reference_ids = np.array(df_ref['uuid'])

    alpha = args.alpha
    k = 10
    ns, ne = args.start, args.end
    chunk_size = 128
    chunks = query_feats.split(chunk_size)
    topk_similarities, topk_indices = [], []
    topn_training_similarities = []
    with torch.no_grad():
        for chunk in tqdm(chunks):
            similarity_matrix = chunk @ reference_feats.t()
            sorted_similarities, indices = torch.sort(similarity_matrix, dim=1, descending=True)
            topk_similarities.append(sorted_similarities[:, : k].detach().cpu())
            topk_indices.append(indices[:, : k].detach().cpu())

            similarity_matrix = chunk @ auxiliary_feats.t()
            sorted_similarities, _ = torch.sort(similarity_matrix, dim=1, descending=True)
            topn_training_similarities.append(sorted_similarities[:, ns : ne].detach().cpu())

    topk_similarities = torch.cat(topk_similarities, dim=0)
    topk_indices = torch.cat(topk_indices, dim=0)
    topn_training_similarities = torch.cat(topn_training_similarities, dim=0)
    bias = torch.mean(topn_training_similarities, dim=1)

    scores = []
    for i in tqdm(range(query_feats.shape[0])):
        _scores = [
            query_ids[i],
            [[uuid, float(s) - alpha * float(bias[i])] for uuid, s in zip(reference_ids[topk_indices[i].tolist()], topk_similarities[i])]
        ]
        scores.append(_scores)
    
    with open(args.output_path, 'w') as file:
        json.dump(scores, file)


if __name__ == "__main__":
    main()

