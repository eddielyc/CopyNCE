import torch
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from loguru import logger


def main():
    exp_name = "descriptor/dual-copynce"
    eval_cfg = "isc_vits_descriptor_e20_lin"
    query_csv = 'data/isc_25k_matching_dev_set_query.csv'

    logger.info(f"Loading aux features...")
    training_feats = torch.load(f"outputs/{exp_name}/eval/{eval_cfg}/train_feat_ep-30.pth")["features"]
    logger.info(f"Loading query features...")
    query_feats = torch.load(f"outputs/{exp_name}/eval/{eval_cfg}/query_feat_ep-30.pth")
    logger.info(f"Loading reference features...")
    reference_feats = torch.load(f"outputs/{exp_name}/eval/{eval_cfg}/reference_feat_ep-30.pth")

    training_feats = F.normalize(training_feats, dim=1, p=2).cuda()
    query_feats = F.normalize(query_feats, dim=1, p=2).cuda()
    reference_feats = F.normalize(reference_feats, dim=1, p=2).cuda()
    
    df_que = pd.read_csv(query_csv)
    df_ref = pd.read_csv('data/isc_matching_test_set_reference.csv')
    query_ids = np.array(df_que['uuid'])
    reference_ids = np.array(df_ref['uuid'])

    alpha = 2.5
    k = 5
    chunk_size = 128
    chunks = query_feats.split(chunk_size)
    _query_feats = []
    with torch.no_grad():
        for chunk in tqdm(chunks):
            similarity_matrix = chunk @ training_feats.t()
            sorted_similarities, _ = torch.sort(similarity_matrix, dim=1, descending=True)
            stretching_ratio = sorted_similarities[:, : k].detach().mean(dim=1, keepdim=True)
            _query_feats.append(alpha * stretching_ratio * chunk)

    query_feats = torch.cat(_query_feats, dim=0)

    k = 10
    scores = []
    chunks = query_feats.split(chunk_size)
    with torch.no_grad():
        for c_i, chunk in tqdm(enumerate(chunks)):
            dist_matrix = torch.cdist(chunk, reference_feats, p=2)
            sorted_dist, sorted_indices = torch.sort(dist_matrix, dim=1, descending=False)
            sorted_dist = sorted_dist[:, : k].detach().cpu().numpy()
            sorted_indices = sorted_indices[:, : k].detach().cpu().numpy()

            for _i, (dists, indices) in enumerate(zip(sorted_dist, sorted_indices)):
                i = c_i * chunk_size + _i
                _scores = [
                    query_ids[i],
                    [[uuid, -float(dist)] for uuid, dist in
                     zip(reference_ids[indices.tolist()], dists)]
                ]
                scores.append(_scores)

    with open("scores.json", 'w') as file:
        json.dump(scores, file)


if __name__ == "__main__":
    main()
