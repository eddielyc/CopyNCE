import numpy as np

from typing import Dict, List, Tuple, Set, Iterable, Optional, Sequence, Any


def argsort(seq: Sequence[Any]):
    """Like np.argsort but for 1D sequences. Based on https://stackoverflow.com/a/3382369/3853462"""
    return sorted(range(len(seq)), key=seq.__getitem__)


def find_true_match(results: List[Tuple[str, List[Tuple[str, float]]]], gt: Dict[str, set]):
    gt_scores = []
    gt_pairs = []
    for q, l in results:
        for r, s in l:
            if r in gt[q]:
                gt_scores.append(s)
                # there's only one ground truth
                gt_pairs.append((q, r))
                break
    return np.array(gt_scores, dtype=np.float32), np.array(gt_pairs)


def sort_pair_result(
        results: List[Tuple[str, List[Tuple[str, float]]]],
        gt: Dict[str, set],
        topk=1):
    positive_queries = [(q, l[:topk]) for q, l in results if q in gt]

    gt_scores, gt_pairs = find_true_match(positive_queries, gt)

    negative_samples = [((q, l[0][0]), l[0][1]) for q, l in results if len(gt.get(q, [])) == 0]
    neg_pairs, neg_scores = map(np.array, zip(*negative_samples))

    scores = np.concatenate([gt_scores, neg_scores])
    pairs = np.concatenate([gt_pairs, neg_pairs])

    # 1 for true match, 0 for false match
    labels = np.concatenate([np.ones(gt_scores.shape, dtype=bool), np.zeros(neg_scores.shape, dtype=bool)])

    order = argsort(list(zip(scores, ~labels)))
    order = order[::-1]  # sort by decreasing score

    scores = scores[order]
    labels = labels[order]
    pairs = pairs[order]
    return scores, labels, pairs
