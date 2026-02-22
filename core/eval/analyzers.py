from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image


class ComposedFilter(object):
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, result, gt, *args, **kwargs):
        return all(func(result, gt, *args, **kwargs) for func in self.funcs)


def is_false_match(result: Tuple[str, List[Tuple[str, float]]], gt: Dict[str, set], topk=1, *args, **kwargs):
    q, l = result
    if q not in gt or len(gt[q]) == 0:
        return False
    elif not any(r in gt[q] for r, s in l[:topk]):
        if not any(r in gt[q] for r, s in l):
            return False
        else:
            return True
    return False


class Analyzer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_root = Path(cfg.eval.data_root)
        ref_csv = pd.read_csv(cfg.eval.ref_csv_path)
        self.ref_ids = {uuid: path.strip() for uuid, path in ref_csv[["uuid", "path"]].values}
        que_csv = pd.read_csv(cfg.eval.que_csv_path)
        self.que_ids = {uuid: path.strip() for uuid, path in que_csv[["uuid", "path"]].values}

    def get_reference_path(self, uuid):
        return self.data_root / self.ref_ids[uuid]

    def get_query_path(self, uuid):
        return self.data_root / self.que_ids[uuid]


class ClsAnalyzer(Analyzer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def pixel_wise_different_from_groundtruth(
            self,
            result: Tuple[str, List[Tuple[str, float]]], gt: Dict[str, set],
            *args,
            **kwargs
    ):
        """
        Be placed after is_false_match
        """
        def pixel_wise_different(image_a, image_b):
            image_a = np.array(image_a, dtype=float)
            image_b = np.array(image_b, dtype=float)
            return np.sum(np.abs(image_a - image_b)) > 10

        q, l = result

        gt_ref_paths = [self.get_reference_path(ref) for ref in gt[q]]
        gt_refs = [
            Image.open(gt_ref_path).convert("RGB").resize(size=(224, 224), resample=Image.LANCZOS)
            for gt_ref_path in gt_ref_paths
        ]

        ref_path = self.get_reference_path(l[0][0])
        ref = Image.open(ref_path).convert("RGB").resize(size=(224, 224), resample=Image.LANCZOS)

        return all(pixel_wise_different(ref, gt_ref) for gt_ref in gt_refs)

    def false_match(self, results: List[Tuple[str, List[Tuple[str, float]]]], gt: Dict[str, set], topk=1):
        composed_filter = ComposedFilter(
            is_false_match,
            # self.pixel_wise_different_from_groundtruth
        )
        false_matches = filter(lambda result: composed_filter(result, gt, topk=topk), results)
        false_matches = list(false_matches)
        print(f"Number of false matches: {len(false_matches)}.")

        data, serializable_data = [], []
        for (uuid, result) in false_matches:
            references = []
            scores = []
            matches = []
            for ref, score in result:
                references.append(self.get_reference_path(ref))
                scores.append(score)
                match = True if ref in gt.get(uuid, set()) else False
                matches.append(match)
            data.append(
                {
                    "query": self.get_query_path(uuid),
                    "references": references,
                    "scores": scores,
                    "matches": matches,
                }
            )
            serializable_data.append(
                {
                    "query": uuid,
                    "references": list(zip(*result))[0],
                    "scores": scores,
                    "matches": matches,
                }
            )

        return data, serializable_data
