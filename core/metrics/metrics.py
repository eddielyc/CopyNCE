from pathlib import Path

from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

from .common import *
from typing import Union
from operator import le, ge
from loguru import logger

from core.metrics.build import register_metric


def relevant_indicator(
                       retrieved_data: Iterable[Tuple[str, float]],
                       relevant_set: Set[str]) -> np.ndarray:
    """
    indicator function determines if each item in retrieved_data is
    in the relevant_set

    Parameters
    ----------
    retrieved_data : Iterable[Tuple[str, float]]
        result retrieved for a query, each item is a tuple of (id, similarity_score)
    relevant_set : Set[str]
        a set of item ids relevant to the query

    Returns
    -------
    numpy.ndarray, shape (N, )
    """
    return np.array([1 if k in relevant_set else 0 for k, v in retrieved_data])


def average_precision(rel_inds: np.ndarray, rel_cnt: int) -> float:
    """
    Average precision

    Parameters
    ----------
    rel_inds : numpy.ndarray
        array with values in {0, 1} where 1 indicates a true relevant item
    rel_cnt : int
        the total number of true relevant items

    Returns
    -------
    float
    """

    precision_at_r = np.cumsum(rel_inds) / np.arange(1, len(rel_inds) + 1)
    return np.sum(rel_inds * precision_at_r) / rel_cnt


def mean_average_precision(results: Iterable[Tuple[str, List[Tuple[str, float]]]],
                           gt: Dict[str, Set]) -> float:
    """
    Mean Average Precision

    Parameters
    ----------
    results
    gt

    Returns
    -------

    """
    if isinstance(results, dict):
        results = results.items()

    mean_ap = []
    for query_key, retrieved_list in results:
        if query_key not in gt or len(gt[query_key]) == 0:
            continue
        rel_inds = relevant_indicator(retrieved_list, gt[query_key])

        ap = average_precision(rel_inds, len(gt[query_key]))
        mean_ap.append(ap)

    pos_query_num = len([k for k in gt.values() if len(k) != 0])
    return np.sum(mean_ap) / pos_query_num


def dir_far(results: List[Tuple[str, List[Tuple[str, float]]]],
            gt: Dict[str, set], topk=1):

    scores, labels, pairs = sort_pair_result(results, gt, topk)

    negative_query_num = len([s for s in gt.values() if len(s) == 0])
    positive_query_num = len(gt) - negative_query_num
    print(f'pos: {positive_query_num} neg: {negative_query_num}')

    dirs = np.cumsum(labels) / positive_query_num
    fars = np.cumsum(~labels) / negative_query_num

    return fars, dirs, scores, labels, pairs


def find_operating_point(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, required_x: float, op: le
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Find the highest y (and corresponding z) with x at least or at most `required_x`.

    Returns
    -------
    x, y, z
        The best operating point (highest y) with x at least or at most `required_x`.
        If we can't find a point with the required x value, return
        x=required_x, y=None, z=None
    """
    valid_points = op(x, required_x)
    if not np.any(valid_points):
        return required_x, None, None

    valid_x = x[valid_points]
    valid_y = y[valid_points]
    valid_z = z[valid_points]
    best_idx = np.argmax(valid_y)
    return valid_x[best_idx], valid_y[best_idx], valid_z[best_idx]


def dir_at_far(
        results: List[Tuple[str, List[Tuple[str, float]]]],
        gt: Dict[str, Union[set, list]],
        required_far: float,
        topk=1
) -> Tuple[float, Optional[float], Optional[float], Dict[str, np.ndarray]]:
    """
        Find the highest dir (and corresponding threshold) with far at most `required_far`.

        Returns
        -------
        far, dir, threshold
            The best operating point (highest y) with x at least `required_x`.
            If we can't find a point with the required x value, return
            far=required_x, dir=None, threshold=None
        """
    fars, dirs, scores, labels, pairs = dir_far(results, gt, topk)

    x, y, z = find_operating_point(fars, dirs, scores, required_far, le)
    if y is None:
        print(f"Can't find a point with the required FAR {required_far}", )
    else:
        print(f"best DIR={y:.2%} @FAR={x:.2%}, threshold: {z:.3f}")
    return x, y, z, dict(fars=fars, dirs=dirs, scores=scores, labels=labels, pairs=pairs)


def log_auc(scores, gt, topk=1, min_logx=-5, max_logx=0):
    # take logarithmic range (10 ** min_logx, 10 ** max_logx)
    _, _, _, details = dir_at_far(scores, gt, topk=topk, required_far=0.01)
    fars, dirs = details['fars'], details['dirs']
    min_x, max_x = 10 ** min_logx, 10 ** max_logx
    inds = (fars >= min_x) & (fars <= max_x)
    fars = fars[inds]
    dirs = dirs[inds]

    # pad first fars, dir value keeps constant between [minx, fars[0]
    if fars[0] > min_x:
        fars = np.concatenate([[min_x], fars])
        dirs = np.concatenate([dirs[0:1], dirs])

    dx = np.diff(np.log10(fars))
    dy = (dirs[1:] + dirs[:-1]) / 2
    return np.sum(dx * dy) / (max_logx - min_logx)


def precision_recall(
    y_true: np.ndarray, probas_pred: np.ndarray, num_positives: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precisions, recalls, and thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Binary label of each prediction (0 or 1). Shape [n, k] or [n*k, ]
    probas_pred : np.ndarray
        Score of each prediction (higher score == images more similar, ie not a distance)
        Shape [n, k] or [n*k, ]
    num_positives : int
        Number of positives in the groundtruth.

    Returns
    -------
    precisions, recalls, thresholds
        ordered by increasing recall_at_threshold, as for a precision-recall_at_threshold curve
    """
    probas_pred = probas_pred.flatten()
    y_true = y_true.flatten()
    # to handle duplicates scores, we sort (score, NOT(jugement)) for predictions
    # eg,the final order will be (0.5, False), (0.5, False), (0.5, True), (0.4, False), ...
    # This allows to have the worst possible AP.
    # It prevents participants from putting the same score for all predictions to get a good AP.
    order = argsort(list(zip(probas_pred, ~y_true)))
    order = order[::-1]  # sort by decreasing score
    probas_pred = probas_pred[order]
    y_true = y_true[order]

    ntp = np.cumsum(y_true)  # number of true positives <= threshold
    nres = np.arange(len(y_true)) + 1  # number of results

    precisions = ntp / nres
    recalls = ntp / num_positives
    return precisions, recalls, probas_pred


def micro_average_precision(
        results: Union[List[Tuple[str, List[Tuple[str, float]]]], Dict[str, List[Tuple[str, float]]]],
        gt: Dict[str, Union[set, list]]
):
    if isinstance(results, dict):
        results = results.items()

    # convert retrieved lists in response to queries to a list of pairs (query, reference)
    pair_list = []
    labels = []
    scores = []
    for q, rel in results:
        for r, s in rel:
            pair_list.append((q, r))
            label = 1 if r in gt.get(q, []) else 0
            labels.append(label)
            scores.append(s)

    labels = np.array(labels)
    scores = np.array(scores)

    pair_gt_cnt = sum(map(len, gt.values()))
    precisions, recalls, t = precision_recall(labels, scores, pair_gt_cnt)

    # Micro-average precision

    # Check that it's ordered by increasing recall_at_threshold
    if not np.all(recalls[:-1] <= recalls[1:]):
        raise ValueError("recalls array must be sorted before passing in")
    return ((recalls - np.concatenate([[0], recalls[:-1]])) * precisions).sum()


def rp90(
        results: Union[List[Tuple[str, List[Tuple[str, float]]]], Dict[str, List[Tuple[str, float]]]],
        gt: Dict[str, Union[set, list]]
):
    if isinstance(results, dict):
        results = results.items()

    # convert retrieved lists in response to queries to a list of pairs (query, reference)
    pair_list = []
    labels = []
    scores = []
    for q, rel in results:
        for r, s in rel:
            pair_list.append((q, r))
            label = 1 if r in gt.get(q, []) else 0
            labels.append(label)
            scores.append(s)

    labels = np.array(labels)
    scores = np.array(scores)

    pair_gt_cnt = sum(map(len, gt.values()))
    precisions, recalls, t = precision_recall(labels, scores, pair_gt_cnt)

    pp90, rp90, tp90 = find_operating_point(precisions, recalls, t, required_x=0.9, op=ge)

    return rp90, tp90


def roc_auc(
        results: Union[List[Tuple[str, List[Tuple[str, float]]]], Dict[str, List[Tuple[str, float]]]],
        gt: Dict[str, Union[set, list]]
):
    if isinstance(results, dict):
        results = list(results.items())
    y_score, y_true = [], []
    for que, scores in results:
        for ref, score in scores:
            y_score.append(score)
            y_true.append(ref in gt.get(que, []) or que in gt.get(ref, []))

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr, thresholds


def far(
        results: List[Tuple[str, List[Tuple[str, float]]]],
        gt: Dict[str, Union[set, list]],
        threshold=0.5
) -> float:
    preds, labels = [], []
    for q, rs in results:
        for r, s in rs:
            labels.append(1. if r in gt.get(q, []) else 0.)
            if s > threshold:
                preds.append(1.)
            else:
                preds.append(0.)
    preds = np.array(preds, dtype=bool)
    labels = np.array(labels, dtype=bool)

    negative_pair_num = np.sum(~labels)
    false_alarm_num = np.sum(preds & ~labels)

    false_alarm_rate = false_alarm_num / negative_pair_num

    return float(false_alarm_rate)


def recall_at_threshold(
        results: List[Tuple[str, List[Tuple[str, float]]]],
        gt: Dict[str, Union[set, list]],
        threshold=0.5
) -> float:
    preds, labels = [], []
    for q, rs in results:
        for r, s in rs:
            labels.append(1. if r in gt.get(q, []) else 0.)
            if s > threshold:
                preds.append(1.)
            else:
                preds.append(0.)
    preds = np.array(preds, dtype=bool)
    labels = np.array(labels, dtype=bool)

    recall_rate = np.sum(preds & labels) / np.sum(labels)

    return float(recall_rate)


def recall_at_topk(
        results: List[Tuple[str, List[Tuple[str, float]]]],
        gt: Dict[str, Union[set, list]],
        k=None
) -> float:
    if k is None:
        k = max([len(rs) for q, rs in results])
    positive_n = sum([len(gt.get(q, [])) > 0 for q, rs in results])
    recalled_n = 0
    for q, rs in results:
        gt_rs = set(gt.get(q, []))
        refs = set(r for r, s in rs[:k])
        if len(gt_rs.intersection(refs)) > 0:
            recalled_n += 1
    recall_rate = recalled_n / positive_n
    return float(recall_rate)


class MetricWrapper(object):
    def __init__(self, eval_func, dump_dir=None, epoch=None):
        self.eval_func = eval_func
        self.dump_dir = dump_dir
        self.epoch = epoch

    def __call__(self, *eval_args, **eval_hparams):
        result = self.eval_func(*eval_args, **eval_hparams)
        metric, s = self.dump(result, **eval_hparams)
        return metric, s

    def dump(self, result, epoch=None, **eval_hparams):
        s = f" -- {self.eval_func.__name__}={result:.2%} "

        if len(eval_hparams):
            _s = ", ".join([f"{key}={value}" for key, value in eval_hparams.items()])
            s += f" ({_s}) \n"
        else:
            s += " \n"
        logger.info(s)
        if self.dump_dir is not None:
            with open(Path(self.dump_dir) / f"metrics{f'_ep-{epoch}' if epoch is not None else ''}.eval", mode="a") as file:
                file.write(s)
        return result, s


@register_metric("dir@far")
class DIRatFAR(MetricWrapper):
    def __init__(self, dump_dir=None, epoch=None):
        super().__init__(dir_at_far, dump_dir, epoch)

    def dump(self, results, epoch=None, **eval_hparams):
        required_far = eval_hparams["required_far"]
        topk = eval_hparams["topk"]

        false_alarm_rate, det_id_rate, th, details = results
        s = f" -- DIR@FAR={det_id_rate:.2%} (FAR={required_far:.3%}, topk={topk}, threshold={th:.4f}) \n"

        logger.info(s)
        if self.dump_dir is not None:
            with open(Path(self.dump_dir) / f"metrics{f'_ep-{epoch}' if epoch is not None else ''}.eval", mode="a") as file:
                file.write(s)

            np.savez(Path(self.dump_dir) / f"metrics{f'_ep-{epoch}' if epoch is not None else ''}.npz", **details)

            fig, ax = plt.subplots()
            ax.plot(details['fars'], details['dirs'])
            ax.set_xscale("log")
            ax.set(xlabel="FAR", ylabel="DIR")
            plt.savefig(Path(self.dump_dir) / f"auc{f'_ep-{epoch}' if epoch is not None else ''}.png", dpi=600)

        return det_id_rate, s


@register_metric("map")
class MeanAveragePrecision(MetricWrapper):
    def __init__(self, dump_dir=None, epoch=None):
        super().__init__(mean_average_precision, dump_dir, epoch)


@register_metric("recall_at_topk")
class RecallAtTopK(MetricWrapper):
    def __init__(self, dump_dir=None, epoch=None):
        super().__init__(recall_at_topk, dump_dir, epoch)


@register_metric("rp90")
class RP90(MetricWrapper):
    def __init__(self, dump_dir=None, epoch=None):
        super().__init__(rp90, dump_dir, epoch)

    def dump(self, results, epoch=None, **eval_hparams):
        recall_at_precision_90, th = results
        s = f" -- RP90={recall_at_precision_90:.2%} (threshold={th:.4f}) \n"

        logger.info(s)
        if self.dump_dir is not None:
            with open(Path(self.dump_dir) / f"metrics{f'_ep-{epoch}' if epoch is not None else ''}.eval", mode="a") as file:
                file.write(s)

        return recall_at_precision_90, s


@register_metric("uap")
class MicroAveragePrecision(MetricWrapper):
    def __init__(self, dump_dir=None, epoch=None):
        super().__init__(micro_average_precision, dump_dir, epoch)


@register_metric("log-auc")
class LogAUC(MetricWrapper):
    def __init__(self, dump_dir=None, epoch=None):
        super().__init__(log_auc, dump_dir, epoch)


@register_metric("roc-auc")
class ROCAUC(MetricWrapper):
    def __init__(self, dump_dir=None, epoch=None):
        super().__init__(roc_auc, dump_dir, epoch)

    def dump(self, results, epoch=None, **eval_hparams):
        roc_auc, fpr, tpr, thresholds = results
        s = f" -- ROC-AUC={roc_auc:.2%} \n"

        logger.info(s)
        if self.dump_dir is not None:
            with open(Path(self.dump_dir) / f"metrics{f'_ep-{epoch}' if epoch is not None else ''}.eval", mode="a") as file:
                file.write(s)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver operating characteristic')
            ax.legend(loc="lower right")
            plt.savefig(Path(self.dump_dir) / f"roc-auc{f'_ep-{epoch}' if epoch is not None else ''}.png", dpi=600)

        return roc_auc, s
