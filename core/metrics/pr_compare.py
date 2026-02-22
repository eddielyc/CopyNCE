import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.utils.multiclass import type_of_target
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils import check_consistent_length, assert_all_finite
from sklearn.utils.validation import column_or_1d
from typing import Dict, List, Tuple, Set, Iterable, Optional

from .common import *


def binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in ('O', 'U', 'S') or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError("y_true takes value in {{{classes_repr}}} and "
                         "pos_label is not specified: either make y_true "
                         "take value in {{0, 1}} or {{-1, 1}} or "
                         "pass pos_label explicitly.".format(
                             classes_repr=classes_repr))
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def black_white_pr(y_label, y_score):
    threshs = [i * 0.001 for i in range(900, 1000)]
    prec = []
    recall = []
    y_label = np.array(y_label)
    y_score = np.array(y_score)
    for t in threshs:
        label = y_label[y_score >= t]
        # black recall
        br = np.sum(label == 1) * 1.0 / np.sum(y_label == 1)
        # white recall
        wr = 1.0 - np.sum(label == 0) * 1.0 / np.sum(y_label == 0)
        # print(t, br, wr)
        prec.append(br)
        recall.append(wr)
    pr_display = PrecisionRecallDisplay(precision=np.array(prec), recall=np.array(recall))
    return pr_display


def print_pr(y_label, y_score, thresh_inv=False):
    threshs = [i * 0.02 for i in range(0, 50)]
    prec = []
    recall = []
    y_label = np.array(y_label)
    y_score = np.array(y_score)
    for t in threshs:
        y_pred = list()
        for x in y_score:
            y = 0
            if x >= t: y = 1
            y_pred.append(y)
        y_pred = np.array(y_pred)

        label = y_label[y_score >= t]
        br = np.sum(label == 1) * 1.0 / np.sum(y_label == 1)
        bp = np.sum(label == 1) * 1.0 / np.sum(y_pred == 1) * 1.0

        if thresh_inv:
            t = 1.0 - t
        # print(t, bp, br)


def draw_pr(save_path):
    if not save_path.endswith('/'):
        save_path = save_path + '/'
    y_true_np = np.load(save_path + "y_true_pos.npy")
    y_score_np = np.load(save_path + "y_score_pos.npy")
    y_true_neg_np = np.load(save_path + "y_true_neg.npy")
    y_score_neg_np = np.load(save_path + "y_score_neg.npy")

    print_pr(y_true_np, y_score_np)

    fps, tps, thresholds = binary_clf_curve(y_true_np, y_score_np)
    tns = fps[-1] - fps
    white_wrong_prec = 1 - (fps / (tns + fps))
    white_wrong_prec = np.r_[white_wrong_prec, 0]
    white_wrong_prec = white_wrong_prec[::-1]

    prec, recall, thresholds = precision_recall_curve(y_true_np, y_score_np)
    neg_prec, neg_recall, neg_thresholds = precision_recall_curve(y_true_neg_np, y_score_neg_np)

    pr_pos_display = PrecisionRecallDisplay(precision=prec, recall=recall)
    pr_neg_display = PrecisionRecallDisplay(precision=neg_prec, recall=neg_recall)
    white_wrong_pr_display = PrecisionRecallDisplay(precision=white_wrong_prec, recall=recall)

    high_thresh_black_white = black_white_pr(y_true_np, y_score_np)
    low_thresh_white_black = black_white_pr(y_true_neg_np, y_score_neg_np)
    return pr_pos_display, pr_neg_display, white_wrong_pr_display, high_thresh_black_white, low_thresh_white_black


def draw_multi_pr(save_path_list, labels, save_name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    color_list = ['blue', 'orange', 'green', 'pink', 'red', 'yellow', 'black', 'purple', 'cyan', 'grey', 'orchid']
    pr_pos_display = list()
    pr_neg_display = list()
    white_wrong_pr_display = list()
    high_thresh_black_display = list()
    low_thresh_white_display = list()
    for save_path in save_path_list:
        candi_pr_pos_display, candi_pr_neg_display, candi_white_wrong_pr_display, high_thresh_black_white, low_thresh_white_black = draw_pr(
            save_path)
        pr_pos_display.append(candi_pr_pos_display)
        pr_neg_display.append(candi_pr_neg_display)
        white_wrong_pr_display.append(candi_white_wrong_pr_display)
        high_thresh_black_display.append(high_thresh_black_white)
        low_thresh_white_display.append(low_thresh_white_black)

    for idx, one_pr_pos_display in enumerate(pr_pos_display):
        one_pr_pos_display.plot(ax=ax1, name=labels[idx], color=color_list[idx], linestyle='-')

    # for idx, one_white_wrong_display in enumerate(white_wrong_pr_display):
    #    one_white_wrong_display.plot(ax=ax1, name=labels[idx]+"_white_correct", color=color_list[idx], linestyle='--')

    for idx, one_pr_neg_display in enumerate(pr_neg_display):
        one_pr_neg_display.plot(ax=ax2, color=color_list[idx], name=labels[idx])

    for idx, one_high_display in enumerate(high_thresh_black_display):
        one_high_display.plot(ax=ax3, color=color_list[idx], name=labels[idx])

    for idx, one_low_display in enumerate(low_thresh_white_display):
        one_low_display.plot(ax=ax4, color=color_list[idx], name=labels[idx])

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax1.set_xlabel("black_recall")
    ax1.set_ylabel("black_precision")
    ax2.set_xlabel("white_recall")
    ax2.set_ylabel("white_precision")
    ax3.set_xlabel("inv_white_recall")
    ax3.set_ylabel("black_recall")
    ax4.set_xlabel("inv_black_recall")
    ax4.set_ylabel("white_recall")
    # save_name = "test_" + "_".join(labels) + "_with_thresh.png"
    plt.savefig(save_name)
    plt.show()


# def false_match(query_labels, gallery_labels, indexes, scores = None):
#     root = 'false_match'
#     D = scores
#
#     matches = (gallery_labels[indexes] == query_labels[:, np.newaxis]).astype(np.int32)
#     matches = matches[:, 0].reshape(len(matches), 1)
#
#     y_true = []
#     y_true_neg = []
#     y_true_score = []
#     y_true_neg_score = []
#     for i in range(len(matches)):
#         if sum(matches[i]) > 0:
#             y_true.append(1)
#             y_true_neg.append(0)
#             for j in range(len(matches[i])):
#                 if matches[i][j] == 1:
#                     y_true_score.append(D[i][0])
#                     y_true_neg_score.append(1.0-D[i][0])
#                     break
#         else:
#             y_true.append(0)
#             y_true_neg.append(1)
#             y_true_score.append(D[i][0])
#             y_true_neg_score.append(1.0-D[i][0])
#     subfix = '/res_34'
#     np.save(root+subfix+'/y_true_pos.npy', np.asarray(y_true))
#     np.save(root+subfix+'/y_true_neg.npy', np.asarray(y_true_neg))
#     np.save(root+subfix+'/y_score_pos.npy', np.asarray(y_true_score))
#     np.save(root+subfix+'/y_score_neg.npy', np.asarray(y_true_neg_score))


def binary_cls_match(
        results: List[Tuple[str, List[Tuple[str, float]]]],
        gt: Dict[str, set],
        output_dir: str,
        ):
    scores, labels, pairs = sort_pair_result(results, gt)

    y_true = labels
    y_true_neg = ~labels
    y_true_score = scores
    y_true_neg_score = 1 - scores

    np.save(os.path.join(output_dir, 'y_true_pos.npy'), np.asarray(y_true))
    np.save(os.path.join(output_dir, 'y_true_neg.npy'), np.asarray(y_true_neg))
    np.save(os.path.join(output_dir, 'y_score_pos.npy'), np.asarray(y_true_score))
    np.save(os.path.join(output_dir, 'y_score_neg.npy'), np.asarray(y_true_neg_score))


def main():
    labels = ['sigmoid_test']  # ['akaze','delf', 'global']
    # labels(directory name):y_true_pos.npy y_score_pos.npy y_true_neg.npy y_score_neg.npy
    save_path_list = [str(x) + '/' for x in labels]
    save_name = "test_" + "_".join(labels) + "_with_thresh.png"

    draw_multi_pr(save_path_list, labels, save_name)


if __name__ == "__main__":
    main()