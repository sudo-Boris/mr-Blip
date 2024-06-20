import logging
import numpy as np
import ast
import re
from collections import defaultdict
import multiprocessing as mp


def r1_and_mIoU(submission, iou_thresholds=[0.3, 0.5, 0.7]):
    """Compute metrics for the moment retrieval task.

    Args:
        submission (list): a list of dictionaries, each containing the following keys:
            - pred_relevant_windows (list): a list of predicted relevant windows
            - relevant_windows (list): a list of ground truth relevant windows
        iou_thresholds (list): a list of iou thresholds, e.g. [0.3, 0.5, 0.7]

    Returns:
        dict: r1, a dictionary with iou thresholds as keys and r1 as values
        float: r1_avg, the average r1 over all iou thresholds
        float: mIoU, the mean IoU over all predictions
        int: invalid_pred_num, the number of invalid predictions
    """

    total_num = len(submission)
    r1 = {}
    iou_list = []
    invalid_pred_num = 0
    for t in iou_thresholds:
        r1[t] = 0
    for r in submission:
        predictions, targets = r["pred_relevant_windows"], r["relevant_windows"]

        _iou = []
        _r1 = {}
        for t in iou_thresholds:
            _r1[t] = 0

        if predictions == [[-1, -1]]:
            invalid_pred_num += 1
            continue

        # There can be multiple relevant windows as targets
        for i in range(len(targets)):
            # If the model has not predicted all relevant windows, the IoU for those are 0.
            if i >= len(predictions):
                _iou.extend([0] * (len(targets) - i))
                break

            # There appears to be a predicted window and a target window
            # Compute IoU between them
            pred = predictions[i]
            target = targets[i]
            try:
                out = compute_IoU(pred, target)
                _iou.append(out)
            except:
                # If the model has not learned to predict the right format, the IoU is 0.
                logging.warning(
                    f"Error when computing IoU between pred: {pred} and target: {target}"
                )
                _iou.append(0)

        # If there are more predictions than targets, the IoU for the extra predictions are 0
        # if len(predictions) > len(targets):
        #     _iou.extend([0] * (len(predictions) - len(targets)))

        if len(_iou) > 0:
            # collect all IoU scores for final mIoU computation
            iou_list.extend(_iou)

            # compute r1 for this current video
            # there are multiple r1 thresholds
            for t in iou_thresholds:
                # compute r1 for this current video with potential multiple relevant windows
                for iou in _iou:
                    if iou >= t:
                        _r1[t] += 1
                _r1[t] /= len(_iou)

                # add the r1 for this video to the total r1
                r1[t] += _r1[t]

    # total number of predictions should be equal to the number of valid predictions
    # plus the number of invalid predictions
    # assert total_num == len(r1[r1_tresholds[0]]) + invalid_pred_num

    # compute mIoU
    if len(iou_list) == 0:
        mIoU = 0
    else:
        mIoU = sum(iou_list) / len(iou_list)

    # compute r1
    r1 = {str(k): v / total_num for k, v in r1.items()}
    r1_avg = sum(r1.values()) / len(r1)

    return r1, r1_avg, mIoU, invalid_pred_num


def moment_str_to_list(m):
    """Convert a string of moments to a list of moments.
    If predicted string is not a list, it means that the model has not yet learned to predict the right format.
    In that case, we return [[-1, -1]] to represent an error.
    This will then lead to an IoU of 0.
    Args:
        m (str): a string of moments, e.g. "[[0, 1], [4, 7]]"
    Returns:
        list: a list of moments, e.g. [[0, 1], [4, 7]]
    """
    # check if the string has the right format of a nested list using regex
    # the list should look like this: [[0, 1], [4, 7], ...]
    # if not, return [[-1, -1]]
    if not re.match(r"\[\[.*\]\]", m):
        return [[-1, -1]]

    try:
        _m = ast.literal_eval(m)
    except:
        return [[-1, -1]]

    # if _m is not a list, it means that the model has not predicted any relevant windows
    # return error
    if not isinstance(_m, list):
        # raise ValueError()
        return [[-1, -1]]

    # if a sublist of _m has more than 2 elements, it means that the model has not learned to predict the right format
    # substitute that sublist with [-1, -1]
    for i in range(len(_m)):
        if len(_m[i]) != 2:
            # print(f"Got a sublist with more or less than 2 elements!{_m[i]}")
            _m[i] = [-1, -1]

    return _m


def compute_IoU(pred, target):
    """Compute IoU between two windows.
    Args:
        pred (list): a list of start and end time of a window, e.g. [0.0, 0.5]
        target (list): a list of start and end time of a window, e.g. [1.0, 2.0]
    Returns:
        float: IoU between pred and target
    """

    def compute_overlap(pred, target):
        if pred[0] > target[1] or pred[1] < target[0]:
            return 0
        else:
            return min(pred[1], target[1]) - max(pred[0], target[0])

    def compute_union(pred, target):
        if pred[0] > target[1] or pred[1] < target[0]:
            return 0
        else:
            return max(pred[1], target[1]) - min(pred[0], target[0])

    try:
        union = compute_union(pred, target)
    except:
        print(f"Union error: {pred}, {target}")
        union = 0

    if union == 0:
        return 0

    try:
        overlap = compute_overlap(pred, target)
    except:
        print(f"Overlap error: {pred}, {target}")
        overlap = 0

    return overlap / union


##############################################################################################################
### Mean average precision for moment retrieval
##############################################################################################################


def compute_mr_ap(
    submission,
    iou_thds=np.linspace(0.5, 0.95, 10),
    max_gt_windows=None,
    max_pred_windows=None,
    num_workers=8,
    chunksize=50,
):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    gt_qid2data = defaultdict(list)
    for d in submission:
        qid = d["qid"]

        # get predicted windows
        pred_windows = (
            d["pred_relevant_windows"][:max_pred_windows]
            if max_pred_windows is not None
            else d["pred_relevant_windows"]
        )
        for w in pred_windows:
            pred_qid2data[qid].append(
                {
                    "video-id": d["qid"],  # in order to use the API
                    "t-start": w[0],
                    "t-end": w[1],
                }
            )

        # get target windows
        gt_windows = (
            d["relevant_windows"][:max_gt_windows]
            if max_gt_windows is not None
            else d["relevant_windows"]
        )
        for w in gt_windows:
            gt_qid2data[qid].append(
                {"video-id": d["qid"], "t-start": w[0], "t-end": w[1]}
            )

    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [
        [qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data
    ]
    from functools import partial

    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds
    )

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(
                compute_ap_from_triple, data_triples, chunksize=chunksize
            ):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_average_precision_detection_wrapper(
    input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds
    )
    return qid, scores


def compute_average_precision_detection(
    ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (list[dict]): List containing the ground truth instances
            (dictionaries). Required keys are 'video-id', 't-start' and
            't-end'.
        prediction (list[dict]): List containing the prediction instances
            (dictionaries). Required keys are: 'video-id', 't-start', and 't-end'.
        tiou_thresholds (np.ndarray): A 1darray indicates the temporal
            intersection over union threshold, which is optional.
            Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        dict: Average precision at different iou thresholds.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    # Adaptation to query faster
    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item["index"] = i
        ground_truth_by_videoid.setdefault(item["video-id"], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred["video-id"] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred["video-id"]]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array(
            [
                [pred["t-start"], pred["t-end"]],
            ]
        )
        _gt = np.array([[gt["t-start"], gt["t-end"]] for gt in gts])
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0]

        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]["index"]] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(
            precision_cumsum[t_idx, :], recall_cumsum[t_idx, :]
        )
    return ap


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def compute_temporal_iou_batch_cross(spans1, spans2):
    """
    Args:
        spans1: (N, 2) np.ndarray, each row defines a span [st, ed]
        spans2: (M, 2) np.ndarray, ...

    Returns:
        iou: (N, M) np.ndarray
        union: (N, M) np.ndarray
    >>> spans1 = np.array([[0, 0.2, 0.9], [0.5, 1.0, 0.2]])
    >>> spans2 = np.array([[0, 0.3], [0., 1.0]])
    >>> compute_temporal_iou_batch_cross(spans1, spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = np.maximum(spans1[:, None, 0], spans2[None, :, 0])  # (N, M)
    right = np.minimum(spans1[:, None, 1], spans2[None, :, 1])  # (N, M)

    inter = np.clip(right - left, 0, None)  # (N, M)
    union = areas1[:, None] + areas2[None, :] - inter  # (N, M)

    iou = inter / union
    return iou, union
