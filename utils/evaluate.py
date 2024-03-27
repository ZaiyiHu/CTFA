import numpy as np
import sklearn.metrics as metrics

def multilabel_score(y_true, y_pred):

    return metrics.f1_score(y_true, y_pred,zero_division=0)


import numpy as np


def _fast_hist(label_true, label_pred, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
    )
    return hist.reshape(num_classes, num_classes)


def scores(label_trues, label_preds, num_classes=6):
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)

    acc = np.diag(hist).sum() / hist.sum()
    _acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(_acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    cls_iu = dict(zip(range(num_classes), iu))

    # Overall Accuracy (OA)
    oa = np.diag(hist).sum() / hist.sum()

    # F1 Score
    precision = np.diag(hist) / hist.sum(axis=1)
    recall = np.diag(hist) / hist.sum(axis=0)
    f1 = 2 * (precision * recall) / (precision + recall)
    valid_f1 = f1[valid]
    mean_f1 = np.nanmean(valid_f1)
    return {
        "pAcc": acc,
        "mAcc": acc_cls,
        "miou": mean_iu,
        "iou": cls_iu,
        "OA": oa,
        "F1": mean_f1,
    }


def pseudo_scores(label_trues, label_preds, num_classes=16):
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        lt = lt.flatten()
        lp = lp.flatten()
        lt[lp==255] = 255
        lp[lp==255] = 0
        hist += _fast_hist(lt, lp, num_classes)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    cls_iu = dict(zip(range(num_classes), iu))

    return {
        "pAcc": acc,
        "mAcc": acc_cls,
        "miou": mean_iu,
        "iou": cls_iu,
    }