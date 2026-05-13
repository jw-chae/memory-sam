from __future__ import annotations

from typing import Dict

import numpy as np


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    tp = float(np.logical_and(pred_b, gt_b).sum())
    tn = float(np.logical_and(~pred_b, ~gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())
    eps = 1e-8
    iou_fg = tp / max(tp + fp + fn, eps)
    iou_bg = tn / max(tn + fp + fn, eps)
    pa_fg = tp / max(tp + fn, eps)
    pa_bg = tn / max(tn + fp, eps)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    dice = (2.0 * tp) / max(2.0 * tp + fp + fn, eps)
    acc = (tp + tn) / max(tp + tn + fp + fn, eps)
    return {
        "mIoU": float(0.5 * (iou_fg + iou_bg)),
        "mPA": float(0.5 * (pa_fg + pa_bg)),
        "Acc": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        "Dice": float(dice),
        "IoU_fg": float(iou_fg),
        "IoU_bg": float(iou_bg),
    }
