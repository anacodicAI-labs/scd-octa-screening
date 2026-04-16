from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    accuracy: float
    auc: float
    tp: int
    tn: int
    fp: int
    fn: int
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "sensitivity": float(self.sensitivity),
            "specificity": float(self.specificity),
            "precision": float(self.precision),
            "f1": float(self.f1),
            "accuracy": float(self.accuracy),
            "auc": float(self.auc),
            "tp": int(self.tp),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "n": int(self.n),
        }


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def specificity_from_cm(cm: np.ndarray) -> float:
    tn, fp, fn, tp = cm.ravel()
    denom = (tn + fp)
    return float(tn / denom) if denom else float("nan")


def pick_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float,
) -> float:
    """
    Choose the HIGHEST threshold that still achieves target sensitivity.
    """
    thresholds = np.unique(np.round(y_prob, 6))
    thresholds = np.concatenate([thresholds, np.array([0.0, 1.0])])
    thresholds = np.unique(thresholds)
    thresholds.sort()

    best = 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, y_pred, zero_division=0)
        if sens >= target_sensitivity:
            best = float(t)
    return best


def evaluate_binary_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> BinaryMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = specificity_from_cm(cm)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = safe_auc(y_true, y_prob)

    return BinaryMetrics(
        threshold=float(threshold),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision),
        f1=float(f1),
        accuracy=float(acc),
        auc=float(auc),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        n=int(len(y_true)),
    )

