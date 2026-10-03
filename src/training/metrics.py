"""Shared classification/statistics helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def confusion_matrix(y_true, y_pred, num_classes: int | None = None) -> np.ndarray:
    labels = list(range(num_classes)) if num_classes is not None else None
    return _confusion_matrix(y_true, y_pred, labels=labels)


def roc_auc_ovr(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))


def cohens_d(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    std = np.std(diff)
    if std == 0:
        return 0.0
    return float(np.mean(diff) / std)
