"""Statistical tests for H1, H2 and H3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "hypothesis"


def h1_paired_ttest(yolo26_maps, yolov8_maps) -> dict:
    """H1: YOLO26 mAP50 exceeds the YOLOv8 baseline, paired across seeds.

    ``alternative="greater"`` makes this one-sided — the hypothesis claims
    YOLO26 is better, so a significant result in the *other* direction does
    not count as support.
    """
    if len(yolo26_maps) != len(yolov8_maps) or len(yolo26_maps) < 2:
        raise ValueError("H1 needs paired mAP scores from at least 2 seeds.")
    stat, p = ttest_rel(yolo26_maps, yolov8_maps, alternative="greater")
    diff = np.asarray(yolo26_maps) - np.asarray(yolov8_maps)
    d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0
    return {
        "test": "paired t-test",
        "statistic": float(stat),
        "p_value": float(p),
        "cohens_d": d,
        "decision": bool(p < 0.05),
    }


def h2_mcnemar(y_true, pred_finetuned, pred_scratch) -> dict:
    y_true = np.asarray(y_true)
    pred_finetuned = np.asarray(pred_finetuned)
    pred_scratch = np.asarray(pred_scratch)
    if not (len(y_true) == len(pred_finetuned) == len(pred_scratch)):
        raise ValueError("H2 needs paired predictions on the same samples.")
    b = int(np.sum((pred_finetuned == y_true) & (pred_scratch != y_true)))
    c = int(np.sum((pred_finetuned != y_true) & (pred_scratch == y_true)))
    table = [[0, b], [c, 0]]
    from statsmodels.stats.contingency_tables import mcnemar

    result = mcnemar(table, exact=True)
    return {
        "test": "McNemar",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "decision": bool(result.pvalue < 0.05),
    }


def h3_wilcoxon(cnn_f1s, svm_f1s) -> dict:
    if len(cnn_f1s) != len(svm_f1s) or len(cnn_f1s) < 2:
        raise ValueError("H3 needs paired F1 scores from at least 2 runs.")
    stat, p = wilcoxon(cnn_f1s, svm_f1s, alternative="greater")
    diff = np.asarray(cnn_f1s) - np.asarray(svm_f1s)
    d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0
    return {
        "test": "Wilcoxon signed-rank",
        "statistic": float(stat),
        "p_value": float(p),
        "cohens_d": d,
        "decision": bool(p < 0.05),
    }


def save_result(name: str, payload: dict, out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
