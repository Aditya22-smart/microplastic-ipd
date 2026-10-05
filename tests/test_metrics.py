"""Tests for the shared classification/statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.training.metrics import (
    cohens_d,
    confusion_matrix,
    macro_f1,
    roc_auc_ovr,
)


def test_macro_f1_is_one_for_a_perfect_prediction() -> None:
    y = [0, 1, 2, 0, 1, 2]
    assert macro_f1(y, y) == pytest.approx(1.0)


def test_macro_f1_weights_classes_equally_not_by_frequency() -> None:
    """Macro averaging must not be swamped by a dominant class.

    90 samples of class 0 predicted perfectly, 10 of class 1 predicted as
    class 0. Accuracy is 0.90, but macro-F1 is near 0.47 — which is the whole
    point of reporting macro-F1 for the imbalanced morphology set.
    """
    y_true = [0] * 90 + [1] * 10
    y_pred = [0] * 90 + [0] * 10
    assert macro_f1(y_true, y_pred) == pytest.approx(0.4737, abs=1e-3)


def test_confusion_matrix_orientation_is_true_by_predicted() -> None:
    matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], num_classes=2)
    assert matrix.shape == (2, 2)
    # row = truth, column = prediction
    assert matrix[0, 0] == 1  # true 0 -> pred 0
    assert matrix[0, 1] == 1  # true 0 -> pred 1 (the error)
    assert matrix[1, 1] == 2


def test_confusion_matrix_includes_absent_classes_when_asked() -> None:
    """A class absent from both inputs still gets a row when num_classes is set."""
    matrix = confusion_matrix([0, 0], [0, 0], num_classes=5)
    assert matrix.shape == (5, 5)
    assert matrix.sum() == 2


def test_roc_auc_ovr_is_one_for_perfectly_separated_scores() -> None:
    y_true = [0, 0, 1, 1, 2, 2]
    # One-hot rows blurred a little, then normalised: sklearn's one-vs-rest AUC
    # expects a probability vector per sample, so the rows must sum to 1.
    scores = np.eye(3)[y_true] * 0.7 + 0.1
    scores /= scores.sum(axis=1, keepdims=True)
    assert np.allclose(scores.sum(axis=1), 1.0)
    assert roc_auc_ovr(y_true, scores) == pytest.approx(1.0)


def test_cohens_d_is_zero_for_identical_samples() -> None:
    assert cohens_d([1, 2, 3], [1, 2, 3]) == 0.0


def test_cohens_d_sign_follows_the_direction_of_the_difference() -> None:
    """The differences must vary, or the standard deviation is 0 and d collapses."""
    higher = cohens_d([3, 4, 6, 7], [1, 3, 4, 5])
    lower = cohens_d([1, 3, 4, 5], [3, 4, 6, 7])
    assert higher > 0
    assert lower < 0
    assert higher == pytest.approx(-lower)


def test_cohens_d_is_zero_when_the_shift_is_constant() -> None:
    """A perfectly uniform offset carries no variance, so d is undefined as 0."""
    assert cohens_d([3, 4, 5, 6], [1, 2, 3, 4]) == 0.0
