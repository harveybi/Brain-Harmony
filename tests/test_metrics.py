import numpy as np
from sklearn.metrics import balanced_accuracy_score

from modules.harmonizer.stage2_finetune.engine_finetune import (
    compute_balanced_accuracy,
    compute_mae,
    compute_mse,
)


def test_balanced_accuracy_matches_sklearn():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    expected = balanced_accuracy_score(y_true, y_pred)
    actual = compute_balanced_accuracy(y_true, y_pred)
    assert actual == expected


def test_mae_matches_expected():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])
    expected = np.mean(np.abs(y_true - y_pred))
    actual = compute_mae(y_true, y_pred)
    assert actual == expected


def test_mse_matches_expected():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])
    expected = np.mean((y_true - y_pred) ** 2)
    actual = compute_mse(y_true, y_pred)
    assert actual == expected
