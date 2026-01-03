import numpy as np
from sklearn.metrics import balanced_accuracy_score

from modules.harmonizer.stage2_finetune.engine_finetune import (
    compute_balanced_accuracy,
)


def test_balanced_accuracy_matches_sklearn():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    expected = balanced_accuracy_score(y_true, y_pred)
    actual = compute_balanced_accuracy(y_true, y_pred)
    assert actual == expected
