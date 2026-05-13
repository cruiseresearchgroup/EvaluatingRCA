"""Evaluation metrics — top@k and Avg@k.

Two metrics, matching the reporting in the paper's Tables 5 and 6:

  top@k   — 1 if any ground-truth root cause appears in the top-k predictions.
  Avg@k   — mean over (1..k) of top@j; the area under the top-k curve.
"""

import numpy as np


def top_at_k(y_true: list[str], y_pred: list[str], k: int) -> int:
    """top@k — 1 if any ground-truth item appears in the top-k predictions."""
    return int(any(t in y_pred[:k] for t in y_true))


def avg_at_k(
    y_true_list: list[list[str]],
    y_pred_list: list[list[str]],
    k: int,
) -> float:
    """Avg@k — mean over scenarios of sum(top@1 .. top@k) / k."""
    scores = []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        scores.append(sum(top_at_k(y_true, y_pred, j) for j in range(1, k + 1)) / k)
    return float(np.mean(scores)) if scores else 0.0
