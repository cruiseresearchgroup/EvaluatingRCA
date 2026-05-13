"""Shared preprocessing utilities for PyRCA-based RCA adapters."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def preprocess(data: pd.DataFrame, patch: int = 100) -> pd.DataFrame:
    """Drop zero-variance columns and aggregate into patches by summation."""
    selector = VarianceThreshold(threshold=0)
    X = data.values
    X_var = selector.fit_transform(X)
    cols = data.columns[selector.get_support(indices=True)]

    sample = X_var.shape[0] // patch
    X_trimmed = X_var[: patch * sample, :]
    X_patched = np.sum(X_trimmed.reshape(-1, patch, X_var.shape[1]), axis=1)
    return pd.DataFrame(X_patched, columns=cols)


def preprocess_pair(
    train: pd.DataFrame, test: pd.DataFrame, patch: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patch-aggregate train and test together, keeping a column if it has
    variance in *either* window.

    Independent variance-thresholding per window (the original ``preprocess``)
    yields different column sets when an actuator is constant in the baseline
    but flips during the attack — PyRCA's ``find_root_causes`` then crashes
    with index/shape mismatches. This helper aligns the columns first so the
    two arrays can be compared.
    """
    common = [c for c in train.columns if c in test.columns]
    train = train[common]
    test = test[common]

    train_var = train.var(numeric_only=True) > 0
    test_var  = test.var(numeric_only=True)  > 0
    keep = (train_var | test_var)
    cols = list(keep.index[keep.values])

    def _patchify(df: pd.DataFrame) -> pd.DataFrame:
        X = df[cols].to_numpy()
        if X.shape[0] < patch or not cols:
            return pd.DataFrame(columns=cols)
        sample = X.shape[0] // patch
        X_trimmed = X[: patch * sample, :]
        X_patched = np.sum(X_trimmed.reshape(-1, patch, X.shape[1]), axis=1)
        return pd.DataFrame(X_patched, columns=cols)

    train_p = _patchify(train)
    test_p  = _patchify(test)

    # PyRCA's EpsilonDiagnosis computes ``np.cov(normal_col, abnormal_col)``
    # which requires equal row counts. Our 30-min baseline gives a fixed
    # 18 patches, but attack windows on SWaT range from 2 to 342 patches.
    # Truncate both to ``min(n)`` so the per-column covariance is well-defined.
    n = min(len(train_p), len(test_p))
    if n == 0:
        return train_p.iloc[:0], test_p.iloc[:0]
    return train_p.iloc[:n].reset_index(drop=True), test_p.iloc[:n].reset_index(drop=True)


def ranked_names(results: list) -> list[str]:
    """Extract root-cause names from pyrca result list, ordered by score."""
    return [r["root_cause"] for r in results]
