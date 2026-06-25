"""
Forecast metrics. Tested in tests/test_metrics.py.

Three accuracy metrics, chosen deliberately for retail demand forecasting:

  - rmse   : familiar, but zero-heavy series make its mean dominated by zeros
             -> we report it but don't pick the champion by it.

  - wmape  : weighted MAPE = sum|err| / sum|y|.  THE retail-industry standard.
             Robust to zeros (since the numerator and denominator are sums, not
             per-row ratios), interpretable as "average % error weighted by volume".

  - mase   : mean abs err divided by seasonal-naive's training in-sample error.
             Lower is better; <1 means we beat seasonal-naive, >1 means we lost.
             The scaling makes MASE comparable ACROSS series of different scales.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error. Returns 0..1 (multiply by 100 for %)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")        # undefined when truth is all zero
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int = 7) -> float:
    """
    Mean Absolute Scaled Error.

    Scale = mean abs error of the in-sample seasonal-naive forecast on y_train.
    For daily retail, season=7 (weekly).

    Returns >0; <1 means we beat seasonal-naive, =1 ties it, >1 we lost.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= season:
        return float("nan")
    naive_err = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if naive_err == 0:
        return float("nan")        # flat training series; MASE undefined
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae / naive_err)
