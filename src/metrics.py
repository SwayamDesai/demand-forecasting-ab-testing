"""
Forecast metrics, chosen for intermittent weekly retail demand. Pure functions,
unit-tested in tests/test_metrics.py.

  wmape : sum|y-yhat| / sum|y|. The retail standard. Robust to zero weeks (numerator
          and denominator are sums, not per-row ratios). Pooling it across all
          series-weeks makes it inherently VOLUME-WEIGHTED -- the units that matter
          dominate, which is what we want for a business that holds inventory.
  rmse  : reported, but not used to pick the champion (squared error over a zero-heavy
          target rewards timid forecasts).
  bias_pct : sum(yhat-y)/sum(y)*100. Sign matters: negative = systematic under-forecast
          (the failure mode that drives stockouts). We watch this everywhere.
  mase  : MAE scaled by the in-sample seasonal-naive MAE. <1 beats seasonal-naive.
          Comparable across series of different scale; we report the median.
"""
from __future__ import annotations

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def wmape(y_true, y_pred) -> float:
    y, p = _arr(y_true), _arr(y_pred)
    denom = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denom) if denom > 0 else float("nan")


def rmse(y_true, y_pred) -> float:
    y, p = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def bias_pct(y_true, y_pred) -> float:
    """Total signed error as a % of total actuals. <0 = under-forecast."""
    y, p = _arr(y_true), _arr(y_pred)
    denom = np.abs(y).sum()
    return float((p - y).sum() / denom * 100) if denom > 0 else float("nan")


def seasonal_naive_scale(y_train, season: int = 52) -> float:
    """Mean abs error of the in-sample seasonal-naive forecast on the train series."""
    y = _arr(y_train)
    if len(y) <= season:
        return float("nan")
    err = np.abs(y[season:] - y[:-season])
    m = float(np.mean(err)) if len(err) else float("nan")
    return m if m and m > 0 else float("nan")


def mase(y_true, y_pred, scale: float) -> float:
    """MAE / precomputed seasonal-naive scale. NaN if scale undefined."""
    if not scale or np.isnan(scale):
        return float("nan")
    y, p = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(y - p)) / scale)
