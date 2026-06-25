"""Real assertions for src.metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import mase, rmse, wmape


def test_rmse_perfect_prediction_is_zero():
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    # errors: 1, -1, 1, -1 -> squared 1,1,1,1 -> mean 1 -> sqrt 1
    y_true = np.array([1, 2, 3, 4], dtype=float)
    y_pred = np.array([0, 3, 2, 5], dtype=float)
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_wmape_perfect_prediction_is_zero():
    y = np.array([1, 2, 3, 4], dtype=float)
    assert wmape(y, y) == 0.0


def test_wmape_known_value():
    # |err| = [1,1,1,1] sum=4 ; |y|=[1,2,3,4] sum=10 -> 0.4
    y_true = np.array([1, 2, 3, 4], dtype=float)
    y_pred = np.array([0, 3, 2, 5], dtype=float)
    assert wmape(y_true, y_pred) == pytest.approx(0.4)


def test_wmape_robust_to_zero_truth_rows():
    # individual zero-truth rows do NOT blow up wmape (unlike plain MAPE)
    y_true = np.array([0, 0, 5, 0, 5], dtype=float)
    y_pred = np.array([1, 0, 5, 0, 5], dtype=float)
    # |err|=1, sum|y|=10 -> 0.1
    assert wmape(y_true, y_pred) == pytest.approx(0.1)


def test_wmape_nan_when_truth_all_zero():
    y_true = np.zeros(5)
    y_pred = np.array([1, 0, 0, 0, 0], dtype=float)
    assert math.isnan(wmape(y_true, y_pred))


def test_mase_equals_one_for_seasonal_naive_replay():
    # If pred = seasonal-naive (y_t = y_{t-season}) of train, MAE matches the scale
    rng = np.random.default_rng(0)
    season = 7
    y_train = rng.integers(0, 20, size=200).astype(float)
    # use the same scaling formula as the forecast error -> MASE ~ 1
    y_true = y_train[-14:]
    y_pred = y_train[-14 - season : -season]  # seasonal-naive on the same data
    val = mase(y_true, y_pred, y_train, season=season)
    # not exactly 1 because train slice differs, but should be close to it
    assert 0.5 <= val <= 2.0


def test_mase_lt_one_when_we_beat_naive():
    # noisy weekly series so the seasonal-naive scale is nonzero
    rng = np.random.default_rng(42)
    season = 7
    y_train = (np.tile([1, 2, 3, 4, 5, 6, 7], 30).astype(float)
               + rng.normal(0, 1, 210))
    y_true = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
    y_pred = y_true.copy()  # perfect forecast
    val = mase(y_true, y_pred, y_train, season=season)
    assert val == 0.0


def test_mase_nan_for_flat_training_series():
    y_train = np.ones(100, dtype=float)
    y_true = np.array([1, 1, 1], dtype=float)
    y_pred = np.array([1, 1, 1], dtype=float)
    assert math.isnan(mase(y_true, y_pred, y_train))
