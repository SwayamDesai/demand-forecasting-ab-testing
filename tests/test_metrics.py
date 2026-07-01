"""Unit tests for the metrics module -- the numbers everything else is judged by."""
import numpy as np
import pytest

from src import metrics


def test_wmape_perfect_is_zero():
    y = [0, 1, 5, 0, 3]
    assert metrics.wmape(y, y) == 0.0


def test_wmape_known_value():
    # errors sum to 2 over actuals summing to 8 -> 0.25
    y = [2, 2, 2, 2]
    p = [3, 2, 2, 1]
    assert metrics.wmape(y, p) == pytest.approx(2 / 8)


def test_wmape_all_zero_truth_is_nan():
    assert np.isnan(metrics.wmape([0, 0, 0], [1, 2, 3]))


def test_bias_sign_under_and_over():
    y = [10, 10]
    assert metrics.bias_pct(y, [8, 8]) == pytest.approx(-20.0)   # under-forecast
    assert metrics.bias_pct(y, [12, 12]) == pytest.approx(+20.0)  # over-forecast


def test_seasonal_naive_scale_and_mase():
    # season=2 perfectly periodic -> in-sample seasonal-naive error 0 -> scale NaN
    y_train = [1, 5, 1, 5, 1, 5]
    assert np.isnan(metrics.seasonal_naive_scale(y_train, season=2))
    # non-trivial scale
    y_train = [1, 2, 3, 4, 5, 6]            # |t - (t-2)| = 2 everywhere -> scale 2
    scale = metrics.seasonal_naive_scale(y_train, season=2)
    assert scale == pytest.approx(2.0)
    # MASE: MAE 1 vs scale 2 -> 0.5
    assert metrics.mase([10, 10], [9, 11], scale) == pytest.approx(0.5)


def test_mase_undefined_scale_is_nan():
    assert np.isnan(metrics.mase([1, 2], [1, 2], float("nan")))


def test_rmse_known_value():
    assert metrics.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt((9 + 16) / 2))
