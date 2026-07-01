"""Backtest fold construction: correct windows, no overlap, train precedes test."""
import pandas as pd

from src import backtest


def _weeks(n):
    return pd.Series(pd.date_range("2020-01-04", periods=n, freq="7D"))


def test_three_nonoverlapping_4w_folds():
    weeks = _weeks(60)
    folds = backtest.make_folds(weeks, horizon=4, n_folds=3)
    assert [f.name for f in folds] == ["fold_1", "fold_2", "fold_3"]
    # each test window is exactly 4 weeks
    for f in folds:
        span = (f.test_end - f.test_start).days // 7 + 1
        assert span == 4
        assert f.train_end < f.test_start          # no leakage: train strictly before test
    # newest fold ends on the last available week
    assert folds[-1].test_end == weeks.iloc[-1]
    # folds step back by exactly the horizon and don't overlap
    assert folds[2].test_start > folds[1].test_end
    assert folds[1].test_start > folds[0].test_end
    assert (folds[2].test_start - folds[1].test_start).days == 28


def test_cutoffs_match_train_ends():
    weeks = _weeks(40)
    folds = backtest.make_folds(weeks, horizon=4, n_folds=3)
    assert backtest.cutoffs(weeks, horizon=4, n_folds=3) == [f.train_end for f in folds]
