"""
Rolling-origin backtest at the WEEKLY grain. One source of truth for the test
windows so Phases 5-8 (classical, ML, DL, A/B) all evaluate on the *identical*
(series, week) cells -- otherwise their numbers aren't comparable.

We mirror statsforecast's cross_validation convention exactly:
    n_folds windows of `horizon` weeks each, stepping back by `horizon`,
    anchored at the last available week. Fold 1 is the oldest, fold n the newest.

Visual (horizon=H, 3 folds, last week = T):
    train .............|--H--|                fold_1  cutoff = T-2H
    train ...................|--H--|          fold_2  cutoff = T-1H
    train .........................|--H--|    fold_3  cutoff = T   (newest)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass(frozen=True)
class WeekFold:
    name: str
    train_end: pd.Timestamp     # last week used for training (inclusive)
    test_start: pd.Timestamp    # first test week
    test_end: pd.Timestamp      # last test week


def make_folds(week_ends: pd.Series,
               horizon: int = config.HORIZON_WEEKS,
               n_folds: int = config.N_FOLDS) -> list[WeekFold]:
    """Build folds from the global ordered list of week-ending dates."""
    weeks = pd.Index(sorted(pd.Index(week_ends.unique())))
    folds: list[WeekFold] = []
    for i in range(n_folds):
        # newest fold first, then we reverse so fold_1 is oldest
        end_pos = len(weeks) - 1 - i * horizon
        start_pos = end_pos - horizon + 1
        if start_pos - 1 < 0:
            raise ValueError("not enough weeks for the requested folds/horizon")
        folds.append(WeekFold(
            name=f"fold_{n_folds - i}",
            train_end=weeks[start_pos - 1],
            test_start=weeks[start_pos],
            test_end=weeks[end_pos],
        ))
    return list(reversed(folds))


def cutoffs(week_ends: pd.Series, horizon=config.HORIZON_WEEKS,
            n_folds=config.N_FOLDS) -> list[pd.Timestamp]:
    """The statsforecast-style cutoffs (= each fold's train_end)."""
    return [f.train_end for f in make_folds(week_ends, horizon, n_folds)]
