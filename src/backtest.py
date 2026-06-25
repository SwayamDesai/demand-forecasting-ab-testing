"""
Rolling-origin (time-based) backtesting -- TS rule #3.

We simulate "every H days you retrain and forecast the next H days" by sliding
the cutoff forward `n_folds` times. NEVER a random split.

Visual (3 folds, horizon=H, step=H, latest date = T):

       train ------------------------|--- test ---|         fold 1 (newest)
                                  ^cutoff_1
       train -------------------|--- test ---|              fold 2
                              ^cutoff_2
       train ----------------|--- test ---|                 fold 3 (oldest)
                          ^cutoff_3

Returned splits cover the LAST `n_folds * step` days of the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (f"Fold({self.name}: train<= {self.train_end.date()} | "
                f"test {self.test_start.date()}..{self.test_end.date()})")


def rolling_origin_splits(
    last_date: pd.Timestamp,
    n_folds: int = 3,
    horizon: int = 28,
    step: int = 28,
) -> list[Fold]:
    """
    Build `n_folds` test windows of `horizon` days each, sliding by `step` days.

    Returned in chronological order (oldest fold first).
    """
    last_date = pd.Timestamp(last_date)
    folds: list[Fold] = []
    for i in range(n_folds):
        test_end = last_date - pd.Timedelta(days=i * step)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)
        folds.append(Fold(f"fold_{n_folds - i}", train_end, test_start, test_end))
    return list(reversed(folds))


def split_frame(df: pd.DataFrame, fold: Fold,
                date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice df into (train, test) for one fold. Both sorted by id, date."""
    train = df[df[date_col] <= fold.train_end]
    test = df[(df[date_col] >= fold.test_start) & (df[date_col] <= fold.test_end)]
    return (train.sort_values(["id", date_col]).reset_index(drop=True),
            test.sort_values(["id", date_col]).reset_index(drop=True))
