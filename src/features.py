"""
Feature engineering with strict no-future-leakage discipline.

THE rule (TS rule #2): a feature row for date t may only use information
available strictly BEFORE t. We enforce this by shifting every lag/rolling
computation by 1 before applying .rolling().

Functions are pure: take a sorted long dataframe, add columns, return it.
The leakage guarantee is unit-tested in tests/test_features.py.

Expected input columns:
  id, date, sales, sell_price, cat_id, has_snap, has_event, is_weekend,
  wday, month, year, is_active
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- atomic feature builders ------------------------------------------------

def add_lag_features(df: pd.DataFrame, lags=(1, 7, 14, 28), col: str = "sales") -> pd.DataFrame:
    """Pure lag: lag_k[t] = col[t-k] within each id."""
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    g = df.groupby("id", sort=False)[col]
    for k in lags:
        df[f"{col}_lag_{k}"] = g.shift(k)
    return df


def add_rolling_features(df: pd.DataFrame, windows=(7, 28),
                         col: str = "sales", shift: int = 1) -> pd.DataFrame:
    """
    Rolling stats AFTER shifting by `shift` -- so the window for date t spans
    [t-shift-window+1 .. t-shift] and never includes t itself.
    """
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    def _shifted_roll(s: pd.Series, w: int, fn: str) -> pd.Series:
        sh = s.shift(shift)
        r = sh.rolling(w, min_periods=1)
        return getattr(r, fn)()

    for w in windows:
        df[f"{col}_rmean_{w}"] = (
            df.groupby("id", sort=False)[col]
              .transform(lambda s, w=w: _shifted_roll(s, w, "mean"))
        )
        df[f"{col}_rstd_{w}"] = (
            df.groupby("id", sort=False)[col]
              .transform(lambda s, w=w: _shifted_roll(s, w, "std"))
        )
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price-derived features (no leakage):
      - price_change_pct : % change of sell_price vs previous row (per id)
      - price_ratio_28d  : sell_price relative to its trailing 28-day mean (shifted)
    """
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    g = df.groupby("id", sort=False)["sell_price"]
    df["price_change_pct"] = g.pct_change()
    df["price_rmean_28"] = g.transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).mean()
    )
    df["price_ratio_28d"] = df["sell_price"] / df["price_rmean_28"]
    return df.drop(columns=["price_rmean_28"])


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight calendar encodings. (wday, month, year already present from src.data.)

      - wday_sin/cos : cyclical encoding of weekday (so Sat/Sun aren't 'far' from Fri)
      - snap_x_food  : SNAP x FOODS interaction (the strongest interaction in EDA)
    """
    df = df.copy()
    # wday in M5: 1=Sat,...,7=Fri ; encode as cyclic
    df["wday_sin"] = np.sin(2 * np.pi * df["wday"] / 7)
    df["wday_cos"] = np.cos(2 * np.pi * df["wday"] / 7)
    df["snap_x_food"] = (df["has_snap"] * (df["cat_id"] == "FOODS").astype(int)).astype("int8")
    return df


# ---- one-call pipeline ------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline used by Phase 2 baselines + Phase 3 deep model.

    NOTE: caller is responsible for filtering is_active=True BEFORE training,
    not here -- we keep the feature builder pure so EDA can show the pre-launch
    distribution if needed.
    """
    df = add_lag_features(df,    lags=(1, 7, 14, 28))
    df = add_rolling_features(df, windows=(7, 28))
    df = add_price_features(df)
    df = add_calendar_features(df)
    return df
