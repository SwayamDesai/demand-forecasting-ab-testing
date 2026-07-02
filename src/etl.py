"""
ETL: raw M5 -> clean daily -> weekly grain -> leakage-safe features.

Pure functions shared by scripts/prepare_data.py and the tests. The leakage
guarantee (a feature row for week t uses only weeks < t) is unit-tested in
tests/test_features.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
STATIC_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
LAGS = (1, 2, 4, 8, 52)
ROLL_WINDOWS = (4, 8, 13)


# ---- daily cleaning -----------------------------------------------------------

def melt_long(sample_wide: pd.DataFrame) -> pd.DataFrame:
    """Wide d_1..d_1941 columns -> one (id, d, sales) row per series-day."""
    day_cols = [c for c in sample_wide.columns if c.startswith("d_")]
    long = sample_wide.melt(
        id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales"
    )
    long["sales"] = long["sales"].astype("int32")
    return long


def join_calendar(long: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    keep = ["d", "date", "wm_yr_wk", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"]
    return long.merge(cal[keep], on="d", how="left")


def join_prices(long_cal: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return long_cal.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """is_active (launched from first priced week) + state-aware calendar flags."""
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    df["is_active"] = (
        df.groupby("id")["sell_price"].transform(lambda s: s.notna().cummax()).astype(bool)
    )

    df["is_weekend"] = df["wday"].isin(config.WEEKEND_WDAYS).astype("int8")
    df["has_event"] = df["event_name_1"].notna().astype("int8")

    df["has_snap"] = 0
    for state, col in config.SNAP_BY_STATE.items():
        m = df["state_id"] == state
        df.loc[m, "has_snap"] = df.loc[m, col]
    df["has_snap"] = df["has_snap"].astype("int8")
    return df


# ---- weekly aggregation ---------------------------------------------------------

def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Sum sales to (id, wm_yr_wk); carry week-level calendar/price attributes."""
    daily = daily.copy()
    daily["has_snap"] = daily["has_snap"].astype(int)
    daily["has_event"] = daily["has_event"].astype(int)
    daily["is_active_i"] = daily["is_active"].astype(int)

    grp = daily.groupby(["id", "wm_yr_wk"], sort=False)
    weekly = grp.agg(
        sales=("sales", "sum"),
        sell_price=("sell_price", "mean"),
        week_start_date=("date", "min"),
        week_end_date=("date", "max"),
        n_days=("date", "size"),
        n_active_days=("is_active_i", "sum"),
        snap_days=("has_snap", "sum"),
        event_days=("has_event", "sum"),
        **{c: (c, "first") for c in STATIC_COLS},
    ).reset_index()
    weekly["year"] = weekly["week_end_date"].dt.year.astype("int16")
    weekly["month"] = weekly["week_end_date"].dt.month.astype("int8")
    return weekly


def keep_full_active(weekly_all: pd.DataFrame) -> pd.DataFrame:
    """Keep weeks that are full (7 calendar days) AND fully active (launched)."""
    mask = (weekly_all["n_days"] == 7) & (weekly_all["n_active_days"] == 7)
    return weekly_all[mask].reset_index(drop=True)


# ---- leakage-safe weekly features -----------------------------------------------

def _weeks_since_last_sale(sales: pd.Series) -> np.ndarray:
    """Weeks since the most recent week (< t) with a sale. NaN until the first sale."""
    s = sales.to_numpy()
    out = np.full(len(s), np.nan)
    last = None
    for i in range(len(s)):
        if last is not None:
            out[i] = i - last
        if s[i] > 0:
            last = i
    return out


def add_features(w: pd.DataFrame) -> pd.DataFrame:
    """Every lag/rolling is .shift()-ed: week t sees only weeks strictly before t."""
    w = w.sort_values(["id", "week_end_date"]).reset_index(drop=True)
    g = w.groupby("id", sort=False)

    for k in LAGS:
        w[f"sales_lag_{k}"] = g["sales"].shift(k)

    for win in ROLL_WINDOWS:
        w[f"sales_rmean_{win}"] = g["sales"].transform(
            lambda s, win=win: s.shift(1).rolling(win, min_periods=1).mean())
        w[f"sales_rstd_{win}"] = g["sales"].transform(
            lambda s, win=win: s.shift(1).rolling(win, min_periods=2).std())

    w["trailing_zero_rate_8"] = g["sales"].transform(
        lambda s: (s.shift(1) == 0).rolling(8, min_periods=1).mean())
    w["weeks_since_last_sale"] = g["sales"].transform(
        lambda s: pd.Series(_weeks_since_last_sale(s), index=s.index))

    gp = g["sell_price"]
    w["price_change_pct"] = gp.pct_change()
    w["price_ratio_8w"] = w["sell_price"] / gp.transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean())

    woy = w["week_end_date"].dt.isocalendar().week.astype(int)
    w["weekofyear"] = woy.astype("int16")
    w["woy_sin"] = np.sin(2 * np.pi * woy / config.SEASON_WEEKS)
    w["woy_cos"] = np.cos(2 * np.pi * woy / config.SEASON_WEEKS)
    return w


# ---- weekly panel loader ---------------------------------------------------------

def load_weekly() -> pd.DataFrame:
    """Concat the per-store weekly parquets written by scripts/prepare_data.py."""
    parts = sorted((config.PROCESSED / "weekly").glob("weekly_*.parquet"))
    assert parts, "no weekly parquets found -- run scripts/prepare_data.py first"
    w = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for c in ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id"):
        w[c] = w[c].astype("category")
    return w
