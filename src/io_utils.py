"""
Raw M5 loading + the stratified sample carve. Shared by every phase that needs
to start from raw (really only Phase 1). Kept separate so the sampling logic
lives in exactly one place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def load_calendar() -> pd.DataFrame:
    """One row per date. Parse the date; tighten a few dtypes for cheap merges."""
    cal = pd.read_csv(config.RAW / "calendar.csv", parse_dates=["date"])
    cal["wday"] = cal["wday"].astype("int8")
    cal["month"] = cal["month"].astype("int8")
    cal["year"] = cal["year"].astype("int16")
    for c in ("snap_CA", "snap_TX", "snap_WI"):
        cal[c] = cal[c].astype("int8")
    return cal


def load_sales_wide() -> pd.DataFrame:
    """Sales in the original wide layout (one row per SKU, one col per day)."""
    return pd.read_csv(config.RAW / "sales_train_evaluation.csv")


def load_prices() -> pd.DataFrame:
    """Weekly sell prices keyed by (store_id, item_id, wm_yr_wk)."""
    return pd.read_csv(config.RAW / "sell_prices.csv")


def carve_sample(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified pick: within each (store, category) draw ITEMS_PER_STORE_CAT items
    with a fixed seed. Independent per store so the final series count is exact.
    The full date range is always kept -- never subsample time.
    """
    sub = sales_wide[sales_wide["store_id"].isin(config.SAMPLE_STORES)].copy()
    rng = np.random.default_rng(config.SAMPLE_SEED)
    keep: set[tuple[str, str]] = set()
    for (store, cat), grp in sub.groupby(["store_id", "cat_id"]):
        items = grp["item_id"].to_numpy()
        n = min(config.ITEMS_PER_STORE_CAT, len(items))
        for item in rng.choice(items, size=n, replace=False):
            keep.add((store, item))
    mask = [(s, i) in keep for s, i in zip(sub["store_id"], sub["item_id"])]
    return sub[mask].reset_index(drop=True)
