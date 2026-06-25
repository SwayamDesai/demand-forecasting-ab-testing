"""
M5 data loading + cleaning (pandas only).

Pipeline:
  load_calendar() ─┐
  load_sales()    ─┼─> wide_to_long() ─> join_calendar() ─> join_prices() ─> clean()
  load_prices()   ─┘                                                            │
                                                                                ▼
                                                                  data/processed/m5_long_sample.parquet

CLI: `python -m src.data --make-sample`
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# ---- dev sample knobs (tweak here only) ------------------------------------
# 10x scale: 3 stores (one per state) x 3 cats x 100 items per (store, cat)
# = ~900-2,400 series (depending on overlap). With ~600-1,200 series per A/B arm,
# MDE drops to ~13% of control mean (vs ~40% at the original 300-series sample).
# Faster than 20x; still enough power to detect a real cost difference.
SAMPLE_STORES = ["CA_1", "TX_1", "WI_1"]
ITEMS_PER_CAT = 100          # per store, per category
SAMPLE_SEED = 7              # reproducible item picks
# ---------------------------------------------------------------------------


def load_calendar() -> pd.DataFrame:
    """Calendar with one row per date. Parses dates; leaves event/snap cols as-is."""
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    # tighten dtypes (saves memory; helps merges)
    cal["wday"] = cal["wday"].astype("int8")
    cal["month"] = cal["month"].astype("int8")
    cal["year"] = cal["year"].astype("int16")
    for c in ["snap_CA", "snap_TX", "snap_WI"]:
        cal[c] = cal[c].astype("int8")
    return cal


def load_sales_wide() -> pd.DataFrame:
    """Sales in original wide format (1 row per SKU, 1 col per day)."""
    return pd.read_csv(RAW / "sales_train_evaluation.csv")


def load_prices() -> pd.DataFrame:
    """Weekly sell-prices: (store_id, item_id, wm_yr_wk) -> sell_price."""
    return pd.read_csv(RAW / "sell_prices.csv")


def pick_sample_items(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Carve a representative dev subset.

    Strategy (industry-pragmatic):
      * 3 stores (one per state CA/TX/WI) for geographic diversity and a multi-state
        SNAP signal -- mirrors a real regional rollout
      * within each store x category, pick `ITEMS_PER_CAT` items via a deterministic
        random sample (seed) -- preserves a realistic mix of fast/slow movers
      * keep the FULL date range (NEVER subsample by time -- TS rule #1)
    """
    # Pick exactly ITEMS_PER_CAT (store, item) pairs per (store, cat),
    # independently per store -- so the final count is deterministic.
    sub = sales_wide[sales_wide["store_id"].isin(SAMPLE_STORES)].copy()
    rng = np.random.default_rng(SAMPLE_SEED)
    keep_keys: set[tuple[str, str]] = set()
    for (store, cat), grp in sub.groupby(["store_id", "cat_id"]):
        items = grp["item_id"].to_numpy()
        chosen = rng.choice(items, size=min(ITEMS_PER_CAT, len(items)), replace=False)
        for item in chosen:
            keep_keys.add((store, item))
    mask = [(s, i) in keep_keys for s, i in zip(sub["store_id"], sub["item_id"])]
    return sub[mask].reset_index(drop=True)


def wide_to_long(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Melt 1941 day-columns into a single 'd' column.

    Result: one row per (id, item_id, dept_id, cat_id, store_id, state_id, d, sales)
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    long = sales_wide.melt(
        id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales"
    )
    long["sales"] = long["sales"].astype("int32")
    return long


def join_calendar(long: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Attach real date + calendar features via the 'd' join key."""
    keep = ["d", "date", "wm_yr_wk", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"]
    return long.merge(cal[keep], on="d", how="left")


def join_prices(long_cal: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Attach weekly sell_price via (store_id, item_id, wm_yr_wk).

    Result: many rows will have NaN sell_price -- that is the "item not yet sold
    at this store in this week" signal we keep and use in clean().
    """
    return long_cal.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Targeted cleaning -- only what is justifiable, no kitchen-sink.

      1. `is_active` flag: a SKU is considered 'launched' from the first week it has
         a non-null sell_price. Rows before that get is_active=False and are kept
         for visibility but should be EXCLUDED before modeling.
      2. Trim leading inactive rows per SKU (the typical fix). We keep them for now
         and only drop in feature step -- so EDA can show the launch effect.
      3. Add helpful derived cols: dow_name, is_weekend, has_event, has_snap (state-aware).
    """
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    # is_active: True from the first non-null price row onward, per series
    df["is_active"] = (
        df.groupby("id")["sell_price"]
          .transform(lambda s: s.notna().cummax())
          .astype(bool)
    )

    df["dow_name"] = df["date"].dt.day_name()
    df["is_weekend"] = df["wday"].isin([1, 2]).astype("int8")   # M5: wday 1=Sat, 2=Sun
    df["has_event"] = df["event_name_1"].notna().astype("int8")

    # state-aware SNAP (since we have stores from multiple states in the full data)
    snap_lookup = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    df["has_snap"] = 0
    for state, col in snap_lookup.items():
        mask = df["state_id"] == state
        df.loc[mask, "has_snap"] = df.loc[mask, col]
    df["has_snap"] = df["has_snap"].astype("int8")

    return df


def build_sample() -> pd.DataFrame:
    """Full pipeline -> tidy long dataframe for the dev sample."""
    print("[1/5] loading raw files...")
    cal = load_calendar()
    sales_wide = load_sales_wide()
    prices = load_prices()

    print(f"[2/5] picking sample: stores={SAMPLE_STORES}, "
          f"items_per_cat={ITEMS_PER_CAT}, seed={SAMPLE_SEED}")
    sample_wide = pick_sample_items(sales_wide)
    print(f"      -> {len(sample_wide)} series selected "
          f"(categories: {sorted(sample_wide['cat_id'].unique())})")

    print("[3/5] reshape wide -> long...")
    long = wide_to_long(sample_wide)
    print(f"      -> {len(long):,} rows ({len(sample_wide)} series x "
          f"{long['d'].nunique()} days)")

    print("[4/5] joining calendar + prices...")
    long = join_calendar(long, cal)
    long = join_prices(long, prices)

    print("[5/5] cleaning + derived cols...")
    df = clean(long)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-sample", action="store_true",
                        help="build the dev sample and write to data/processed/")
    args = parser.parse_args()

    if not args.make_sample:
        parser.error("nothing to do; pass --make-sample")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_sample()

    out = PROCESSED / "m5_long_sample.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(df):,} rows)")

    # quick sanity print
    print("\n=== schema ===")
    print(df.dtypes.to_string())
    print("\n=== head ===")
    print(df.head(3).to_string())
    print("\n=== summary ===")
    print(f"date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"series:     {df['id'].nunique()}")
    print(f"cats:       {dict(df.groupby('cat_id')['id'].nunique())}")
    print(f"active%:    {df['is_active'].mean() * 100:.1f}%")
    print(f"zero-sales% (active only): "
          f"{(df.loc[df['is_active'], 'sales'] == 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
