"""
Phase 1 -- Data cleaning.

Goal: turn the raw M5 files into ONE clean, tidy DAILY table we trust, and prove
we trust it with an explicit quality gate. No weekly aggregation, no features, no
modeling here -- those are later phases. Cleaning only.

Pipeline:
    raw sales (wide) --carve sample--> melt to long --join calendar--> join prices
        --> mark pre-launch (is_active) --> derived calendar flags --> QUALITY GATE

Decisions (and why):
  * is_active: a store-item is "launched" from the first week it has a sell price.
    M5 simply has no price row before an item is listed, so a null price = not yet
    sold here. We FLAG pre-launch rows (keep them visible) rather than dropping in
    cleaning; modeling phases drop them. This keeps cleaning lossless and auditable.
  * We keep the full daily calendar for every series (1941 days). Zero-sales days
    are real demand observations, not missing data -- we never impute them.
  * The only "fill" we do is price: pre-launch prices stay NaN (correct); we do not
    invent prices for inactive periods.

Outputs:
  data/processed/daily_clean.parquet
  reports/phase1_cleaning/data_quality.csv
  reports/phase1_cleaning/PHASE1_SUMMARY.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, io_utils

OUT_DIR = config.REPORTS / "phase1_cleaning"
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


# ---- build steps ------------------------------------------------------------

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
    """is_active (launched-onward) + a few state-aware calendar flags."""
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    # launched from the first non-null price onward, per series
    df["is_active"] = (
        df.groupby("id")["sell_price"].transform(lambda s: s.notna().cummax()).astype(bool)
    )

    df["is_weekend"] = df["wday"].isin(config.WEEKEND_WDAYS).astype("int8")
    df["has_event"] = df["event_name_1"].notna().astype("int8")

    # SNAP is state-specific -- pick the column for each row's state
    df["has_snap"] = 0
    for state, col in config.SNAP_BY_STATE.items():
        m = df["state_id"] == state
        df.loc[m, "has_snap"] = df.loc[m, col]
    df["has_snap"] = df["has_snap"].astype("int8")
    return df


# ---- quality gate -----------------------------------------------------------

def quality_checks(df: pd.DataFrame, n_days_expected: int) -> pd.DataFrame:
    """A battery of hard checks. Each row: (check, passed, detail)."""
    checks: list[tuple[str, bool, str]] = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    n_series = df["id"].nunique()

    # 1. exactly 900 series, balanced by store and category
    add("series_count_900", n_series == 900, f"{n_series} series")
    by_cat = df.groupby("cat_id")["id"].nunique().to_dict()
    by_store = df.groupby("store_id")["id"].nunique().to_dict()
    add("balanced_by_category", set(by_cat.values()) == {300}, str(by_cat))
    add("balanced_by_store", set(by_store.values()) == {300}, str(by_store))

    # 2. no duplicate series-day rows
    dups = df.duplicated(["id", "date"]).sum()
    add("no_duplicate_id_date", dups == 0, f"{dups} dup rows")

    # 3. every series has the full daily calendar, no gaps
    per = df.groupby("id")["date"].agg(["count", "nunique"])
    add("full_calendar_per_series",
        (per["count"] == n_days_expected).all() and (per["nunique"] == n_days_expected).all(),
        f"expected {n_days_expected}/series; min={per['count'].min()}, max={per['count'].max()}")

    # 4. sales are non-null, non-negative integers
    add("sales_non_null", df["sales"].notna().all(), f"{df['sales'].isna().sum()} nulls")
    add("sales_non_negative", (df["sales"] >= 0).all(), f"min={df['sales'].min()}")

    # 5. dates parsed and within the M5 window
    add("date_is_datetime", pd.api.types.is_datetime64_any_dtype(df["date"]), str(df["date"].dtype))

    # 6. price is present for ALL active rows except the legitimate first-week edge,
    #    and NEVER present before launch (would contradict the is_active definition).
    active = df[df["is_active"]]
    active_price_null = active["sell_price"].isna().mean()
    add("active_rows_have_price", active_price_null < 0.01,
        f"{active_price_null*100:.3f}% of active rows missing price")
    pre = df[~df["is_active"]]
    add("prelaunch_has_no_price", pre["sell_price"].isna().all(),
        f"{(~pre['sell_price'].isna()).sum()} pre-launch rows with a price")

    # 7. is_active is monotonic (never reactivates) per series
    reactivated = (
        df.groupby("id")["is_active"]
          .apply(lambda s: (s.astype(int).diff() < 0).any())
          .sum()
    )
    add("is_active_monotonic", reactivated == 0, f"{reactivated} series toggle off after launch")

    # 8. calendar joins fully resolved (no null dates/wday)
    add("calendar_fully_joined",
        df[["date", "wday", "wm_yr_wk"]].notna().all().all(),
        f"nulls: {df[['date','wday','wm_yr_wk']].isna().sum().to_dict()}")

    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


# ---- driver -----------------------------------------------------------------

def main() -> None:
    config.ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading raw M5 files...")
    cal = io_utils.load_calendar()
    sales_wide = io_utils.load_sales_wide()
    prices = io_utils.load_prices()
    n_days = sum(c.startswith("d_") for c in sales_wide.columns)

    print(f"[2/6] carving sample: stores={config.SAMPLE_STORES}, "
          f"{config.ITEMS_PER_STORE_CAT} items/(store,cat), seed={config.SAMPLE_SEED}")
    sample_wide = io_utils.carve_sample(sales_wide)
    print(f"      -> {len(sample_wide)} series")

    print("[3/6] melt wide -> long...")
    long = melt_long(sample_wide)
    print(f"      -> {len(long):,} daily rows ({len(sample_wide)} series x {n_days} days)")

    print("[4/6] join calendar + prices...")
    long = join_calendar(long, cal)
    long = join_prices(long, prices)

    print("[5/6] flags (is_active, snap, event, weekend)...")
    df = add_flags(long)

    print("[6/6] quality gate...")
    qc = quality_checks(df, n_days_expected=n_days)
    n_pass = int(qc["passed"].sum())
    for _, r in qc.iterrows():
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"      [{mark}] {r['check']:28s} {r['detail']}")
    qc.to_csv(OUT_DIR / "data_quality.csv", index=False)

    # write the clean daily table
    out = config.PROCESSED / "daily_clean.parquet"
    df.to_parquet(out, index=False)

    # ---- summary numbers --------------------------------------------------
    active = df[df["is_active"]]
    daily_zero = (active["sales"] == 0).mean()
    summary = {
        "series": df["id"].nunique(),
        "daily_rows": len(df),
        "active_daily_rows": len(active),
        "active_pct": active.shape[0] / df.shape[0] * 100,
        "daily_zero_rate_active": daily_zero * 100,
        "date_min": df["date"].min().date(),
        "date_max": df["date"].max().date(),
        "by_cat": df.groupby("cat_id")["id"].nunique().to_dict(),
        "by_store": df.groupby("store_id")["id"].nunique().to_dict(),
        "checks_passed": f"{n_pass}/{len(qc)}",
    }

    md = OUT_DIR / "PHASE1_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 1 -- Data Cleaning\n\n")
        f.write("Raw M5 -> clean daily tidy table, gated by explicit quality checks. "
                "No weekly aggregation / features / modeling yet.\n\n")
        f.write("## Result\n\n")
        f.write(f"- Quality checks passed: **{n_pass}/{len(qc)}**\n")
        f.write(f"- Series: {summary['series']} "
                f"(by category {summary['by_cat']}; by store {summary['by_store']})\n")
        f.write(f"- Daily rows: {summary['daily_rows']:,} "
                f"({summary['active_daily_rows']:,} active = {summary['active_pct']:.1f}%)\n")
        f.write(f"- Date range: {summary['date_min']} -> {summary['date_max']}\n")
        f.write(f"- Daily zero-sales rate (active rows): "
                f"**{summary['daily_zero_rate_active']:.1f}%**  <- the noise weekly will smooth\n\n")
        f.write("## Checks\n\n| check | passed | detail |\n|---|---|---|\n")
        for _, r in qc.iterrows():
            f.write(f"| {r['check']} | {'PASS' if r['passed'] else 'FAIL'} | {r['detail']} |\n")
        f.write(f"\n## Outputs\n\n- `data/processed/daily_clean.parquet`\n"
                f"- `reports/phase1_cleaning/data_quality.csv`\n")

    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(df):,} rows)")
    print(f"wrote {md}")
    print(f"\n=== Phase 1 summary ===")
    print(f"checks passed     : {n_pass}/{len(qc)}")
    print(f"series            : {summary['series']}  {summary['by_cat']}")
    print(f"active daily rows : {summary['active_daily_rows']:,} "
          f"({summary['active_pct']:.1f}%)")
    print(f"daily zero-rate   : {summary['daily_zero_rate_active']:.1f}% (active)  "
          f"<- key motivation for weekly")
    print(f"date range        : {summary['date_min']} -> {summary['date_max']}")

    if n_pass != len(qc):
        raise SystemExit("QUALITY GATE FAILED -- fix before proceeding to Phase 2")


if __name__ == "__main__":
    main()
