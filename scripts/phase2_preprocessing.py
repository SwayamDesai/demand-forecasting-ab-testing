"""
Phase 2 -- Preprocessing: weekly aggregation + leakage-safe features.

This is where the noise drops. We roll the clean DAILY table up to the Walmart
weekly grain (wm_yr_wk = Sat..Fri), keep only full active weeks, then attach
features that a row at week t may compute using ONLY weeks < t.

Two decisions worth stating:
  * A "full active week" must (a) have all 7 calendar days present in our data
    (drops the partial week at each boundary of the M5 window) and (b) have the
    series active on all 7 days (drops the partial week in which an item launches).
    Zero-sales weeks among active weeks are REAL demand and are kept.
  * Aggregation is lossless arithmetic: total weekly sales == total daily sales.
    We assert that before trusting anything downstream.

Feature groups (all leakage-safe -- every lag/rolling is .shift()-ed):
  - lags of weekly sales: 1, 2, 4, 8, 52 weeks
  - trailing rolling mean/std over 4, 8, 13 weeks (window ends at t-1)
  - intermittency: trailing zero-rate (8w) and weeks-since-last-sale
  - price: level, week-over-week % change, ratio to trailing 8w mean
  - calendar (known in advance, no leakage): month, week-of-year + sin/cos,
    snap_days (0-7), event_days (0-7)

Outputs:
  data/processed/weekly_features.parquet
  reports/phase2_preprocessing/data_quality.csv
  reports/phase2_preprocessing/PHASE2_SUMMARY.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

OUT_DIR = config.REPORTS / "phase2_preprocessing"

STATIC_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
LAGS = (1, 2, 4, 8, 52)
ROLL_WINDOWS = (4, 8, 13)


# ---- 1) daily -> weekly aggregation -----------------------------------------

def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Sum sales to (id, wm_yr_wk). Carry week-level calendar/price attributes.
    Returns ALL weeks (full + partial); filtering happens in keep_full_active.
    """
    daily = daily.copy()
    daily["has_snap"] = daily["has_snap"].astype(int)
    daily["has_event"] = daily["has_event"].astype(int)
    daily["is_active_i"] = daily["is_active"].astype(int)

    grp = daily.groupby(["id", "wm_yr_wk"], sort=False)
    weekly = grp.agg(
        sales=("sales", "sum"),
        sell_price=("sell_price", "mean"),       # constant within a wm_yr_wk
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


# ---- 2) leakage-safe weekly features ----------------------------------------

def _weeks_since_last_sale(sales: pd.Series) -> np.ndarray:
    """
    For each week t, how many weeks since the most recent week (< t) with a sale.
    Uses only weeks strictly before t (leakage-safe). NaN until the first sale.
    """
    s = sales.to_numpy()
    out = np.full(len(s), np.nan)
    last = None
    for i in range(len(s)):
        if last is not None:
            out[i] = i - last            # distance from t to last sale (both < t handled below)
        if s[i] > 0:
            last = i
    return out


def add_features(w: pd.DataFrame) -> pd.DataFrame:
    w = w.sort_values(["id", "week_end_date"]).reset_index(drop=True)
    g = w.groupby("id", sort=False)

    # lags of weekly sales
    for k in LAGS:
        w[f"sales_lag_{k}"] = g["sales"].shift(k)

    # trailing rolling stats: window ends at t-1 (shift(1) first)
    for win in ROLL_WINDOWS:
        w[f"sales_rmean_{win}"] = g["sales"].transform(
            lambda s, win=win: s.shift(1).rolling(win, min_periods=1).mean())
        w[f"sales_rstd_{win}"] = g["sales"].transform(
            lambda s, win=win: s.shift(1).rolling(win, min_periods=2).std())

    # intermittency (leakage-safe)
    w["trailing_zero_rate_8"] = g["sales"].transform(
        lambda s: (s.shift(1) == 0).rolling(8, min_periods=1).mean())
    w["weeks_since_last_sale"] = g["sales"].transform(
        lambda s: pd.Series(_weeks_since_last_sale(s), index=s.index))

    # price features
    gp = g["sell_price"]
    w["price_change_pct"] = gp.pct_change()
    w["price_ratio_8w"] = w["sell_price"] / gp.transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean())

    # calendar / seasonality (known in advance)
    woy = w["week_end_date"].dt.isocalendar().week.astype(int)
    w["weekofyear"] = woy.astype("int16")
    w["woy_sin"] = np.sin(2 * np.pi * woy / config.SEASON_WEEKS)
    w["woy_cos"] = np.cos(2 * np.pi * woy / config.SEASON_WEEKS)
    return w


# ---- 3) quality gate --------------------------------------------------------

def quality_checks(daily: pd.DataFrame, weekly_all: pd.DataFrame,
                   weekly: pd.DataFrame) -> pd.DataFrame:
    checks: list[tuple[str, bool, str]] = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    # 1. aggregation is lossless: weekly total (all weeks) == daily total
    d_tot = int(daily["sales"].sum())
    w_tot = int(weekly_all["sales"].sum())
    add("weekly_reconciles_daily", d_tot == w_tot, f"daily={d_tot:,} weekly={w_tot:,}")

    # 2. no duplicate (id, week)
    dups = weekly.duplicated(["id", "wm_yr_wk"]).sum()
    add("no_duplicate_id_week", dups == 0, f"{dups} dup rows")

    # 3. every kept week is full + active
    add("kept_weeks_full_active",
        ((weekly["n_days"] == 7) & (weekly["n_active_days"] == 7)).all(),
        f"n_days in {sorted(weekly['n_days'].unique())}, "
        f"n_active in {sorted(weekly['n_active_days'].unique())}")

    # 4. weeks are contiguous (7-day step) within each series after launch
    def max_gap(s):
        d = s.sort_values().diff().dt.days.dropna()
        return d.max() if len(d) else 7
    gaps = weekly.groupby("id")["week_end_date"].apply(max_gap)
    add("weeks_contiguous_7d", (gaps == 7).all(),
        f"max within-series gap = {int(gaps.max())} days (expect 7)")

    # 5. active weeks have a price
    null_price = weekly["sell_price"].isna().mean()
    add("kept_weeks_have_price", null_price == 0, f"{null_price*100:.3f}% null price")

    # 6. LEAKAGE TEST: independently recompute lag_1 and rmean_4 for a sample of
    #    series and confirm the stored features match exactly (where not NaN).
    rng = np.random.default_rng(0)
    sample_ids = rng.choice(weekly["id"].unique(), size=25, replace=False)
    mismatches = 0
    for sid in sample_ids:
        s = weekly[weekly["id"] == sid].sort_values("week_end_date")
        exp_lag1 = s["sales"].shift(1)
        exp_rmean4 = s["sales"].shift(1).rolling(4, min_periods=1).mean()
        m1 = ~exp_lag1.isna()
        m4 = ~exp_rmean4.isna()
        if not np.allclose(s.loc[m1, "sales_lag_1"], exp_lag1[m1]):
            mismatches += 1
        if not np.allclose(s.loc[m4, "sales_rmean_4"], exp_rmean4[m4], equal_nan=True):
            mismatches += 1
    add("features_leakage_free", mismatches == 0,
        f"{mismatches} mismatches over {len(sample_ids)} sampled series")

    # 7. lag_1[t] must never equal current sales by construction (extra guard):
    #    correlation sanity -- lag_1 should not be a copy of sales
    sub = weekly.dropna(subset=["sales_lag_1"])
    identical = (sub["sales_lag_1"] == sub["sales"]).mean()
    add("lag1_not_current_week", identical < 0.9,
        f"{identical*100:.1f}% rows where lag_1==sales (coincidental zeros ok)")

    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


# ---- driver -----------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] load clean daily table...")
    daily = pd.read_parquet(config.PROCESSED / "daily_clean.parquet")

    print("[2/5] aggregate daily -> weekly (wm_yr_wk)...")
    weekly_all = aggregate_weekly(daily)
    weekly = keep_full_active(weekly_all)
    print(f"      weeks: {len(weekly_all):,} total -> {len(weekly):,} full+active")

    print("[3/5] build leakage-safe weekly features...")
    weekly = add_features(weekly)

    print("[4/5] quality gate...")
    qc = quality_checks(daily, weekly_all, weekly)
    n_pass = int(qc["passed"].sum())
    for _, r in qc.iterrows():
        print(f"      [{'PASS' if r['passed'] else 'FAIL'}] {r['check']:26s} {r['detail']}")
    qc.to_csv(OUT_DIR / "data_quality.csv", index=False)

    print("[5/5] write model-ready weekly table...")
    out = config.PROCESSED / "weekly_features.parquet"
    weekly.to_parquet(out, index=False)

    # ---- summary ----------------------------------------------------------
    weekly_zero = (weekly["sales"] == 0).mean() * 100
    daily_active = daily[daily["is_active"]]
    daily_zero = (daily_active["sales"] == 0).mean() * 100
    wk_per_series = weekly.groupby("id").size()

    md = OUT_DIR / "PHASE2_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 2 -- Preprocessing (weekly aggregation + features)\n\n")
        f.write("## The payoff: noise collapses at the weekly grain\n\n")
        f.write(f"- Daily zero-sales rate (active): **{daily_zero:.1f}%**\n")
        f.write(f"- Weekly zero-sales rate (active full weeks): **{weekly_zero:.1f}%**\n\n")
        f.write("## Result\n\n")
        f.write(f"- Quality checks passed: **{n_pass}/{len(qc)}**\n")
        f.write(f"- Model-ready weekly rows: {len(weekly):,}  (series={weekly['id'].nunique()})\n")
        f.write(f"- Weeks per series: median {int(wk_per_series.median())}, "
                f"min {int(wk_per_series.min())}, max {int(wk_per_series.max())}\n")
        f.write(f"- Week range: {weekly['week_end_date'].min().date()} -> "
                f"{weekly['week_end_date'].max().date()}\n")
        f.write(f"- Feature columns: {sum(c.startswith(('sales_lag','sales_rmean','sales_rstd')) for c in weekly.columns)} "
                f"lag/rolling + price + calendar + intermittency\n\n")
        f.write("## Checks\n\n| check | passed | detail |\n|---|---|---|\n")
        for _, r in qc.iterrows():
            f.write(f"| {r['check']} | {'PASS' if r['passed'] else 'FAIL'} | {r['detail']} |\n")
        f.write("\n## Outputs\n\n- `data/processed/weekly_features.parquet`\n"
                "- `reports/phase2_preprocessing/data_quality.csv`\n")

    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(weekly):,} rows)")
    print(f"\n=== Phase 2 summary ===")
    print(f"checks passed     : {n_pass}/{len(qc)}")
    print(f"weekly rows       : {len(weekly):,}  (900 series)")
    print(f"weeks/series      : median {int(wk_per_series.median())} "
          f"(min {int(wk_per_series.min())}, max {int(wk_per_series.max())})")
    print(f"DAILY  zero-rate  : {daily_zero:.1f}%  (active)")
    print(f"WEEKLY zero-rate  : {weekly_zero:.1f}%  (active full weeks)  <- the noise drop")

    if n_pass != len(qc):
        raise SystemExit("QUALITY GATE FAILED -- fix before Phase 3")


if __name__ == "__main__":
    main()
