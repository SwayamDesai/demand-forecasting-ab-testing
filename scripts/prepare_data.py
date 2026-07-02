"""
Step 1 -- data: raw M5 -> weekly features, processed ONE STORE AT A TIME.

Melting all 30,490 series at once creates a ~59M-row daily frame whose merges need
20-30GB of RAM. Per store it's a 5.9M-row frame (~0.5GB peak), aggregated to weekly
(~700K rows) and released before the next store -- the per-store discipline the M5
winners used, and it keeps the whole step under ~4GB on a 16GB machine.

Checkpointed: a store whose parquet already exists is skipped, so a crash resumes.

Outputs:
  data/processed/weekly/weekly_{STORE}.parquet   (one per store, feature-complete)
  reports/data_validation.csv                    (per-store reconciliation)
"""
from __future__ import annotations

import gc
import resource

import pandas as pd

from src import config, etl

pd.options.mode.copy_on_write = True

OUT_DATA = config.PROCESSED / "weekly"
OUT_REP = config.REPORTS


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def downcast(w: pd.DataFrame) -> pd.DataFrame:
    for c in w.columns:
        if w[c].dtype == "float64":
            w[c] = w[c].astype("float32")
    for c in ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id"):
        if c in w.columns:
            w[c] = w[c].astype("category")
    return w


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REP.mkdir(parents=True, exist_ok=True)

    print("[load] calendar + prices + sales (wide, int16)...")
    cal = pd.read_csv(config.RAW / "calendar.csv", parse_dates=["date"])
    for c in ("snap_CA", "snap_TX", "snap_WI"):
        cal[c] = cal[c].astype("int8")
    cal["wday"] = cal["wday"].astype("int8")
    cal["month"] = cal["month"].astype("int8")
    cal["year"] = cal["year"].astype("int16")

    prices = pd.read_csv(config.RAW / "sell_prices.csv",
                         dtype={"store_id": "category", "item_id": "category",
                                "wm_yr_wk": "int32", "sell_price": "float32"})

    hdr = pd.read_csv(config.RAW / "sales_train_evaluation.csv", nrows=0)
    day_cols = [c for c in hdr.columns if c.startswith("d_")]
    sales_wide = pd.read_csv(config.RAW / "sales_train_evaluation.csv",
                             dtype={c: "int16" for c in day_cols})
    stores = sorted(sales_wide["store_id"].unique())
    print(f"      {len(sales_wide)} series, {len(stores)} stores  (rss {rss_gb():.1f} GB)")

    val_rows = []
    for store in stores:
        out = OUT_DATA / f"weekly_{store}.parquet"
        if out.exists():
            print(f"[{store}] checkpoint exists -- skip")
            continue

        print(f"[{store}] melt -> join -> clean -> weekly -> features ...")
        sub = sales_wide[sales_wide["store_id"] == store]
        long = etl.melt_long(sub)
        long = etl.join_calendar(long, cal)
        p_store = prices[prices["store_id"] == store]
        p_store = p_store.assign(store_id=p_store["store_id"].astype(str),
                                 item_id=p_store["item_id"].astype(str))
        long = etl.join_prices(long, p_store)
        daily = etl.add_flags(long)
        del long
        gc.collect()

        weekly_all = etl.aggregate_weekly(daily)
        d_total = int(daily["sales"].sum())
        w_total = int(weekly_all["sales"].sum())
        del daily
        gc.collect()

        weekly = etl.keep_full_active(weekly_all)
        n_weeks_all = len(weekly_all)
        del weekly_all
        weekly = etl.add_features(weekly)
        weekly = downcast(weekly)
        weekly.to_parquet(out, index=False)

        val_rows.append(dict(
            store=store, daily_units=d_total, weekly_units=w_total,
            reconciles=d_total == w_total, series=weekly["id"].nunique(),
            weekly_rows_all=n_weeks_all, weekly_rows_model=len(weekly),
            zero_rate=float((weekly["sales"] == 0).mean()),
        ))
        print(f"      ok: {weekly['id'].nunique()} series, {len(weekly):,} weekly rows, "
              f"reconciles={d_total == w_total}  (rss {rss_gb():.1f} GB)")
        del weekly
        gc.collect()

    if val_rows:
        val = pd.DataFrame(val_rows)
        vp = OUT_REP / "data_validation.csv"
        if vp.exists():
            val = pd.concat([pd.read_csv(vp), val], ignore_index=True).drop_duplicates("store")
        val.to_csv(vp, index=False)

    print("\n[audit] all stores:")
    tot_series, tot_rows = 0, 0
    for store in stores:
        w = pd.read_parquet(OUT_DATA / f"weekly_{store}.parquet", columns=["id", "sales"])
        tot_series += w["id"].nunique(); tot_rows += len(w)
        del w
    print(f"      series total : {tot_series:,} (expect 30,490)")
    print(f"      weekly rows  : {tot_rows:,}")
    print(f"      peak RSS     : {rss_gb():.1f} GB")
    assert tot_series == 30490, "series count mismatch"


if __name__ == "__main__":
    main()
