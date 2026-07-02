"""
Full-M5 scale, step 2 -- fast baselines + per-store LightGBM (30,490 series).

Scale decisions (and why):
  * NO statsforecast / ETS / ARIMA here: per-series fitting at 30K series is hours of
    CPU and the multiprocessing memory bomb that kills 16GB machines. The fast
    baselines below are pure vectorized pandas (seconds) and give the accuracy floor.
  * NO LSTM: it lost to LightGBM at 900-series scale, adds no decision value, and is
    the one component that could blow the memory/time budget.
  * LightGBM is trained PER STORE (10 models per fold) -- the pattern the M5 winners
    used: each store frame is ~670K rows, fits in a couple hundred MB, trains in
    seconds-to-minutes, and store-level demand patterns get their own model.
  * Checkpointed per store; a crash resumes where it left off.

Baselines (vectorized): naive_last, seasonal_naive_52 (fallback naive), ma4, ma8.
Evaluation: identical rolling-origin protocol as the 900-series pipeline
(3 folds x 4-week horizon), pooled volume-weighted WMAPE + bias + MASE.

Outputs:
  reports/full_m5/baseline_predictions.parquet
  reports/full_m5/lgbm_mean/preds_{STORE}.parquet   (checkpoints)
  reports/full_m5/leaderboard_summary.csv, leaderboard_per_store.csv
  reports/full_m5/01_leaderboard.png, 02_per_store_wmape.png
  reports/full_m5/FULL2_SUMMARY.md
"""
from __future__ import annotations

import gc
import resource
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import backtest, config, metrics
from scripts.phase6_lightgbm import (EXOG, STATIC, add_codes, fit_fold,
                                     recursive_forecast)

pd.options.mode.copy_on_write = True
warnings.simplefilter("ignore")

DATA = config.PROCESSED / "full"
OUT = config.REPORTS / "full_m5"
LGBM_DIR = OUT / "lgbm_mean"


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load_weekly() -> pd.DataFrame:
    parts = sorted(DATA.glob("weekly_*.parquet"))
    assert len(parts) == 10, f"expected 10 store parquets, found {len(parts)}"
    w = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for c in ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id"):
        w[c] = w[c].astype("category")
    return w


# ---- vectorized baselines ----------------------------------------------------

def baseline_preds(w: pd.DataFrame, folds) -> pd.DataFrame:
    out = []
    hist = w[["id", "week_end_date", "sales"]]
    for f in folds:
        train = hist[hist["week_end_date"] <= f.train_end]
        cells = w.loc[(w["week_end_date"] >= f.test_start) &
                      (w["week_end_date"] <= f.test_end),
                      ["id", "week_end_date", "sales", "store_id"]].rename(
                      columns={"sales": "y"})

        last = train.groupby("id", observed=True)["sales"].last().rename("naive_last")
        cells = cells.merge(last, on="id", how="left")

        for k in (4, 8):
            lo = f.train_end - pd.Timedelta(weeks=k - 1)
            win = train[train["week_end_date"] >= lo]
            ma = win.groupby("id", observed=True)["sales"].mean().rename(f"ma{k}")
            cells = cells.merge(ma, on="id", how="left")

        lut = hist.rename(columns={"sales": "snaive_52"}).copy()
        lut["week_end_date"] = lut["week_end_date"] + pd.Timedelta(weeks=52)
        cells = cells.merge(lut, on=["id", "week_end_date"], how="left")
        cells["snaive_52"] = cells["snaive_52"].fillna(cells["naive_last"])

        cells["fold"] = f.name
        out.append(cells)
    preds = pd.concat(out, ignore_index=True)
    # series with zero training history (launched inside the test window) can't be
    # forecast by anything -- drop those cells from ALL models equally.
    n0 = preds["naive_last"].isna().sum()
    preds = preds.dropna(subset=["naive_last"]).reset_index(drop=True)
    for c in ("ma4", "ma8", "snaive_52"):
        preds[c] = preds[c].fillna(preds["naive_last"])
    print(f"      baseline cells: {len(preds):,}  (dropped {n0} no-history cells)")
    return preds


# ---- driver -------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LGBM_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] load weekly panel...")
    w = load_weekly()
    print(f"      {w['id'].nunique():,} series, {len(w):,} rows  (rss {rss_gb():.1f} GB)")

    folds = backtest.make_folds(w["week_end_date"], config.HORIZON_WEEKS, config.N_FOLDS)
    for f in folds:
        print(f"      {f.name}: train<= {f.train_end.date()} | "
              f"test {f.test_start.date()}..{f.test_end.date()}")

    print("[2/4] vectorized baselines...")
    base = baseline_preds(w, folds)
    base.to_parquet(OUT / "baseline_predictions.parquet", index=False)

    print("[3/4] per-store LightGBM (Tweedie mean, recursive)...")
    lookups = {c: {v: i for i, v in enumerate(sorted(w[c].astype(str).unique()))}
               for c in ["cat_id", "dept_id", "store_id", "item_id"]}
    stores = sorted(w["store_id"].astype(str).unique())
    for store in stores:
        ck = LGBM_DIR / f"preds_{store}.parquet"
        if ck.exists():
            print(f"  [{store}] checkpoint exists -- skip")
            continue
        ws = w[w["store_id"] == store].copy()
        for c in ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id"):
            ws[c] = ws[c].astype(str)
        ws = add_codes(ws, lookups)
        exog_codes = ws[["id", "week_end_date"] + EXOG + STATIC]
        store_preds = []
        for f in folds:
            train = ws[ws["week_end_date"] <= f.train_end].dropna(subset=["sales"])
            model = fit_fold(train)
            work = ws[["id", "week_end_date", "sales"]].copy()
            work["sales_work"] = np.where(work["week_end_date"] <= f.train_end,
                                          work["sales"], np.nan)
            test_weeks = ws.loc[(ws["week_end_date"] >= f.test_start) &
                                (ws["week_end_date"] <= f.test_end),
                                "week_end_date"].unique()
            fp = recursive_forecast(model, work, exog_codes, test_weeks)
            fp["fold"] = f.name
            store_preds.append(fp)
        sp = pd.concat(store_preds, ignore_index=True)
        sp.to_parquet(ck, index=False)
        print(f"  [{store}] done: {len(sp):,} cells  (rss {rss_gb():.1f} GB)")
        del ws, exog_codes, work, sp
        gc.collect()

    print("[4/4] evaluate...")
    lgbm = pd.concat([pd.read_parquet(p) for p in sorted(LGBM_DIR.glob("preds_*.parquet"))],
                     ignore_index=True).rename(columns={"y_pred": "lightgbm"})
    tbl = base.merge(lgbm[["id", "week_end_date", "lightgbm"]],
                     on=["id", "week_end_date"], how="left")
    assert tbl["lightgbm"].notna().all(), "lightgbm missing cells"

    first_test = folds[0].test_start
    scales = (w[w["week_end_date"] < first_test]
              .groupby("id", observed=True)["sales"]
              .apply(lambda s: metrics.seasonal_naive_scale(s.to_numpy(), config.SEASON_WEEKS)))

    models = ["lightgbm", "ma4", "ma8", "snaive_52", "naive_last"]
    rows = []
    for m in models:
        per = (tbl.assign(ae=np.abs(tbl["y"] - tbl[m]))
                  .groupby("id", observed=True)["ae"].mean())
        mase_s = (per / scales).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(model=m, wmape=metrics.wmape(tbl["y"], tbl[m]),
                         rmse=metrics.rmse(tbl["y"], tbl[m]),
                         bias_pct=metrics.bias_pct(tbl["y"], tbl[m]),
                         mase_median=float(mase_s.median())))
    lb = pd.DataFrame(rows).sort_values("wmape").reset_index(drop=True)
    lb.to_csv(OUT / "leaderboard_summary.csv", index=False)

    per_store = []
    for store, g in tbl.groupby("store_id", observed=True):
        for m in ["lightgbm", "ma4", "snaive_52"]:
            per_store.append(dict(store=store, model=m, wmape=metrics.wmape(g["y"], g[m])))
    ps = pd.DataFrame(per_store)
    ps.to_csv(OUT / "leaderboard_per_store.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#55A868" if i == 0 else "#4C72B0" for i in range(len(lb))]
    ax.barh(lb["model"], lb["wmape"], color=colors)
    for i, (v, b) in enumerate(zip(lb["wmape"], lb["bias_pct"])):
        ax.text(v, i, f"  {v:.3f} (bias {b:+.0f}%)", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("WMAPE")
    ax.set_title(f"Full M5 ({tbl['id'].nunique():,} series) -- champion in green")
    fig.tight_layout(); fig.savefig(OUT / "01_leaderboard.png", dpi=110); plt.close(fig)

    piv = ps.pivot(index="store", columns="model", values="wmape")
    fig, ax = plt.subplots(figsize=(9, 4.2)); piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("WMAPE"); ax.set_title("WMAPE by store"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(OUT / "02_per_store_wmape.png", dpi=110); plt.close(fig)

    champ = lb.iloc[0]["model"]
    with open(OUT / "FULL2_SUMMARY.md", "w") as f:
        f.write("# Full M5 -- Baselines + per-store LightGBM\n\n")
        f.write(f"Panel: {tbl['id'].nunique():,} series, {len(tbl):,} test cells "
                f"(3 folds x 4-week horizon).\n\n")
        f.write("| model | WMAPE | RMSE | bias % | MASE (med) |\n|---|---|---|---|---|\n")
        for _, r in lb.iterrows():
            star = " **<- champion**" if r["model"] == champ else ""
            f.write(f"| {r['model']}{star} | {r['wmape']:.4f} | {r['rmse']:.3f} | "
                    f"{r['bias_pct']:+.1f}% | {r['mase_median']:.3f} |\n")
        f.write(f"\nPer-store WMAPE in `leaderboard_per_store.csv`. ETS/ARIMA/LSTM "
                f"deliberately excluded at this scale (see script docstring).\n")

    print("\n=== Full M5 leaderboard ===")
    print(lb.to_string(index=False))
    print(f"\nchampion: {champ} | peak RSS {rss_gb():.1f} GB")


if __name__ == "__main__":
    main()
