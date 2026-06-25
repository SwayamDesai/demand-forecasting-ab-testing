"""
Phase 2 orchestrator: run all baselines through rolling-origin backtest,
compute WMAPE/RMSE/MASE (overall + by category), pick a champion, emit plots.

Run: python -m scripts.phase2_baselines
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest import rolling_origin_splits, split_frame
from src.features import build_features
from src.metrics import mase, rmse, wmape
from src.models.baseline import run_arima, run_ets, run_lightgbm, run_seasonal_naive

DATA = Path("data/processed/m5_long_sample.parquet")
OUT = Path("reports/phase2_baselines")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

# all phase 2 / phase 3 runs land in the same SQLite-backed MLflow DB
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("phase2_baselines")


# ---- per-series scoring -----------------------------------------------------

def score_per_series(preds: pd.DataFrame, test: pd.DataFrame,
                     train: pd.DataFrame, season: int = 7) -> pd.DataFrame:
    """
    Returns one row per series: id, wmape, rmse, mase, sum_actual, sum_pred.
    """
    merged = preds.merge(test[["id", "date", "sales"]], on=["id", "date"], how="inner")
    rows = []
    for sid, g in merged.groupby("id", sort=False):
        y = g["sales"].to_numpy(dtype=float)
        yh = g["y_pred"].to_numpy(dtype=float)
        yt = train.loc[train["id"] == sid, "sales"].to_numpy(dtype=float)
        rows.append({
            "id": sid,
            "wmape": wmape(y, yh),
            "rmse":  rmse(y, yh),
            "mase":  mase(y, yh, yt, season=season),
            "sum_actual": float(y.sum()),
            "sum_pred":   float(yh.sum()),
        })
    return pd.DataFrame(rows)


def aggregate_scores(per_series: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Compute volume-weighted + unweighted aggregates, overall and by category."""
    df = per_series.merge(meta[["id", "cat_id"]].drop_duplicates(), on="id", how="left")
    valid = df.dropna(subset=["wmape"])
    w = valid["sum_actual"].clip(lower=0).to_numpy()
    out = {
        "wmape_unweighted":      float(valid["wmape"].mean()),
        "wmape_volume_weighted": float(np.average(valid["wmape"], weights=w) if w.sum() > 0 else np.nan),
        "rmse_mean":             float(valid["rmse"].mean()),
        "mase_median":           float(valid["mase"].dropna().median()),
        "n_series":              int(len(valid)),
    }
    for cat, sub in valid.groupby("cat_id"):
        wc = sub["sum_actual"].clip(lower=0).to_numpy()
        out[f"wmape_vw_{cat}"] = float(np.average(sub["wmape"], weights=wc)) if wc.sum() > 0 else np.nan
    return out


# ---- main -------------------------------------------------------------------

def main():
    print(f"loading {DATA}...")
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_active"]].copy()   # drop pre-launch rows (final modeling cut)
    print(f"  {len(df):,} active rows, {df['id'].nunique()} series, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    # Build features once on the full active frame.
    # SAFE: features are .shift()-ed; the rolling-origin train/test split below
    # respects time order, and per the leakage guard test, a feature for date t
    # uses only sales<t (which the per-fold split honors).
    print("building features (lag/rolling/calendar/price)...")
    feat = build_features(df)

    folds = rolling_origin_splits(last_date=df["date"].max(),
                                  n_folds=3, horizon=28, step=28)
    print("rolling-origin folds:")
    for f in folds:
        print(f"  {f}")

    leaderboard_rows = []
    all_preds = []          # for plotting

    for fold in folds:
        print(f"\n=== {fold.name}: train<= {fold.train_end.date()} | "
              f"test {fold.test_start.date()}..{fold.test_end.date()} ===")
        train_f, test_f = split_frame(feat, fold)
        train_r, test_r = split_frame(df,   fold)    # raw (for statsforecast/seasonal-naive)

        # ARIMA dropped at the 20x scale run: AutoARIMA didn't complete fold_1 in
        # 57 min (would take 9+ hours total). At 300 series it ran fine in ~6 min
        # and lost to LightGBM (0.861 vs 0.789 WMAPE-vw). It cannot share info
        # across series the way LightGBM does, so it gets relatively worse at scale.
        # Implementation stays in src/models/baseline.py for the small-data case.
        for model_name in ("seasonal_naive", "ets", "lightgbm"):
            t0 = time.time()
            with mlflow.start_run(run_name=f"{model_name}_{fold.name}"):
                mlflow.log_params({"model": model_name, "fold": fold.name,
                                   "horizon": 28, "n_train_series": train_r["id"].nunique()})
                mlflow.set_tags({"phase": "2", "model_family": model_name})
                try:
                    if model_name == "seasonal_naive":
                        preds = run_seasonal_naive(train_r, test_r)
                    elif model_name == "ets":
                        preds = run_ets(train_r, test_r)
                    elif model_name == "arima":
                        preds = run_arima(train_r, test_r)
                    else:
                        preds = run_lightgbm(train_f, test_f, full_aux=df)
                except Exception as e:
                    dt = time.time() - t0
                    mlflow.set_tag("status", f"FAILED: {type(e).__name__}")
                    print(f"  [{model_name:14s}] FAILED in {dt:.1f}s -- {type(e).__name__}: {e}")
                    continue
                dt = time.time() - t0

                per_series = score_per_series(preds, test_r, train_r, season=7)
                agg = aggregate_scores(per_series, df)
                agg.update({"model": model_name, "fold": fold.name, "secs": round(dt, 1)})
                leaderboard_rows.append(agg)
                preds["model"] = model_name; preds["fold"] = fold.name
                all_preds.append(preds)

                mlflow.log_metrics({"wmape_vw":  agg["wmape_volume_weighted"],
                                    "wmape_uw":  agg["wmape_unweighted"],
                                    "rmse":      agg["rmse_mean"],
                                    "mase":      agg["mase_median"],
                                    "train_secs": dt})

                print(f"  [{model_name:14s}] WMAPE-vw={agg['wmape_volume_weighted']:.3f}  "
                      f"WMAPE-uw={agg['wmape_unweighted']:.3f}  "
                      f"RMSE={agg['rmse_mean']:.2f}  MASE={agg['mase_median']:.2f}  "
                      f"({dt:.1f}s)")

    # ---- aggregate across folds + save ---------------------------------
    lb = pd.DataFrame(leaderboard_rows)
    lb.to_csv(OUT / "leaderboard_per_fold.csv", index=False)

    summary = (lb.groupby("model")
                 .agg(wmape_vw=("wmape_volume_weighted", "mean"),
                      wmape_uw=("wmape_unweighted", "mean"),
                      rmse=("rmse_mean", "mean"),
                      mase=("mase_median", "median"),
                      secs=("secs", "sum"))
                 .sort_values("wmape_vw"))
    summary.to_csv(OUT / "leaderboard_summary.csv")
    print("\n=== LEADERBOARD (avg across folds, sorted by volume-weighted WMAPE) ===")
    print(summary.round(3).to_string())

    champion = summary.index[0]
    print(f"\n*** CHAMPION: {champion}  (WMAPE-vw = {summary.loc[champion, 'wmape_vw']:.3f}) ***")

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_parquet(OUT / "predictions.parquet", index=False)

    # ---- plots ---------------------------------------------------------
    plot_leaderboard(summary)
    plot_per_fold(lb)
    plot_actuals_vs_pred(df, preds_df, folds[-1])   # plot the newest fold

    print(f"\n  artifacts in {OUT}/")


# ---- plots ------------------------------------------------------------------

def _save(fig, name):
    p = OUT / f"{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def plot_leaderboard(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    summary[["wmape_vw", "wmape_uw"]].plot.barh(ax=ax)
    ax.set_xlabel("WMAPE  (lower is better)")
    ax.set_title("Phase 2 baselines -- WMAPE volume-weighted vs unweighted")
    ax.invert_yaxis()
    _save(fig, "01_leaderboard")


def plot_per_fold(lb: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=lb, x="fold", y="wmape_volume_weighted", hue="model", ax=ax)
    ax.set_ylabel("WMAPE (volume-weighted)"); ax.set_title("Per-fold WMAPE by model")
    _save(fig, "02_per_fold_wmape")


def plot_actuals_vs_pred(raw: pd.DataFrame, preds: pd.DataFrame, fold) -> None:
    # pick top-3 by training-period volume so the plot is readable
    top3 = (raw[raw["date"] <= fold.train_end]
              .groupby("id")["sales"].sum().sort_values(ascending=False).head(3).index.tolist())
    fold_preds = preds[preds["fold"] == fold.name]
    actuals = raw[(raw["date"] >= fold.test_start) & (raw["date"] <= fold.test_end)]

    fig, axes = plt.subplots(len(top3), 1, figsize=(12, 9), sharex=True)
    for ax, sid in zip(axes, top3):
        a = actuals[actuals["id"] == sid]
        ax.plot(a["date"], a["sales"], "k-", lw=1.5, label="actual")
        for m, sub in fold_preds[fold_preds["id"] == sid].groupby("model"):
            ax.plot(sub["date"], sub["y_pred"], lw=1.2, label=m, alpha=0.85)
        ax.set_title(sid); ax.set_ylabel("units")
        ax.legend(fontsize=8, ncol=5, loc="upper right")
    axes[-1].set_xlabel("date")
    _save(fig, "03_actuals_vs_pred_topvol")


if __name__ == "__main__":
    main()
