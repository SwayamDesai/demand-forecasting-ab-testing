"""
Phase 3: train the LSTM challenger and compare to Phase 2 champion.

Each fold's training run is logged as a separate MLflow run under the
'phase3_lstm' experiment. After this runs, `mlflow ui --backend-store-uri
file:./mlruns` shows the runs side-by-side.

Run: python -m scripts.phase3_lstm
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
from src.metrics import mase, rmse, wmape
from src.models.deep import LSTMConfig, train_and_predict

DATA = Path("data/processed/m5_long_sample.parquet")
PHASE2_DIR = Path("reports/phase2_baselines")
OUT = Path("reports/phase3_lstm")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("phase3_lstm")


def score_per_series(preds, test, train, season=7):
    merged = preds.merge(test[["id", "date", "sales"]], on=["id", "date"], how="inner")
    rows = []
    for sid, g in merged.groupby("id", sort=False):
        y = g["sales"].to_numpy(dtype=float)
        yh = g["y_pred"].to_numpy(dtype=float)
        yt = train.loc[train["id"] == sid, "sales"].to_numpy(dtype=float)
        rows.append({"id": sid,
                     "wmape": wmape(y, yh), "rmse": rmse(y, yh),
                     "mase": mase(y, yh, yt, season=season),
                     "sum_actual": float(y.sum()), "sum_pred": float(yh.sum())})
    return pd.DataFrame(rows)


def aggregate(per_series, meta):
    df = per_series.merge(meta[["id", "cat_id"]].drop_duplicates(), on="id", how="left")
    valid = df.dropna(subset=["wmape"])
    w = valid["sum_actual"].clip(lower=0).to_numpy()
    return {"wmape_vw": float(np.average(valid["wmape"], weights=w) if w.sum() > 0 else np.nan),
            "wmape_uw": float(valid["wmape"].mean()),
            "rmse_mean": float(valid["rmse"].mean()),
            "mase_median": float(valid["mase"].dropna().median()),
            "n_series": int(len(valid))}


def main():
    print(f"loading {DATA}...")
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_active"]].copy()
    print(f"  {len(df):,} active rows, {df['id'].nunique()} series")

    folds = rolling_origin_splits(df["date"].max(), n_folds=3, horizon=28, step=28)
    cfg = LSTMConfig()

    leaderboard_rows = []
    all_preds = []
    all_losses = {}

    for fold in folds:
        print(f"\n=== {fold.name}: train<= {fold.train_end.date()} | "
              f"test {fold.test_start.date()}..{fold.test_end.date()} ===")
        train, test = split_frame(df, fold)

        with mlflow.start_run(run_name=f"lstm_{fold.name}"):
            mlflow.log_params({
                "fold": fold.name, "horizon": 28,
                "L": cfg.L, "stride": cfg.stride,
                "hidden": cfg.hidden, "item_embed_dim": cfg.item_embed_dim,
                "dropout": cfg.dropout, "lr": cfg.lr,
                "batch_size": cfg.batch_size, "epochs": cfg.epochs,
                "n_train_series": train["id"].nunique(),
            })

            t0 = time.time()
            preds, info = train_and_predict(train, test, df, cfg)
            train_secs = time.time() - t0

            per_series = score_per_series(preds, test, train, season=7)
            agg = aggregate(per_series, df)
            agg["fold"] = fold.name; agg["secs"] = round(train_secs, 1)
            agg["model"] = "lstm"
            leaderboard_rows.append(agg)
            preds["model"] = "lstm"; preds["fold"] = fold.name
            all_preds.append(preds)
            all_losses[fold.name] = info["epoch_losses"]

            # log to MLflow
            for ep, loss in enumerate(info["epoch_losses"], 1):
                mlflow.log_metric("train_loss", loss, step=ep)
            mlflow.log_metrics({"wmape_vw": agg["wmape_vw"],
                                "wmape_uw": agg["wmape_uw"],
                                "rmse": agg["rmse_mean"],
                                "mase": agg["mase_median"],
                                "train_secs": train_secs,
                                "n_windows": info["n_windows"]})
            mlflow.set_tags({"phase": "3", "model_family": "lstm",
                             "device": info["device"]})

            print(f"  WMAPE-vw={agg['wmape_vw']:.3f}  WMAPE-uw={agg['wmape_uw']:.3f}  "
                  f"RMSE={agg['rmse_mean']:.2f}  MASE={agg['mase_median']:.2f}  "
                  f"({train_secs:.1f}s on {info['device']})")

    # ---- aggregate + save ----------------------------------------------
    lb = pd.DataFrame(leaderboard_rows)
    lb.to_csv(OUT / "lstm_per_fold.csv", index=False)

    summary = pd.DataFrame([{
        "model": "lstm",
        "wmape_vw": lb["wmape_vw"].mean(),
        "wmape_uw": lb["wmape_uw"].mean(),
        "rmse":     lb["rmse_mean"].mean(),
        "mase":     lb["mase_median"].median(),
        "secs":     lb["secs"].sum(),
    }])
    print("\n=== LSTM (avg across folds) ===")
    print(summary.round(3).to_string(index=False))

    # combine with Phase 2 leaderboard for a single comparison view
    p2 = pd.read_csv(PHASE2_DIR / "leaderboard_summary.csv")
    combined = pd.concat([p2, summary], ignore_index=True).sort_values("wmape_vw")
    combined.to_csv(OUT / "leaderboard_combined.csv", index=False)
    print("\n=== COMBINED LEADERBOARD (Phase 2 + Phase 3) ===")
    print(combined.round(3).to_string(index=False))

    champ = combined.iloc[0]["model"]
    chall = "lstm"
    print(f"\n*** Phase 2 champion: lightgbm   Phase 3 challenger: lstm ***")
    print(f"    overall champion by WMAPE-vw: {champ}")

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_parquet(OUT / "lstm_predictions.parquet", index=False)

    # ---- plots ---------------------------------------------------------
    plot_combined_leaderboard(combined)
    plot_per_fold(lb, p2_per_fold=PHASE2_DIR / "leaderboard_per_fold.csv")
    plot_loss_curves(all_losses)
    plot_actuals_vs_pred(df, preds_df, folds[-1])

    print(f"\n  artifacts in {OUT}/")
    print("  view MLflow: .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db")


# ---- plots ------------------------------------------------------------------

def _save(fig, name):
    p = OUT / f"{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def plot_combined_leaderboard(combined: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    combined.set_index("model")[["wmape_vw", "wmape_uw"]].plot.barh(ax=ax)
    ax.set_xlabel("WMAPE (lower is better)")
    ax.set_title("Combined leaderboard: Phase 2 baselines + Phase 3 LSTM challenger")
    ax.invert_yaxis()
    _save(fig, "01_combined_leaderboard")


def plot_per_fold(lstm_lb: pd.DataFrame, p2_per_fold: Path):
    p2 = pd.read_csv(p2_per_fold)
    p2_min = p2.rename(columns={"wmape_volume_weighted": "wmape_vw"})[["model", "fold", "wmape_vw"]]
    lstm_min = lstm_lb[["model", "fold", "wmape_vw"]]
    combined = pd.concat([p2_min, lstm_min], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=combined, x="fold", y="wmape_vw", hue="model", ax=ax)
    ax.set_ylabel("WMAPE (volume-weighted)")
    ax.set_title("Per-fold WMAPE -- all 5 models")
    _save(fig, "02_per_fold_all_models")


def plot_loss_curves(losses: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    for fold_name, ls in losses.items():
        ax.plot(range(1, len(ls) + 1), ls, marker="o", label=fold_name)
    ax.set_xlabel("epoch"); ax.set_ylabel("train MSE (log1p scale)")
    ax.set_title("LSTM training loss per fold")
    ax.legend()
    _save(fig, "03_training_loss")


def plot_actuals_vs_pred(raw: pd.DataFrame, lstm_preds: pd.DataFrame, fold) -> None:
    # overlay champion (lightgbm) for visual comparison
    p2_preds = pd.read_parquet(PHASE2_DIR / "predictions.parquet")
    p2_preds = p2_preds[(p2_preds["model"] == "lightgbm") & (p2_preds["fold"] == fold.name)]

    top3 = (raw[raw["date"] <= fold.train_end]
              .groupby("id")["sales"].sum().sort_values(ascending=False).head(3).index.tolist())
    actuals = raw[(raw["date"] >= fold.test_start) & (raw["date"] <= fold.test_end)]

    fig, axes = plt.subplots(len(top3), 1, figsize=(12, 9), sharex=True)
    for ax, sid in zip(axes, top3):
        a = actuals[actuals["id"] == sid]
        ax.plot(a["date"], a["sales"], "k-", lw=1.5, label="actual")
        lg = p2_preds[p2_preds["id"] == sid]
        ax.plot(lg["date"], lg["y_pred"], lw=1.2, label="lightgbm", alpha=0.85)
        ls = lstm_preds[(lstm_preds["id"] == sid) & (lstm_preds["fold"] == fold.name)]
        ax.plot(ls["date"], ls["y_pred"], lw=1.2, label="lstm", alpha=0.85)
        ax.set_title(sid); ax.set_ylabel("units")
        ax.legend(fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xlabel("date")
    _save(fig, "04_actuals_lstm_vs_lightgbm")


if __name__ == "__main__":
    main()
