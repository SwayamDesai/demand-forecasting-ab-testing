"""
Phase 3c: re-train the seq2seq LSTM with PINBALL (quantile) loss at tau=0.80.

Why: Phase 4 A/B showed LSTM mean predictions caused MORE stockouts than
LightGBM because of a slight under-prediction bias under asymmetric (5:1)
stockout-vs-holding costs. Pinball loss at tau=0.80 deliberately biases
predictions UP toward the 80th-percentile demand (close to the newsvendor
optimum 0.833 for our cost ratio).

MLflow runs are tagged lstm_seq2seq_q80_fold_X so they compare side-by-side
with the MSE-trained seq2seq runs.

Run: python -m scripts.phase3_lstm_quantile
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
from src.models.deep import Seq2SeqConfig, train_and_predict_seq2seq

DATA = Path("data/processed/m5_long_sample.parquet")
PHASE2_DIR = Path("reports/phase2_baselines")
SEQ2SEQ_DIR = Path("reports/phase3_lstm_seq2seq")
OUT = Path("reports/phase3_lstm_quantile")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("phase3_lstm")

QUANTILE = 0.80


def score_per_series(preds, test, train, season=7):
    merged = preds.merge(test[["id", "date", "sales"]], on=["id", "date"], how="inner")
    rows = []
    for sid, g in merged.groupby("id", sort=False):
        y  = g["sales"].to_numpy(dtype=float)
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
            "bias_units": float((valid["sum_pred"] - valid["sum_actual"]).sum()),
            "n_series": int(len(valid))}


def main():
    print(f"loading {DATA}...")
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_active"]].copy()
    print(f"  {len(df):,} active rows, {df['id'].nunique()} series")

    folds = rolling_origin_splits(df["date"].max(), n_folds=3, horizon=28, step=28)
    cfg = Seq2SeqConfig(quantile=QUANTILE)

    rows, all_preds, losses = [], [], {}
    for fold in folds:
        print(f"\n=== {fold.name}: train<= {fold.train_end.date()} | "
              f"test {fold.test_start.date()}..{fold.test_end.date()} ===")
        train, test = split_frame(df, fold)

        with mlflow.start_run(run_name=f"lstm_seq2seq_q80_{fold.name}"):
            mlflow.log_params({
                "fold": fold.name, "horizon": cfg.H, "L": cfg.L, "stride": cfg.stride,
                "hidden": cfg.hidden, "item_embed_dim": cfg.item_embed_dim,
                "dropout": cfg.dropout, "lr": cfg.lr,
                "batch_size": cfg.batch_size, "epochs": cfg.epochs,
                "loss": "pinball", "quantile": cfg.quantile,
                "n_train_series": train["id"].nunique(),
                "arch": "encoder_decoder_lstm",
            })

            t0 = time.time()
            preds, info = train_and_predict_seq2seq(train, test, df, cfg)
            dt = time.time() - t0

            per_series = score_per_series(preds, test, train, season=7)
            agg = aggregate(per_series, df)
            agg.update({"model": "lstm_seq2seq_q80", "fold": fold.name,
                        "secs": round(dt, 1)})
            rows.append(agg)
            preds["model"] = "lstm_seq2seq_q80"; preds["fold"] = fold.name
            all_preds.append(preds)
            losses[fold.name] = info["epoch_losses"]

            for ep, l in enumerate(info["epoch_losses"], 1):
                mlflow.log_metric("train_loss", l, step=ep)
            mlflow.log_metrics({"wmape_vw": agg["wmape_vw"],
                                "wmape_uw": agg["wmape_uw"],
                                "rmse": agg["rmse_mean"],
                                "mase": agg["mase_median"],
                                "bias_units": agg["bias_units"],
                                "train_secs": dt,
                                "n_windows": info["n_windows"]})
            mlflow.set_tags({"phase": "3", "model_family": "lstm_seq2seq_q80",
                             "device": info["device"]})
            print(f"  WMAPE-vw={agg['wmape_vw']:.3f}  WMAPE-uw={agg['wmape_uw']:.3f}  "
                  f"RMSE={agg['rmse_mean']:.2f}  MASE={agg['mase_median']:.2f}  "
                  f"BIAS={agg['bias_units']:+.0f} units  ({dt:.1f}s on {info['device']})")

    # ---- aggregate & save -------------------------------------------------
    lb = pd.DataFrame(rows)
    lb.to_csv(OUT / "q80_per_fold.csv", index=False)
    summary = pd.DataFrame([{
        "model": "lstm_seq2seq_q80",
        "wmape_vw": lb["wmape_vw"].mean(),
        "wmape_uw": lb["wmape_uw"].mean(),
        "rmse":     lb["rmse_mean"].mean(),
        "mase":     lb["mase_median"].median(),
        "bias_units": lb["bias_units"].sum(),
        "secs":     lb["secs"].sum(),
    }])
    print("\n=== LSTM-seq2seq-q80 (avg across folds) ===")
    print(summary.round(3).to_string(index=False))

    # Bring in Phase 2 + Phase 3 (mse seq2seq) for a comparison view
    p2 = pd.read_csv(PHASE2_DIR / "leaderboard_summary.csv").assign(bias_units=np.nan)
    p3_seq2seq = pd.read_csv(SEQ2SEQ_DIR / "seq2seq_per_fold.csv")
    p3_summary = pd.DataFrame([{
        "model": "lstm_seq2seq_mse",
        "wmape_vw": p3_seq2seq["wmape_vw"].mean(),
        "wmape_uw": p3_seq2seq["wmape_uw"].mean(),
        "rmse":     p3_seq2seq["rmse_mean"].mean(),
        "mase":     p3_seq2seq["mase_median"].median(),
        "bias_units": np.nan,
        "secs":     p3_seq2seq["secs"].sum(),
    }])
    combined = (pd.concat([p2, p3_summary, summary], ignore_index=True)
                  .sort_values("wmape_vw"))
    combined.to_csv(OUT / "leaderboard_combined.csv", index=False)
    print("\n=== COMBINED LEADERBOARD ===")
    print(combined.round(3).to_string(index=False))

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_parquet(OUT / "q80_predictions.parquet", index=False)

    # ---- plots ------------------------------------------------------------
    plot_combined_leaderboard(combined)
    plot_bias_comparison(preds_df, df, folds)
    plot_loss_curves(losses)
    plot_actuals_vs_pred(df, preds_df, folds[-1])
    print(f"\n  artifacts in {OUT}/")
    print("  view MLflow: .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db")


# ---- plots ------------------------------------------------------------------

def _save(fig, name):
    p = OUT / f"{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def plot_combined_leaderboard(combined):
    fig, ax = plt.subplots(figsize=(10, 4))
    combined.set_index("model")[["wmape_vw", "wmape_uw"]].plot.barh(ax=ax)
    ax.set_xlabel("WMAPE (lower is better)")
    ax.set_title("All models incl. quantile-trained LSTM (q80)")
    ax.invert_yaxis()
    _save(fig, "01_combined_leaderboard")


def plot_bias_comparison(q80_preds, raw_df, folds):
    """Show pred-actual bias for q80 vs the existing models.
    Positive bias = model predicts higher than actual (good for inventory)."""
    test_dates = []
    for fold in folds:
        test_dates.extend(pd.date_range(fold.test_start, fold.test_end))
    test_dates = pd.DatetimeIndex(test_dates).unique()
    actuals_in_test = raw_df[raw_df["date"].isin(test_dates)][["id", "date", "sales"]]

    p2 = pd.read_parquet(PHASE2_DIR / "predictions.parquet")
    p2 = p2[p2["model"] == "lightgbm"][["id", "date", "y_pred"]].assign(model="lightgbm")
    p3 = pd.read_parquet(SEQ2SEQ_DIR / "seq2seq_predictions.parquet"
                         )[["id", "date", "y_pred"]].assign(model="lstm_mse")
    p3q = q80_preds[["id", "date", "y_pred"]].assign(model="lstm_q80")
    allp = pd.concat([p2, p3, p3q], ignore_index=True)
    allp = allp.merge(actuals_in_test, on=["id", "date"], how="inner")
    allp["err"] = allp["y_pred"] - allp["sales"]   # positive = over-predict

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=allp, x="model", y="err", showfliers=False, ax=ax)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("y_pred - actual (positive = over-predict)")
    ax.set_title("Prediction bias by model (we WANT positive bias under 5:1 cost)")
    _save(fig, "02_bias_comparison")

    print(f"\n  median bias per row -- lightgbm: "
          f"{allp[allp['model']=='lightgbm']['err'].median():+.2f}, "
          f"lstm_mse: {allp[allp['model']=='lstm_mse']['err'].median():+.2f}, "
          f"lstm_q80: {allp[allp['model']=='lstm_q80']['err'].median():+.2f}")


def plot_loss_curves(losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    for fold_name, ls in losses.items():
        ax.plot(range(1, len(ls) + 1), ls, marker="o", label=fold_name)
    ax.set_xlabel("epoch"); ax.set_ylabel("pinball loss (log1p scale)")
    ax.set_title(f"Quantile-LSTM training loss per fold (tau={QUANTILE})")
    ax.legend()
    _save(fig, "03_training_loss")


def plot_actuals_vs_pred(raw, q80_preds, fold):
    p2 = pd.read_parquet(PHASE2_DIR / "predictions.parquet")
    p2 = p2[(p2["model"] == "lightgbm") & (p2["fold"] == fold.name)]
    p3 = pd.read_parquet(SEQ2SEQ_DIR / "seq2seq_predictions.parquet")
    p3 = p3[p3["fold"] == fold.name]

    top3 = (raw[raw["date"] <= fold.train_end]
              .groupby("id")["sales"].sum().sort_values(ascending=False).head(3).index.tolist())
    actuals = raw[(raw["date"] >= fold.test_start) & (raw["date"] <= fold.test_end)]

    fig, axes = plt.subplots(len(top3), 1, figsize=(12, 9), sharex=True)
    for ax, sid in zip(axes, top3):
        a = actuals[actuals["id"] == sid]
        ax.plot(a["date"], a["sales"], "k-", lw=1.5, label="actual")
        lg = p2[p2["id"] == sid]
        ax.plot(lg["date"], lg["y_pred"], lw=1.0, label="lightgbm", alpha=0.7)
        lm = p3[p3["id"] == sid]
        ax.plot(lm["date"], lm["y_pred"], lw=1.0, label="lstm_mse", alpha=0.7)
        lq = q80_preds[(q80_preds["id"] == sid) & (q80_preds["fold"] == fold.name)]
        ax.plot(lq["date"], lq["y_pred"], lw=1.5, label=f"lstm_q{int(QUANTILE*100)}", alpha=0.95)
        ax.set_title(sid); ax.set_ylabel("units")
        ax.legend(fontsize=8, ncol=4, loc="upper right")
    axes[-1].set_xlabel("date")
    _save(fig, "04_actuals_q80_vs_others")


if __name__ == "__main__":
    main()
