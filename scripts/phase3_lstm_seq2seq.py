"""
Phase 3b: train the seq2seq LSTM challenger (the fix for the recursive collapse)
and compare to Phase 2 champion + the recursive LSTM run.

Each fold's training run logs to MLflow under experiment 'phase3_lstm' with
run_name=lstm_seq2seq_fold_X so it shows side-by-side with the recursive runs.

Run: python -m scripts.phase3_lstm_seq2seq
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
PHASE3_DIR = Path("reports/phase3_lstm")
OUT = Path("reports/phase3_lstm_seq2seq")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("phase3_lstm")


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
            "n_series": int(len(valid))}


def main():
    print(f"loading {DATA}...")
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_active"]].copy()
    print(f"  {len(df):,} active rows, {df['id'].nunique()} series")

    folds = rolling_origin_splits(df["date"].max(), n_folds=3, horizon=28, step=28)
    cfg = Seq2SeqConfig()

    rows, all_preds, losses = [], [], {}
    for fold in folds:
        print(f"\n=== {fold.name}: train<= {fold.train_end.date()} | "
              f"test {fold.test_start.date()}..{fold.test_end.date()} ===")
        train, test = split_frame(df, fold)

        with mlflow.start_run(run_name=f"lstm_seq2seq_{fold.name}"):
            mlflow.log_params({
                "fold": fold.name, "horizon": cfg.H,
                "L": cfg.L, "stride": cfg.stride,
                "hidden": cfg.hidden, "item_embed_dim": cfg.item_embed_dim,
                "dropout": cfg.dropout, "lr": cfg.lr,
                "batch_size": cfg.batch_size, "epochs": cfg.epochs,
                "n_train_series": train["id"].nunique(),
                "arch": "encoder_decoder_lstm",
            })

            t0 = time.time()
            preds, info = train_and_predict_seq2seq(train, test, df, cfg)
            dt = time.time() - t0

            per_series = score_per_series(preds, test, train, season=7)
            agg = aggregate(per_series, df)
            agg.update({"model": "lstm_seq2seq", "fold": fold.name, "secs": round(dt, 1)})
            rows.append(agg)
            preds["model"] = "lstm_seq2seq"; preds["fold"] = fold.name
            all_preds.append(preds)
            losses[fold.name] = info["epoch_losses"]

            for ep, l in enumerate(info["epoch_losses"], 1):
                mlflow.log_metric("train_loss", l, step=ep)
            mlflow.log_metrics({"wmape_vw": agg["wmape_vw"],
                                "wmape_uw": agg["wmape_uw"],
                                "rmse": agg["rmse_mean"],
                                "mase": agg["mase_median"],
                                "train_secs": dt,
                                "n_windows": info["n_windows"]})
            mlflow.set_tags({"phase": "3", "model_family": "lstm_seq2seq",
                             "device": info["device"]})
            print(f"  WMAPE-vw={agg['wmape_vw']:.3f}  WMAPE-uw={agg['wmape_uw']:.3f}  "
                  f"RMSE={agg['rmse_mean']:.2f}  MASE={agg['mase_median']:.2f}  "
                  f"({dt:.1f}s on {info['device']})")

    # ---- aggregate + combine with Phase 2 + recursive LSTM --------------
    lb = pd.DataFrame(rows)
    lb.to_csv(OUT / "seq2seq_per_fold.csv", index=False)
    summary = pd.DataFrame([{
        "model": "lstm_seq2seq",
        "wmape_vw": lb["wmape_vw"].mean(),
        "wmape_uw": lb["wmape_uw"].mean(),
        "rmse":     lb["rmse_mean"].mean(),
        "mase":     lb["mase_median"].median(),
        "secs":     lb["secs"].sum(),
    }])
    print("\n=== LSTM-seq2seq (avg across folds) ===")
    print(summary.round(3).to_string(index=False))

    # Save predictions IMMEDIATELY (so a plotting failure doesn't cost us the training run)
    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_parquet(OUT / "seq2seq_predictions.parquet", index=False)

    p2 = pd.read_csv(PHASE2_DIR / "leaderboard_summary.csv")
    pieces = [p2, summary]
    p3_old_path = PHASE3_DIR / "lstm_per_fold.csv"
    if p3_old_path.exists():
        p3_old = pd.read_csv(p3_old_path)
        pieces.insert(1, pd.DataFrame([{
            "model": "lstm_recursive",
            "wmape_vw": p3_old["wmape_vw"].mean(),
            "wmape_uw": p3_old["wmape_uw"].mean(),
            "rmse":     p3_old["rmse_mean"].mean(),
            "mase":     p3_old["mase_median"].median(),
            "secs":     p3_old["secs"].sum(),
        }]))
    combined = pd.concat(pieces, ignore_index=True).sort_values("wmape_vw")
    combined.to_csv(OUT / "leaderboard_combined.csv", index=False)
    print("\n=== COMBINED LEADERBOARD ===")
    print(combined.round(3).to_string(index=False))
    print(f"\n*** OVERALL CHAMPION by WMAPE-vw: {combined.iloc[0]['model']} ***")

    # ---- plots (best-effort: don't crash the run on a plot failure) ----
    plot_combined_leaderboard(combined)
    try:
        plot_per_fold_all(lb,
                          PHASE2_DIR / "leaderboard_per_fold.csv",
                          p3_old_path if p3_old_path.exists() else None)
    except Exception as e:
        print(f"  [warn] per-fold plot skipped: {type(e).__name__}: {e}")
    plot_loss_curves(losses)
    try:
        plot_actuals_vs_pred(df, preds_df, folds[-1])
    except Exception as e:
        print(f"  [warn] actuals-vs-pred plot skipped: {type(e).__name__}: {e}")

    print(f"\n  artifacts in {OUT}/")
    print("  view MLflow: .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db")


# ---- plots ------------------------------------------------------------------

def _save(fig, name):
    p = OUT / f"{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def plot_combined_leaderboard(combined: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    combined.set_index("model")[["wmape_vw", "wmape_uw"]].plot.barh(ax=ax)
    ax.set_xlabel("WMAPE (lower is better)")
    ax.set_title("Phase 2 baselines + Phase 3 LSTM variants")
    ax.invert_yaxis()
    _save(fig, "01_combined_leaderboard")


def plot_per_fold_all(lstm_lb, p2_csv: Path, p3_csv: Path | None):
    p2 = pd.read_csv(p2_csv).rename(columns={"wmape_volume_weighted": "wmape_vw"})[
        ["model", "fold", "wmape_vw"]]
    pieces = [p2]
    if p3_csv is not None and p3_csv.exists():
        p3 = pd.read_csv(p3_csv).assign(model="lstm_recursive")[["model", "fold", "wmape_vw"]]
        pieces.append(p3)
    pieces.append(lstm_lb[["model", "fold", "wmape_vw"]])
    combined = pd.concat(pieces, ignore_index=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.barplot(data=combined, x="fold", y="wmape_vw", hue="model", ax=ax)
    ax.set_ylabel("WMAPE (volume-weighted)")
    ax.set_title("Per-fold WMAPE -- all models")
    _save(fig, "02_per_fold_all_models")


def plot_loss_curves(losses: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    for fold_name, ls in losses.items():
        ax.plot(range(1, len(ls) + 1), ls, marker="o", label=fold_name)
    ax.set_xlabel("epoch"); ax.set_ylabel("train MSE (log1p, full H=28)")
    ax.set_title("Seq2seq training loss per fold")
    ax.legend()
    _save(fig, "03_training_loss")


def plot_actuals_vs_pred(raw, seq2seq_preds, fold):
    p2_preds = pd.read_parquet(PHASE2_DIR / "predictions.parquet")
    # Plot champion (ets) and runner-up (lightgbm) from Phase 2 for context
    p2_champ = p2_preds[(p2_preds["model"] == "ets") & (p2_preds["fold"] == fold.name)]
    p2_lgbm  = p2_preds[(p2_preds["model"] == "lightgbm") & (p2_preds["fold"] == fold.name)]
    p3_recursive_path = PHASE3_DIR / "lstm_predictions.parquet"
    p3_preds = pd.read_parquet(p3_recursive_path) if p3_recursive_path.exists() else None

    top3 = (raw[raw["date"] <= fold.train_end]
              .groupby("id")["sales"].sum().sort_values(ascending=False).head(3).index.tolist())
    actuals = raw[(raw["date"] >= fold.test_start) & (raw["date"] <= fold.test_end)]

    fig, axes = plt.subplots(len(top3), 1, figsize=(12, 9), sharex=True)
    for ax, sid in zip(axes, top3):
        a = actuals[actuals["id"] == sid]
        ax.plot(a["date"], a["sales"], "k-", lw=1.5, label="actual")
        for label, df_ in [("ets (champion)", p2_champ), ("lightgbm", p2_lgbm)]:
            sub = df_[df_["id"] == sid]
            if len(sub):
                ax.plot(sub["date"], sub["y_pred"], lw=1.0, label=label, alpha=0.7)
        if p3_preds is not None:
            lr = p3_preds[(p3_preds["model"] == "lstm") & (p3_preds["fold"] == fold.name)
                          & (p3_preds["id"] == sid)]
            if len(lr):
                ax.plot(lr["date"], lr["y_pred"], lw=1.0, label="lstm_recursive", alpha=0.7)
        ls = seq2seq_preds[(seq2seq_preds["id"] == sid) & (seq2seq_preds["fold"] == fold.name)]
        ax.plot(ls["date"], ls["y_pred"], lw=1.5, label="lstm_seq2seq", alpha=0.9)
        ax.set_title(sid); ax.set_ylabel("units")
        ax.legend(fontsize=8, ncol=4, loc="upper right")
    axes[-1].set_xlabel("date")
    _save(fig, "04_actuals_seq2seq_vs_others")


if __name__ == "__main__":
    main()
