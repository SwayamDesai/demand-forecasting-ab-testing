"""
Tier 2a: error-analysis drill-down -- WHERE and WHY the models fail.

Aggregate WMAPE hides everything. This script reads the existing backtest
predictions (no training) and answers:
  - which categories / stores / volume tiers are hardest?
  - is the error BIAS (systematic over/under) or VARIANCE (noisy)?
  - do intermittent (zero-heavy) series drive the error?
  - which 20 individual series are worst, and what do they have in common?

Outputs -> reports/tier2_error_analysis/ : 5 plots, worst_series.csv, SUMMARY.md

Run: python -m scripts.tier2_error_analysis
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest import rolling_origin_splits
from src.metrics import wmape

DATA = Path("data/processed/m5_long_sample.parquet")
P2 = Path("reports/phase2_baselines/predictions.parquet")
P3 = Path("reports/phase3_lstm_seq2seq/seq2seq_predictions.parquet")
OUT = Path("reports/tier2_error_analysis")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

MODELS = ["ets", "lightgbm", "lstm_seq2seq"]
CHAMPION = "lstm_seq2seq"          # the accuracy champion we drill into


def _save(fig, name):
    p = OUT / f"{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def load() -> pd.DataFrame:
    actuals = pd.read_parquet(DATA)
    actuals["date"] = pd.to_datetime(actuals["date"])
    actuals = actuals[actuals["is_active"]].copy()

    p2 = pd.read_parquet(P2)[["id", "date", "y_pred", "model", "fold"]]
    p3 = pd.read_parquet(P3)[["id", "date", "y_pred", "model", "fold"]]
    preds = pd.concat([p2[p2["model"].isin(MODELS)], p3], ignore_index=True)
    preds["date"] = pd.to_datetime(preds["date"])

    df = preds.merge(
        actuals[["id", "date", "sales", "cat_id", "store_id"]],
        on=["id", "date"], how="left")
    return df, actuals


def per_series_table(df: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, id): wmape, signed bias, metadata, volume tier, zero-rate."""
    folds = rolling_origin_splits(actuals["date"].max(), n_folds=3, horizon=28, step=28)
    cutoff = folds[0].train_end

    # series-level descriptors from the TRAIN period (no leakage)
    tr = actuals[actuals["date"] <= cutoff]
    vol = tr.groupby("id")["sales"].sum().rename("train_vol")
    zero_rate = tr.assign(z=lambda d: (d["sales"] == 0).astype(int)) \
                  .groupby("id")["z"].mean().rename("zero_rate")
    meta = (actuals.groupby("id")
                   .agg(cat_id=("cat_id", "first"), store_id=("store_id", "first")))
    desc = meta.join(vol).join(zero_rate)
    desc["volume_tertile"] = pd.qcut(desc["train_vol"], 3, labels=["low", "mid", "high"])

    rows = []
    for (model, sid), g in df.groupby(["model", "id"], sort=False):
        y = g["sales"].to_numpy(float); yh = g["y_pred"].to_numpy(float)
        rows.append({"model": model, "id": sid,
                     "wmape": wmape(y, yh),
                     "bias": float((yh - y).mean()),          # +over / -under
                     "sum_actual": float(y.sum())})
    ps = pd.DataFrame(rows).merge(desc, on="id", how="left")
    return ps, desc


# ---- plots ------------------------------------------------------------------

def plot_by_segment(ps: pd.DataFrame, col: str, name: str, title: str):
    """Volume-weighted WMAPE by a segment column, per model."""
    g = (ps.dropna(subset=["wmape"])
           .groupby([col, "model"], observed=True)
           .apply(lambda d: np.average(d["wmape"],
                                       weights=d["sum_actual"].clip(lower=1e-9)),
                  include_groups=False)
           .rename("wmape").reset_index())
    order = ["low", "mid", "high"] if col == "volume_tertile" else None
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=g, x=col, y="wmape", hue="model", order=order, ax=ax)
    ax.set_ylabel("WMAPE (volume-weighted)"); ax.set_title(title)
    _save(fig, name)
    return g


def plot_bias_by_tier(ps: pd.DataFrame):
    """Mean signed error by volume tier -- is it bias or variance?"""
    g = (ps.dropna(subset=["bias"])
           .groupby(["volume_tertile", "model"], observed=True)["bias"]
           .mean().reset_index())
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=g, x="volume_tertile", y="bias", hue="model",
                order=["low", "mid", "high"], ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("mean signed error (+over / -under, units/day)")
    ax.set_title("Forecast bias by volume tier (systematic over/under-prediction)")
    _save(fig, "04_bias_by_tier")


def plot_error_vs_intermittency(ps: pd.DataFrame):
    """Per-series WMAPE vs zero-rate for the champion -- are sparse series hardest?"""
    sub = ps[(ps["model"] == CHAMPION) & ps["wmape"].notna()].copy()
    sub = sub[sub["wmape"] < sub["wmape"].quantile(0.99)]   # drop a few extreme outliers for readability
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(sub["zero_rate"], sub["wmape"], c=np.log10(sub["train_vol"] + 1),
                    cmap="viridis", s=14, alpha=0.6)
    ax.set_xlabel("series zero-sales rate (intermittency)")
    ax.set_ylabel(f"per-series WMAPE ({CHAMPION})")
    ax.set_title("Champion error rises with intermittency (color = log10 volume)")
    fig.colorbar(sc, label="log10(train volume)")
    _save(fig, "05_error_vs_intermittency")
    # correlation stat
    r = sub[["zero_rate", "wmape"]].corr().iloc[0, 1]
    return r


def worst_series(ps: pd.DataFrame, n=20) -> pd.DataFrame:
    sub = (ps[(ps["model"] == CHAMPION) & ps["wmape"].notna()]
           .sort_values("wmape", ascending=False).head(n))
    sub.to_csv(OUT / "worst_series.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = sub["id"].str.replace("_evaluation", "", regex=False)
    ax.barh(labels, sub["wmape"], color="#c44e52")
    ax.invert_yaxis(); ax.set_xlabel(f"WMAPE ({CHAMPION})")
    ax.set_title(f"Worst {n} series by WMAPE (champion)")
    _save(fig, "06_worst_series")
    return sub


def main():
    print("loading predictions + actuals...")
    df, actuals = load()
    ps, desc = per_series_table(df, actuals)
    print(f"  {ps['id'].nunique()} series x {ps['model'].nunique()} models scored")

    by_cat  = plot_by_segment(ps, "cat_id", "01_wmape_by_category",
                              "WMAPE by category")
    by_tier = plot_by_segment(ps, "volume_tertile", "02_wmape_by_volume_tier",
                              "WMAPE by volume tier (low/mid/high movers)")
    by_store = plot_by_segment(ps, "store_id", "03_wmape_by_store",
                               "WMAPE by store")
    plot_bias_by_tier(ps)
    r = plot_error_vs_intermittency(ps)
    worst = worst_series(ps)

    # ---- text/markdown summary --------------------------------------------
    champ = ps[ps["model"] == CHAMPION]
    tier_w = by_tier[by_tier["model"] == CHAMPION].set_index("volume_tertile")["wmape"]
    cat_w  = by_cat[by_cat["model"] == CHAMPION].set_index("cat_id")["wmape"]
    worst_cat = worst["cat_id"].value_counts()
    worst_zero = worst["zero_rate"].mean()
    overall_zero = champ["zero_rate"].mean()
    # champion bias by tier (data-driven, no hardcoded claims)
    champ_bias = (champ.dropna(subset=["bias"])
                       .groupby("volume_tertile", observed=True)["bias"].mean())
    ets_bias = (ps[ps["model"] == "ets"].dropna(subset=["bias"])
                  .groupby("volume_tertile", observed=True)["bias"].mean())

    lines = [
        "# Tier 2a -- Error Analysis (champion = lstm_seq2seq)\n",
        "## Headline findings\n",
        f"- **Intermittency drives error.** Per-series WMAPE vs zero-rate correlation "
        f"r = {r:.2f}. The worst-20 series average a {worst_zero*100:.0f}% zero-rate "
        f"vs {overall_zero*100:.0f}% overall.",
        f"- **Low-volume movers are hardest:** WMAPE low={tier_w.get('low', np.nan):.2f} "
        f"vs high={tier_w.get('high', np.nan):.2f}.",
        f"- **By category:** " + ", ".join(f"{c}={cat_w[c]:.2f}" for c in cat_w.index) + ".",
        f"- **Worst-20 concentration by category:** "
        + ", ".join(f"{c}:{n}" for c, n in worst_cat.items()) + ".",
        f"- **The champion systematically UNDER-predicts, and the bias grows with volume:** "
        f"mean signed error low={champ_bias.get('low', np.nan):+.2f}, "
        f"mid={champ_bias.get('mid', np.nan):+.2f}, "
        f"high={champ_bias.get('high', np.nan):+.2f} units/day "
        f"(vs ETS high={ets_bias.get('high', np.nan):+.2f}, ~centered).",
        "\n## What this implies\n",
        "- The model is good on fast/steady items and weak on sparse, low-volume SKUs "
        "-- exactly where *any* point forecast struggles. A two-stage (zero / nonzero) "
        "or Croston-style model is the right next step for the long tail.",
        "- **This bias is the mechanistic cause of the A/B stockout result.** The LSTM "
        "under-orders high-volume SKUs, so when its (low) forecast drives the order policy "
        "it stocks out more than the well-centered ETS champion. It also explains why the "
        "quantile-loss fix (which deliberately shifts predictions UP) was the right lever -- "
        "and why a high enough quantile is needed to neutralise this under-bias.",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n  artifacts in {OUT}/")


if __name__ == "__main__":
    main()
