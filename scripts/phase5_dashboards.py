"""
Phase 5: build dashboard inputs + mock dashboard PNGs.

Outputs (under reports/phase5_dashboards/):
  - forecasts_long.csv          Tableau-ready: actuals + every model's forecast
                                in long format, with hierarchy + fold + arm cols
  - ab_readout.csv              Power BI-ready: per-series A/B outcomes for both
                                experiment versions (v1 = LSTM MSE, v2 = LSTM q80)
                                with assignment, costs, stockouts, and lift columns

  - mock_tableau_forecasts.png  static preview of the Tableau dashboard layout
  - mock_powerbi_ab.png         static preview of the Power BI A/B readout

  - TABLEAU_BUILD.md            step-by-step recipe to recreate as a real .twbx
  - POWERBI_BUILD.md            step-by-step recipe for the .pbix

Run: python -m scripts.phase5_dashboards
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT = Path("reports/phase5_dashboards")
OUT.mkdir(parents=True, exist_ok=True)

ACTUALS  = Path("data/processed/m5_long_sample.parquet")
P2_PREDS = Path("reports/phase2_baselines/predictions.parquet")
P3_MSE   = Path("reports/phase3_lstm_seq2seq/seq2seq_predictions.parquet")
P3_Q80   = Path("reports/phase3_lstm_quantile/q80_predictions.parquet")
P4_V1_LONG = Path("reports/phase4_experiment/experiment_long_table.csv")
P4_V2_LONG = Path("reports/phase4_experiment_v2/experiment_long_table.csv")
P4_V1_SUMMARY = Path("reports/phase4_experiment/summary.csv")
P4_V2_SUMMARY = Path("reports/phase4_experiment_v2/summary.csv")

sns.set_theme(style="whitegrid", context="notebook")


# ============================================================================
# 1) Tableau-ready forecasts CSV (actuals + every model in long format)
# ============================================================================

def build_forecasts_csv() -> pd.DataFrame:
    print("[1/4] building forecasts_long.csv ...")
    actuals = pd.read_parquet(ACTUALS, columns=[
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "date", "sales", "is_active"
    ])
    actuals["date"] = pd.to_datetime(actuals["date"])
    actuals = actuals[actuals["is_active"]].copy()

    p2 = pd.read_parquet(P2_PREDS)[["id", "date", "fold", "model", "y_pred"]]
    p2["date"] = pd.to_datetime(p2["date"])
    p3m = pd.read_parquet(P3_MSE)[["id", "date", "fold", "model", "y_pred"]]
    p3m["date"] = pd.to_datetime(p3m["date"])
    p3q = pd.read_parquet(P3_Q80)[["id", "date", "fold", "model", "y_pred"]]
    p3q["date"] = pd.to_datetime(p3q["date"])
    all_pred = pd.concat([p2, p3m, p3q], ignore_index=True)

    test_dates = all_pred["date"].unique()
    actuals_test = actuals[actuals["date"].isin(test_dates)]

    long = all_pred.merge(
        actuals_test[["id", "item_id", "dept_id", "cat_id", "store_id",
                      "state_id", "date", "sales"]],
        on=["id", "date"], how="left",
    )
    long.rename(columns={"sales": "actual"}, inplace=True)
    long["error"] = long["y_pred"] - long["actual"]
    long["abs_error"] = long["error"].abs()

    path = OUT / "forecasts_long.csv"
    long.to_csv(path, index=False)
    print(f"     wrote {path}  ({len(long):,} rows, {long['model'].nunique()} models)")
    return long


# ============================================================================
# 2) Power BI-ready A/B readout CSV
# ============================================================================

def build_ab_csv() -> pd.DataFrame:
    print("[2/4] building ab_readout.csv ...")
    v1 = pd.read_csv(P4_V1_LONG); v1["version"] = "v1_lstm_mse"
    v2 = pd.read_csv(P4_V2_LONG); v2["version"] = "v2_lstm_q80"
    long = pd.concat([v1, v2], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])

    s1 = pd.read_csv(P4_V1_SUMMARY).assign(version="v1_lstm_mse")
    s2 = pd.read_csv(P4_V2_SUMMARY).assign(version="v2_lstm_q80")
    summary = pd.concat([s1, s2], ignore_index=True)

    long_path = OUT / "ab_readout.csv"
    long.to_csv(long_path, index=False)
    print(f"     wrote {long_path}  ({len(long):,} rows)")

    sum_path = OUT / "ab_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"     wrote {sum_path}  ({len(summary)} rows, 3 metrics x 2 versions)")
    return summary


# ============================================================================
# 3) Mock Tableau dashboard: 4 panels in one PNG
# ============================================================================

def mock_tableau_forecasts(long: pd.DataFrame):
    print("[3/4] rendering mock_tableau_forecasts.png ...")
    top3_ids = (long.groupby("id")["actual"].sum()
                    .sort_values(ascending=False).head(3).index.tolist())

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           height_ratios=[0.7, 1.8, 1.4],
                           hspace=0.55, wspace=0.35)

    # ----- header strip ----------------------------------------------------
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    ax0.text(0.01, 0.7, "Forecasting Dashboard  --  M5 (3 stores × 3 categories)",
             fontsize=20, weight="bold")
    ax0.text(0.01, 0.18,
             "Tableau MOCK  --  Filters: cat_id | store_id | model | fold    "
             "Drill: cat -> dept -> item",
             fontsize=11, color="#555")

    # ----- (a) actuals vs predictions, 3 top SKUs --------------------------
    for i, sid in enumerate(top3_ids):
        ax = fig.add_subplot(gs[1, i])
        sub = long[long["id"] == sid].sort_values("date")
        actual = sub[["date", "actual"]].drop_duplicates("date").sort_values("date")
        ax.plot(actual["date"], actual["actual"], "k-", lw=1.5, label="actual")
        for model, color in [("ets",          "#4c72b0"),
                             ("lightgbm",     "#dd8452"),
                             ("lstm_seq2seq", "#55a868"),
                             ("lstm_seq2seq_q80", "#c44e52")]:
            m = sub[sub["model"] == model]
            if len(m):
                ax.plot(m["date"], m["y_pred"], lw=1.1, label=model, alpha=0.85, color=color)
        ax.set_title(sid, fontsize=10)
        ax.set_ylabel("units"); ax.tick_params(axis="x", rotation=30)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2)

    # ----- (b) error heatmap by store x category x model -------------------
    ax_h = fig.add_subplot(gs[2, 0])
    pivot = (long.assign(wmape=lambda d: d["abs_error"] / d["actual"].replace(0, np.nan))
                 .groupby(["model", "store_id"])["abs_error"].sum()
                 .div(long.groupby(["model", "store_id"])["actual"].sum())
                 .unstack("store_id"))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
                cbar_kws={"label": "WMAPE"}, ax=ax_h)
    ax_h.set_title("WMAPE by model x store")

    # ----- (c) leaderboard bar ---------------------------------------------
    ax_lb = fig.add_subplot(gs[2, 1])
    lb = (long.groupby("model")
              .apply(lambda d: d["abs_error"].sum() / d["actual"].sum(),
                     include_groups=False)
              .sort_values()
              .rename("WMAPE")
              .to_frame())
    colors = ["#55a868" if m.startswith("lstm") else "#4c72b0" for m in lb.index]
    ax_lb.barh(lb.index, lb["WMAPE"], color=colors)
    ax_lb.set_xlabel("WMAPE (lower is better)"); ax_lb.invert_yaxis()
    ax_lb.set_title("Model leaderboard")

    # ----- (d) error by category -------------------------------------------
    ax_cat = fig.add_subplot(gs[2, 2])
    cat = (long.groupby(["cat_id", "model"])
               .apply(lambda d: d["abs_error"].sum() / d["actual"].sum(),
                      include_groups=False)
               .unstack("model"))
    cat.plot.bar(ax=ax_cat, width=0.85)
    ax_cat.set_ylabel("WMAPE"); ax_cat.tick_params(axis="x", rotation=0)
    ax_cat.set_title("WMAPE by category"); ax_cat.legend(fontsize=7)

    p = OUT / "mock_tableau_forecasts.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"     wrote {p}")


# ============================================================================
# 4) Mock Power BI dashboard: 4 panels in one PNG
# ============================================================================

def mock_powerbi_ab(summary: pd.DataFrame):
    print("[4/4] rendering mock_powerbi_ab.png ...")
    s = summary.copy()

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           height_ratios=[0.7, 1.4, 1.4],
                           hspace=0.55, wspace=0.35)

    # header
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
    ax0.text(0.01, 0.7, "A/B Experiment Readout  --  Champion (ETS) vs LSTM",
             fontsize=20, weight="bold")
    ax0.text(0.01, 0.18,
             "Power BI MOCK  --  Decision rule: SHIP if cost reduction >= 5% AND "
             "p<0.05 AND stockout-rate change <= 2 pp",
             fontsize=11, color="#555")

    # ---- decision cards (2, one per version) ------------------------------
    for i, version in enumerate(["v1_lstm_mse", "v2_lstm_q80"]):
        ax = fig.add_subplot(gs[1, i])
        ax.axis("off")
        sv = s[s["version"] == version]
        cost  = sv[sv["metric"] == "total_cost_per_series"].iloc[0]
        stk   = sv[sv["metric"] == "stockout_day_rate"].iloc[0]
        cost_ok  = cost["pct_change"] <= -5
        sig_ok   = cost["p_value"] < 0.05
        guard_ok = (stk["diff"] * 100) <= 2.0
        decision = "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD"
        color    = "#55a868" if decision == "SHIP" else "#c44e52"

        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=True,
                                   facecolor=color, alpha=0.10,
                                   edgecolor=color, linewidth=2))
        ax.text(0.5, 0.88, version.replace("_", " "),
                ha="center", fontsize=14, weight="bold")
        ax.text(0.5, 0.55, decision, ha="center", fontsize=46,
                weight="bold", color=color)
        ax.text(0.5, 0.30,
                f"cost: {cost['pct_change']:+.2f}%   "
                f"p={cost['p_value']:.4f}",
                ha="center", fontsize=11)
        ax.text(0.5, 0.18,
                f"stockout-rate: {stk['diff']*100:+.2f} pp",
                ha="center", fontsize=11)

    # ---- "what does each version do" sidebar ------------------------------
    ax_legend = fig.add_subplot(gs[1, 2]); ax_legend.axis("off")
    ax_legend.text(0.02, 0.92, "Variants", fontsize=14, weight="bold")
    ax_legend.text(0.02, 0.74,
                   "v1_lstm_mse:  point forecast + standard safety stock",
                   fontsize=10)
    ax_legend.text(0.02, 0.64,
                   "                       trained on MSE, mean-equivalent",
                   fontsize=9, color="#555")
    ax_legend.text(0.02, 0.42,
                   "v2_lstm_q80:  80th-pctile forecast, no extra safety",
                   fontsize=10)
    ax_legend.text(0.02, 0.32,
                   "                       trained on pinball loss (tau=0.80)",
                   fontsize=9, color="#555")
    ax_legend.text(0.02, 0.10,
                   "Newsvendor-optimal quantile for 5:1 cost ratio = 0.833",
                   fontsize=9, color="#555")

    # ---- lift bars with CI error bars -------------------------------------
    ax_l = fig.add_subplot(gs[2, 0])
    labels, vals, ci_lo, ci_hi, colors = [], [], [], [], []
    for v in ["v1_lstm_mse", "v2_lstm_q80"]:
        cost = s[(s["version"] == v) & (s["metric"] == "total_cost_per_series")].iloc[0]
        labels.append(v.replace("_lstm_", "\n").replace("_", " "))
        vals.append(cost["pct_change"])
        if cost["mean_control"] > 0:
            ci_lo.append(cost["ci95_low"] / cost["mean_control"] * 100)
            ci_hi.append(cost["ci95_high"] / cost["mean_control"] * 100)
        else:
            ci_lo.append(np.nan); ci_hi.append(np.nan)
        colors.append("#55a868" if vals[-1] < 0 else "#c44e52")
    err_lo = [v - lo if not np.isnan(lo) else 0 for v, lo in zip(vals, ci_lo)]
    err_hi = [hi - v if not np.isnan(hi) else 0 for v, hi in zip(vals, ci_hi)]
    ax_l.bar(labels, vals, color=colors, alpha=0.75)
    ax_l.errorbar(labels, vals, yerr=[err_lo, err_hi], fmt="none",
                  color="black", capsize=8, lw=1.4)
    ax_l.axhline(0, color="black", lw=0.6)
    ax_l.set_ylabel("cost % change (treatment vs control)")
    ax_l.set_title("Cost lift  +  95% CI")

    # ---- stockout-rate panel ---------------------------------------------
    ax_s = fig.add_subplot(gs[2, 1])
    so_vals, so_colors = [], []
    for v in ["v1_lstm_mse", "v2_lstm_q80"]:
        stk = s[(s["version"] == v) & (s["metric"] == "stockout_day_rate")].iloc[0]
        so_vals.append(stk["diff"] * 100)   # in pp
        so_colors.append("#55a868" if so_vals[-1] <= 2 else "#c44e52")
    ax_s.bar(labels, so_vals, color=so_colors, alpha=0.75)
    ax_s.axhline(2, color="grey", ls="--", label="guardrail 2 pp")
    ax_s.axhline(0, color="black", lw=0.6)
    ax_s.set_ylabel("stockout-rate change (pp)")
    ax_s.set_title("Stockout-rate change  vs  guardrail")
    ax_s.legend(fontsize=8)

    # ---- summary text box -------------------------------------------------
    ax_t = fig.add_subplot(gs[2, 2]); ax_t.axis("off")
    ax_t.text(0.02, 0.92, "Verdict", fontsize=14, weight="bold")
    ax_t.text(0.02, 0.74,
              "v1 (MSE LSTM): mild cost win (~2.5%), not significant.\n"
              "                       Direction looks right; needs more data.",
              fontsize=10)
    ax_t.text(0.02, 0.40,
              "v2 (q80 LSTM): large cost win (-12.9%), p=0.001\n"
              "                       BUT stockouts +8.9 pp -> service degrades.",
              fontsize=10)
    ax_t.text(0.02, 0.10,
              "Next iteration: try tau=0.65 with normal safety stock.",
              fontsize=10, color="#555", style="italic")

    p = OUT / "mock_powerbi_ab.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"     wrote {p}")


def main():
    long = build_forecasts_csv()
    summary = build_ab_csv()
    mock_tableau_forecasts(long)
    mock_powerbi_ab(summary)
    print(f"\nartifacts in {OUT}/")


if __name__ == "__main__":
    main()
