"""
Phase 5 -- professional dashboard inputs + mock previews, on the LATEST results.

Builds everything from the 900-series run and the Tier-2 outcome (deployed
challenger = LSTM quantile tau=0.90, evaluated with the PAIRED A/B test).

Outputs (reports/phase5_dashboards/):
  forecasts_long.csv        Tableau: actuals + every model's forecast (incl. the
                            deployed tau=0.90 challenger), tidy long, with hierarchy
  ab_paired_readout.csv     Power BI: per-(SKU, fold) champion vs challenger cost +
                            stockout-rate, with segments (category/store/volume tier)
  ab_summary.csv            Power BI: headline cards (paired cost %, Wilcoxon p, CI,
                            stockout pp, % SKUs cheaper, decision) for fold_3 + all
  mock_tableau_forecasts.png   polished static preview of the forecasting dashboard
  mock_powerbi_ab.png          polished static preview of the A/B readout

Run: python -m scripts.phase5_dashboards
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from src.backtest import rolling_origin_splits
from src.experiment import (
    SERVICE_Z_DEFAULT,
    compute_safety_stock,
    paired_diff_test,
    simulate_costs,
)

# ---- paths ------------------------------------------------------------------
DATA = Path("data/processed/m5_long_sample.parquet")
P2 = Path("reports/phase2_baselines/predictions.parquet")          # sn / ets / lgbm
P3_MSE = Path("reports/phase3_lstm_seq2seq/seq2seq_predictions.parquet")
Q90 = Path("reports/tier2_quantile_sweep/preds_q90.parquet")        # deployed challenger
OUT = Path("reports/phase5_dashboards")
OUT.mkdir(parents=True, exist_ok=True)

FOLDS = ["fold_1", "fold_2", "fold_3"]
CONFIRMATORY_FOLD = "fold_3"
COST_RULE_PCT = 5.0
ALPHA = 0.05
STOCKOUT_GUARDRAIL_PP = 2.0

# ---- a clean, consistent visual identity for the mocks ----------------------
INK = "#1f2937"; MUTE = "#6b7280"; GRID = "#e5e7eb"; PANEL = "#f8fafc"
CHAMP = "#5b7fa6"        # champion (ETS) = muted blue
CHALL = "#2a9d8f"        # challenger (LSTM q90) = teal
GOOD = "#2e7d52"; BAD = "#c0392b"; AMBER = "#b8860b"
PALETTE = {"seasonal_naive": "#b0b7c3", "ets": CHAMP, "lightgbm": "#dd8452",
           "lstm_seq2seq": "#8172b2", "lstm_q90": CHALL}
MODEL_LABEL = {"seasonal_naive": "Seasonal-naive", "ets": "ETS (champion)",
               "lightgbm": "LightGBM", "lstm_seq2seq": "LSTM (accuracy)",
               "lstm_q90": "LSTM q90 (deployed)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "figure.facecolor": "white",
})


# ============================================================================
# load + assemble
# ============================================================================
def load_all():
    actuals = pd.read_parquet(DATA)
    actuals["date"] = pd.to_datetime(actuals["date"])
    actuals = actuals[actuals["is_active"]].copy()

    def _load(path, model_name=None, model_filter=None):
        d = pd.read_parquet(path)
        if model_filter:
            d = d[d["model"] == model_filter]
        d = d[["id", "date", "y_pred", "fold"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        d["model"] = model_name
        return d

    preds = pd.concat([
        _load(P2, "seasonal_naive", "seasonal_naive"),
        _load(P2, "ets", "ets"),
        _load(P2, "lightgbm", "lightgbm"),
        _load(P3_MSE, "lstm_seq2seq"),
        _load(Q90, "lstm_q90"),
    ], ignore_index=True)
    return actuals, preds


# ============================================================================
# 1) Tableau CSV
# ============================================================================
def build_forecasts_csv(actuals, preds):
    meta = actuals[["id", "item_id", "dept_id", "cat_id", "store_id",
                    "state_id", "date", "sales"]]
    long = preds.merge(meta, on=["id", "date"], how="left").rename(columns={"sales": "actual"})
    long["error"] = long["y_pred"] - long["actual"]
    long["abs_error"] = long["error"].abs()
    long = long[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                 "date", "fold", "model", "y_pred", "actual", "error", "abs_error"]]
    long.to_csv(OUT / "forecasts_long.csv", index=False)
    print(f"  wrote forecasts_long.csv  ({len(long):,} rows, {long['model'].nunique()} models)")
    return long


# ============================================================================
# 2) Power BI paired readout + 3) summary
# ============================================================================
def _per_id_fold(preds_one, actuals, safety, label):
    out = []
    for fold in FOLDS:
        p = preds_one[preds_one["fold"] == fold][["id", "date", "y_pred"]]
        sim = simulate_costs(p, actuals, safety)
        g = sim.groupby("id")
        f = pd.DataFrame({f"{label}_cost": g["cost"].sum(),
                          f"{label}_stockout_rate": g["stockouts"].apply(lambda s: float((s > 0).mean()))})
        f["fold"] = fold
        out.append(f.reset_index())
    return pd.concat(out, ignore_index=True)


def build_ab_readout(actuals, preds):
    folds = rolling_origin_splits(actuals["date"].max(), n_folds=3, horizon=28, step=28)
    safety = compute_safety_stock(actuals, folds[0].train_end, z=SERVICE_Z_DEFAULT)
    zero = pd.Series(0, index=safety.index, name="safety_stock")

    champ = _per_id_fold(preds[preds["model"] == "ets"], actuals, safety, "champion")
    chall = _per_id_fold(preds[preds["model"] == "lstm_q90"], actuals, zero, "challenger")

    seg = (actuals[actuals["date"] <= folds[0].train_end]
           .groupby("id").agg(sales_sum=("sales", "sum"),
                              cat_id=("cat_id", "first"),
                              store_id=("store_id", "first")).reset_index())
    seg["volume_tertile"] = pd.qcut(seg["sales_sum"], 3, labels=["low", "mid", "high"])

    readout = (champ.merge(chall, on=["id", "fold"])
                    .merge(seg[["id", "cat_id", "store_id", "volume_tertile"]], on="id"))
    readout["cost_diff"] = readout["challenger_cost"] - readout["champion_cost"]
    readout["challenger_cheaper"] = (readout["cost_diff"] < 0).astype(int)
    readout["is_confirmatory_fold"] = (readout["fold"] == CONFIRMATORY_FOLD).astype(int)
    readout.to_csv(OUT / "ab_paired_readout.csv", index=False)
    print(f"  wrote ab_paired_readout.csv  ({len(readout):,} rows)")

    # ---- summary (paired) for fold_3 (confirmatory) and all folds ----------
    rows = []
    for scope, sub in [("fold_3 (confirmatory)", readout[readout["fold"] == CONFIRMATORY_FOLD]),
                       ("all folds (context)", readout.groupby("id").agg(
                           champion_cost=("champion_cost", "sum"),
                           challenger_cost=("challenger_cost", "sum"),
                           champion_stockout_rate=("champion_stockout_rate", "mean"),
                           challenger_stockout_rate=("challenger_stockout_rate", "mean")).reset_index())]:
        res = paired_diff_test(sub["champion_cost"].to_numpy(), sub["challenger_cost"].to_numpy())
        so_c = sub["champion_stockout_rate"].mean() * 100
        so_t = sub["challenger_stockout_rate"].mean() * 100
        cost_ok = res.pct_change <= -COST_RULE_PCT
        sig_ok = res.wilcoxon_p < ALPHA
        guard_ok = (so_t - so_c) <= STOCKOUT_GUARDRAIL_PP
        rows.append({
            "scope": scope, "n_skus": res.n,
            "champion_mean_cost": round(res.mean_a, 2),
            "challenger_mean_cost": round(res.mean_b, 2),
            "cost_pct_change": round(res.pct_change, 2),
            "wilcoxon_p": res.wilcoxon_p,
            "ci95_low": round(res.ci95_low, 2), "ci95_high": round(res.ci95_high, 2),
            "pct_skus_cheaper": round(res.pct_units_b_better, 1),
            "stockout_champion_pct": round(so_c, 2),
            "stockout_challenger_pct": round(so_t, 2),
            "stockout_change_pp": round(so_t - so_c, 2),
            "decision": "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "ab_summary.csv", index=False)
    print(f"  wrote ab_summary.csv")
    print(summary[["scope", "cost_pct_change", "wilcoxon_p", "stockout_change_pp", "decision"]]
          .to_string(index=False))
    return readout, summary


# ============================================================================
# mock dashboards (professional static previews)
# ============================================================================
def _card(ax, x, y, w, h, fc=PANEL, ec=GRID):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.2, transform=ax.transAxes, clip_on=False))


def _kpi(ax, x, w, title, value, sub, color=INK):
    _card(ax, x, 0.0, w, 1.0)
    ax.text(x + w / 2, 0.74, title, ha="center", va="center", fontsize=8.5,
            color=MUTE, transform=ax.transAxes)
    ax.text(x + w / 2, 0.44, value, ha="center", va="center", fontsize=20,
            weight="bold", color=color, transform=ax.transAxes)
    ax.text(x + w / 2, 0.17, sub, ha="center", va="center", fontsize=7.5,
            color=MUTE, transform=ax.transAxes)


def mock_tableau(long):
    top3 = long.groupby("id")["actual"].sum().sort_values(ascending=False).head(3).index.tolist()
    fig = plt.figure(figsize=(16, 9.5), facecolor="white")
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[0.5, 1.5, 1.25],
                           hspace=0.5, wspace=0.28, left=0.05, right=0.97, top=0.94, bottom=0.07)

    head = fig.add_subplot(gs[0, :]); head.axis("off")
    head.text(0, 0.62, "Demand Forecasting  —  Accuracy Dashboard", fontsize=20, weight="bold")
    head.text(0, 0.12, "M5 · 3 stores × 3 categories · 900 SKUs · rolling-origin backtest (3×28-day folds)   |   "
                       "Tableau preview · filters: category · store · model · fold",
              fontsize=9.5, color=MUTE)

    for i, sid in enumerate(top3):
        ax = fig.add_subplot(gs[1, i])
        sub = long[long["id"] == sid].sort_values("date")
        a = sub[["date", "actual"]].drop_duplicates("date")
        ax.fill_between(a["date"], a["actual"], color="#cbd5e1", alpha=0.5, label="actual")
        for m in ["ets", "lightgbm", "lstm_q90"]:
            mm = sub[sub["model"] == m]
            if len(mm):
                ax.plot(mm["date"], mm["y_pred"], lw=1.6, color=PALETTE[m], label=MODEL_LABEL[m])
        ax.set_title(sid.replace("_evaluation", ""), fontsize=9.5, color=INK)
        ax.tick_params(axis="x", rotation=25, labelsize=7); ax.tick_params(axis="y", labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
            ax.set_ylabel("units / day", fontsize=8)

    # leaderboard
    axlb = fig.add_subplot(gs[2, 0])
    wm = (long.groupby("model").apply(lambda d: d["abs_error"].sum() / d["actual"].sum(),
                                      include_groups=False).sort_values())
    axlb.barh([MODEL_LABEL[m] for m in wm.index], wm.values,
              color=[PALETTE[m] for m in wm.index])
    axlb.invert_yaxis(); axlb.set_xlabel("WMAPE", fontsize=8); axlb.tick_params(labelsize=7.5)
    axlb.set_title("Model leaderboard (lower = better)", fontsize=9.5)

    # WMAPE by category
    axc = fig.add_subplot(gs[2, 1])
    cat = (long.groupby(["cat_id", "model"]).apply(
        lambda d: d["abs_error"].sum() / d["actual"].sum(), include_groups=False).unstack("model"))
    cat = cat[["ets", "lightgbm", "lstm_q90"]]
    cat.plot.bar(ax=axc, color=[PALETTE[c] for c in cat.columns], width=0.8, legend=False)
    axc.set_title("WMAPE by category", fontsize=9.5); axc.set_xlabel("")
    axc.tick_params(axis="x", rotation=0, labelsize=8); axc.tick_params(axis="y", labelsize=7.5)

    # WMAPE by store heatmap
    axh = fig.add_subplot(gs[2, 2])
    piv = (long.groupby(["model", "store_id"]).apply(
        lambda d: d["abs_error"].sum() / d["actual"].sum(), include_groups=False).unstack("store_id"))
    im = axh.imshow(piv.values, cmap="RdYlGn_r", aspect="auto", vmin=0.7, vmax=1.0)
    axh.set_xticks(range(len(piv.columns))); axh.set_xticklabels(piv.columns, fontsize=8)
    axh.set_yticks(range(len(piv.index))); axh.set_yticklabels([MODEL_LABEL[m] for m in piv.index], fontsize=7.5)
    for (r, c), v in np.ndenumerate(piv.values):
        axh.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    axh.set_title("WMAPE by store × model", fontsize=9.5); axh.grid(False)

    fig.savefig(OUT / "mock_tableau_forecasts.png", dpi=110, bbox_inches="tight")
    plt.close(fig); print("  wrote mock_tableau_forecasts.png")


def mock_powerbi(readout, summary):
    conf = summary[summary["scope"].str.startswith("fold_3")].iloc[0]
    decision = conf["decision"]; dcol = GOOD if decision == "SHIP" else BAD
    d3 = readout[readout["fold"] == CONFIRMATORY_FOLD]
    diff = d3["cost_diff"].to_numpy()

    fig = plt.figure(figsize=(16, 9.5), facecolor="white")
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[0.95, 1.25, 1.15],
                           hspace=0.55, wspace=0.35, left=0.05, right=0.97, top=0.93, bottom=0.07)

    head = fig.add_subplot(gs[0, :]); head.axis("off")
    head.text(0, 0.78, "Champion–Challenger A/B  —  Experiment Readout", fontsize=20, weight="bold")
    head.text(0, 0.42, "Champion: ETS + safety stock   vs   Challenger: LSTM quantile τ=0.90   |   "
                       "paired counterfactual test · held-out fold_3 · n=%d SKUs" % int(conf["n_skus"]),
              fontsize=9.5, color=MUTE)
    head.text(0, 0.08, "Decision rule (pre-registered): SHIP iff cost ≤ −5%  AND  p < 0.05  AND  "
                       "stockout change ≤ +2.0 pp", fontsize=8.5, color=MUTE, style="italic")

    # KPI strip
    kp = fig.add_subplot(gs[1, :]); kp.axis("off")
    cost_col = GOOD if conf["cost_pct_change"] <= -COST_RULE_PCT else BAD
    so_col = GOOD if conf["stockout_change_pp"] <= STOCKOUT_GUARDRAIL_PP else BAD
    _kpi(kp, 0.00, 0.185, "COST CHANGE", f"{conf['cost_pct_change']:+.1f}%",
         f"95% CI [{conf['ci95_low']:+.0f}, {conf['ci95_high']:+.0f}]", cost_col)
    _kpi(kp, 0.205, 0.185, "SIGNIFICANCE", f"p={conf['wilcoxon_p']:.0e}",
         "Wilcoxon signed-rank", GOOD if conf["wilcoxon_p"] < ALPHA else BAD)
    _kpi(kp, 0.41, 0.185, "STOCKOUT Δ", f"{conf['stockout_change_pp']:+.2f} pp",
         f"guardrail ≤ {STOCKOUT_GUARDRAIL_PP:.1f} pp", so_col)
    _kpi(kp, 0.615, 0.185, "SKUs CHEAPER", f"{conf['pct_skus_cheaper']:.0f}%",
         "challenger < champion", INK)
    _kpi(kp, 0.82, 0.18, "DECISION", decision, "held-out verdict", dcol)

    # cost-difference distribution
    ax1 = fig.add_subplot(gs[2, 0:2])
    lo, hi = np.percentile(diff, [1, 99]); dc = diff[(diff >= lo) & (diff <= hi)]
    ax1.hist(dc, bins=55, color=CHALL, alpha=0.85)
    ax1.axvline(0, color=INK, lw=1)
    ax1.axvline(diff.mean(), color=BAD, lw=2, label=f"mean {diff.mean():+.1f}")
    ax1.axvspan(conf["ci95_low"], conf["ci95_high"], color=BAD, alpha=0.12,
                label=f"95% CI [{conf['ci95_low']:+.0f},{conf['ci95_high']:+.0f}]")
    ax1.set_title("Per-SKU cost difference  (challenger − champion; < 0 = cheaper)", fontsize=10)
    ax1.set_xlabel("cost difference"); ax1.set_ylabel("# SKUs"); ax1.legend(fontsize=8)

    # cost by category (champion vs challenger means)
    ax2 = fig.add_subplot(gs[2, 2])
    by = d3.groupby("cat_id")[["champion_cost", "challenger_cost"]].mean()
    by.plot.bar(ax=ax2, color=[CHAMP, CHALL], width=0.8, legend=True)
    ax2.set_title("Mean cost / SKU by category", fontsize=10); ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=0, labelsize=8)
    ax2.legend(["Champion", "Challenger"], fontsize=7.5)

    # stockout vs guardrail
    ax3 = fig.add_subplot(gs[2, 3])
    ax3.bar(["Champion", "Challenger"], [conf["stockout_champion_pct"], conf["stockout_challenger_pct"]],
            color=[CHAMP, CHALL], width=0.6)
    ax3.axhline(conf["stockout_champion_pct"] + STOCKOUT_GUARDRAIL_PP, color=BAD, ls="--", lw=1.3,
                label=f"guardrail (+{STOCKOUT_GUARDRAIL_PP:.0f}pp)")
    ax3.set_title("Stockout day-rate (%)", fontsize=10); ax3.legend(fontsize=7.5)
    ax3.tick_params(axis="x", labelsize=8)

    fig.text(0.05, 0.015,
             "Verdict: challenger cuts cost ~%.0f%% (significant, CI excludes 0) but breaches the "
             "service guardrail by %.2f pp → HOLD-and-retune (τ≈0.87)." %
             (-conf["cost_pct_change"], conf["stockout_change_pp"] - STOCKOUT_GUARDRAIL_PP),
             fontsize=8.5, color=MUTE, style="italic")
    fig.savefig(OUT / "mock_powerbi_ab.png", dpi=110, bbox_inches="tight")
    plt.close(fig); print("  wrote mock_powerbi_ab.png")


def main():
    print("loading latest results (900 series, tau=0.90 deployed challenger)...")
    actuals, preds = load_all()
    long = build_forecasts_csv(actuals, preds)
    readout, summary = build_ab_readout(actuals, preds)
    mock_tableau(long)
    mock_powerbi(readout, summary)
    print(f"\n  artifacts in {OUT}/")


if __name__ == "__main__":
    main()
