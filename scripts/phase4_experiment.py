"""
Phase 4 orchestrator: champion vs challenger A/B test, end-to-end.

Champion  = LightGBM   (predictions from reports/phase2_baselines/)
Challenger = LSTM seq2seq (predictions from reports/phase3_lstm_seq2seq/)

Output: reports/phase4_experiment/ with results CSV (for Power BI) and 4 plots.

Run: python -m scripts.phase4_experiment
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest import rolling_origin_splits
from src.experiment import (
    DECISION_ALPHA,
    DECISION_MAX_WMAPE_DEGRADATION_PCT,
    DECISION_MIN_COST_REDUCTION_PCT,
    H_HOLDING_DEFAULT,
    H_STOCKOUT_DEFAULT,
    SERVICE_Z_DEFAULT,
    analyze_cost,
    compute_safety_stock,
    minimum_detectable_effect,
    recommend,
    simulate_costs,
    stratified_assignment,
    two_proportion_z_test,
)
from src.metrics import wmape

DATA = Path("data/processed/m5_long_sample.parquet")
CHAMP_PREDS = Path("reports/phase2_baselines/predictions.parquet")
CHALL_PREDS = Path("reports/phase3_lstm_seq2seq/seq2seq_predictions.parquet")
OUT = Path("reports/phase4_experiment")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    p = OUT / f"{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def main():
    # ---- 1) load everything ------------------------------------------------
    print("loading actuals + predictions...")
    actuals = pd.read_parquet(DATA)
    actuals["date"] = pd.to_datetime(actuals["date"])
    actuals = actuals[actuals["is_active"]].copy()

    champ = pd.read_parquet(CHAMP_PREDS)
    champ = champ[champ["model"] == "ets"][["id", "date", "y_pred", "fold"]]   # 20x champion is ETS
    champ["date"] = pd.to_datetime(champ["date"])
    chall = pd.read_parquet(CHALL_PREDS)[["id", "date", "y_pred", "fold"]]
    chall["date"] = pd.to_datetime(chall["date"])

    folds = rolling_origin_splits(actuals["date"].max(), n_folds=3, horizon=28, step=28)
    train_cutoff = folds[0].train_end          # everything before first test fold

    # ---- 2) safety stock & stratification info (computed on TRAIN ONLY) ----
    safety = compute_safety_stock(actuals, train_cutoff, z=SERVICE_Z_DEFAULT)
    print(f"  safety stock: z={SERVICE_Z_DEFAULT} (95% svc), median ss = {safety.median():.0f}")

    units = (actuals[actuals["date"] <= train_cutoff]
             .groupby("id", as_index=False)
             .agg(sales_sum=("sales", "sum"),
                  cat_id=("cat_id", "first")))
    # 3-tertile bucket so high/mid/low-volume series are balanced across arms
    units["volume_tertile"] = pd.qcut(units["sales_sum"], q=3,
                                      labels=["low", "mid", "high"])

    # ---- 3) stratified assignment ------------------------------------------
    assignment = stratified_assignment(units, "volume_tertile", seed=42)
    assignment = assignment.merge(units[["id", "cat_id", "volume_tertile", "sales_sum"]],
                                  on="id", how="left")
    n_arm = assignment["arm"].value_counts()
    print(f"  assignment: control={n_arm.get('control', 0)}  "
          f"treatment={n_arm.get('treatment', 0)}")
    print("  stratum balance:")
    print(assignment.groupby(["volume_tertile", "arm"], observed=True)
                    .size().unstack(fill_value=0).to_string())
    assignment.to_csv(OUT / "assignment.csv", index=False)

    # ---- 4) simulate costs for BOTH models (every series gets both) --------
    print(f"  cost ratio stockout:holding = {H_STOCKOUT_DEFAULT}:{H_HOLDING_DEFAULT}")
    champ_sim = simulate_costs(champ, actuals, safety).merge(assignment[["id", "arm"]], on="id")
    chall_sim = simulate_costs(chall, actuals, safety).merge(assignment[["id", "arm"]], on="id")

    # apply the assignment: control sees champion's outcomes, treatment sees challenger's
    control_rows   = champ_sim[champ_sim["arm"] == "control"]
    treatment_rows = chall_sim[chall_sim["arm"] == "treatment"]

    # ---- 5) per-series outcomes (one number per series) --------------------
    control_costs   = control_rows.groupby("id")["cost"].sum()
    treatment_costs = treatment_rows.groupby("id")["cost"].sum()
    print(f"\n  per-series cost: control mean={control_costs.mean():.1f}  "
          f"treatment mean={treatment_costs.mean():.1f}")

    # ---- 6) power analysis (retrospective MDE at our actual n & sigma) -----
    sigma_pool = float(np.concatenate([control_costs, treatment_costs]).std(ddof=1))
    mde = minimum_detectable_effect(n_per_arm=min(len(control_costs), len(treatment_costs)),
                                    sigma=sigma_pool, alpha=0.05, power=0.8)
    mde_pct = mde / control_costs.mean() * 100
    print(f"  power: at n_per_arm~{min(len(control_costs), len(treatment_costs))}, "
          f"sigma={sigma_pool:.1f} -> MDE={mde:.1f} ({mde_pct:.1f}% of control mean)")

    # ---- 7) statistical tests ----------------------------------------------
    cost = analyze_cost(control_costs.to_numpy(), treatment_costs.to_numpy())
    print(f"\n  COST TEST  ({cost.test_used}, normality_ok={cost.normality_ok})")
    print(f"    mean_control   = {cost.mean_control:.2f}")
    print(f"    mean_treatment = {cost.mean_treatment:.2f}")
    print(f"    diff           = {cost.diff:+.2f}  ({cost.pct_change:+.2f}%)")
    print(f"    95% CI of diff = [{cost.ci95_low:+.2f}, {cost.ci95_high:+.2f}]")
    print(f"    Cohen's d      = {cost.cohens_d:+.3f}")
    print(f"    p-value        = {cost.p_value:.4f}")

    # stockout day-rate
    so_c = float((control_rows["stockouts"] > 0).mean())
    so_t = float((treatment_rows["stockouts"] > 0).mean())
    so_test = two_proportion_z_test(so_c, len(control_rows), so_t, len(treatment_rows))
    print(f"\n  STOCKOUT-RATE TEST (two-proportion z)")
    print(f"    p_control      = {so_c*100:.2f}%   "
          f"p_treatment    = {so_t*100:.2f}%   "
          f"lift           = {so_test['lift_pp']:+.2f} pp   "
          f"p-value        = {so_test['p_value']:.4f}")

    # ---- 8) guardrail: WMAPE in each arm -----------------------------------
    def arm_wmape(rows: pd.DataFrame) -> float:
        y  = rows["sales"].to_numpy(dtype=float)
        yh = rows["y_pred"].to_numpy(dtype=float)
        return float(wmape(y, yh))
    wmape_c = arm_wmape(control_rows)
    wmape_t = arm_wmape(treatment_rows)
    wmape_change_pct = (wmape_t / wmape_c - 1) * 100
    print(f"\n  GUARDRAIL: WMAPE control={wmape_c:.3f}  treatment={wmape_t:.3f}  "
          f"({wmape_change_pct:+.2f}%)")

    # ---- 9) decision -------------------------------------------------------
    decision_rule = (f"SHIP if cost_reduction >= {DECISION_MIN_COST_REDUCTION_PCT:.0f}% "
                     f"AND p < {DECISION_ALPHA} AND wmape_degradation <= "
                     f"{DECISION_MAX_WMAPE_DEGRADATION_PCT:.0f}%")
    decision, why = recommend(cost, wmape_change_pct)
    print(f"\n  DECISION RULE: {decision_rule}")
    print(f"  >>> {decision}  ({why})")

    # ---- 10) persist results for Power BI ----------------------------------
    summary = pd.DataFrame([{
        "metric": "total_cost_per_series",
        "mean_control": cost.mean_control, "mean_treatment": cost.mean_treatment,
        "diff": cost.diff, "pct_change": cost.pct_change,
        "ci95_low": cost.ci95_low, "ci95_high": cost.ci95_high,
        "cohens_d": cost.cohens_d, "p_value": cost.p_value, "test": cost.test_used,
    }, {
        "metric": "stockout_day_rate",
        "mean_control": so_c, "mean_treatment": so_t,
        "diff": so_t - so_c, "pct_change": (so_t / so_c - 1) * 100 if so_c else float("nan"),
        "ci95_low": float("nan"), "ci95_high": float("nan"),
        "cohens_d": float("nan"), "p_value": so_test["p_value"], "test": "two_prop_z",
    }, {
        "metric": "wmape_guardrail",
        "mean_control": wmape_c, "mean_treatment": wmape_t,
        "diff": wmape_t - wmape_c, "pct_change": wmape_change_pct,
        "ci95_low": float("nan"), "ci95_high": float("nan"),
        "cohens_d": float("nan"), "p_value": float("nan"), "test": "n/a",
    }])
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame({"id": control_costs.index, "arm": "control",
                  "total_cost": control_costs.values}).to_csv(OUT / "per_series_control.csv", index=False)
    pd.DataFrame({"id": treatment_costs.index, "arm": "treatment",
                  "total_cost": treatment_costs.values}).to_csv(OUT / "per_series_treatment.csv", index=False)

    # Power BI-friendly long table of every series-day outcome
    cols = ["id", "date", "fold", "arm", "y_pred", "sales",
            "safety_stock", "order", "stockouts", "holding", "cost"]
    long_table = pd.concat([
        control_rows.assign(model="champion_lightgbm")[cols + ["model"]],
        treatment_rows.assign(model="challenger_lstm_seq2seq")[cols + ["model"]],
    ])
    long_table.to_csv(OUT / "experiment_long_table.csv", index=False)

    # ---- 11) plots ---------------------------------------------------------
    plot_assignment_balance(assignment)
    plot_cost_distribution(control_costs, treatment_costs)
    plot_stockout_rate_by_category(control_rows, treatment_rows)
    plot_decision_summary(cost, so_test, wmape_change_pct, decision, mde_pct)
    print(f"\n  artifacts in {OUT}/")


# ---- plots ------------------------------------------------------------------

def plot_assignment_balance(assignment: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.countplot(data=assignment, x="volume_tertile", hue="arm",
                  order=["low", "mid", "high"], ax=axes[0])
    axes[0].set_title("Assignment balance by volume tertile")
    axes[0].set_xlabel("training-period volume tertile"); axes[0].set_ylabel("# series")

    pivot = (assignment.groupby(["cat_id", "arm"]).size()
             .unstack(fill_value=0))
    pivot.plot.bar(stacked=False, ax=axes[1])
    axes[1].set_title("Assignment by category"); axes[1].set_ylabel("# series")
    _save(fig, "01_assignment_balance")


def plot_cost_distribution(control: pd.Series, treatment: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    df = pd.concat([
        pd.DataFrame({"arm": "control",   "total_cost": control.values}),
        pd.DataFrame({"arm": "treatment", "total_cost": treatment.values}),
    ])
    sns.boxplot(data=df, x="arm", y="total_cost", showfliers=False, ax=axes[0])
    axes[0].set_title("Per-series total cost by arm (84-day test horizon)")

    axes[1].hist(control.values, bins=30, alpha=0.6, label="control")
    axes[1].hist(treatment.values, bins=30, alpha=0.6, label="treatment")
    axes[1].axvline(control.mean(), color="C0", ls="--")
    axes[1].axvline(treatment.mean(), color="C1", ls="--")
    axes[1].set_title("Per-series total cost histogram"); axes[1].legend()
    axes[1].set_xlabel("total cost")
    _save(fig, "02_cost_distribution")


def plot_stockout_rate_by_category(c_rows: pd.DataFrame, t_rows: pd.DataFrame):
    def cat_rate(rows, label):
        out = (rows.assign(stk=lambda d: (d["stockouts"] > 0).astype(int))
                   .merge(rows[["id"]].drop_duplicates(), on="id"))
        return (rows.assign(stk=lambda d: (d["stockouts"] > 0).astype(int))
                    .groupby("id")["stk"].mean()
                    .to_frame("stockout_rate").assign(arm=label))
    overall = pd.concat([
        c_rows.assign(stk=lambda d: (d["stockouts"] > 0).astype(int))
              .merge(c_rows[["id"]].drop_duplicates(), on="id")
              .groupby("id")["stk"].mean().to_frame("stockout_rate").assign(arm="control"),
        t_rows.assign(stk=lambda d: (d["stockouts"] > 0).astype(int))
              .merge(t_rows[["id"]].drop_duplicates(), on="id")
              .groupby("id")["stk"].mean().to_frame("stockout_rate").assign(arm="treatment"),
    ]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=overall, x="arm", y="stockout_rate", showfliers=False, ax=ax)
    ax.set_ylabel("per-series stockout day-rate")
    ax.set_title("Stockout day-rate by arm (lower = fewer stockouts)")
    _save(fig, "03_stockout_rate")


def plot_decision_summary(cost, so_test, wmape_change_pct, decision, mde_pct):
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["cost\n(% change)", "stockout-rate\n(pp change)", "WMAPE\n(% change)"]
    vals = [cost.pct_change, so_test["lift_pp"], wmape_change_pct]
    ci_low  = [cost.ci95_low  / cost.mean_control * 100, np.nan, np.nan]
    ci_high = [cost.ci95_high / cost.mean_control * 100, np.nan, np.nan]
    err_low  = [v - lo if not np.isnan(lo) else 0 for v, lo in zip(vals, ci_low)]
    err_high = [hi - v if not np.isnan(hi) else 0 for v, hi in zip(vals, ci_high)]
    colors = ["green" if v < 0 else "red" for v in vals]   # negative = good for cost & stockouts
    ax.bar(metrics, vals, color=colors, alpha=0.7)
    ax.errorbar(metrics, vals,
                yerr=[err_low, err_high], fmt="none", color="black", capsize=6, lw=1.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("treatment vs control")
    sig_marker = "*" if cost.p_value < DECISION_ALPHA else "n.s."
    ax.set_title(f"A/B summary -- decision: {decision}    "
                 f"(cost p={cost.p_value:.3f} {sig_marker}, "
                 f"MDE ~ {mde_pct:.1f}% of control mean)")
    _save(fig, "04_decision_summary")


if __name__ == "__main__":
    main()
