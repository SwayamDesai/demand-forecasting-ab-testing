"""
Phase 4: champion vs challenger A/B test (one parametrized script for both variants).

Champion (control) = ETS (the Phase 2 leaderboard winner at this scale).
Challenger (treatment) = an LSTM variant, selected with --challenger:

  --challenger mse   LSTM seq2seq (MSE-trained), order = forecast + safety stock,
                     guardrail = WMAPE degradation.   -> reports/phase4_experiment/
  --challenger q80   LSTM seq2seq (pinball tau=0.80), order = forecast (the quantile
                     IS the buffer, no extra safety stock), guardrail = stockout-rate.
                                                       -> reports/phase4_experiment_v2/

Run both:
  python -m scripts.phase4_experiment --challenger mse
  python -m scripts.phase4_experiment --challenger q80
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest import rolling_origin_splits
from src.experiment import (
    DECISION_ALPHA,
    DECISION_MIN_COST_REDUCTION_PCT,
    H_HOLDING_DEFAULT,
    H_STOCKOUT_DEFAULT,
    SERVICE_Z_DEFAULT,
    analyze_cost,
    compute_safety_stock,
    minimum_detectable_effect,
    simulate_costs,
    stratified_assignment,
    two_proportion_z_test,
)
from src.metrics import wmape

DATA = Path("data/processed/m5_long_sample.parquet")
CHAMP_PREDS = Path("reports/phase2_baselines/predictions.parquet")
CHAMPION_MODEL = "ets"                      # Phase 2 leaderboard winner
STOCKOUT_GUARDRAIL_MAX_PP = 2.0             # service may not worsen by more than this
WMAPE_GUARDRAIL_MAX_PCT = 5.0

sns.set_theme(style="whitegrid", context="notebook")


@dataclass(frozen=True)
class Variant:
    name: str
    preds: Path
    challenger_label: str
    challenger_gets_safety_stock: bool      # quantile model already bakes in the buffer
    guardrail: str                          # "wmape" or "stockout"
    out: Path


VARIANTS = {
    "mse": Variant(
        name="v1_lstm_mse",
        preds=Path("reports/phase3_lstm_seq2seq/seq2seq_predictions.parquet"),
        challenger_label="challenger_lstm_seq2seq",
        challenger_gets_safety_stock=True,
        guardrail="wmape",
        out=Path("reports/phase4_experiment"),
    ),
    "q80": Variant(
        name="v2_lstm_q80",
        preds=Path("reports/phase3_lstm_quantile/q80_predictions.parquet"),
        challenger_label="challenger_lstm_q80",
        challenger_gets_safety_stock=False,
        guardrail="stockout",
        out=Path("reports/phase4_experiment_v2"),
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", choices=list(VARIANTS), default="mse")
    v = VARIANTS[ap.parse_args().challenger]
    v.out.mkdir(parents=True, exist_ok=True)
    print(f"=== A/B: champion={CHAMPION_MODEL}  vs  challenger={v.name} "
          f"(guardrail={v.guardrail}) ===")

    # ---- 1) load --------------------------------------------------------
    actuals = pd.read_parquet(DATA)
    actuals["date"] = pd.to_datetime(actuals["date"])
    actuals = actuals[actuals["is_active"]].copy()

    champ = pd.read_parquet(CHAMP_PREDS)
    champ = champ[champ["model"] == CHAMPION_MODEL][["id", "date", "y_pred", "fold"]]
    champ["date"] = pd.to_datetime(champ["date"])
    chall = pd.read_parquet(v.preds)[["id", "date", "y_pred", "fold"]]
    chall["date"] = pd.to_datetime(chall["date"])

    folds = rolling_origin_splits(actuals["date"].max(), n_folds=3, horizon=28, step=28)
    train_cutoff = folds[0].train_end

    # ---- 2) safety stock (TRAIN-only) + per-arm order policy ------------
    safety_champ = compute_safety_stock(actuals, train_cutoff, z=SERVICE_Z_DEFAULT)
    if v.challenger_gets_safety_stock:
        safety_chall = safety_champ
        print(f"  order policy: both arms = forecast + safety stock (z={SERVICE_Z_DEFAULT})")
    else:
        safety_chall = pd.Series(0, index=safety_champ.index, name="safety_stock")
        print(f"  order policy: champion = forecast + safety stock (z={SERVICE_Z_DEFAULT}); "
              f"challenger = forecast only (quantile is the buffer)")

    # ---- 3) stratified assignment (balanced by volume tertile) ---------
    units = (actuals[actuals["date"] <= train_cutoff]
             .groupby("id", as_index=False)
             .agg(sales_sum=("sales", "sum"), cat_id=("cat_id", "first")))
    units["volume_tertile"] = pd.qcut(units["sales_sum"], q=3, labels=["low", "mid", "high"])
    assignment = stratified_assignment(units, "volume_tertile", seed=42)
    assignment = assignment.merge(units[["id", "cat_id", "volume_tertile", "sales_sum"]],
                                  on="id", how="left")
    n_arm = assignment["arm"].value_counts()
    print(f"  assignment: control={n_arm.get('control', 0)}  treatment={n_arm.get('treatment', 0)}")
    assignment.to_csv(v.out / "assignment.csv", index=False)

    # ---- 4) simulate costs, apply assignment ---------------------------
    print(f"  cost ratio stockout:holding = {H_STOCKOUT_DEFAULT}:{H_HOLDING_DEFAULT}")
    champ_sim = simulate_costs(champ, actuals, safety_champ).merge(assignment[["id", "arm"]], on="id")
    chall_sim = simulate_costs(chall, actuals, safety_chall).merge(assignment[["id", "arm"]], on="id")
    control_rows   = champ_sim[champ_sim["arm"] == "control"]
    treatment_rows = chall_sim[chall_sim["arm"] == "treatment"]

    control_costs   = control_rows.groupby("id")["cost"].sum()
    treatment_costs = treatment_rows.groupby("id")["cost"].sum()
    print(f"  per-series cost: control mean={control_costs.mean():.1f}  "
          f"treatment mean={treatment_costs.mean():.1f}")

    # ---- 5) power analysis (retrospective MDE) -------------------------
    sigma_pool = float(np.concatenate([control_costs, treatment_costs]).std(ddof=1))
    n_per_arm = min(len(control_costs), len(treatment_costs))
    mde = minimum_detectable_effect(n_per_arm=n_per_arm, sigma=sigma_pool, alpha=0.05, power=0.8)
    mde_pct = mde / control_costs.mean() * 100
    print(f"  power: n/arm~{n_per_arm}, sigma={sigma_pool:.1f} -> MDE={mde:.1f} ({mde_pct:.1f}%)")

    # ---- 6) tests ------------------------------------------------------
    cost = analyze_cost(control_costs.to_numpy(), treatment_costs.to_numpy())
    print(f"\n  COST TEST ({cost.test_used}): diff={cost.diff:+.2f} ({cost.pct_change:+.2f}%)  "
          f"CI95=[{cost.ci95_low:+.1f},{cost.ci95_high:+.1f}]  d={cost.cohens_d:+.3f}  "
          f"p={cost.p_value:.4f}")

    so_c = float((control_rows["stockouts"] > 0).mean())
    so_t = float((treatment_rows["stockouts"] > 0).mean())
    so_test = two_proportion_z_test(so_c, len(control_rows), so_t, len(treatment_rows))
    print(f"  STOCKOUT-RATE: control={so_c*100:.2f}%  treatment={so_t*100:.2f}%  "
          f"lift={so_test['lift_pp']:+.2f}pp  p={so_test['p_value']:.4f}")

    wmape_c = wmape(control_rows["sales"].to_numpy(float), control_rows["y_pred"].to_numpy(float))
    wmape_t = wmape(treatment_rows["sales"].to_numpy(float), treatment_rows["y_pred"].to_numpy(float))
    wmape_change_pct = (wmape_t / wmape_c - 1) * 100

    # ---- 7) decision (guardrail depends on variant) --------------------
    cost_ok = cost.pct_change <= -DECISION_MIN_COST_REDUCTION_PCT
    sig_ok  = cost.p_value < DECISION_ALPHA
    if v.guardrail == "stockout":
        guard_val = so_test["lift_pp"]
        guard_ok = guard_val <= STOCKOUT_GUARDRAIL_MAX_PP
        guard_desc = f"stockout-rate change {guard_val:+.2f}pp <= {STOCKOUT_GUARDRAIL_MAX_PP}pp"
        print(f"  [info] WMAPE {wmape_c:.3f}->{wmape_t:.3f} ({wmape_change_pct:+.1f}%, "
              f"not the guardrail for a quantile model)")
    else:
        guard_val = wmape_change_pct
        guard_ok = guard_val <= WMAPE_GUARDRAIL_MAX_PCT
        guard_desc = f"WMAPE degradation {guard_val:+.2f}% <= {WMAPE_GUARDRAIL_MAX_PCT}%"
    decision = "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD"
    print(f"  GUARDRAIL: {guard_desc}  -> {'OK' if guard_ok else 'FAIL'}")
    print(f"\n  DECISION RULE: SHIP if cost_reduction>={DECISION_MIN_COST_REDUCTION_PCT}% "
          f"AND p<{DECISION_ALPHA} AND {v.guardrail} guardrail holds")
    print(f"  >>> {decision}  (cost_ok={cost_ok}, sig_ok={sig_ok}, guard_ok={guard_ok})")

    # ---- 8) persist (schema unchanged -> Phase 5 dashboards still read it)
    _write_outputs(v, cost, so_c, so_t, so_test, wmape_c, wmape_t, wmape_change_pct,
                   control_costs, treatment_costs, control_rows, treatment_rows)

    # ---- 9) plots ------------------------------------------------------
    plot_assignment_balance(v, assignment)
    plot_cost_distribution(v, control_costs, treatment_costs)
    plot_stockout_rate(v, control_rows, treatment_rows)
    plot_decision_summary(v, cost, so_test, wmape_change_pct, decision, mde_pct)
    if v.name == "v2_lstm_q80":
        plot_v1_vs_v2(v)
    print(f"\n  artifacts in {v.out}/")


def _write_outputs(v, cost, so_c, so_t, so_test, wmape_c, wmape_t, wmape_change_pct,
                   control_costs, treatment_costs, control_rows, treatment_rows):
    summary = pd.DataFrame([
        {"metric": "total_cost_per_series",
         "mean_control": cost.mean_control, "mean_treatment": cost.mean_treatment,
         "diff": cost.diff, "pct_change": cost.pct_change,
         "ci95_low": cost.ci95_low, "ci95_high": cost.ci95_high,
         "cohens_d": cost.cohens_d, "p_value": cost.p_value, "test": cost.test_used},
        {"metric": "stockout_day_rate",
         "mean_control": so_c, "mean_treatment": so_t,
         "diff": so_t - so_c, "pct_change": (so_t / so_c - 1) * 100 if so_c else float("nan"),
         "ci95_low": float("nan"), "ci95_high": float("nan"),
         "cohens_d": float("nan"), "p_value": so_test["p_value"], "test": "two_prop_z"},
        {"metric": "wmape_guardrail",
         "mean_control": wmape_c, "mean_treatment": wmape_t,
         "diff": wmape_t - wmape_c, "pct_change": wmape_change_pct,
         "ci95_low": float("nan"), "ci95_high": float("nan"),
         "cohens_d": float("nan"), "p_value": float("nan"), "test": "n/a"},
    ])
    summary.to_csv(v.out / "summary.csv", index=False)
    pd.DataFrame({"id": control_costs.index, "arm": "control",
                  "total_cost": control_costs.values}).to_csv(v.out / "per_series_control.csv", index=False)
    pd.DataFrame({"id": treatment_costs.index, "arm": "treatment",
                  "total_cost": treatment_costs.values}).to_csv(v.out / "per_series_treatment.csv", index=False)
    cols = ["id", "date", "fold", "arm", "y_pred", "sales",
            "safety_stock", "order", "stockouts", "holding", "cost"]
    pd.concat([
        control_rows.assign(model=f"champion_{CHAMPION_MODEL}")[cols + ["model"]],
        treatment_rows.assign(model=v.challenger_label)[cols + ["model"]],
    ]).to_csv(v.out / "experiment_long_table.csv", index=False)


# ---- plots ------------------------------------------------------------------

def _save(v, fig, name):
    p = v.out / f"{name}.png"
    fig.tight_layout(); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


def plot_assignment_balance(v, assignment):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.countplot(data=assignment, x="volume_tertile", hue="arm",
                  order=["low", "mid", "high"], ax=axes[0])
    axes[0].set_title("Assignment balance by volume tertile"); axes[0].set_ylabel("# series")
    assignment.groupby(["cat_id", "arm"]).size().unstack(fill_value=0).plot.bar(ax=axes[1])
    axes[1].set_title("Assignment by category"); axes[1].set_ylabel("# series")
    _save(v, fig, "01_assignment_balance")


def plot_cost_distribution(v, control, treatment):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    df = pd.concat([pd.DataFrame({"arm": "control", "total_cost": control.values}),
                    pd.DataFrame({"arm": "treatment", "total_cost": treatment.values})])
    sns.boxplot(data=df, x="arm", y="total_cost", showfliers=False, ax=axes[0])
    axes[0].set_title("Per-series total cost by arm (84-day horizon)")
    axes[1].hist(control.values, bins=30, alpha=0.6, label="control")
    axes[1].hist(treatment.values, bins=30, alpha=0.6, label="treatment")
    axes[1].axvline(control.mean(), color="C0", ls="--")
    axes[1].axvline(treatment.mean(), color="C1", ls="--")
    axes[1].set_title("Per-series total cost histogram"); axes[1].legend()
    _save(v, fig, "02_cost_distribution")


def plot_stockout_rate(v, c_rows, t_rows):
    overall = pd.concat([
        c_rows.assign(stk=lambda d: (d["stockouts"] > 0).astype(int))
              .groupby("id")["stk"].mean().to_frame("stockout_rate").assign(arm="control"),
        t_rows.assign(stk=lambda d: (d["stockouts"] > 0).astype(int))
              .groupby("id")["stk"].mean().to_frame("stockout_rate").assign(arm="treatment"),
    ]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=overall, x="arm", y="stockout_rate", showfliers=False, ax=ax)
    ax.set_ylabel("per-series stockout day-rate")
    ax.set_title("Stockout day-rate by arm (lower = fewer stockouts)")
    _save(v, fig, "03_stockout_rate")


def plot_decision_summary(v, cost, so_test, wmape_change_pct, decision, mde_pct):
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["cost\n(% change)", "stockout-rate\n(pp change)", "WMAPE\n(% change)"]
    vals = [cost.pct_change, so_test["lift_pp"], wmape_change_pct]
    ci_low  = [cost.ci95_low / cost.mean_control * 100, np.nan, np.nan]
    ci_high = [cost.ci95_high / cost.mean_control * 100, np.nan, np.nan]
    err_low  = [val - lo if not np.isnan(lo) else 0 for val, lo in zip(vals, ci_low)]
    err_high = [hi - val if not np.isnan(hi) else 0 for val, hi in zip(vals, ci_high)]
    colors = ["green" if val < 0 else "red" for val in vals]
    ax.bar(metrics, vals, color=colors, alpha=0.7)
    ax.errorbar(metrics, vals, yerr=[err_low, err_high], fmt="none",
                color="black", capsize=6, lw=1.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("treatment vs control")
    sig = "*" if cost.p_value < DECISION_ALPHA else "n.s."
    ax.set_title(f"A/B {v.name} -- decision: {decision}   "
                 f"(cost p={cost.p_value:.3f} {sig}, MDE ~ {mde_pct:.1f}%)")
    _save(v, fig, "04_decision_summary")


def plot_v1_vs_v2(v):
    """Only for q80: compare against the mse variant if it's been run."""
    v1_summary = VARIANTS["mse"].out / "summary.csv"
    if not v1_summary.exists():
        print("  [skip] v1 summary not found; run --challenger mse first for the comparison plot")
        return
    v1 = pd.read_csv(v1_summary); v1["version"] = "v1 (LSTM MSE)"
    v2 = pd.read_csv(v.out / "summary.csv"); v2["version"] = "v2 (LSTM q80)"
    combined = pd.concat([v1, v2], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m, title in zip(
        axes,
        ["total_cost_per_series", "stockout_day_rate", "wmape_guardrail"],
        ["Cost % change", "Stockout-rate change", "WMAPE % change"],
    ):
        sub = combined[combined["metric"] == m]
        ax.bar(sub["version"], sub["pct_change"],
               color=["#dd8452" if x > 0 else "#55a868" for x in sub["pct_change"]])
        ax.axhline(0, color="black", lw=0.5); ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Phase 4 v1 vs v2 -- did the quantile-loss fix flip the verdict?", y=1.02)
    _save(v, fig, "05_v1_vs_v2")


if __name__ == "__main__":
    main()
