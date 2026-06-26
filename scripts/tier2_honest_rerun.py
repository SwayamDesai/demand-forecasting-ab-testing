"""
Tier 2c -- the HONEST, leakage-free SHIP test.

Fixes two integrity problems in the earlier "SHIP at tau=0.90" result:

  (#1) tau was chosen on the SAME backtest its significance was reported on
       -> selection bias / uncorrected multiple comparisons.
  (#2) the stockout guardrail was adopted AFTER seeing q80 fail the old one
       -> goalpost-moving.

This script fixes both with a pre-registered, split protocol:

  PRE-REGISTERED DECISION RULE (declared here, BEFORE looking at any result):
      SHIP iff  cost_reduction >= 5%
            AND cost p-value (Mann-Whitney) < 0.05
            AND stockout-rate change <= +2.0 pp   <- guardrail = SERVICE, fixed up-front

  PRE-REGISTERED SELECTION POLICY:
      Among tau candidates whose stockout change <= +2.0 pp on the SELECTION set,
      pick the one with the largest cost reduction. (Service first, then cost.)

  DATA SPLIT (time-based, no leakage):
      SELECTION  set = fold_1 + fold_2   (allowed to "shop" for tau here)
      CONFIRMATION set = fold_3          (untouched during selection; evaluated ONCE)

  The confirmation verdict on fold_3 is the FINAL answer -- whatever it says.

Champion (control)  = ETS,    order = forecast + safety stock (z=1.645, 95% svc)
Challenger (treat.) = LSTM@tau, order = tau-quantile forecast, NO extra safety stock

Run: python -m scripts.tier2_honest_rerun
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import rolling_origin_splits
from src.experiment import (
    SERVICE_Z_DEFAULT,
    analyze_cost,
    compute_safety_stock,
    minimum_detectable_effect,
    simulate_costs,
    stratified_assignment,
    two_proportion_z_test,
)

DATA = Path("data/processed/m5_long_sample.parquet")
ETS_PREDS = Path("reports/phase2_baselines/predictions.parquet")
OUT = Path("reports/tier2_honest_rerun")
OUT.mkdir(parents=True, exist_ok=True)

# τ candidates with existing fold-complete predictions
TAU_PREDS = {
    0.80: Path("reports/phase3_lstm_quantile/q80_predictions.parquet"),
    0.90: Path("reports/tier2_quantile_sweep/preds_q90.parquet"),
    0.95: Path("reports/tier2_quantile_sweep/preds_q95.parquet"),
}

# ---- PRE-REGISTERED rule + guardrail (fixed BEFORE evaluation) --------------
MIN_COST_REDUCTION_PCT = 5.0
ALPHA = 0.05
STOCKOUT_GUARDRAIL_PP = 2.0

SELECTION_FOLDS = ["fold_1", "fold_2"]
CONFIRMATION_FOLD = "fold_3"


def _per_series_costs(preds: pd.DataFrame, actuals: pd.DataFrame,
                      safety: pd.Series, fold_names, arm_ids) -> pd.DataFrame:
    """Simulate costs for one arm restricted to (folds, arm series)."""
    p = preds[preds["fold"].isin(fold_names)][["id", "date", "y_pred"]]
    sim = simulate_costs(p, actuals, safety)
    return sim[sim["id"].isin(arm_ids)]


def _evaluate(control_sim: pd.DataFrame, treat_sim: pd.DataFrame) -> dict:
    """Cost % change + p-value + stockout pp change for one (fold-set, tau)."""
    c_cost = control_sim.groupby("id")["cost"].sum()
    t_cost = treat_sim.groupby("id")["cost"].sum()
    cost = analyze_cost(c_cost.to_numpy(), t_cost.to_numpy())
    so_c = float((control_sim["stockouts"] > 0).mean())
    so_t = float((treat_sim["stockouts"] > 0).mean())
    so = two_proportion_z_test(so_c, len(control_sim), so_t, len(treat_sim))
    sigma = float(np.concatenate([c_cost, t_cost]).std(ddof=1))
    mde = minimum_detectable_effect(min(len(c_cost), len(t_cost)), sigma)
    return {
        "cost_pct": cost.pct_change, "cost_p": cost.p_value,
        "ci_low": cost.ci95_low, "ci_high": cost.ci95_high,
        "test": cost.test_used,
        "stockout_c": so_c * 100, "stockout_t": so_t * 100,
        "stockout_pp": so["lift_pp"],
        "mde_pct": mde / c_cost.mean() * 100,
        "n_per_arm": int(min(len(c_cost), len(t_cost))),
    }


def main():
    print(__doc__.split("Run:")[0])

    actuals = pd.read_parquet(DATA)
    actuals["date"] = pd.to_datetime(actuals["date"])
    actuals = actuals[actuals["is_active"]].copy()

    ets = pd.read_parquet(ETS_PREDS)
    ets = ets[ets["model"] == "ets"][["id", "date", "y_pred", "fold"]]
    ets["date"] = pd.to_datetime(ets["date"])

    tau_preds = {}
    for tau, path in TAU_PREDS.items():
        d = pd.read_parquet(path)[["id", "date", "y_pred", "fold"]]
        d["date"] = pd.to_datetime(d["date"])
        tau_preds[tau] = d

    # ---- fixed assignment + safety stock (computed BEFORE any test fold) ----
    folds = rolling_origin_splits(actuals["date"].max(), n_folds=3, horizon=28, step=28)
    pre_cutoff = folds[0].train_end
    safety = compute_safety_stock(actuals, pre_cutoff, z=SERVICE_Z_DEFAULT)
    zero_safety = pd.Series(0, index=safety.index, name="safety_stock")

    units = (actuals[actuals["date"] <= pre_cutoff]
             .groupby("id", as_index=False).agg(sales_sum=("sales", "sum")))
    units["volume_tertile"] = pd.qcut(units["sales_sum"], 3, labels=["low", "mid", "high"])
    assign = stratified_assignment(units, "volume_tertile", seed=42)
    control_ids = set(assign.loc[assign["arm"] == "control", "id"])
    treat_ids = set(assign.loc[assign["arm"] == "treatment", "id"])

    # ======================================================================
    # STEP 1 -- SELECTION on folds 1+2 (we are allowed to shop for tau here)
    # ======================================================================
    print("=" * 72)
    print(f"STEP 1  SELECTION on {SELECTION_FOLDS}  (choosing tau)")
    print("=" * 72)
    control_sel = _per_series_costs(ets, actuals, safety, SELECTION_FOLDS, control_ids)
    sel_rows = []
    for tau, preds in tau_preds.items():
        treat_sel = _per_series_costs(preds, actuals, zero_safety, SELECTION_FOLDS, treat_ids)
        r = _evaluate(control_sel, treat_sel); r["tau"] = tau
        sel_rows.append(r)
        passes = r["stockout_pp"] <= STOCKOUT_GUARDRAIL_PP
        print(f"  tau={tau:.2f}  cost {r['cost_pct']:+6.2f}%  p={r['cost_p']:.4f}  "
              f"stockout {r['stockout_pp']:+.2f}pp  guardrail={'OK' if passes else 'FAIL'}")
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(OUT / "selection_folds12.csv", index=False)

    # pre-registered selection policy: among guardrail-passers, max cost reduction
    eligible = sel[sel["stockout_pp"] <= STOCKOUT_GUARDRAIL_PP]
    if eligible.empty:
        print("\n  no tau passes the service guardrail on the selection set -> "
              "no shippable candidate. STOP.")
        return
    chosen = eligible.sort_values("cost_pct").iloc[0]          # most negative = best
    chosen_tau = float(chosen["tau"])
    print(f"\n  --> selected tau = {chosen_tau:.2f}  "
          f"(best cost reduction among guardrail-passers on selection set)")

    # ======================================================================
    # STEP 2 -- CONFIRMATION on fold_3 (UNTOUCHED during selection; once only)
    # ======================================================================
    print("\n" + "=" * 72)
    print(f"STEP 2  CONFIRMATION on {CONFIRMATION_FOLD}  (chosen tau only, evaluated ONCE)")
    print("=" * 72)
    control_conf = _per_series_costs(ets, actuals, safety, [CONFIRMATION_FOLD], control_ids)
    treat_conf = _per_series_costs(tau_preds[chosen_tau], actuals, zero_safety,
                                   [CONFIRMATION_FOLD], treat_ids)
    conf = _evaluate(control_conf, treat_conf)

    cost_ok = conf["cost_pct"] <= -MIN_COST_REDUCTION_PCT
    sig_ok = conf["cost_p"] < ALPHA
    guard_ok = conf["stockout_pp"] <= STOCKOUT_GUARDRAIL_PP
    decision = "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD"

    print(f"  tau                 = {chosen_tau:.2f}")
    print(f"  n per arm           = {conf['n_per_arm']}")
    print(f"  cost change         = {conf['cost_pct']:+.2f}%   "
          f"(rule: <= -{MIN_COST_REDUCTION_PCT}%)   -> {'OK' if cost_ok else 'FAIL'}")
    print(f"  cost p-value        = {conf['cost_p']:.4f} ({conf['test']})   "
          f"(rule: < {ALPHA})   -> {'OK' if sig_ok else 'FAIL'}")
    print(f"  cost 95% CI         = [{conf['ci_low']:+.1f}, {conf['ci_high']:+.1f}]  (mean diff, abs)")
    print(f"  stockout change     = {conf['stockout_pp']:+.2f} pp   "
          f"(guardrail: <= {STOCKOUT_GUARDRAIL_PP} pp)   -> {'OK' if guard_ok else 'FAIL'}")
    print(f"  MDE on this fold    = {conf['mde_pct']:.1f}% of control mean")
    print(f"\n  >>> HONEST VERDICT (held-out fold_3): {decision}")

    # ---- persist ---------------------------------------------------------
    pd.DataFrame([{**conf, "tau": chosen_tau, "decision": decision,
                   "cost_ok": cost_ok, "sig_ok": sig_ok, "guard_ok": guard_ok}]
                 ).to_csv(OUT / "confirmation_fold3.csv", index=False)

    _plot(sel, chosen_tau, conf, decision)
    print(f"\n  artifacts in {OUT}/")


def _plot(sel: pd.DataFrame, chosen_tau: float, conf: dict, decision: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # left: selection-set frontier (cost vs stockout) per tau
    ax1.axhline(STOCKOUT_GUARDRAIL_PP, color="grey", ls="--", label=f"guardrail {STOCKOUT_GUARDRAIL_PP}pp")
    for _, r in sel.iterrows():
        c = "#55a868" if r["tau"] == chosen_tau else "#4c72b0"
        ax1.scatter(r["cost_pct"], r["stockout_pp"], s=120, color=c, zorder=3)
        ax1.annotate(f"τ={r['tau']:.2f}", (r["cost_pct"], r["stockout_pp"]),
                     textcoords="offset points", xytext=(6, 6))
    ax1.axvline(-MIN_COST_REDUCTION_PCT, color="grey", ls=":", alpha=0.6)
    ax1.set_xlabel("cost % change (more negative = better)")
    ax1.set_ylabel("stockout change (pp)")
    ax1.set_title("STEP 1 — selection on folds 1+2\n(green = chosen τ)")
    ax1.legend(fontsize=8)

    # right: confirmation verdict bars vs thresholds
    metrics = ["cost %\n(rule ≤ -5)", "stockout pp\n(rule ≤ 2)"]
    vals = [conf["cost_pct"], conf["stockout_pp"]]
    colors = ["#55a868" if (vals[0] <= -MIN_COST_REDUCTION_PCT) else "#c44e52",
              "#55a868" if (vals[1] <= STOCKOUT_GUARDRAIL_PP) else "#c44e52"]
    ax2.bar(metrics, vals, color=colors, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.6)
    sig = "*" if conf["cost_p"] < ALPHA else "n.s."
    ax2.set_title(f"STEP 2 — confirmation on held-out fold_3\n"
                  f"τ={chosen_tau:.2f}  cost p={conf['cost_p']:.3f} {sig}  "
                  f"-> {decision}")
    ax2.set_ylabel("treatment vs control")
    fig.tight_layout()
    p = OUT / "honest_rerun.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
