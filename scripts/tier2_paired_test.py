"""
Tier 2d -- PAIRED counterfactual A/B test (the statistically-correct design
for an OFFLINE experiment).

The unpaired test (analyze_cost) threw away information: because this is a
simulation, every SKU has a cost under BOTH the champion and the challenger.
The unpaired design compared two disjoint groups, so between-SKU variance (a
few huge SKUs) swamped the signal -> wide CI that crossed zero.

The paired design compares each SKU to ITSELF:
    d_i = cost_challenger_i - cost_champion_i
and tests whether d is < 0. Between-SKU variance cancels, and all ~900 SKUs
contribute (no arm split). Both effects tighten the CI dramatically.

Honesty preserved: tau=0.90 was selected on folds 1+2 (see tier2_honest_rerun);
the HEADLINE paired test here is on the untouched CONFIRMATION fold_3. We also
report all-folds for context (labelled as including the selection folds).

Champion  = ETS,     order = forecast + safety stock (z=1.645)
Challenger = LSTM@0.90, order = 90th-pctile forecast, no extra safety stock

Run: python -m scripts.tier2_paired_test
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import rolling_origin_splits
from src.experiment import (
    SERVICE_Z_DEFAULT,
    compute_safety_stock,
    paired_diff_test,
    simulate_costs,
)

DATA = Path("data/processed/m5_long_sample.parquet")
ETS_PREDS = Path("reports/phase2_baselines/predictions.parquet")
Q90_PREDS = Path("reports/tier2_quantile_sweep/preds_q90.parquet")
UNPAIRED_CONF = Path("reports/tier2_honest_rerun/confirmation_fold3.csv")
OUT = Path("reports/tier2_paired_test")
OUT.mkdir(parents=True, exist_ok=True)

CHOSEN_TAU = 0.90
MIN_COST_REDUCTION_PCT = 5.0
ALPHA = 0.05
STOCKOUT_GUARDRAIL_PP = 2.0
CONFIRMATION_FOLD = "fold_3"


def _per_series(preds: pd.DataFrame, actuals: pd.DataFrame, safety: pd.Series,
                fold_names) -> pd.DataFrame:
    """Per-series total cost + stockout-day-rate for one model over given folds."""
    p = preds[preds["fold"].isin(fold_names)][["id", "date", "y_pred"]]
    sim = simulate_costs(p, actuals, safety)
    g = sim.groupby("id")
    return pd.DataFrame({
        "cost": g["cost"].sum(),
        "stockout_rate": g["stockouts"].apply(lambda s: float((s > 0).mean())),
    })


def _paired_frame(actuals, ets, q90, safety, zero_safety, fold_names):
    champ = _per_series(ets, actuals, safety, fold_names)
    chall = _per_series(q90, actuals, zero_safety, fold_names)
    j = champ.join(chall, lsuffix="_champ", rsuffix="_chall", how="inner")
    return j.dropna()


def _run(label, j):
    cost = paired_diff_test(j["cost_champ"].to_numpy(), j["cost_chall"].to_numpy())
    so_champ = j["stockout_rate_champ"].mean() * 100
    so_chall = j["stockout_rate_chall"].mean() * 100
    so_pp = so_chall - so_champ
    print(f"\n--- {label}  (n={cost.n} paired SKUs) ---")
    print(f"  champion mean cost   = {cost.mean_a:.2f}")
    print(f"  challenger mean cost = {cost.mean_b:.2f}")
    print(f"  mean diff            = {cost.mean_diff:+.2f}  ({cost.pct_change:+.2f}%)")
    print(f"  median diff          = {cost.median_diff:+.2f}")
    print(f"  SKUs challenger cheaper = {cost.pct_units_b_better:.1f}%")
    print(f"  Wilcoxon p (paired)  = {cost.wilcoxon_p:.2e}")
    print(f"  paired t-test p      = {cost.ttest_rel_p:.2e}")
    print(f"  95% CI of mean diff  = [{cost.ci95_low:+.2f}, {cost.ci95_high:+.2f}]")
    print(f"  stockout change      = {so_pp:+.2f} pp ({so_champ:.2f}% -> {so_chall:.2f}%)")
    return cost, so_pp


def main():
    print(__doc__.split("Run:")[0])

    actuals = pd.read_parquet(DATA)
    actuals["date"] = pd.to_datetime(actuals["date"])
    actuals = actuals[actuals["is_active"]].copy()

    ets = pd.read_parquet(ETS_PREDS)
    ets = ets[ets["model"] == "ets"][["id", "date", "y_pred", "fold"]]
    ets["date"] = pd.to_datetime(ets["date"])
    q90 = pd.read_parquet(Q90_PREDS)[["id", "date", "y_pred", "fold"]]
    q90["date"] = pd.to_datetime(q90["date"])

    folds = rolling_origin_splits(actuals["date"].max(), n_folds=3, horizon=28, step=28)
    safety = compute_safety_stock(actuals, folds[0].train_end, z=SERVICE_Z_DEFAULT)
    zero_safety = pd.Series(0, index=safety.index, name="safety_stock")

    # HEADLINE: paired on held-out fold_3
    j_conf = _paired_frame(actuals, ets, q90, safety, zero_safety, [CONFIRMATION_FOLD])
    cost_conf, so_conf = _run("PAIRED on held-out fold_3 (CONFIRMATORY)", j_conf)

    # CONTEXT: paired on all folds (includes the tau-selection folds)
    j_all = _paired_frame(actuals, ets, q90, safety, zero_safety,
                          ["fold_1", "fold_2", "fold_3"])
    cost_all, so_all = _run("PAIRED on all 3 folds (context; includes selection)", j_all)

    # ---- decision on the confirmatory paired test --------------------------
    cost_ok = cost_conf.pct_change <= -MIN_COST_REDUCTION_PCT
    sig_ok = cost_conf.wilcoxon_p < ALPHA
    ci_excludes_zero = cost_conf.ci95_high < 0
    guard_ok = so_conf <= STOCKOUT_GUARDRAIL_PP
    decision = "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD"
    print("\n" + "=" * 72)
    print("PRE-REGISTERED RULE: SHIP iff cost<=-5% AND p<0.05 AND stockout<=+2pp")
    print(f"  cost {cost_conf.pct_change:+.2f}% ({'OK' if cost_ok else 'FAIL'})   "
          f"Wilcoxon p={cost_conf.wilcoxon_p:.2e} ({'OK' if sig_ok else 'FAIL'})   "
          f"stockout {so_conf:+.2f}pp ({'OK' if guard_ok else 'FAIL'})")
    print(f"  >>> PAIRED VERDICT (fold_3): {decision}")
    print(f"  >>> mean-cost 95% CI now {'EXCLUDES' if ci_excludes_zero else 'still includes'} zero")

    # ---- compare to the unpaired confirmation ------------------------------
    cmp_txt = ""
    if UNPAIRED_CONF.exists():
        u = pd.read_csv(UNPAIRED_CONF).iloc[0]
        # unpaired CI was reported as absolute mean-diff bounds
        print("\n--- unpaired vs paired (fold_3) ---")
        print(f"  unpaired: cost {u['cost_pct']:+.2f}%  p={u['cost_p']:.2e}  "
              f"CI(meandiff)=[{u['ci_low']:+.1f}, {u['ci_high']:+.1f}]  (crosses 0: "
              f"{u['ci_low'] < 0 < u['ci_high']})")
        print(f"  paired:   cost {cost_conf.pct_change:+.2f}%  p={cost_conf.wilcoxon_p:.2e}  "
              f"CI(meandiff)=[{cost_conf.ci95_low:+.1f}, {cost_conf.ci95_high:+.1f}]  (crosses 0: "
              f"{cost_conf.ci95_low < 0 < cost_conf.ci95_high})")
        cmp_txt = (f"unpaired CI width = {u['ci_high'] - u['ci_low']:.0f}  ->  "
                   f"paired CI width = {cost_conf.ci95_high - cost_conf.ci95_low:.0f}")
        print(f"  {cmp_txt}")

    # ---- save + plot -------------------------------------------------------
    pd.DataFrame([{
        "scope": "fold_3", "tau": CHOSEN_TAU, "n": cost_conf.n,
        "pct_change": cost_conf.pct_change, "wilcoxon_p": cost_conf.wilcoxon_p,
        "ttest_rel_p": cost_conf.ttest_rel_p,
        "ci_low": cost_conf.ci95_low, "ci_high": cost_conf.ci95_high,
        "stockout_pp": so_conf, "decision": decision,
    }]).to_csv(OUT / "paired_result_fold3.csv", index=False)

    _plot(j_conf, cost_conf, decision, cmp_txt)
    print(f"\n  artifacts in {OUT}/")


def _plot(j_conf, cost, decision, cmp_txt):
    d = (j_conf["cost_chall"] - j_conf["cost_champ"]).to_numpy()
    # clip extreme tail for a readable histogram (note count clipped)
    lo, hi = np.percentile(d, [1, 99])
    d_clip = d[(d >= lo) & (d <= hi)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(d_clip, bins=60, color="#4c72b0", alpha=0.8)
    ax1.axvline(0, color="black", lw=1)
    ax1.axvline(cost.mean_diff, color="#c44e52", lw=2,
                label=f"mean diff = {cost.mean_diff:+.1f}")
    ax1.axvspan(cost.ci95_low, cost.ci95_high, color="#c44e52", alpha=0.15,
                label=f"95% CI [{cost.ci95_low:+.0f},{cost.ci95_high:+.0f}]")
    ax1.set_xlabel("per-SKU cost difference  (challenger − champion; <0 = cheaper)")
    ax1.set_ylabel("# SKUs")
    ax1.set_title(f"Paired cost differences, held-out fold_3\n"
                  f"{cost.pct_units_b_better:.0f}% of SKUs cheaper under challenger")
    ax1.legend(fontsize=8)

    ax2.axis("off")
    sig = "*" if cost.wilcoxon_p < ALPHA else "n.s."
    lines = [
        f"PAIRED A/B  (held-out fold_3, n={cost.n})",
        "",
        f"cost change    : {cost.pct_change:+.2f}%",
        f"Wilcoxon p     : {cost.wilcoxon_p:.2e} {sig}",
        f"95% CI (mean)  : [{cost.ci95_low:+.1f}, {cost.ci95_high:+.1f}]",
        f"CI excludes 0  : {cost.ci95_high < 0}",
        "",
        cmp_txt,
        "",
        f"DECISION       : {decision}",
    ]
    color = "#55a868" if decision == "SHIP" else "#c44e52"
    ax2.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=12)
    ax2.text(0.02, 0.06, decision, fontsize=44, weight="bold", color=color)
    fig.tight_layout()
    p = OUT / "paired_test.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
