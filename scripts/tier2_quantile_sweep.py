"""
Tier 2b: quantile sweep to hunt for a defensible SHIP.

Tier 2a showed the LSTM systematically UNDER-orders (bias grows with volume),
which is why tau=0.80 reduced cost but tripped the +2pp stockout guardrail.
The fix: sweep HIGHER quantiles that lift predictions toward the champion's
service level, and find the tau that minimises cost WITHOUT breaching the
stockout guardrail.

Steps:
  1. Get challenger predictions for tau in {0.80 (reuse), 0.90, 0.95}.
  2. Build a cost-vs-service frontier over the full population
     (challenger = quantile forecast, NO safety stock; champion = ETS + safety).
  3. Pick the best tau = lowest cost s.t. stockout-rate <= champion + 2pp.
  4. Run the FORMAL stratified A/B for that tau -> SHIP / HOLD.

Run: python -m scripts.tier2_quantile_sweep
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
from src.experiment import (
    DECISION_ALPHA,
    DECISION_MIN_COST_REDUCTION_PCT,
    SERVICE_Z_DEFAULT,
    analyze_cost,
    compute_safety_stock,
    minimum_detectable_effect,
    simulate_costs,
    stratified_assignment,
    two_proportion_z_test,
)
from src.models.deep import Seq2SeqConfig, train_and_predict_seq2seq

DATA = Path("data/processed/m5_long_sample.parquet")
CHAMP_PREDS = Path("reports/phase2_baselines/predictions.parquet")
Q80_PREDS = Path("reports/phase3_lstm_quantile/q80_predictions.parquet")  # reuse tau=0.80
OUT = Path("reports/tier2_quantile_sweep")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("tier2_quantile_sweep")

CHAMPION_MODEL = "ets"
TAUS_TO_TRAIN = [0.90, 0.95]
STOCKOUT_GUARDRAIL_MAX_PP = 2.0


def train_tau(tau: float, df: pd.DataFrame, folds) -> pd.DataFrame:
    cfg = Seq2SeqConfig(quantile=tau)
    all_preds = []
    for fold in folds:
        train, test = split_frame(df, fold)
        with mlflow.start_run(run_name=f"sweep_q{int(tau*100)}_{fold.name}"):
            mlflow.log_params({"tau": tau, "fold": fold.name, "loss": "pinball",
                               "arch": "encoder_decoder_lstm", "early_stopping": True})
            t0 = time.time()
            preds, info = train_and_predict_seq2seq(train, test, df, cfg)
            mlflow.log_metrics({"train_secs": time.time() - t0,
                                "best_epoch": info.get("best_epoch", 0),
                                "n_windows": info["n_windows"]})
        preds["fold"] = fold.name
        all_preds.append(preds[["id", "date", "y_pred", "fold"]])
    out = pd.concat(all_preds, ignore_index=True)
    out.to_parquet(OUT / f"preds_q{int(tau*100)}.parquet", index=False)
    return out


def population_outcome(preds, actuals, safety) -> dict:
    """Total cost + stockout day-rate over the FULL population (paired frontier)."""
    sim = simulate_costs(preds, actuals, safety)
    return {"total_cost": float(sim["cost"].sum()),
            "cost_per_series": float(sim.groupby("id")["cost"].sum().mean()),
            "stockout_rate": float((sim["stockouts"] > 0).mean())}


def main():
    print("loading actuals + champion preds...")
    actuals = pd.read_parquet(DATA)
    actuals["date"] = pd.to_datetime(actuals["date"])
    actuals = actuals[actuals["is_active"]].copy()

    champ = pd.read_parquet(CHAMP_PREDS)
    champ = champ[champ["model"] == CHAMPION_MODEL][["id", "date", "y_pred", "fold"]]
    champ["date"] = pd.to_datetime(champ["date"])

    folds = rolling_origin_splits(actuals["date"].max(), n_folds=3, horizon=28, step=28)
    cutoff = folds[0].train_end

    # order policies: champion = forecast + safety stock; challenger = quantile, NO safety
    safety = compute_safety_stock(actuals, cutoff, z=SERVICE_Z_DEFAULT)
    zero_safety = pd.Series(0, index=safety.index, name="safety_stock")

    # ---- 1) gather challenger predictions per tau -------------------------
    tau_preds = {0.80: pd.read_parquet(Q80_PREDS)[["id", "date", "y_pred", "fold"]]}
    tau_preds[0.80]["date"] = pd.to_datetime(tau_preds[0.80]["date"])
    for tau in TAUS_TO_TRAIN:
        print(f"\n=== training challenger at tau={tau} ===")
        tau_preds[tau] = train_tau(tau, actuals, folds)

    # ---- 2) frontier over full population ---------------------------------
    champ_out = population_outcome(champ, actuals, safety)
    print(f"\nchampion (ETS + safety): cost/series={champ_out['cost_per_series']:.1f}  "
          f"stockout={champ_out['stockout_rate']*100:.2f}%")

    rows = [{"model": "champion_ets", "tau": np.nan, **champ_out}]
    for tau in sorted(tau_preds):
        o = population_outcome(tau_preds[tau], actuals, zero_safety)
        o.update({"model": f"challenger_q{int(tau*100)}", "tau": tau})
        rows.append(o)
        print(f"challenger q{int(tau*100):d}: cost/series={o['cost_per_series']:.1f}  "
              f"stockout={o['stockout_rate']*100:.2f}%  "
              f"(cost {(o['cost_per_series']/champ_out['cost_per_series']-1)*100:+.1f}% vs champ)")
    frontier = pd.DataFrame(rows)
    frontier.to_csv(OUT / "frontier.csv", index=False)

    # ---- 3) pick best tau: min cost s.t. stockout <= champ + 2pp ----------
    guard_max = champ_out["stockout_rate"] + STOCKOUT_GUARDRAIL_MAX_PP / 100
    elig = frontier[(frontier["tau"].notna()) &
                    (frontier["stockout_rate"] <= guard_max)]
    if len(elig):
        best = elig.sort_values("cost_per_series").iloc[0]
        best_tau = float(best["tau"])
        print(f"\nbest tau under guardrail (stockout <= {guard_max*100:.2f}%): "
              f"q{int(best_tau*100)} (cost/series={best['cost_per_series']:.1f})")
    else:
        best_tau = max(tau_preds)  # none pass -> still run A/B on the highest tau
        print(f"\nNo tau meets the stockout guardrail; running A/B on highest tau "
              f"q{int(best_tau*100)} for the record.")

    plot_frontier(frontier, champ_out, guard_max)

    # ---- 4) formal stratified A/B for best tau ----------------------------
    decision = run_formal_ab(actuals, champ, tau_preds[best_tau], best_tau,
                             safety, zero_safety, cutoff)

    print(f"\n  artifacts in {OUT}/")
    return decision


def run_formal_ab(actuals, champ, chall, tau, safety, zero_safety, cutoff):
    units = (actuals[actuals["date"] <= cutoff]
             .groupby("id", as_index=False)
             .agg(sales_sum=("sales", "sum"), cat_id=("cat_id", "first")))
    units["volume_tertile"] = pd.qcut(units["sales_sum"], 3, labels=["low", "mid", "high"])
    assignment = stratified_assignment(units, "volume_tertile", seed=42)

    champ_sim = simulate_costs(champ, actuals, safety).merge(assignment, on="id")
    chall_sim = simulate_costs(chall, actuals, zero_safety).merge(assignment, on="id")
    control = champ_sim[champ_sim["arm"] == "control"]
    treatment = chall_sim[chall_sim["arm"] == "treatment"]

    c_cost = control.groupby("id")["cost"].sum()
    t_cost = treatment.groupby("id")["cost"].sum()
    cost = analyze_cost(c_cost.to_numpy(), t_cost.to_numpy())

    so_c = float((control["stockouts"] > 0).mean())
    so_t = float((treatment["stockouts"] > 0).mean())
    so = two_proportion_z_test(so_c, len(control), so_t, len(treatment))

    sigma = float(np.concatenate([c_cost, t_cost]).std(ddof=1))
    mde_pct = minimum_detectable_effect(min(len(c_cost), len(t_cost)), sigma) / c_cost.mean() * 100

    cost_ok = cost.pct_change <= -DECISION_MIN_COST_REDUCTION_PCT
    sig_ok = cost.p_value < DECISION_ALPHA
    guard_ok = so["lift_pp"] <= STOCKOUT_GUARDRAIL_MAX_PP
    decision = "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD"

    print(f"\n=== FORMAL A/B: champion=ETS vs challenger=q{int(tau*100)} ===")
    print(f"  cost: {cost.pct_change:+.2f}%  p={cost.p_value:.4f}  "
          f"CI=[{cost.ci95_low:+.1f},{cost.ci95_high:+.1f}]  MDE~{mde_pct:.1f}%")
    print(f"  stockout: {so_c*100:.2f}%->{so_t*100:.2f}% ({so['lift_pp']:+.2f}pp)  "
          f"guardrail {'OK' if guard_ok else 'FAIL'} (<= {STOCKOUT_GUARDRAIL_MAX_PP}pp)")
    print(f"  >>> {decision}  (cost_ok={cost_ok}, sig_ok={sig_ok}, guard_ok={guard_ok})")

    pd.DataFrame([{
        "best_tau": tau, "decision": decision,
        "cost_pct_change": cost.pct_change, "cost_p": cost.p_value,
        "cost_ci_low": cost.ci95_low, "cost_ci_high": cost.ci95_high,
        "stockout_control": so_c, "stockout_treatment": so_t,
        "stockout_lift_pp": so["lift_pp"], "mde_pct": mde_pct,
    }]).to_csv(OUT / "formal_ab_result.csv", index=False)
    return decision


def plot_frontier(frontier, champ_out, guard_max):
    fig, ax = plt.subplots(figsize=(9, 6))
    ch = frontier[frontier["model"].str.startswith("challenger")].sort_values("tau")
    ax.plot(ch["stockout_rate"] * 100, ch["cost_per_series"], "-o", color="#4c72b0",
            label="LSTM quantile challenger")
    for _, r in ch.iterrows():
        ax.annotate(f"τ={r['tau']:.2f}", (r["stockout_rate"] * 100, r["cost_per_series"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.scatter([champ_out["stockout_rate"] * 100], [champ_out["cost_per_series"]],
               color="#c44e52", s=90, zorder=5, label="champion (ETS + safety stock)")
    ax.axvline(guard_max * 100, ls="--", color="grey",
               label=f"stockout guardrail ({guard_max*100:.1f}%)")
    ax.set_xlabel("stockout day-rate (%)  -- service")
    ax.set_ylabel("cost per series  -- lower is better")
    ax.set_title("Cost-vs-service frontier: which quantile lands a SHIP?")
    ax.legend()
    p = OUT / "01_cost_service_frontier.png"
    fig.tight_layout(); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
