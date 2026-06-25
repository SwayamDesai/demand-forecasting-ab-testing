"""
Champion-challenger A/B test (Phase 4). Industry-standard, kept lean.

Unit of analysis: store-SKU (n=300 in our sample).
Population is split: stratified 50/50 by demand-volume tertile.
  - control arm   -> orders driven by Phase 2 champion (LightGBM)
  - treatment arm -> orders driven by Phase 3 challenger (LSTM seq2seq)

Each series-day we run a newsvendor simulation:
  order      = max(0, round(forecast + safety_stock))
  stockouts  = max(0, actual_demand - order)
  holding    = max(0, order - actual_demand)
  cost       = h_stockout * stockouts + h_holding * holding

Industry defaults (per user): h_stockout : h_holding = 5 : 1, service level 95% -> z=1.645.

We then compare arms via (a) Welch's t-test on per-series total cost,
(b) two-proportion z-test on stockout day-rate, plus a guardrail (WMAPE) so
we don't celebrate cost wins that came from accuracy losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# ---- industry defaults ------------------------------------------------------
H_STOCKOUT_DEFAULT = 5.0           # cost per stocked-out unit
H_HOLDING_DEFAULT  = 1.0           # cost per held unit
SERVICE_Z_DEFAULT  = 1.645         # 95% service level -> z = 1.645

DECISION_MIN_COST_REDUCTION_PCT = 5.0
DECISION_MAX_WMAPE_DEGRADATION_PCT = 5.0
DECISION_ALPHA = 0.05


# ---- safety stock per series (TS rule #2: cutoff strictly before test) ------

def compute_safety_stock(active_df: pd.DataFrame, cutoff_date: pd.Timestamp,
                         z: float = SERVICE_Z_DEFAULT) -> pd.Series:
    """Per-series safety stock = z * std(sales) on data with date <= cutoff."""
    train = active_df[active_df["date"] <= cutoff_date]
    sigma = train.groupby("id")["sales"].std().fillna(0)
    ss = (z * sigma).clip(lower=0).round().astype(int)
    ss.name = "safety_stock"
    return ss


# ---- simulation -------------------------------------------------------------

def simulate_costs(predictions: pd.DataFrame,
                   actuals: pd.DataFrame,
                   safety_stocks: pd.Series,
                   h_stockout: float = H_STOCKOUT_DEFAULT,
                   h_holding: float = H_HOLDING_DEFAULT) -> pd.DataFrame:
    """
    Returns the predictions df joined to actuals with these columns added:
      safety_stock, order, stockouts, holding, cost
    One row per (id, date).
    """
    df = predictions.merge(actuals[["id", "date", "sales"]],
                           on=["id", "date"], how="inner")
    df = df.merge(safety_stocks, on="id", how="left")
    df["safety_stock"] = df["safety_stock"].fillna(0)
    df["order"] = (df["y_pred"] + df["safety_stock"]).clip(lower=0).round()
    df["stockouts"] = (df["sales"] - df["order"]).clip(lower=0)
    df["holding"]   = (df["order"] - df["sales"]).clip(lower=0)
    df["cost"] = h_stockout * df["stockouts"] + h_holding * df["holding"]
    return df


# ---- stratified assignment --------------------------------------------------

def stratified_assignment(units: pd.DataFrame,
                          stratify_col: str = "volume_tertile",
                          seed: int = 42) -> pd.DataFrame:
    """50/50 random assignment within each stratum.  Returns df[id, arm]."""
    rng = np.random.default_rng(seed)
    rows = []
    for stratum, g in units.groupby(stratify_col, observed=True):
        ids = g["id"].to_numpy().copy()
        rng.shuffle(ids)
        n_treat = len(ids) // 2
        treat = set(ids[:n_treat])
        for sid in g["id"]:
            rows.append({"id": sid, "arm": "treatment" if sid in treat else "control"})
    return pd.DataFrame(rows)


# ---- power analysis ---------------------------------------------------------

def minimum_detectable_effect(n_per_arm: int, sigma: float,
                              alpha: float = 0.05, power: float = 0.8) -> float:
    """
    Two-sample two-sided test MDE (Cohen's formula):
      MDE = (z_{alpha/2} + z_{1-beta}) * sigma * sqrt(2 / n)
    Returns the MDE in the same units as sigma.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * sigma * np.sqrt(2.0 / n_per_arm))


# ---- analysis ---------------------------------------------------------------

@dataclass
class CostAnalysis:
    mean_control: float; mean_treatment: float
    diff: float; pct_change: float
    cohens_d: float
    ci95_low: float; ci95_high: float
    test_used: str
    p_value: float
    normality_ok: bool


def analyze_cost(control_costs: np.ndarray, treatment_costs: np.ndarray,
                 n_bootstrap: int = 2000, seed: int = 0) -> CostAnalysis:
    """
    Per-series total cost, control vs treatment.

    Picks Welch's t-test when both arms look ~normal (Shapiro p > 0.05),
    Mann-Whitney U otherwise. Reports Cohen's d and a bootstrap 95% CI for
    the difference in means.
    """
    control_costs = np.asarray(control_costs, dtype=float)
    treatment_costs = np.asarray(treatment_costs, dtype=float)
    # normality
    _, p_c = stats.shapiro(control_costs[: min(len(control_costs), 5000)])
    _, p_t = stats.shapiro(treatment_costs[: min(len(treatment_costs), 5000)])
    normal_ok = (p_c > 0.05) and (p_t > 0.05)
    if normal_ok:
        stat, p = stats.ttest_ind(treatment_costs, control_costs, equal_var=False)
        test_used = "welch_t"
    else:
        stat, p = stats.mannwhitneyu(treatment_costs, control_costs, alternative="two-sided")
        test_used = "mann_whitney_u"

    mean_c = float(control_costs.mean())
    mean_t = float(treatment_costs.mean())
    diff   = mean_t - mean_c
    pooled = np.sqrt((control_costs.var(ddof=1) + treatment_costs.var(ddof=1)) / 2.0)
    d = float(diff / pooled) if pooled > 0 else 0.0

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        bc = rng.choice(control_costs, size=len(control_costs), replace=True)
        bt = rng.choice(treatment_costs, size=len(treatment_costs), replace=True)
        boots[i] = bt.mean() - bc.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return CostAnalysis(
        mean_control=mean_c, mean_treatment=mean_t,
        diff=diff, pct_change=(diff / mean_c * 100) if mean_c else float("nan"),
        cohens_d=d,
        ci95_low=float(lo), ci95_high=float(hi),
        test_used=test_used, p_value=float(p),
        normality_ok=bool(normal_ok),
    )


def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> dict:
    """Standard two-sample proportion z-test, two-sided."""
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)))
    z = (p2 - p1) / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"p_control": p1, "p_treatment": p2,
            "lift_pp": (p2 - p1) * 100, "z": z, "p_value": p_value}


# ---- decision rule (stated up-front, evaluated at end) ----------------------

def recommend(cost: CostAnalysis, wmape_change_pct: float) -> tuple[str, dict]:
    """
    Stated decision rule:
      SHIP if (cost reduction >= 5%) AND (p < 0.05) AND (WMAPE degrades by <= 5%).
    Otherwise HOLD.
    """
    cost_ok = cost.pct_change <= -DECISION_MIN_COST_REDUCTION_PCT
    sig_ok  = cost.p_value < DECISION_ALPHA
    guard_ok = wmape_change_pct <= DECISION_MAX_WMAPE_DEGRADATION_PCT
    decision = "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD"
    return decision, {
        "cost_reduction_pct": -cost.pct_change,
        "cost_ok": cost_ok,
        "p_value": cost.p_value,
        "sig_ok": sig_ok,
        "wmape_change_pct": wmape_change_pct,
        "guard_ok": guard_ok,
    }
