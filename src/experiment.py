"""
A/B-test machinery for Phase 8: a newsvendor business simulation plus the
statistics to judge it honestly. Pure functions, unit-tested.

The decision a forecast actually drives is an ORDER. So we score forecasts by the
money that order policy costs, not by error alone:

  newsvendor: order = forecast + z(tau*) * sigma_series,   tau* = cu / (cu + co)
              cost  = cu * understock_units + co * overstock_units

cu = stockout (underage) cost/unit, co = holding (overage) cost/unit. The SAME z and
sigma are applied to both model arms, so the arms differ only by their forecast --
a fair test of "which forecast makes the cheaper order".

Two comparisons are provided:
  * unpaired  -> mimics a live randomized experiment (each series sees ONE model)
  * paired    -> the counterfactual every-series-sees-both design; higher power and
                 the correct way to read an offline backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def newsvendor_z(cu: float, co: float) -> float:
    """Safety-stock multiplier for the critical ratio tau* = cu/(cu+co)."""
    tau = cu / (cu + co)
    return float(stats.norm.ppf(tau))


def simulate(df: pd.DataFrame, pred_col: str, sigma: pd.Series,
             cu: float, co: float) -> pd.DataFrame:
    """
    Per (series, week): order, understock/overstock units, cost.
    `df` needs columns [id, y, pred_col]; `sigma` is per-series demand std.
    """
    z = newsvendor_z(cu, co)
    out = df[["id", "y", pred_col]].copy()
    s = out["id"].map(sigma).fillna(0.0).to_numpy()
    order = np.clip(np.round(out[pred_col].to_numpy() + z * s), 0, None)
    out["order"] = order
    out["understock"] = np.clip(out["y"].to_numpy() - order, 0, None)
    out["overstock"] = np.clip(order - out["y"].to_numpy(), 0, None)
    out["cost"] = cu * out["understock"] + co * out["overstock"]
    out["stockout"] = (out["understock"] > 0).astype(int)
    return out


def per_series(sim: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-week sim to one row per series."""
    g = sim.groupby("id")
    return pd.DataFrame({
        "cost": g["cost"].sum(),
        "understock_units": g["understock"].sum(),
        "overstock_units": g["overstock"].sum(),
        "stockout_rate": g["stockout"].mean(),
        "demand": g["y"].sum(),
    }).reset_index()


def stratified_assignment(units: pd.DataFrame, strata_cols: list[str],
                          seed: int = 42) -> pd.DataFrame:
    """50/50 assignment within each stratum -> df[id, arm]."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, g in units.groupby(strata_cols, observed=True):
        ids = g["id"].to_numpy().copy(); rng.shuffle(ids)
        k = len(ids) // 2
        treat = set(ids[:k])
        for sid in g["id"]:
            rows.append({"id": sid, "arm": "treatment" if sid in treat else "control"})
    return pd.DataFrame(rows)


def mde(n_per_arm: int, sigma: float, alpha=0.05, power=0.8) -> float:
    """Two-sample MDE (absolute units): (z_a/2 + z_b) * sigma * sqrt(2/n)."""
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    return float((za + zb) * sigma * np.sqrt(2.0 / n_per_arm))


@dataclass
class TestResult:
    n: int
    mean_control: float
    mean_treatment: float
    pct_change: float
    p_value: float
    ci_low: float
    ci_high: float
    test: str
    extra: dict


def unpaired_cost_test(control_costs, treatment_costs, seed=0, nb=2000) -> TestResult:
    c = np.asarray(control_costs, float); t = np.asarray(treatment_costs, float)
    _, p = stats.mannwhitneyu(t, c, alternative="two-sided")
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(t, len(t), True).mean() - rng.choice(c, len(c), True).mean()
                      for _ in range(nb)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mc = c.mean()
    return TestResult(min(len(c), len(t)), float(mc), float(t.mean()),
                      float((t.mean() - mc) / mc * 100) if mc else float("nan"),
                      float(p), float(lo), float(hi), "mann_whitney_u",
                      {"mde_pct": mde(min(len(c), len(t)), c.std()) / mc * 100 if mc else np.nan})


def paired_cost_test(champ_costs, chall_costs, seed=0, nb=2000) -> TestResult:
    """Paired: every series has both costs. d = challenger - champion (<0 = cheaper)."""
    a = np.asarray(champ_costs, float); b = np.asarray(chall_costs, float)
    assert len(a) == len(b)
    d = b - a
    try:
        _, p = stats.wilcoxon(d, alternative="two-sided")
    except ValueError:
        p = float("nan")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(d))
    boots = np.array([d[rng.choice(idx, len(d), True)].mean() for _ in range(nb)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ma = a.mean()
    return TestResult(len(d), float(ma), float(b.mean()),
                      float(d.mean() / ma * 100) if ma else float("nan"),
                      float(p), float(lo), float(hi), "wilcoxon_signed_rank",
                      {"pct_series_cheaper": float((d < 0).mean() * 100),
                       "median_diff": float(np.median(d))})


def two_proportion_z(p1, n1, p2, n2) -> dict:
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    return {"lift_pp": (p2 - p1) * 100, "z": float(z),
            "p_value": float(2 * (1 - stats.norm.cdf(abs(z))))}
