"""Unit tests for the A/B newsvendor + stats machinery."""
import numpy as np
import pandas as pd
import pytest

from src import experiment as ex


def test_newsvendor_z_symmetric_and_monotonic():
    assert ex.newsvendor_z(1, 1) == pytest.approx(0.0, abs=1e-9)   # tau*=0.5
    assert ex.newsvendor_z(3, 1) == pytest.approx(0.6744, abs=1e-3)
    assert ex.newsvendor_z(9, 1) > ex.newsvendor_z(5, 1) > ex.newsvendor_z(3, 1)


def test_simulate_cost_arithmetic():
    df = pd.DataFrame({"id": ["a", "a"], "y": [10, 2], "pred": [6, 5]})
    sigma = pd.Series({"a": 0.0})                       # no safety stock -> order = round(pred)
    sim = ex.simulate(df, "pred", sigma, cu=5, co=1)
    # week1: demand 10 order 6 -> understock 4, cost 20 ; week2: demand 2 order 5 -> overstock 3, cost 3
    assert list(sim["understock"]) == [4, 0]
    assert list(sim["overstock"]) == [0, 3]
    assert list(sim["cost"]) == [20, 3]
    assert list(sim["stockout"]) == [1, 0]


def test_per_series_aggregation():
    sim = pd.DataFrame({"id": ["a", "a"], "cost": [20, 3], "understock": [4, 0],
                        "overstock": [0, 3], "stockout": [1, 0], "y": [10, 2]})
    ps = ex.per_series(sim)
    assert ps.loc[0, "cost"] == 23
    assert ps.loc[0, "stockout_rate"] == 0.5


def test_paired_test_detects_cheaper_challenger():
    champ = np.array([100, 120, 80, 90, 110], float)
    chall = champ - 10                                  # uniformly cheaper
    r = ex.paired_cost_test(champ, chall)
    assert r.pct_change < 0
    assert r.extra["pct_series_cheaper"] == 100.0
    assert r.ci_high < 0                                # CI excludes zero


def test_mde_positive():
    assert ex.mde(100, 50.0) > 0


def test_simulate_zero_sigma_orders_equal_rounded_prediction():
    """Phase 8b treatment: with empty/zero sigma the order IS the (rounded) forecast."""
    df = pd.DataFrame({"id": ["a", "a", "b"], "y": [5, 0, 3], "p": [4.4, 1.6, 3.0]})
    sim = ex.simulate(df, "p", pd.Series(dtype=float), cu=5, co=1)
    assert sim["order"].tolist() == [4.0, 2.0, 3.0]          # round(pred), no safety stock
    assert sim.loc[0, "understock"] == 1 and sim.loc[0, "cost"] == 5.0
    assert sim.loc[1, "overstock"] == 2 and sim.loc[1, "cost"] == 2.0
