"""The leakage guard: features for week t may use only weeks strictly before t.

Two checks on the Phase-2 feature builder:
  1. lags/rolling stats match a from-scratch recomputation, and
  2. perturbing sales at week t changes NO feature at week t (the leak test).
"""
import numpy as np
import pandas as pd
import pytest

from scripts.phase2_preprocessing import add_features

FEATURE_COLS = [
    "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_8", "sales_lag_52",
    "sales_rmean_4", "sales_rstd_4", "sales_rmean_8", "sales_rstd_8",
    "sales_rmean_13", "sales_rstd_13", "trailing_zero_rate_8",
    "weeks_since_last_sale", "price_change_pct", "price_ratio_8w",
]


def _toy_panel(n_weeks=60, seed=3):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2015-01-02", periods=n_weeks, freq="7D")
    frames = []
    for sid in ["a", "b"]:
        frames.append(pd.DataFrame({
            "id": sid,
            "week_end_date": weeks,
            "sales": rng.poisson(4, n_weeks),
            "sell_price": 3.0 + rng.normal(0, 0.1, n_weeks).round(2),
        }))
    return pd.concat(frames, ignore_index=True)


def test_lags_and_rolling_match_manual_recompute():
    w = add_features(_toy_panel())
    g = w[w["id"] == "a"].sort_values("week_end_date")
    exp_lag1 = g["sales"].shift(1)
    exp_rmean4 = g["sales"].shift(1).rolling(4, min_periods=1).mean()
    assert np.allclose(g["sales_lag_1"].dropna(), exp_lag1.dropna())
    assert np.allclose(g["sales_rmean_4"].dropna(), exp_rmean4.dropna())


def test_no_future_leakage_current_week_cannot_influence_its_features():
    base = _toy_panel()
    feats_before = add_features(base.copy())

    # blow up sales in the LAST week of series 'a' -- a huge, unmissable change
    poked = base.copy()
    last_ix = poked[poked["id"] == "a"].index[-1]
    poked.loc[last_ix, "sales"] = 10_000

    feats_after = add_features(poked)

    # the features AT the poked week must be identical: they may only use past weeks
    a_before = feats_before[(feats_before["id"] == "a")].iloc[-1][FEATURE_COLS]
    a_after = feats_after[(feats_after["id"] == "a")].iloc[-1][FEATURE_COLS]
    pd.testing.assert_series_equal(a_before, a_after, check_names=False)

    # and the OTHER series is untouched everywhere
    b_before = feats_before[feats_before["id"] == "b"][FEATURE_COLS].reset_index(drop=True)
    b_after = feats_after[feats_after["id"] == "b"][FEATURE_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(b_before, b_after)


def test_leak_would_be_caught():
    """Sanity that the guard has teeth: a deliberately leaky feature DOES differ."""
    base = _toy_panel()
    poked = base.copy()
    last_ix = poked[poked["id"] == "a"].index[-1]
    poked.loc[last_ix, "sales"] = 10_000
    # an unshifted rolling mean (leaky by construction) must change at week t
    leaky_before = base.groupby("id")["sales"].transform(lambda s: s.rolling(4, 1).mean())
    leaky_after = poked.groupby("id")["sales"].transform(lambda s: s.rolling(4, 1).mean())
    assert leaky_before.iloc[last_ix] != leaky_after.iloc[last_ix]
