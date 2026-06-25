"""
Real assertions for src.features -- with THE critical no-future-leakage guard.

The leakage test is the most important one in this whole project. If it fails,
your offline metrics are a lie and the model dies in production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_calendar_features,
    add_lag_features,
    add_price_features,
    add_rolling_features,
    build_features,
)


# ---- fixtures ---------------------------------------------------------------

def _toy(n_per_id: int = 30, n_ids: int = 2, start: str = "2020-01-01") -> pd.DataFrame:
    """A tiny, fully-controlled long dataframe."""
    rows = []
    for sid in range(n_ids):
        dates = pd.date_range(start, periods=n_per_id, freq="D")
        rows.append(pd.DataFrame({
            "id": f"sku_{sid}",
            "date": dates,
            "sales": np.arange(1, n_per_id + 1, dtype=float),
            "sell_price": 10.0 + 0.1 * np.arange(n_per_id),
            "cat_id": "FOODS",
            "has_snap": (np.arange(n_per_id) % 4 == 0).astype(int),
            "wday": dates.dayofweek.values.astype(int) + 1,
            "month": dates.month.values.astype(int),
            "year": dates.year.values.astype(int),
            "has_event": 0,
            "is_weekend": 0,
            "is_active": True,
        }))
    return pd.concat(rows, ignore_index=True)


# ---- atomic feature tests ---------------------------------------------------

def test_lag_features_have_correct_values_and_nans():
    df = add_lag_features(_toy(n_per_id=10, n_ids=1), lags=(1, 7))
    sub = df[df["id"] == "sku_0"].reset_index(drop=True)
    # sales = 1..10 ; lag_1 should be [NaN, 1, 2, ..., 9]
    assert sub["sales_lag_1"].iloc[0] != sub["sales_lag_1"].iloc[0]  # NaN
    assert (sub["sales_lag_1"].iloc[1:].values == sub["sales"].iloc[:-1].values).all()
    # lag_7: first 7 are NaN, then [1, 2, 3]
    assert sub["sales_lag_7"].iloc[:7].isna().all()
    assert (sub["sales_lag_7"].iloc[7:].values == np.arange(1, 4)).all()


def test_rolling_mean_is_shifted_and_correct():
    # rmean_3 with shift=1 for sales=[1..10]:
    # row t uses sales[t-3..t-1] -> NaN, 1, 1.5, 2, 3, 4, 5, 6, 7, 8
    df = add_rolling_features(_toy(n_per_id=10, n_ids=1), windows=(3,))
    sub = df[df["id"] == "sku_0"].reset_index(drop=True)
    expected = [np.nan, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    got = sub["sales_rmean_3"].tolist()
    assert np.isnan(got[0]) and np.isnan(expected[0])
    np.testing.assert_allclose(got[1:], expected[1:])


# ---- THE leakage guard ------------------------------------------------------

def test_no_future_leakage_in_any_lag_or_rolling_feature():
    """
    For EVERY row in the engineered frame, EVERY engineered sales-derived feature
    must be a function of rows strictly EARLIER than that row.

    Procedure (works regardless of feature internals):
      1. Compute features on the full series.
      2. For each cut date c, recompute features using only sales[date <= c-1]
         padded with NaN for date >= c (i.e., 'as if' future were unknown).
      3. The features for rows with date < c must match between the two runs.
         If any future-derived feature exists, this will diverge.
    """
    df = _toy(n_per_id=40, n_ids=2)

    full = add_rolling_features(add_lag_features(df.copy(), lags=(1, 7, 14)),
                                windows=(7, 28))
    feat_cols = [c for c in full.columns if c.startswith("sales_")]

    cut = pd.Timestamp("2020-01-25")
    masked = df.copy()
    masked.loc[masked["date"] >= cut, "sales"] = np.nan
    rebuilt = add_rolling_features(add_lag_features(masked, lags=(1, 7, 14)),
                                   windows=(7, 28))

    before_cut = full["date"] < cut
    for col in feat_cols:
        a = full.loc[before_cut, col].to_numpy()
        b = rebuilt.loc[before_cut, col].to_numpy()
        # both NaN counts as equal; otherwise must match exactly
        equal = (np.isnan(a) & np.isnan(b)) | (a == b)
        assert equal.all(), f"leakage detected in {col}"


# ---- price + calendar -------------------------------------------------------

def test_price_features_run_without_error():
    df = add_price_features(_toy())
    assert {"price_change_pct", "price_ratio_28d"}.issubset(df.columns)
    # first row of each id has no previous price -> NaN
    first_rows = df.groupby("id").head(1)
    assert first_rows["price_change_pct"].isna().all()


def test_calendar_features_cyclic_encoding_bounds():
    df = add_calendar_features(_toy())
    assert df["wday_sin"].between(-1.0, 1.0).all()
    assert df["wday_cos"].between(-1.0, 1.0).all()
    assert df["snap_x_food"].dtype.kind in ("i", "u")


def test_build_features_pipeline_adds_expected_columns():
    out = build_features(_toy())
    expected = {
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "sales_rmean_7", "sales_rstd_7", "sales_rmean_28", "sales_rstd_28",
        "price_change_pct", "price_ratio_28d",
        "wday_sin", "wday_cos", "snap_x_food",
    }
    assert expected.issubset(out.columns)
