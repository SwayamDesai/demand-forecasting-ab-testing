"""
Baseline forecasters (Phase 2). All return a predictions DataFrame:
    columns = ['id', 'date', 'y_pred']  with one row per (series, test date).

Models implemented:
  - SeasonalNaive (weekly)                     -- the floor every model must beat
  - AutoETS via statsforecast                  -- fast classical, handles trend+seasonality
  - AutoARIMA via statsforecast (SARIMA d=1,D=1,s=7) -- textbook Box-Jenkins
  - LightGBM global model with recursive multi-step  -- modern ML

All clipped to >= 0 (negative sales are meaningless).

The LightGBM recursive forecast is the only nontrivial bit -- see docstring.
"""
from __future__ import annotations

import warnings
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd


# ---- 1) Seasonal Naive ------------------------------------------------------

def run_seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, season: int = 7) -> pd.DataFrame:
    """
    y_hat[t+k] = y[t+k-season], where t is the last train date.

    If the lookup date is missing for a series (e.g., series started recently),
    we fall back to that series' training-period mean.
    """
    # last `season` days of training, one row per (id, date)
    last_dates = pd.date_range(end=train["date"].max(), periods=season, freq="D")
    template = (train[train["date"].isin(last_dates)]
                .loc[:, ["id", "date", "sales"]]
                .rename(columns={"sales": "y_lookup"}))
    # offset dates by `season` days -> they line up with the test horizon
    template["date"] = template["date"] + pd.Timedelta(days=season)

    # repeat for as many full weeks as needed to cover test horizon
    n_weeks = int(np.ceil(test["date"].nunique() / season)) + 1
    blocks = [template.assign(date=lambda d, off=i: d["date"] + pd.Timedelta(days=off * season))
              for i in range(n_weeks)]
    expanded = pd.concat(blocks, ignore_index=True)

    out = test[["id", "date"]].merge(expanded, on=["id", "date"], how="left")
    # series-mean fallback
    means = train.groupby("id")["sales"].mean().rename("y_mean")
    out = out.merge(means, on="id", how="left")
    out["y_pred"] = out["y_lookup"].fillna(out["y_mean"]).clip(lower=0)
    return out[["id", "date", "y_pred"]]


# ---- 2) statsforecast: AutoETS + AutoARIMA ----------------------------------

def _statsforecast_predict(train: pd.DataFrame, test: pd.DataFrame, model) -> pd.DataFrame:
    """Generic wrapper around a single statsforecast model object."""
    from statsforecast import StatsForecast

    sf_train = (train.rename(columns={"id": "unique_id", "date": "ds", "sales": "y"})
                     [["unique_id", "ds", "y"]])
    horizon = test["date"].nunique()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf = StatsForecast(models=[model], freq="D", n_jobs=-1)
        sf.fit(df=sf_train)
        fc = sf.predict(h=horizon)
    fc = fc.rename(columns={"unique_id": "id", "ds": "date"})
    model_col = [c for c in fc.columns if c not in ("id", "date")][0]
    out = test[["id", "date"]].merge(fc[["id", "date", model_col]], on=["id", "date"], how="left")
    out["y_pred"] = out[model_col].clip(lower=0).fillna(0.0)
    return out[["id", "date", "y_pred"]]


def run_ets(train, test):
    from statsforecast.models import AutoETS
    return _statsforecast_predict(train, test, AutoETS(season_length=7))


def run_arima(train, test):
    from statsforecast.models import AutoARIMA
    # constrain search space so 300 series x 3 folds doesn't run forever
    model = AutoARIMA(season_length=7, max_p=2, max_q=2, max_P=1, max_Q=1, d=1, D=1)
    return _statsforecast_predict(train, test, model)


# ---- 3) LightGBM (global model, recursive multi-step) -----------------------

LGB_FEATURE_COLS_NONLAG = [
    "sell_price", "price_change_pct", "price_ratio_28d",
    "wday", "month", "year", "wday_sin", "wday_cos",
    "has_snap", "has_event", "is_weekend", "snap_x_food",
    "cat_id_code", "dept_id_code", "item_id_code",
]
LGB_FEATURE_COLS_LAG = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rmean_7", "sales_rstd_7", "sales_rmean_28", "sales_rstd_28",
]
LGB_FEATURE_COLS = LGB_FEATURE_COLS_NONLAG + LGB_FEATURE_COLS_LAG
LGB_CATEGORICAL = ["cat_id_code", "dept_id_code", "item_id_code",
                   "wday", "month", "has_snap", "has_event", "snap_x_food"]


def _encode_categoricals(df: pd.DataFrame, lookups: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Stable integer encoding for cat_id, dept_id, item_id."""
    df = df.copy()
    if lookups is None:
        lookups = {}
        for col in ("cat_id", "dept_id", "item_id"):
            uniq = sorted(df[col].astype(str).unique())
            lookups[col] = {v: i for i, v in enumerate(uniq)}
    for col in ("cat_id", "dept_id", "item_id"):
        df[f"{col}_code"] = df[col].astype(str).map(lookups[col]).astype("int32")
    return df, lookups


def _lag_features_recursive(
    work: pd.DataFrame, target_date: pd.Timestamp, sales_col: str = "sales_work"
) -> pd.DataFrame:
    """
    Compute lag + rolling features for rows where date == target_date,
    using `sales_col` (which holds actuals for train + already-predicted future).
    Returns a frame with one row per id, indexed by id.
    """
    out_rows = work.loc[work["date"] == target_date, ["id"]].set_index("id")

    # pure lags
    for lag in (1, 7, 14, 28):
        lag_d = target_date - pd.Timedelta(days=lag)
        vals = (work.loc[work["date"] == lag_d, ["id", sales_col]]
                    .set_index("id")[sales_col]
                    .rename(f"sales_lag_{lag}"))
        out_rows = out_rows.join(vals)

    # rolling mean/std over [t-window .. t-1]
    for w in (7, 28):
        start = target_date - pd.Timedelta(days=w)
        end   = target_date - pd.Timedelta(days=1)
        past = work[(work["date"] >= start) & (work["date"] <= end)]
        agg = past.groupby("id")[sales_col].agg(["mean", "std"])
        agg.columns = [f"sales_rmean_{w}", f"sales_rstd_{w}"]
        out_rows = out_rows.join(agg)

    return out_rows


def run_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    full_aux: pd.DataFrame,
    n_estimators: int = 500,
) -> pd.DataFrame:
    """
    Global LightGBM with recursive multi-step prediction.

    Args:
      train     -- rows with date <= train_end and is_active=True, including the
                   pre-built features (lag/rolling/calendar/price)
      test      -- rows with test_start <= date <= test_end and is_active=True
                   (used only for the id/date pairs we must predict)
      full_aux  -- the FULL long frame with id, date, sales, sell_price, calendar,
                   etc. (no features needed) -- this is what we 'walk' during
                   recursive forecasting.
    """
    train, lookups = _encode_categoricals(train)

    # train target uses raw 'sales'; features were pre-built upstream.
    # drop any row with NaN in feature cols (early warm-up rows have NaN lags)
    keep = train.dropna(subset=LGB_FEATURE_COLS)
    X = keep[LGB_FEATURE_COLS]
    y = keep["sales"]

    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.1,
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, feature_fraction=0.9, bagging_fraction=0.8,
        bagging_freq=1, verbosity=-1, n_jobs=-1, random_state=0,
    )
    model.fit(X, y, categorical_feature=LGB_CATEGORICAL)

    # ---- recursive forecast ----
    # build a working frame that holds true sales up to train_end, NaN beyond
    work = (full_aux.sort_values(["id", "date"])
                    .loc[:, ["id", "date", "sales", "sell_price",
                             "cat_id", "dept_id", "item_id",
                             "wday", "month", "year",
                             "has_snap", "has_event", "is_weekend"]]
                    .copy())
    train_end = train["date"].max()
    work["sales_work"] = np.where(work["date"] <= train_end, work["sales"], np.nan)

    # encode + cyclic + interaction up-front (they don't depend on lags)
    work, _ = _encode_categoricals(work, lookups=lookups)
    work["wday_sin"] = np.sin(2 * np.pi * work["wday"] / 7)
    work["wday_cos"] = np.cos(2 * np.pi * work["wday"] / 7)
    work["snap_x_food"] = (work["has_snap"] * (work["cat_id"] == "FOODS").astype(int)).astype("int8")
    # price-derived (computed once; uses sell_price which is known in advance)
    g = work.groupby("id", sort=False)["sell_price"]
    work["price_change_pct"] = g.pct_change()
    work["price_ratio_28d"] = (
        work["sell_price"]
        / g.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    )

    test_dates = sorted(test["date"].unique())
    all_preds = []
    for d in test_dates:
        d = pd.Timestamp(d)
        lag_feats = _lag_features_recursive(work, d, sales_col="sales_work")
        rows = (work[work["date"] == d]
                .set_index("id")
                .join(lag_feats, how="left")
                .reset_index())
        # fill remaining NaN feature values (warm-up edges) with 0
        rows[LGB_FEATURE_COLS] = rows[LGB_FEATURE_COLS].fillna(0)
        yhat = np.clip(model.predict(rows[LGB_FEATURE_COLS]), a_min=0, a_max=None)
        rows["y_pred"] = yhat
        all_preds.append(rows[["id", "date", "y_pred"]])
        # feed predictions back as future "actuals" for next iteration
        pred_map = dict(zip(rows["id"], yhat))
        mask = work["date"] == d
        work.loc[mask, "sales_work"] = work.loc[mask, "id"].map(pred_map)

    return pd.concat(all_preds, ignore_index=True)


# ---- registry ---------------------------------------------------------------

BASELINES: dict[str, Callable] = {
    "seasonal_naive": "naive",      # special; needs only train+test
    "ets":            "stats",      # statsforecast
    "arima":          "stats",
    "lightgbm":       "lgb",
}
