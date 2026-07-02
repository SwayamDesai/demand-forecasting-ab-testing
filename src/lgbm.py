"""
LightGBM helpers: feature list, per-store fitting (Tweedie mean + pinball quantile),
and recursive multi-step forecasting at the weekly grain.

Design (the pattern the M5 winners used):
  * one model per store -- each store frame is small enough for a 16GB machine and
    store-level demand patterns get their own model;
  * Tweedie objective for the MEAN model (zero-inflation + multiplicative variance);
  * quantile (pinball) objective for the ORDER model -- it predicts the newsvendor
    tau-quantile of demand directly, which IS the order quantity;
  * recursive forecasting feeds back the MEAN model's predictions to build lag
    features over the horizon (feeding back a high quantile would inflate the lags).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

LAG_ROLL = ["sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_8", "sales_lag_52",
            "sales_rmean_4", "sales_rstd_4", "sales_rmean_8", "sales_rstd_8",
            "sales_rmean_13", "sales_rstd_13", "trailing_zero_rate_8", "weeks_since_last_sale"]
EXOG = ["sell_price", "price_change_pct", "price_ratio_8w",
        "month", "weekofyear", "woy_sin", "woy_cos", "snap_days", "event_days"]
STATIC = ["cat_code", "dept_code", "store_code", "item_code"]
FEATURES = LAG_ROLL + EXOG + STATIC
CATEGORICAL = ["month", "cat_code", "dept_code", "store_code", "item_code"]

LAGS = (1, 2, 4, 8, 52)
ROLLS = (4, 8, 13)

_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
               min_child_samples=40, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.9, random_state=0, n_jobs=-1, verbosity=-1)


def tau_name(t: float) -> str:
    return f"q{t:.3f}".rstrip("0").rstrip(".")


def add_codes(df: pd.DataFrame, lookups: dict) -> pd.DataFrame:
    df = df.copy()
    for col, code in [("cat_id", "cat_code"), ("dept_id", "dept_code"),
                      ("store_id", "store_code"), ("item_id", "item_code")]:
        df[code] = df[col].astype(str).map(lookups[col]).astype("int32")
    return df


def fit_mean(train: pd.DataFrame) -> lgb.LGBMRegressor:
    """Tweedie mean model -- drives the accuracy leaderboard + recursion feed."""
    model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, **_PARAMS)
    model.fit(train[FEATURES], train["sales"], categorical_feature=CATEGORICAL)
    return model


def fit_quantile(train: pd.DataFrame, tau: float) -> lgb.LGBMRegressor:
    """Pinball-loss model predicting the tau demand quantile = the order quantity."""
    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, **_PARAMS)
    model.fit(train[FEATURES], train["sales"], categorical_feature=CATEGORICAL)
    return model


def sales_features_at(work: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    """Recompute the sales-derived features for every series at week `d` from the
    working sales column (actuals up to cutoff + predictions after)."""
    out = work.loc[work["week_end_date"] == d, ["id"]].copy()

    for k in LAGS:
        ref = (work.loc[work["week_end_date"] == d - pd.Timedelta(weeks=k), ["id", "sales_work"]]
                   .rename(columns={"sales_work": f"sales_lag_{k}"}))
        out = out.merge(ref, on="id", how="left")

    for wdw in ROLLS:
        lo, hi = d - pd.Timedelta(weeks=wdw), d - pd.Timedelta(weeks=1)
        win = work[(work["week_end_date"] >= lo) & (work["week_end_date"] <= hi)]
        agg = win.groupby("id")["sales_work"].agg(["mean", "std"])
        agg.columns = [f"sales_rmean_{wdw}", f"sales_rstd_{wdw}"]
        out = out.merge(agg, on="id", how="left")

    win8 = work[(work["week_end_date"] >= d - pd.Timedelta(weeks=8)) &
                (work["week_end_date"] <= d - pd.Timedelta(weeks=1))]
    tzr = (win8.assign(z=(win8["sales_work"] == 0).astype(float))
                .groupby("id")["z"].mean().rename("trailing_zero_rate_8"))
    out = out.merge(tzr, on="id", how="left")

    hist = work[(work["week_end_date"] < d) &
                (work["week_end_date"] >= d - pd.Timedelta(weeks=104))]
    last_sale = (hist[hist["sales_work"] > 0].groupby("id")["week_end_date"].max())
    wsls = ((d - last_sale).dt.days // 7).rename("weeks_since_last_sale")
    out = out.merge(wsls, on="id", how="left")
    return out


def recursive_forecast(model, work, exog_codes, test_weeks) -> pd.DataFrame:
    """Walk the horizon week by week, feeding predictions back as future 'sales'."""
    preds = []
    for d in sorted(test_weeks):
        d = pd.Timestamp(d)
        sfeat = sales_features_at(work, d)
        row = exog_codes[exog_codes["week_end_date"] == d].merge(sfeat, on="id", how="left")
        yhat = np.clip(model.predict(row[FEATURES]), 0, None)
        pred_map = dict(zip(row["id"], yhat))
        work.loc[work["week_end_date"] == d, "sales_work"] = \
            work.loc[work["week_end_date"] == d, "id"].map(pred_map)
        preds.append(pd.DataFrame({"id": row["id"].values, "week_end_date": d, "y_pred": yhat}))
    return pd.concat(preds, ignore_index=True)
