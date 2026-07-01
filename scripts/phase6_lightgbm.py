"""
Phase 6 -- LightGBM ML challenger.

One GLOBAL gradient-boosted model over all 900 series. Choices follow Phase 4:
  - Tweedie objective (variance_power=1.2): handles zero-inflation + the
    multiplicative mean-variance relationship we measured (slope 0.74).
  - Features = the Phase-2 weekly features (lags 1/2/4/8/52, trailing mean/std
    4/8/13, intermittency, price, calendar incl. snap_days COUNT + week-of-year)
    plus static item/dept/store/category codes so one model can tell series apart.
  - Forecasting = RECURSIVE multi-step over the 4-week horizon (same protocol as
    the classical baselines -> the Phase-8 A/B is apples-to-apples). Only the
    sales-derived features are recomputed from predictions each step; exogenous
    (price/calendar) are known in advance and reused. LightGBM consumes NaN
    natively, so warm-up lags need no imputation.
  - Rolling-origin: refit per fold on data up to that fold's cutoff. No leakage.

Evaluated on the IDENTICAL (series, week) cells as Phase 5 (read from its
predictions.parquet), so WMAPE/bias are directly comparable to the champion.

Outputs: reports/phase6_lightgbm/*.png + *.csv + predictions.parquet + PHASE6_SUMMARY.md
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import backtest, config, metrics

OUT = config.REPORTS / "phase6_lightgbm"

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


def add_codes(df: pd.DataFrame, lookups: dict) -> pd.DataFrame:
    df = df.copy()
    for col, code in [("cat_id", "cat_code"), ("dept_id", "dept_code"),
                      ("store_id", "store_code"), ("item_id", "item_code")]:
        df[code] = df[col].astype(str).map(lookups[col]).astype("int32")
    return df


def sales_features_at(work: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    """Recompute the sales-derived features for every series at week `d`
    from the working sales column (actuals up to cutoff + predictions after)."""
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


def fit_fold(train: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.2,
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.9, random_state=0, n_jobs=-1, verbosity=-1)
    model.fit(train[FEATURES], train["sales"], categorical_feature=CATEGORICAL)
    return model


def recursive_forecast(model, work, exog_codes, test_weeks) -> pd.DataFrame:
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")

    w = pd.read_parquet(config.PROCESSED / "weekly_features.parquet")
    champ_pred = pd.read_parquet(config.REPORTS / "phase5_baselines" / "predictions.parquet")
    lb5 = pd.read_csv(config.REPORTS / "phase5_baselines" / "leaderboard_summary.csv")
    prof = pd.read_csv(config.REPORTS / "phase3_eda" / "series_profile.csv")
    prof["volume_tier"] = pd.qcut(prof["total_units"], 3, labels=["low", "mid", "high"])
    champion = lb5.iloc[0]["model"]

    # stable static code lookups (fit once on all data)
    lookups = {c: {v: i for i, v in enumerate(sorted(w[c].astype(str).unique()))}
               for c in ["cat_id", "dept_id", "store_id", "item_id"]}
    w = add_codes(w, lookups)

    folds = backtest.make_folds(w["week_end_date"], config.HORIZON_WEEKS, config.N_FOLDS)
    exog_codes = w[["id", "week_end_date"] + EXOG + STATIC].copy()

    all_preds = []
    for f in folds:
        print(f"[{f.name}] fit on weeks <= {f.train_end.date()} ...")
        train = w[w["week_end_date"] <= f.train_end]
        model = fit_fold(train)
        # working frame: actual sales up to cutoff, NaN after (to be filled recursively)
        work = w[["id", "week_end_date", "sales"]].copy()
        work["sales_work"] = np.where(work["week_end_date"] <= f.train_end, work["sales"], np.nan)
        test_weeks = w.loc[(w["week_end_date"] >= f.test_start) &
                           (w["week_end_date"] <= f.test_end), "week_end_date"].unique()
        fp = recursive_forecast(model, work, exog_codes, test_weeks)
        fp["fold"] = f.name
        all_preds.append(fp)
    preds = pd.concat(all_preds, ignore_index=True)

    # align to the exact champion test cells + actuals
    truth = champ_pred[["id", "week_end_date", "fold", "y", champion]].rename(
        columns={champion: "champ_pred"})
    m = truth.merge(preds[["id", "week_end_date", "y_pred"]], on=["id", "week_end_date"], how="left")
    assert m["y_pred"].notna().all(), "LightGBM missed some test cells"

    # MASE scale (same as Phase 5)
    first_test = folds[0].test_start
    scales = (w[w["week_end_date"] < first_test].groupby("id")["sales"]
              .apply(lambda s: metrics.seasonal_naive_scale(s.to_numpy(), config.SEASON_WEEKS)))

    def row_metrics(name, yhat):
        per = (m.assign(ae=np.abs(m["y"] - yhat)).groupby("id")["ae"].mean())
        mase_s = (per / scales).replace([np.inf, -np.inf], np.nan).dropna()
        return dict(model=name, wmape=metrics.wmape(m["y"], yhat),
                    rmse=metrics.rmse(m["y"], yhat), bias_pct=metrics.bias_pct(m["y"], yhat),
                    mase_median=float(mase_s.median()))

    lb = pd.DataFrame([row_metrics("lightgbm", m["y_pred"]),
                       row_metrics(f"{champion} (champion)", m["champ_pred"])])
    lb.to_csv(OUT / "leaderboard_summary.csv", index=False)
    m.to_parquet(OUT / "predictions.parquet", index=False)

    # per-segment WMAPE (lightgbm vs champion)
    seg_rows = []
    ms = m.merge(prof[["id", "cat_id", "sbc_class", "volume_tier"]], on="id", how="left")
    for seg_col in ["cat_id", "sbc_class", "volume_tier"]:
        for val, g in ms.groupby(seg_col):
            seg_rows.append(dict(segment=seg_col, value=val,
                                 lightgbm=metrics.wmape(g["y"], g["y_pred"]),
                                 champion=metrics.wmape(g["y"], g["champ_pred"])))
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(OUT / "wmape_by_segment.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(lb["model"], lb["wmape"], color=["#DD8452", "#55A868"])
    for i, (v, b) in enumerate(zip(lb["wmape"], lb["bias_pct"])):
        ax.text(v, i, f"  {v:.4f} (bias {b:+.0f}%)", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("WMAPE"); ax.set_title("LightGBM vs champion")
    fig.tight_layout(); fig.savefig(OUT / "01_lgbm_vs_champion.png", dpi=110); plt.close(fig)

    sc = seg[seg["segment"] == "sbc_class"].set_index("value")[["lightgbm", "champion"]]
    fig, ax = plt.subplots(figsize=(7.5, 4)); sc.plot(kind="bar", ax=ax)
    ax.set_ylabel("WMAPE"); ax.set_title("WMAPE by intermittency class"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(OUT / "02_segment_wmape.png", dpi=110); plt.close(fig)

    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()[-15:]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(fi.index, fi.values, color="#4C72B0")
    ax.set_title("LightGBM feature importance (last fold, top 15)")
    fig.tight_layout(); fig.savefig(OUT / "03_feature_importance.png", dpi=110); plt.close(fig)

    lgbm_w = lb.iloc[0]["wmape"]; champ_w = lb.iloc[1]["wmape"]
    verdict = "BEATS" if lgbm_w < champ_w else "does NOT beat"
    md = OUT / "PHASE6_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 6 -- LightGBM ML Challenger\n\n")
        f.write(f"Global Tweedie LightGBM, recursive 4-week forecast, evaluated on the same "
                f"{len(m):,} test cells as Phase 5.\n\n")
        f.write(f"## Verdict: LightGBM **{verdict}** the `{champion}` champion on WMAPE "
                f"({lgbm_w:.4f} vs {champ_w:.4f}, {(lgbm_w/champ_w-1)*100:+.1f}%).\n\n")
        f.write("| model | WMAPE | RMSE | bias % | MASE (med) |\n|---|---|---|---|---|\n")
        for _, r in lb.iterrows():
            f.write(f"| {r['model']} | {r['wmape']:.4f} | {r['rmse']:.3f} | "
                    f"{r['bias_pct']:+.1f}% | {r['mase_median']:.3f} |\n")
        f.write("\n## WMAPE by segment\n\n| segment | value | lightgbm | champion |\n|---|---|---|---|\n")
        for _, r in seg.iterrows():
            f.write(f"| {r['segment']} | {r['value']} | {r['lightgbm']:.3f} | {r['champion']:.3f} |\n")
        f.write("\n## Outputs\n\n- `predictions.parquet` (aligned to champion cells)\n"
                "- `leaderboard_summary.csv`, `wmape_by_segment.csv`\n- figures 01-03\n")

    print("\n=== Phase 6 result ===")
    print(lb.to_string(index=False))
    print(f"\nLightGBM {verdict} champion ({champion}): "
          f"WMAPE {lgbm_w:.4f} vs {champ_w:.4f} ({(lgbm_w/champ_w-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
