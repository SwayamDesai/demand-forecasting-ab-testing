"""
Phase 5 -- Time-series baselines (the champion bake-off).

Runs the classical contenders through ONE rolling-origin backtest (4-week horizon,
3 folds) and ranks them. The winner is the **champion** every later model -- LightGBM
(Phase 6), LSTM (Phase 7) -- must beat, and the control arm in the A/B test (Phase 8).

Models (all via statsforecast's tested implementations -- we don't re-derive Croston):
  naive_last          Naive()                      -- last value carried forward
  seasonal_naive_52   SeasonalNaive(52)            -- same week last year (the floor)
  moving_average_4/8  WindowAverage(4/8)           -- short trailing means
  croston_sba         CrostonSBA()                 -- intermittent-demand specialist
  ets                 AutoETS(season_length=52)    -- classical trend+seasonal
  arima               AutoARIMA(season_length=52)  -- Box-Jenkins (constrained search)

Why these: Phase 4 found (a) low-order AR + a yearly echo -> seasonal-naive & ETS,
(b) 377 intermittent/lumpy SKUs -> Croston, (c) moderate per-SKU seasonality -> don't
over-rely on seasonal terms. Predictions are clipped to >=0 (negative demand is
meaningless). The exact (series, week) test cells are saved so Phases 6-8 compare
on identical ground.

Outputs: reports/phase5_baselines/*.png + *.csv + predictions.parquet + PHASE5_SUMMARY.md
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (AutoARIMA, AutoETS, CrostonSBA, Naive,
                                  SeasonalNaive, WindowAverage)

from src import backtest, config, metrics

OUT = config.REPORTS / "phase5_baselines"
RUN_ARIMA = True            # toggled off if it proves too slow at full scale
FORCE_CV = False            # True = refit from scratch; False = reuse cached predictions.parquet

MODEL_RENAME = {
    "Naive": "naive_last",
    "SeasonalNaive": "seasonal_naive_52",
    "WindowAverage": "moving_average_4",
    "WindowAverage_4": "moving_average_4",
    "WindowAverage_8": "moving_average_8",
    "CrostonSBA": "croston_sba",
    "AutoETS": "ets",
    "AutoARIMA": "arima",
}


def build_models():
    models = [
        Naive(),
        SeasonalNaive(season_length=config.SEASON_WEEKS),
        WindowAverage(window_size=4, alias="moving_average_4"),
        WindowAverage(window_size=8, alias="moving_average_8"),
        CrostonSBA(),
        AutoETS(season_length=config.SEASON_WEEKS),
    ]
    if RUN_ARIMA:
        # Non-seasonal Box-Jenkins: seasonality is already covered by
        # SeasonalNaive(52) and AutoETS(52); a seasonal m=52 search is the runtime
        # hog and rarely wins on intermittent weekly retail. Phase 4 showed
        # low-order AR structure, which a non-seasonal ARIMA captures cheaply.
        models.append(AutoARIMA(
            season_length=1, max_p=3, max_q=3, max_d=2,
            stepwise=True, alias="arima"))
    return models


def tidy_cv(cv: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalise statsforecast cv output; rename model columns to our names."""
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv.copy()
    cv = cv.rename(columns={"unique_id": "id", "ds": "week_end_date"})
    reserved = {"id", "week_end_date", "cutoff", "y"}
    model_cols = [c for c in cv.columns if c not in reserved]
    rename = {c: MODEL_RENAME.get(c, c) for c in model_cols}
    cv = cv.rename(columns=rename)
    model_cols = [rename[c] for c in model_cols]
    # clip negatives -> 0
    for c in model_cols:
        cv[c] = cv[c].clip(lower=0)
    return cv, model_cols


# ---- evaluation -------------------------------------------------------------

def leaderboard(cv: pd.DataFrame, model_cols: list[str],
                scales: pd.Series) -> pd.DataFrame:
    rows = []
    y = cv["y"].to_numpy()
    for m in model_cols:
        p = cv[m].to_numpy()
        # MASE: per series mean-abs-err / series scale, then median across series
        per = (cv.assign(ae=np.abs(cv["y"] - cv[m]))
                 .groupby("id")["ae"].mean())
        mase_series = (per / scales).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(
            model=m,
            wmape=metrics.wmape(y, p),
            rmse=metrics.rmse(y, p),
            bias_pct=metrics.bias_pct(y, p),
            mase_median=float(mase_series.median()),
        ))
    return pd.DataFrame(rows).sort_values("wmape").reset_index(drop=True)


def per_fold(cv, model_cols):
    rows = []
    for fold, g in cv.groupby("fold"):
        for m in model_cols:
            rows.append(dict(fold=fold, model=m,
                             wmape=metrics.wmape(g["y"], g[m])))
    return pd.DataFrame(rows)


def per_segment(cv, model_cols, prof):
    cv = cv.merge(prof[["id", "cat_id", "sbc_class", "volume_tier"]], on="id", how="left")
    rows = []
    for seg_col in ["cat_id", "sbc_class", "volume_tier"]:
        for seg_val, g in cv.groupby(seg_col):
            for m in model_cols:
                rows.append(dict(segment=seg_col, value=seg_val, model=m,
                                 wmape=metrics.wmape(g["y"], g[m])))
    return pd.DataFrame(rows)


# ---- figures ----------------------------------------------------------------

def fig_leaderboard(lb):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#55A868" if i == 0 else "#4C72B0" for i in range(len(lb))]
    ax.barh(lb["model"], lb["wmape"], color=colors)
    for i, (v, b) in enumerate(zip(lb["wmape"], lb["bias_pct"])):
        ax.text(v, i, f"  {v:.3f} (bias {b:+.0f}%)", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("WMAPE (lower is better)")
    ax.set_title("Phase 5 leaderboard -- champion in green")
    fig.tight_layout(); fig.savefig(OUT / "01_leaderboard.png", dpi=110); plt.close(fig)


def fig_per_fold(pf, champ):
    piv = pf.pivot(index="fold", columns="model", values="wmape")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    piv.plot(marker="o", ax=ax)
    ax.set_ylabel("WMAPE"); ax.set_title("WMAPE per fold (stability check)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "02_per_fold.png", dpi=110); plt.close(fig)


def fig_segment(seg, champ):
    sub = seg[(seg["segment"] == "sbc_class") & (seg["model"].isin(
        [champ, "seasonal_naive_52", "croston_sba"]))]
    piv = sub.pivot(index="value", columns="model", values="wmape")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("WMAPE"); ax.set_title("WMAPE by intermittency class")
    ax.tick_params(axis="x", rotation=0); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "03_segment_wmape.png", dpi=110); plt.close(fig)


# ---- driver -----------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")

    w = pd.read_parquet(config.PROCESSED / "weekly_features.parquet")
    prof = pd.read_csv(config.REPORTS / "phase3_eda" / "series_profile.csv")
    prof["volume_tier"] = pd.qcut(prof["total_units"], 3, labels=["low", "mid", "high"])

    sf_df = (w[["id", "week_end_date", "sales"]]
             .rename(columns={"id": "unique_id", "week_end_date": "ds", "sales": "y"}))

    folds = backtest.make_folds(w["week_end_date"], config.HORIZON_WEEKS, config.N_FOLDS)
    cutoff_to_fold = {f.train_end: f.name for f in folds}
    print("folds:")
    for f in folds:
        print(f"  {f.name}: train<= {f.train_end.date()} | test {f.test_start.date()}..{f.test_end.date()}")

    # Cache the (slow ~8 min) statsforecast CV: reuse it on re-runs unless FORCE_CV.
    pred_path = OUT / "predictions.parquet"
    if pred_path.exists() and not FORCE_CV:
        print("loading cached CV predictions (set FORCE_CV=True to refit)...")
        cv = pd.read_parquet(pred_path)
        model_cols = [c for c in cv.columns
                      if c not in ("id", "week_end_date", "cutoff", "y", "fold")]
    else:
        print(f"\nrunning cross-validation ({len(build_models())} models, "
              f"{config.N_FOLDS} folds, h={config.HORIZON_WEEKS})...")
        # fallback_model: a handful of series are too short for AutoETS(52)/AutoARIMA
        # ("tiny datasets") -- statsforecast substitutes a Naive forecast for those.
        sf = StatsForecast(models=build_models(), freq="7D", n_jobs=-1,
                           fallback_model=Naive())
        cv = sf.cross_validation(df=sf_df, h=config.HORIZON_WEEKS,
                                 step_size=config.HORIZON_WEEKS, n_windows=config.N_FOLDS)
        cv, model_cols = tidy_cv(cv)
        cv["fold"] = cv["cutoff"].map(cutoff_to_fold)

    # Clean: clip negatives to 0, fill any residual NaN (1 ultra-short series whose
    # SeasonalNaive(52)/MA windows can't be filled) with the always-defined Naive.
    for m in model_cols:
        cv[m] = cv[m].clip(lower=0).fillna(cv["naive_last"]).fillna(0.0)

    # MASE scale per series from history strictly before the first test week
    first_test = folds[0].test_start
    scales = (w[w["week_end_date"] < first_test]
              .groupby("id")["sales"]
              .apply(lambda s: metrics.seasonal_naive_scale(s.to_numpy(), config.SEASON_WEEKS)))

    print("evaluating...")
    lb = leaderboard(cv, model_cols, scales)
    pf = per_fold(cv, model_cols)
    seg = per_segment(cv, model_cols, prof)
    champ = lb.iloc[0]["model"]

    lb.to_csv(OUT / "leaderboard_summary.csv", index=False)
    pf.to_csv(OUT / "leaderboard_per_fold.csv", index=False)
    seg.to_csv(OUT / "leaderboard_by_segment.csv", index=False)
    cv.to_parquet(OUT / "predictions.parquet", index=False)

    fig_leaderboard(lb); fig_per_fold(pf, champ); fig_segment(seg, champ)

    # cross-checks (post-cleaning)
    assert len(cv) == 900 * config.HORIZON_WEEKS * config.N_FOLDS, \
        f"expected {900*config.HORIZON_WEEKS*config.N_FOLDS} cv rows, got {len(cv)}"
    assert cv[model_cols].notna().all().all(), "NaN predictions remain"
    assert (cv[model_cols] >= 0).all().all(), "negative predictions remain"

    md = OUT / "PHASE5_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 5 -- Time-Series Baselines (champion bake-off)\n\n")
        f.write(f"Rolling-origin backtest: {config.N_FOLDS} folds x {config.HORIZON_WEEKS}-week "
                f"horizon, last {config.N_FOLDS*config.HORIZON_WEEKS} weeks. Primary metric WMAPE "
                f"(volume-weighted, pooled). Predictions clipped to >=0.\n\n")
        f.write(f"## Champion: `{champ}`  (WMAPE {lb.iloc[0]['wmape']:.4f})\n\n")
        f.write("This is the bar Phases 6-7 must beat and the control arm in Phase 8.\n\n")
        f.write("## Leaderboard\n\n| model | WMAPE | RMSE | bias % | MASE (med) |\n|---|---|---|---|---|\n")
        for _, r in lb.iterrows():
            star = " **<- champion**" if r["model"] == champ else ""
            f.write(f"| {r['model']}{star} | {r['wmape']:.4f} | {r['rmse']:.3f} | "
                    f"{r['bias_pct']:+.1f}% | {r['mase_median']:.3f} |\n")
        f.write("\n## WMAPE by intermittency class (champion vs specialists)\n\n")
        sc = seg[seg["segment"] == "sbc_class"].pivot(index="value", columns="model", values="wmape")
        keep = [c for c in dict.fromkeys([champ, "seasonal_naive_52", "croston_sba", "ets"])
                if c in sc.columns]
        f.write("| class | " + " | ".join(keep) + " |\n|" + "---|" * (len(keep) + 1) + "\n")
        for cls, row in sc[keep].iterrows():
            f.write(f"| {cls} | " + " | ".join(f"{row[c]:.3f}" for c in keep) + " |\n")
        f.write(f"\n## Outputs\n\n- `leaderboard_summary.csv`, `_per_fold.csv`, `_by_segment.csv`\n"
                f"- `predictions.parquet` (canonical test cells for Phases 6-8)\n- figures 01-03\n")

    print("\n=== Phase 5 leaderboard ===")
    print(lb.to_string(index=False))
    print(f"\nCHAMPION: {champ}  (WMAPE {lb.iloc[0]['wmape']:.4f}, "
          f"MASE {lb.iloc[0]['mase_median']:.3f})")
    mase_floor = lb.iloc[0]["mase_median"]
    print(f"beats seasonal-naive? MASE {mase_floor:.3f} {'< 1 yes' if mase_floor<1 else '>= 1 NO'}")


if __name__ == "__main__":
    main()
