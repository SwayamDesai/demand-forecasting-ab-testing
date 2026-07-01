"""
Phase 8c -- Experiment 2: the pre-registered retest of the cost-aware policy.

Phase 8b ended HOLD with a documented lesson: the Wilcoxon (median/typical-series)
test was the wrong primary for a total-dollars question, and the honest next step --
written in PHASE8B_SUMMARY.md BEFORE this experiment -- was:

    "Next confirmation (new data or full M5) should pre-register a mean-based
     paired test, plus a tau ~ 0.85-0.87."

This script runs exactly that follow-up. What makes it a legitimate second
experiment rather than p-hacking:

  * FRESH DATA. All evaluation happens on the 24 weeks ENDING WHERE FOLD 1 BEGAN --
    weeks that no prior phase ever scored a model on. Every model is refit per
    window with train < cutoff, so there is no leakage within this experiment either.
  * THE RULE WAS DECLARED BEFORE THE DATA WAS TOUCHED (in the 8b post-mortem).
  * SELECTION AND CONFIRMATION STAY SEPARATE. tau in {0.85, 0.866, 0.90} is chosen
    on the 3 OLDER windows; the winner is scored ONCE on the 3 NEWER windows.
  * SAME economics and guardrail as before: 5:1, ship needs >=5% mean cost
    reduction, p < 0.05 (paired mean test), stockout increase <= +2pp.

Arms (same as 8b):
  control   = AutoETS(52) forecast + Gaussian newsvendor policy z(0.833)*sigma
  treatment = LightGBM pinball loss predicting the demand quantile = the order
              (lag features over the horizon fed by a Tweedie MEAN model, as in 8b)

Outputs: reports/phase8c_retest/*.png + *.csv + PHASE8C_SUMMARY.md
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from src import backtest, config, experiment as ex, metrics
from scripts.phase6_lightgbm import (CATEGORICAL, EXOG, FEATURES, STATIC,
                                     add_codes, fit_fold, recursive_forecast,
                                     sales_features_at)
from scripts.phase8b_cost_aware import fit_quantile, tau_name

OUT = config.REPORTS / "phase8c_retest"

# ---- PRE-REGISTERED (declared in the 8b post-mortem, before touching this data) ----
CU, CO = 5.0, 1.0
TAUS = [0.85, 0.866, 0.90]                 # the declared range + the 8b winner
N_WINDOWS = 6                              # 24 fresh weeks, none previously evaluated
SELECTION_WINDOWS = ["win_1", "win_2", "win_3"]      # older -> selection
CONFIRM_WINDOWS = ["win_4", "win_5", "win_6"]        # newer -> one confirmation look
MIN_COST_REDUCTION_PCT = 5.0
ALPHA = 0.05
MAX_STOCKOUT_INCREASE_PP = 2.0
SENSITIVITY_RATIOS = [(3, 1), (5, 1), (9, 1)]

ZERO_SIGMA = pd.Series(dtype=float)


def paired_mean_test(champ_costs, chall_costs, seed=0, nb=4000) -> dict:
    """PRIMARY test: paired difference of MEAN cost (total dollars per series).
    p from the paired t-test; CI from a paired bootstrap on the mean."""
    a = np.asarray(champ_costs, float); b = np.asarray(chall_costs, float)
    d = b - a
    _, p = stats.ttest_rel(b, a)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(d))
    boots = np.array([d[rng.choice(idx, len(d), True)].mean() for _ in range(nb)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ma = a.mean()
    return dict(n=len(d), mean_control=float(ma), mean_treatment=float(b.mean()),
                pct_change=float(d.mean() / ma * 100), p_value=float(p),
                ci_low_pct=float(lo / ma * 100), ci_high_pct=float(hi / ma * 100),
                pct_series_cheaper=float((d < 0).mean() * 100))


def ets_forecasts(w_trunc: pd.DataFrame) -> pd.DataFrame:
    """AutoETS(52) rolling-origin over the 6 fresh windows (statsforecast CV)."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, Naive
    sf_df = (w_trunc[["id", "week_end_date", "sales"]]
             .rename(columns={"id": "unique_id", "week_end_date": "ds", "sales": "y"}))
    sf = StatsForecast(models=[AutoETS(season_length=config.SEASON_WEEKS)],
                       freq="7D", n_jobs=-1, fallback_model=Naive())
    cv = sf.cross_validation(df=sf_df, h=config.HORIZON_WEEKS,
                             step_size=config.HORIZON_WEEKS, n_windows=N_WINDOWS)
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    cv = cv.rename(columns={"unique_id": "id", "ds": "week_end_date", "AutoETS": "champ_pred"})
    cv["champ_pred"] = cv["champ_pred"].clip(lower=0).fillna(0.0)
    return cv[["id", "week_end_date", "cutoff", "y", "champ_pred"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")

    w = pd.read_parquet(config.PROCESSED / "weekly_features.parquet")
    lookups = {c: {v: i for i, v in enumerate(sorted(w[c].astype(str).unique()))}
               for c in ["cat_id", "dept_id", "store_id", "item_id"]}
    w = add_codes(w, lookups)

    # fresh experiment data: drop the last 12 weeks (folds 1-3 of experiments 1/2)
    old_folds = backtest.make_folds(w["week_end_date"], config.HORIZON_WEEKS, config.N_FOLDS)
    w_trunc = w[w["week_end_date"] <= old_folds[0].train_end].copy()

    # Exclude series too NEW to evaluate on this period: statsforecast CV needs
    # more history than the 24 test weeks + horizon. Dropped from BOTH arms and
    # documented -- they effectively didn't exist for most of the experiment window.
    min_weeks = N_WINDOWS * config.HORIZON_WEEKS + config.HORIZON_WEEKS + 1   # 29
    weeks_per = w_trunc.groupby("id")["week_end_date"].size()
    too_new = weeks_per[weeks_per < min_weeks].index.tolist()
    if too_new:
        print(f"excluding {len(too_new)} series too new for this window range: "
              f"{[t[:24] for t in too_new]}")
    w_trunc = w_trunc[~w_trunc["id"].isin(too_new)].copy()
    n_series = w_trunc["id"].nunique()
    wins = backtest.make_folds(w_trunc["week_end_date"], config.HORIZON_WEEKS, N_WINDOWS)
    wins = [type(f)(name=f"win_{i+1}", train_end=f.train_end,
                    test_start=f.test_start, test_end=f.test_end)
            for i, f in enumerate(wins)]
    print("fresh windows (never previously evaluated):")
    for f in wins:
        print(f"  {f.name}: train<= {f.train_end.date()} | test {f.test_start.date()}..{f.test_end.date()}")

    # ---- control arm: ETS over the 6 windows -------------------------------
    print("\n[1/3] AutoETS(52) over 6 fresh windows (statsforecast CV)...")
    champ = ets_forecasts(w_trunc)
    cutoff_to_win = {f.train_end: f.name for f in wins}
    champ["window"] = champ["cutoff"].map(cutoff_to_win)

    exog_codes = w_trunc[["id", "week_end_date"] + EXOG + STATIC].copy()

    # ---- treatment arm: mean-fed recursion + quantile orders per window ----
    print("[2/3] LightGBM mean (recursion feed) + quantile orders per window...")
    qcols = [tau_name(t) for t in TAUS]
    all_preds = []
    for f in wins:
        train = w_trunc[w_trunc["week_end_date"] <= f.train_end]
        mean_model = fit_fold(train)                       # Tweedie mean (phase-6 config)
        work = w_trunc[["id", "week_end_date", "sales"]].copy()
        work["sales_work"] = np.where(work["week_end_date"] <= f.train_end,
                                      work["sales"], np.nan)
        test_weeks = w_trunc.loc[(w_trunc["week_end_date"] >= f.test_start) &
                                 (w_trunc["week_end_date"] <= f.test_end),
                                 "week_end_date"].unique()
        # recursion feed: mean predictions fill sales_work over the horizon
        mp = recursive_forecast(mean_model, work, exog_codes, test_weeks)
        qmodels = {t: fit_quantile(train, t) for t in TAUS}
        for d in sorted(test_weeks):
            d = pd.Timestamp(d)
            sfeat = sales_features_at(work, d)             # work already mean-filled
            row = exog_codes[exog_codes["week_end_date"] == d].merge(sfeat, on="id", how="left")
            out = row[["id"]].copy(); out["week_end_date"] = d; out["window"] = f.name
            for t, m in qmodels.items():
                out[tau_name(t)] = np.clip(m.predict(row[FEATURES]), 0, None)
            all_preds.append(out)
        print(f"  {f.name} done")
    preds = pd.concat(all_preds, ignore_index=True)

    tbl = champ.merge(preds, on=["id", "week_end_date", "window"], how="inner")
    assert tbl[qcols].notna().all().all(), "quantile predictions missing"
    assert len(tbl) == n_series * config.HORIZON_WEEKS * N_WINDOWS, \
        f"expected {n_series * config.HORIZON_WEEKS * N_WINDOWS} cells, got {len(tbl)}"

    # sigma for the control policy: history before this experiment's first test week
    first_test = tbl["week_end_date"].min()
    sigma = w_trunc[w_trunc["week_end_date"] < first_test].groupby("id")["sales"].std().fillna(0.0)

    def arm_sims(sub, cu=CU, co=CO):
        res = {"champion": ex.simulate(sub.rename(columns={"champ_pred": "p"}), "p", sigma, cu, co)}
        for q in qcols:
            res[q] = ex.simulate(sub.rename(columns={q: "p"}), "p", ZERO_SIGMA, cu, co)
        return res

    # ---- SELECTION on the 3 older windows ----------------------------------
    print("[3/3] selection -> confirmation...")
    sel = tbl[tbl["window"].isin(SELECTION_WINDOWS)]
    ssim = arm_sims(sel)
    so_c_sel = ssim["champion"]["stockout"].mean()
    sel_rows = []
    for q in qcols:
        cost_pct = (ssim[q]["cost"].sum() / ssim["champion"]["cost"].sum() - 1) * 100
        so_pp = (ssim[q]["stockout"].mean() - so_c_sel) * 100
        sel_rows.append(dict(tau=q, cost_pct_change=cost_pct, stockout_pp=so_pp,
                             guard_ok=so_pp <= MAX_STOCKOUT_INCREASE_PP))
    sel_df = pd.DataFrame(sel_rows)
    ok = sel_df[sel_df["guard_ok"]]
    chosen = (ok if len(ok) else sel_df).sort_values("cost_pct_change").iloc[0]["tau"]
    print("selection (win 1-3):\n" + sel_df.to_string(index=False))
    print(f"chosen tau: {chosen}")

    # ---- CONFIRMATION on the 3 newer windows (one look) ---------------------
    conf = tbl[tbl["window"].isin(CONFIRM_WINDOWS)]
    csim = arm_sims(conf)
    champ_ps = ex.per_series(csim["champion"])
    chall_ps = ex.per_series(csim[chosen])
    primary = paired_mean_test(champ_ps["cost"], chall_ps["cost"])
    # secondary (context): Wilcoxon, as used in experiment 1
    wilcoxon = ex.paired_cost_test(champ_ps["cost"], chall_ps["cost"])
    so_c = csim["champion"]["stockout"].mean(); so_t = csim[chosen]["stockout"].mean()
    stockout_pp = (so_t - so_c) * 100
    guard = ex.two_proportion_z(so_c, len(csim["champion"]), so_t, len(csim[chosen]))

    cost_ok = primary["pct_change"] <= -MIN_COST_REDUCTION_PCT
    sig_ok = primary["p_value"] < ALPHA
    guard_ok = stockout_pp <= MAX_STOCKOUT_INCREASE_PP
    decision = "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD"

    wmape_champ = metrics.wmape(conf["y"], conf["champ_pred"])
    wmape_chall = metrics.wmape(conf["y"], conf[chosen])

    # ---- sensitivity ----------------------------------------------------------
    sens = []
    for cu, co in SENSITIVITY_RATIOS:
        s = arm_sims(conf, cu, co)
        pr = paired_mean_test(ex.per_series(s["champion"])["cost"],
                              ex.per_series(s[chosen])["cost"])
        pp = (s[chosen]["stockout"].mean() - s["champion"]["stockout"].mean()) * 100
        sens.append(dict(ratio=f"{cu:.0f}:{co:.0f}", cost_pct_change=pr["pct_change"],
                         p=pr["p_value"], stockout_pp=pp,
                         decision="SHIP" if (pr["pct_change"] <= -MIN_COST_REDUCTION_PCT
                                             and pr["p_value"] < ALPHA
                                             and pp <= MAX_STOCKOUT_INCREASE_PP) else "HOLD"))
    sens = pd.DataFrame(sens)

    sel_df.to_csv(OUT / "selection_windows123.csv", index=False)
    sens.to_csv(OUT / "confirmation_sensitivity.csv", index=False)
    pd.concat([champ_ps.assign(arm="champion"), chall_ps.assign(arm="treatment")]) \
        .to_csv(OUT / "per_series_costs_confirmation.csv", index=False)
    tbl.to_parquet(OUT / "predictions.parquet", index=False)

    # ---- figures ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot([primary["ci_low_pct"], primary["ci_high_pct"]], [1, 1], color="#C44E52", lw=3, label="95% CI")
    ax.plot([primary["pct_change"]], [1], "o", color="#C44E52", ms=9, label="paired mean effect")
    ax.axvline(0, ls="--", c="grey"); ax.axvline(-5, ls=":", c="green", label="ship threshold -5%")
    ax.set_ylim(0.5, 1.5); ax.set_yticks([])
    ax.set_xlabel("mean cost change % (cost-aware vs champion)")
    ax.set_title(f"Confirmation (3 fresh windows, tau={chosen})  ->  {decision}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "01_confirmation_ci.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sens["ratio"], sens["cost_pct_change"],
           color=["#55A868" if d == "SHIP" else "#C44E52" for d in sens["decision"]])
    ax.axhline(-5, ls=":", c="green"); ax.axhline(0, c="grey", lw=0.7)
    ax.set_ylabel("paired mean cost change %"); ax.set_xlabel("stockout:holding ratio")
    ax.set_title("Cost-ratio sensitivity (green=SHIP)")
    for i, (v, d) in enumerate(zip(sens["cost_pct_change"], sens["decision"])):
        ax.text(i, v, f"{v:+.1f}%\n{d}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "02_sensitivity.png", dpi=110); plt.close(fig)

    # ---- summary -----------------------------------------------------------------
    md = OUT / "PHASE8C_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 8c -- Experiment 2: the pre-registered retest\n\n")
        f.write("Runs the follow-up documented in PHASE8B_SUMMARY.md *before* this data was "
                "touched: mean-based (total-dollars) primary test, tau in {0.85, 0.866, 0.90}, "
                "on 24 fresh weeks never previously evaluated.\n\n")
        f.write("## Why this is honest, not test-shopping\n\n")
        f.write("- Evaluation data: the 24 weeks ending where fold 1 began -- no prior phase ever "
                "scored a model there. Models refit per window with train < cutoff.\n")
        f.write("- The mean-based rule was declared in the 8b post-mortem, before this experiment.\n")
        f.write("- tau chosen on windows 1-3; confirmed once on windows 4-6.\n")
        f.write(f"- Panel: {n_series} of 900 series ({len(too_new)} excluded from BOTH arms as too "
                f"new to have history in this window range).\n\n")
        f.write(f"## Pre-registered rule\n\nSHIP iff: paired MEAN cost reduction >= "
                f"{MIN_COST_REDUCTION_PCT:.0f}% AND paired-t p < {ALPHA} AND stockout increase "
                f"<= +{MAX_STOCKOUT_INCREASE_PP:.0f}pp.\n\n")
        f.write("## Selection (windows 1-3)\n\n| tau | cost vs champion | stockout pp | guardrail |\n|---|---|---|---|\n")
        for _, r in sel_df.iterrows():
            mark = " **<- chosen**" if r["tau"] == chosen else ""
            f.write(f"| {r['tau']}{mark} | {r['cost_pct_change']:+.1f}% | "
                    f"{r['stockout_pp']:+.2f}pp | {'pass' if r['guard_ok'] else 'FAIL'} |\n")
        f.write(f"\n## VERDICT (confirmation windows 4-6): **{decision}**\n\n")
        f.write("| gate | value | pass |\n|---|---|---|\n")
        f.write(f"| mean cost reduction >= 5% | {-primary['pct_change']:.1f}% | {cost_ok} |\n")
        f.write(f"| paired-t p < 0.05 | {primary['p_value']:.2e} | {sig_ok} |\n")
        f.write(f"| stockout increase <= +2pp | {stockout_pp:+.2f}pp (two-prop p={guard['p_value']:.2e}) | {guard_ok} |\n\n")
        f.write(f"- 95% CI on mean cost change: [{primary['ci_low_pct']:.1f}%, {primary['ci_high_pct']:.1f}%]; "
                f"{primary['pct_series_cheaper']:.0f}% of series cheaper.\n")
        f.write(f"- Secondary (context): Wilcoxon p = {wilcoxon.p_value:.2e}.\n")
        f.write(f"- Stockout rate: {so_c*100:.1f}% -> {so_t*100:.1f}%.\n")
        f.write(f"- WMAPE (context, not gated): champion {wmape_champ:.4f} vs treatment {wmape_chall:.4f}.\n\n")
        f.write("## Sensitivity\n\n| ratio | mean cost change | p | stockout pp | decision |\n|---|---|---|---|---|\n")
        for _, r in sens.iterrows():
            f.write(f"| {r['ratio']} | {r['cost_pct_change']:+.1f}% | {r['p']:.2e} | "
                    f"{r['stockout_pp']:+.2f}pp | {r['decision']} |\n")
        f.write("\n## Outputs\n\n- `selection_windows123.csv`, `confirmation_sensitivity.csv`, "
                "`per_series_costs_confirmation.csv`, `predictions.parquet`\n- figures 01-02\n")

    print("\n=== Phase 8c verdict (confirmation windows 4-6) ===")
    print(f"chosen tau={chosen} | DECISION: {decision}")
    print(f"mean cost change {primary['pct_change']:+.1f}% (paired-t p={primary['p_value']:.2e}, "
          f"CI [{primary['ci_low_pct']:.1f}%, {primary['ci_high_pct']:.1f}%])")
    print(f"stockouts {stockout_pp:+.2f}pp | {primary['pct_series_cheaper']:.0f}% series cheaper | "
          f"Wilcoxon (context) p={wilcoxon.p_value:.2e}")
    print("sensitivity:\n" + sens.to_string(index=False))


if __name__ == "__main__":
    main()
