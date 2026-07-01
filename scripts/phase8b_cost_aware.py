"""
Phase 8b -- The cost-aware challenger: order the newsvendor quantile directly.

Phase 8 said HOLD: the accuracy-tuned LightGBM didn't cut cost. Diagnosis: both arms
used the same crude order policy (mean forecast + z*sigma_demand), so we were tuning
the small lever (point accuracy) while the big lever (the order itself) stayed fixed.

The fix tested here -- one idea, no new infrastructure:
    Train the SAME LightGBM with pinball (quantile) loss so it directly predicts the
    cost-optimal ORDER quantity: the tau* = cu/(cu+co) = 5/6 ~ 0.833 demand quantile.

Fair comparison by construction: the control (ETS + z(0.833)*sigma) already targets the
same 0.833 critical ratio via a Gaussian approximation on historical demand sigma. The
treatment learns that quantile per-SKU from features instead. Same target, better
estimator -- that's the hypothesis.

Honest protocol (pre-registered before results):
  * tau sweep {0.75, 0.833, 0.90} evaluated ONLY on folds 1-2 (selection).
  * The best tau under the guardrail is confirmed ONCE on untouched fold 3.
  * SHIP iff (confirmation fold): paired cost reduction >= 5%  AND  Wilcoxon p < 0.05
    AND stockout-rate increase <= +2pp vs champion.
  * NOTE the WMAPE gate from Phase 8 is intentionally NOT applied to the treatment:
    a 0.833-quantile forecast trades point accuracy for cost by design. WMAPE is
    reported as context. Guardrail = service (stockouts), the thing that can hurt.
  * Sensitivity: reprice the SAME orders at 3:1 and 9:1 (policy fixed, assumption varies).

Recursion detail: lag features over the horizon are fed by the Phase-6 MEAN model's
predictions (already computed), not by the quantile predictions -- feeding back a high
quantile would inflate the lags. Mean for state, quantile for the order decision.

Outputs: reports/phase8b_cost_aware/*.png + *.csv + PHASE8B_SUMMARY.md
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import backtest, config, experiment as ex, metrics
from scripts.phase6_lightgbm import (CATEGORICAL, EXOG, FEATURES, STATIC,
                                     add_codes, sales_features_at)

OUT = config.REPORTS / "phase8b_cost_aware"

# ---- PRE-REGISTERED (fixed before seeing results) ----------------------------
CU, CO = 5.0, 1.0
TAUS = [0.75, 5 / 6, 0.90]              # candidate order quantiles (tau* = 5/6)
SELECTION_FOLDS = ["fold_1", "fold_2"]
CONFIRM_FOLD = "fold_3"
MIN_COST_REDUCTION_PCT = 5.0
ALPHA = 0.05
MAX_STOCKOUT_INCREASE_PP = 2.0
SENSITIVITY_RATIOS = [(3, 1), (5, 1), (9, 1)]

ZERO_SIGMA = pd.Series(dtype=float)      # treatment orders ARE the prediction


def tau_name(t: float) -> str:
    return f"q{t:.3f}".rstrip("0").rstrip(".")


def fit_quantile(train: pd.DataFrame, tau: float) -> lgb.LGBMRegressor:
    """Same architecture/hyperparams as Phase 6, only the objective changes."""
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=tau,
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.9, random_state=0, n_jobs=-1, verbosity=-1)
    model.fit(train[FEATURES], train["sales"], categorical_feature=CATEGORICAL)
    return model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")

    w = pd.read_parquet(config.PROCESSED / "weekly_features.parquet")
    champ_pred = pd.read_parquet(config.REPORTS / "phase5_baselines" / "predictions.parquet")
    lb5 = pd.read_csv(config.REPORTS / "phase5_baselines" / "leaderboard_summary.csv")
    mean_pred = pd.read_parquet(config.REPORTS / "phase6_lightgbm" / "predictions.parquet")
    champion = lb5.iloc[0]["model"]

    lookups = {c: {v: i for i, v in enumerate(sorted(w[c].astype(str).unique()))}
               for c in ["cat_id", "dept_id", "store_id", "item_id"]}
    w = add_codes(w, lookups)
    exog_codes = w[["id", "week_end_date"] + EXOG + STATIC].copy()
    folds = backtest.make_folds(w["week_end_date"], config.HORIZON_WEEKS, config.N_FOLDS)

    # Phase-6 mean predictions keyed by (id, week) -- the recursion feed
    mean_map = mean_pred.set_index(["id", "week_end_date"])["y_pred"]

    # ---- train + predict quantile orders per fold --------------------------
    all_preds = []
    for f in folds:
        print(f"[{f.name}] fitting {len(TAUS)} quantile models on weeks <= {f.train_end.date()} ...")
        train = w[w["week_end_date"] <= f.train_end]
        models = {t: fit_quantile(train, t) for t in TAUS}

        # working frame: actuals to cutoff, Phase-6 MEAN preds over this fold's horizon
        work = w[["id", "week_end_date", "sales"]].copy()
        work["sales_work"] = np.where(work["week_end_date"] <= f.train_end,
                                      work["sales"], np.nan)
        in_test = (work["week_end_date"] >= f.test_start) & (work["week_end_date"] <= f.test_end)
        key = pd.MultiIndex.from_frame(work.loc[in_test, ["id", "week_end_date"]])
        work.loc[in_test, "sales_work"] = mean_map.reindex(key).to_numpy()

        test_weeks = sorted(w.loc[in_test, "week_end_date"].unique())
        for d in test_weeks:
            d = pd.Timestamp(d)
            sfeat = sales_features_at(work, d)
            row = exog_codes[exog_codes["week_end_date"] == d].merge(sfeat, on="id", how="left")
            out = row[["id"]].copy()
            out["week_end_date"] = d
            out["fold"] = f.name
            for t, m in models.items():
                out[tau_name(t)] = np.clip(m.predict(row[FEATURES]), 0, None)
            all_preds.append(out)
    preds = pd.concat(all_preds, ignore_index=True)
    qcols = [tau_name(t) for t in TAUS]

    # align with champion predictions + actuals on identical cells
    tbl = (champ_pred[["id", "week_end_date", "fold", "y", champion]]
           .rename(columns={champion: "champ_pred"})
           .merge(preds[["id", "week_end_date"] + qcols], on=["id", "week_end_date"], how="left"))
    assert tbl[qcols].notna().all().all(), "quantile predictions missing on some test cells"

    # champion sigma: same as Phase 8 (history before first test week)
    first_test = tbl["week_end_date"].min()
    sigma = w[w["week_end_date"] < first_test].groupby("id")["sales"].std().fillna(0.0)

    def arm_costs(sub: pd.DataFrame, cu=CU, co=CO):
        """champion per-series costs + each tau's per-series costs on `sub` cells."""
        champ_sim = ex.simulate(sub.rename(columns={"champ_pred": "p"}), "p", sigma, cu, co)
        res = {"champion": champ_sim}
        for q in qcols:
            res[q] = ex.simulate(sub.rename(columns={q: "p"}), "p", ZERO_SIGMA, cu, co)
        return res

    # ---- SELECTION on folds 1-2 --------------------------------------------
    sel = tbl[tbl["fold"].isin(SELECTION_FOLDS)]
    sims = arm_costs(sel)
    so_champ_sel = sims["champion"]["stockout"].mean()
    sel_rows = []
    for q in qcols:
        cost_pct = (sims[q]["cost"].sum() / sims["champion"]["cost"].sum() - 1) * 100
        so_pp = (sims[q]["stockout"].mean() - so_champ_sel) * 100
        sel_rows.append(dict(tau=q, cost_pct_change=cost_pct, stockout_pp=so_pp,
                             guard_ok=so_pp <= MAX_STOCKOUT_INCREASE_PP))
    sel_df = pd.DataFrame(sel_rows)
    ok = sel_df[sel_df["guard_ok"]]
    chosen = (ok if len(ok) else sel_df).sort_values("cost_pct_change").iloc[0]["tau"]
    print("\nselection (folds 1-2):\n" + sel_df.to_string(index=False))
    print(f"chosen tau: {chosen}")

    # ---- CONFIRMATION on fold 3 (one look) ----------------------------------
    conf = tbl[tbl["fold"] == CONFIRM_FOLD]
    csims = arm_costs(conf)
    champ_ps = ex.per_series(csims["champion"])
    chall_ps = ex.per_series(csims[chosen])
    paired = ex.paired_cost_test(champ_ps["cost"], chall_ps["cost"])
    so_c = csims["champion"]["stockout"].mean()
    so_t = csims[chosen]["stockout"].mean()
    stockout_pp = (so_t - so_c) * 100
    guard = ex.two_proportion_z(so_c, len(csims["champion"]), so_t, len(csims[chosen]))

    cost_ok = paired.pct_change <= -MIN_COST_REDUCTION_PCT
    sig_ok = paired.p_value < ALPHA
    guard_ok = stockout_pp <= MAX_STOCKOUT_INCREASE_PP
    decision = "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD"

    # WMAPE context (not a gate -- quantile forecasts trade accuracy for cost)
    wmape_champ = metrics.wmape(conf["y"], conf["champ_pred"])
    wmape_chall = metrics.wmape(conf["y"], conf[chosen])

    # ---- sensitivity: reprice the SAME orders at other ratios ---------------
    sens = []
    for cu, co in SENSITIVITY_RATIOS:
        s = arm_costs(conf, cu, co)
        pr = ex.paired_cost_test(ex.per_series(s["champion"])["cost"],
                                 ex.per_series(s[chosen])["cost"])
        pp = (s[chosen]["stockout"].mean() - s["champion"]["stockout"].mean()) * 100
        sens.append(dict(ratio=f"{cu:.0f}:{co:.0f}", cost_pct_change=pr.pct_change,
                         p=pr.p_value, ci_low=pr.ci_low, ci_high=pr.ci_high, stockout_pp=pp,
                         decision="SHIP" if (pr.pct_change <= -MIN_COST_REDUCTION_PCT
                                             and pr.p_value < ALPHA
                                             and pp <= MAX_STOCKOUT_INCREASE_PP) else "HOLD"))
    sens = pd.DataFrame(sens)

    sel_df.to_csv(OUT / "selection_folds12.csv", index=False)
    sens.to_csv(OUT / "confirmation_sensitivity.csv", index=False)
    pd.concat([champ_ps.assign(arm="champion"), chall_ps.assign(arm="treatment")]) \
        .to_csv(OUT / "per_series_costs_fold3.csv", index=False)
    tbl.to_parquet(OUT / "predictions.parquet", index=False)

    # ---- figures -------------------------------------------------------------
    # 01 frontier on selection folds
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in sel_df.iterrows():
        color = "#55A868" if r["guard_ok"] else "#C44E52"
        ax.scatter(r["stockout_pp"], r["cost_pct_change"], s=90, c=color, zorder=3)
        ax.annotate(r["tau"], (r["stockout_pp"], r["cost_pct_change"]),
                    textcoords="offset points", xytext=(8, 4))
    ax.scatter([0], [0], marker="*", s=220, c="#4C72B0", zorder=3)
    ax.annotate(f"champion ({champion})", (0, 0), textcoords="offset points", xytext=(8, -12))
    ax.axhline(-MIN_COST_REDUCTION_PCT, ls=":", c="green", label="ship threshold -5%")
    ax.axvline(MAX_STOCKOUT_INCREASE_PP, ls=":", c="red", label="guardrail +2pp")
    ax.set_xlabel("stockout-rate change vs champion (pp)")
    ax.set_ylabel("cost change vs champion (%)")
    ax.set_title("Cost/service frontier -- selection folds 1-2")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "01_frontier_selection.png", dpi=110); plt.close(fig)

    # 02 confirmation paired CI
    lo_pct = paired.ci_low / paired.mean_control * 100
    hi_pct = paired.ci_high / paired.mean_control * 100
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot([lo_pct, hi_pct], [1, 1], color="#C44E52", lw=3, label="95% CI")
    ax.plot([paired.pct_change], [1], "o", color="#C44E52", ms=9, label="paired point est.")
    ax.axvline(0, ls="--", c="grey"); ax.axvline(-5, ls=":", c="green", label="ship threshold -5%")
    ax.set_ylim(0.5, 1.5); ax.set_yticks([])
    ax.set_xlabel("cost change % (cost-aware vs champion)")
    ax.set_title(f"Confirmation fold 3, tau={chosen}  ->  {decision}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "02_confirmation_ci.png", dpi=110); plt.close(fig)

    # 03 sensitivity
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sens["ratio"], sens["cost_pct_change"],
           color=["#55A868" if d == "SHIP" else "#C44E52" for d in sens["decision"]])
    ax.axhline(-5, ls=":", c="green"); ax.axhline(0, c="grey", lw=0.7)
    ax.set_ylabel("paired cost change %"); ax.set_xlabel("stockout:holding ratio")
    ax.set_title("Repricing the same orders (green=SHIP)")
    for i, (v, d) in enumerate(zip(sens["cost_pct_change"], sens["decision"])):
        ax.text(i, v, f"{v:+.1f}%\n{d}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "03_sensitivity.png", dpi=110); plt.close(fig)

    # ---- summary ---------------------------------------------------------------
    md = OUT / "PHASE8B_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 8b -- Cost-Aware Challenger (order the quantile directly)\n\n")
        f.write(f"**Control** = `{champion}` + Gaussian newsvendor policy (z(0.833)*sigma).  "
                f"**Treatment** = LightGBM pinball loss predicting the demand quantile as the order.\n\n")
        f.write("Both arms target the SAME 5:1 critical ratio tau*=0.833; they differ only in how "
                "the quantile is estimated (Gaussian assumption vs learned per-SKU).\n\n")
        f.write("## Protocol (pre-registered)\n\n")
        f.write(f"- tau swept on folds 1-2 only; winner confirmed ONCE on untouched fold 3.\n")
        f.write(f"- SHIP iff: paired cost reduction >= {MIN_COST_REDUCTION_PCT:.0f}% AND Wilcoxon "
                f"p < {ALPHA} AND stockout increase <= +{MAX_STOCKOUT_INCREASE_PP:.0f}pp.\n")
        f.write("- WMAPE reported as context, NOT gated (quantile forecasts trade point accuracy "
                "for cost by design).\n\n")
        f.write("## Selection (folds 1-2)\n\n| tau | cost vs champion | stockout pp | guardrail |\n|---|---|---|---|\n")
        for _, r in sel_df.iterrows():
            mark = " **<- chosen**" if r["tau"] == chosen else ""
            f.write(f"| {r['tau']}{mark} | {r['cost_pct_change']:+.1f}% | "
                    f"{r['stockout_pp']:+.2f}pp | {'pass' if r['guard_ok'] else 'FAIL'} |\n")
        f.write(f"\n## VERDICT (confirmation fold 3): **{decision}**\n\n")
        f.write("| gate | value | pass |\n|---|---|---|\n")
        f.write(f"| cost reduction >= 5% | {-paired.pct_change:.1f}% | {cost_ok} |\n")
        f.write(f"| Wilcoxon p < 0.05 | {paired.p_value:.2e} | {sig_ok} |\n")
        f.write(f"| stockout increase <= +2pp | {stockout_pp:+.2f}pp (two-prop p={guard['p_value']:.2e}) | {guard_ok} |\n\n")
        f.write(f"- Paired 95% CI on cost change: [{lo_pct:.1f}%, {hi_pct:.1f}%]; "
                f"{paired.extra['pct_series_cheaper']:.0f}% of series cheaper under treatment.\n")
        f.write(f"- Stockout rate: {so_c*100:.1f}% -> {so_t*100:.1f}%.\n")
        f.write(f"- WMAPE (context only): champion {wmape_champ:.4f} vs treatment {wmape_chall:.4f} "
                f"-- expected to be worse; the treatment optimises cost, not point error.\n\n")
        f.write("## Sensitivity: reprice the same orders\n\n")
        f.write("| ratio | cost change | p | stockout pp | decision |\n|---|---|---|---|---|\n")
        for _, r in sens.iterrows():
            f.write(f"| {r['ratio']} | {r['cost_pct_change']:+.1f}% | {r['p']:.2e} | "
                    f"{r['stockout_pp']:+.2f}pp | {r['decision']} |\n")
        f.write("\n## Outputs\n\n- `selection_folds12.csv`, `confirmation_sensitivity.csv`, "
                "`per_series_costs_fold3.csv`, `predictions.parquet`\n- figures 01-03\n")

    print("\n=== Phase 8b verdict (confirmation fold 3) ===")
    print(f"chosen tau={chosen} | DECISION: {decision}")
    print(f"paired cost change {paired.pct_change:+.1f}% (p={paired.p_value:.2e}, "
          f"CI [{lo_pct:.1f}%, {hi_pct:.1f}%]); stockout {stockout_pp:+.2f}pp; "
          f"{paired.extra['pct_series_cheaper']:.0f}% series cheaper")
    print(f"WMAPE context: champion {wmape_champ:.4f} vs treatment {wmape_chall:.4f}")
    print("sensitivity (same orders, repriced):\n" + sens.to_string(index=False))


if __name__ == "__main__":
    main()
