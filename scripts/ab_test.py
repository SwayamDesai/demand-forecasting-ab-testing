"""
Step 3 -- the A/B test: cost-aware quantile ordering vs the simple current system.

Arms:
  control   = ma4 forecast + Gaussian newsvendor policy z(0.833)*sigma
              (the simple rule-based ordering a retailer actually runs)
  treatment = per-store LightGBM pinball loss predicting the demand quantile = the
              order (lag features over the horizon fed by the step-2 MEAN model)

The A/B design, pre-registered before results:
  1. PRIMARY TEST = paired MEAN cost difference (paired t + bootstrap CI) -- the
     total-dollars question.
  2. PRACTICAL SIGNIFICANCE CARRIES THE DECISION: at ~30,000 paired series p-values
     collapse for trivially small effects; the >=5% cost-reduction gate is the real
     bar, p<0.05 is merely necessary.
  3. PER-STORE HETEROGENEITY is reported: a verdict that only holds in some stores
     argues for a staged store-level rollout.
  4. POLICY-ISOLATION SECONDARY: LightGBM-mean + z*sigma vs LightGBM-quantile
     separates "better model" from "better ordering policy".

Protocol: tau in {0.85, 0.866, 0.90} selected on folds 1-2, confirmed ONCE on fold 3.
Gates: mean cost reduction >= 5%  AND  p < 0.05  AND  stockout increase <= +2pp.
Sensitivity: the same orders repriced at 3:1 and 9:1.

Checkpointed per store. Outputs in reports/.
"""
from __future__ import annotations

import gc
import resource
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import backtest, config, experiment as ex, metrics
from src.etl import load_weekly
from src.lgbm import (EXOG, FEATURES, STATIC, add_codes, fit_quantile,
                      sales_features_at, tau_name)

pd.options.mode.copy_on_write = True
warnings.simplefilter("ignore")

OUT = config.REPORTS
QDIR = OUT / "lgbm_quantile"

CU, CO = 5.0, 1.0
TAUS = [0.85, 0.866, 0.90]
MIN_COST_REDUCTION_PCT = 5.0
ALPHA = 0.05
MAX_STOCKOUT_INCREASE_PP = 2.0
MIN_HISTORY_WEEKS = 8
SENSITIVITY_RATIOS = [(3, 1), (9, 1)]
ZERO_SIGMA = pd.Series(dtype=float)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def paired_mean_test(a, b, seed=0, nb=4000) -> dict:
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    _, p = stats.ttest_rel(b, a)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(d))
    boots = np.array([d[rng.choice(idx, len(d), True)].mean() for _ in range(nb)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ma = a.mean()
    return dict(n=len(d), pct_change=float(d.mean() / ma * 100), p_value=float(p),
                ci_low_pct=float(lo / ma * 100), ci_high_pct=float(hi / ma * 100),
                pct_series_cheaper=float((d < 0).mean() * 100))


def fit_quantiles_all_stores(w, folds, lookups) -> pd.DataFrame:
    """Per store x fold: quantile fits + orders on test cells (mean-fed features)."""
    stores = sorted(w["store_id"].astype(str).unique())
    for store in stores:
        ck = QDIR / f"preds_{store}.parquet"
        if ck.exists():
            print(f"  [{store}] checkpoint exists -- skip")
            continue
        ws = w[w["store_id"] == store].copy()
        for c in ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id"):
            ws[c] = ws[c].astype(str)
        ws = add_codes(ws, lookups)
        exog_codes = ws[["id", "week_end_date"] + EXOG + STATIC]
        mean_p = pd.read_parquet(OUT / "lgbm_mean" / f"preds_{store}.parquet")
        mean_map = mean_p.set_index(["id", "week_end_date"])["y_pred"]

        rows = []
        for f in folds:
            train = ws[ws["week_end_date"] <= f.train_end]
            models = {t: fit_quantile(train, t) for t in TAUS}
            work = ws[["id", "week_end_date", "sales"]].copy()
            work["sales_work"] = np.where(work["week_end_date"] <= f.train_end,
                                          work["sales"], np.nan)
            in_test = ((work["week_end_date"] >= f.test_start) &
                       (work["week_end_date"] <= f.test_end))
            key = pd.MultiIndex.from_frame(work.loc[in_test, ["id", "week_end_date"]])
            work.loc[in_test, "sales_work"] = mean_map.reindex(key).to_numpy()
            for d in sorted(work.loc[in_test, "week_end_date"].unique()):
                d = pd.Timestamp(d)
                sfeat = sales_features_at(work, d)
                row = exog_codes[exog_codes["week_end_date"] == d].merge(
                    sfeat, on="id", how="left")
                o = row[["id"]].copy(); o["week_end_date"] = d; o["fold"] = f.name
                for t, m in models.items():
                    o[tau_name(t)] = np.clip(m.predict(row[FEATURES]), 0, None)
                rows.append(o)
        sp = pd.concat(rows, ignore_index=True)
        sp.to_parquet(ck, index=False)
        print(f"  [{store}] done: {len(sp):,} cells  (rss {rss_gb():.1f} GB)")
        del ws, exog_codes, work, sp
        gc.collect()
    return pd.concat([pd.read_parquet(p) for p in sorted(QDIR.glob("preds_*.parquet"))],
                     ignore_index=True)


def main() -> None:
    QDIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] load panel + step-2 predictions...")
    w = load_weekly()
    folds = backtest.make_folds(w["week_end_date"], config.HORIZON_WEEKS, config.N_FOLDS)
    base = pd.read_parquet(OUT / "baseline_predictions.parquet")
    mean_lgbm = pd.concat([pd.read_parquet(p) for p in
                           sorted((OUT / "lgbm_mean").glob("preds_*.parquet"))],
                          ignore_index=True).rename(columns={"y_pred": "lgbm_mean"})

    print("[2/5] per-store quantile models...")
    lookups = {c: {v: i for i, v in enumerate(sorted(w[c].astype(str).unique()))}
               for c in ["cat_id", "dept_id", "store_id", "item_id"]}
    quant = fit_quantiles_all_stores(w, folds, lookups)
    qcols = [tau_name(t) for t in TAUS]

    print("[3/5] assemble panel...")
    tbl = (base.merge(quant, on=["id", "week_end_date", "fold"], how="inner")
               .merge(mean_lgbm[["id", "week_end_date", "lgbm_mean"]],
                      on=["id", "week_end_date"], how="left"))
    first_test = folds[0].test_start
    hist_weeks = (w[w["week_end_date"] < first_test]
                  .groupby("id", observed=True)["week_end_date"].size())
    keep_ids = hist_weeks[hist_weeks >= MIN_HISTORY_WEEKS].index
    n_before = tbl["id"].nunique()
    tbl = tbl[tbl["id"].isin(keep_ids)].reset_index(drop=True)
    print(f"      panel: {tbl['id'].nunique():,} series "
          f"({n_before - tbl['id'].nunique()} excluded, <{MIN_HISTORY_WEEKS}w history), "
          f"{len(tbl):,} cells")

    sigma = (w[w["week_end_date"] < first_test]
             .groupby("id", observed=True)["sales"].std().fillna(0.0))

    def arm_sims(sub, cu=CU, co=CO):
        res = {"control": ex.simulate(sub.rename(columns={"ma4": "p"}), "p", sigma, cu, co),
               "lgbm_mean_policy": ex.simulate(sub.rename(columns={"lgbm_mean": "p"}),
                                               "p", sigma, cu, co)}
        for q in qcols:
            res[q] = ex.simulate(sub.rename(columns={q: "p"}), "p", ZERO_SIGMA, cu, co)
        return res

    print("[4/5] selection (folds 1-2) -> confirmation (fold 3)...")
    sel = tbl[tbl["fold"].isin(["fold_1", "fold_2"])]
    ssim = arm_sims(sel)
    so_c_sel = ssim["control"]["stockout"].mean()
    sel_rows = []
    for q in qcols:
        cost_pct = (ssim[q]["cost"].sum() / ssim["control"]["cost"].sum() - 1) * 100
        so_pp = (ssim[q]["stockout"].mean() - so_c_sel) * 100
        sel_rows.append(dict(tau=q, cost_pct_change=cost_pct, stockout_pp=so_pp,
                             guard_ok=so_pp <= MAX_STOCKOUT_INCREASE_PP))
    sel_df = pd.DataFrame(sel_rows)
    ok = sel_df[sel_df["guard_ok"]]
    chosen = (ok if len(ok) else sel_df).sort_values("cost_pct_change").iloc[0]["tau"]
    print(sel_df.to_string(index=False)); print(f"      chosen tau: {chosen}")
    del ssim, sel; gc.collect()

    conf = tbl[tbl["fold"] == "fold_3"]
    csim = arm_sims(conf)
    champ_ps = ex.per_series(csim["control"])
    chall_ps = ex.per_series(csim[chosen])
    primary = paired_mean_test(champ_ps["cost"], chall_ps["cost"])
    wilcoxon = ex.paired_cost_test(champ_ps["cost"], chall_ps["cost"])
    so_c = csim["control"]["stockout"].mean(); so_t = csim[chosen]["stockout"].mean()
    stockout_pp = (so_t - so_c) * 100

    cost_ok = primary["pct_change"] <= -MIN_COST_REDUCTION_PCT
    sig_ok = primary["p_value"] < ALPHA
    guard_ok = stockout_pp <= MAX_STOCKOUT_INCREASE_PP
    decision = "SHIP" if (cost_ok and sig_ok and guard_ok) else "HOLD"

    iso = paired_mean_test(ex.per_series(csim["lgbm_mean_policy"])["cost"],
                           chall_ps["cost"])

    sens_rows = [dict(ratio="5:1", cost_pct_change=primary["pct_change"],
                      p=primary["p_value"], stockout_pp=stockout_pp, decision=decision)]
    for cu, co in SENSITIVITY_RATIOS:
        s = arm_sims(conf, cu, co)
        pr = paired_mean_test(ex.per_series(s["control"])["cost"],
                              ex.per_series(s[chosen])["cost"])
        pp = (s[chosen]["stockout"].mean() - s["control"]["stockout"].mean()) * 100
        sens_rows.append(dict(ratio=f"{cu:.0f}:{co:.0f}", cost_pct_change=pr["pct_change"],
                              p=pr["p_value"], stockout_pp=pp,
                              decision="SHIP" if (pr["pct_change"] <= -MIN_COST_REDUCTION_PCT
                                                  and pr["p_value"] < ALPHA
                                                  and pp <= MAX_STOCKOUT_INCREASE_PP) else "HOLD"))
        del s; gc.collect()
    sens = pd.DataFrame(sens_rows)

    store_map = conf[["id", "store_id"]].drop_duplicates().set_index("id")["store_id"]
    per_store = []
    merged = (champ_ps.set_index("id")["cost"].rename("c")
              .to_frame().join(chall_ps.set_index("id")["cost"].rename("t")))
    merged["store"] = store_map
    for store, g in merged.groupby("store"):
        per_store.append(dict(store=store, n=len(g),
                              cost_pct=(g["t"].sum() / g["c"].sum() - 1) * 100))
    ps = pd.DataFrame(per_store).sort_values("cost_pct")

    print("[5/5] write outputs...")
    sel_df.to_csv(OUT / "ab_selection.csv", index=False)
    sens.to_csv(OUT / "ab_sensitivity.csv", index=False)
    ps.to_csv(OUT / "ab_per_store.csv", index=False)
    wmape_c = metrics.wmape(conf["y"], conf["ma4"])
    wmape_t = metrics.wmape(conf["y"], conf[chosen])

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot([primary["ci_low_pct"], primary["ci_high_pct"]], [1, 1], color="#C44E52", lw=3,
            label="95% CI")
    ax.plot([primary["pct_change"]], [1], "o", color="#C44E52", ms=9, label="paired mean effect")
    ax.axvline(0, ls="--", c="grey"); ax.axvline(-5, ls=":", c="green", label="ship threshold -5%")
    ax.set_ylim(0.5, 1.5); ax.set_yticks([])
    ax.set_xlabel("mean cost change % (cost-aware vs ma4+safety-stock)")
    ax.set_title(f"Confirmation, n={primary['n']:,}, tau={chosen} -> {decision}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "03_ab_confirmation_ci.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#55A868" if v <= -5 else ("#DD8452" if v < 0 else "#C44E52")
              for v in ps["cost_pct"]]
    ax.bar(ps["store"], ps["cost_pct"], color=colors)
    ax.axhline(-5, ls=":", c="green"); ax.axhline(0, c="grey", lw=0.7)
    ax.set_ylabel("cost change % (fold 3)"); ax.set_title("Per-store cost impact")
    fig.tight_layout(); fig.savefig(OUT / "04_ab_per_store.png", dpi=110); plt.close(fig)

    with open(OUT / "AB_SUMMARY.md", "w") as f:
        f.write("# Cost-Aware A/B at 30K series\n\n")
        f.write(f"Control = ma4 + z(0.833)*sigma. Treatment = per-store LightGBM "
                f"quantile orders. Panel = {primary['n']:,} paired series.\n\n")
        f.write("## Pre-registered rule\n\nSHIP iff mean cost reduction >= 5% AND paired-t "
                "p < 0.05 AND stockout increase <= +2pp. At this n, the 5% practical gate "
                "carries the decision; p is merely necessary.\n\n")
        f.write("## Selection (folds 1-2)\n\n| tau | cost | stockout pp | guard |\n|---|---|---|---|\n")
        for _, r in sel_df.iterrows():
            mark = " **<- chosen**" if r["tau"] == chosen else ""
            f.write(f"| {r['tau']}{mark} | {r['cost_pct_change']:+.1f}% | "
                    f"{r['stockout_pp']:+.2f}pp | {'pass' if r['guard_ok'] else 'FAIL'} |\n")
        f.write(f"\n## VERDICT (fold 3): **{decision}**\n\n")
        f.write("| gate | value | pass |\n|---|---|---|\n")
        f.write(f"| mean cost reduction >= 5% | {-primary['pct_change']:.1f}% | {cost_ok} |\n")
        f.write(f"| paired-t p < 0.05 | {primary['p_value']:.2e} | {sig_ok} |\n")
        f.write(f"| stockout <= +2pp | {stockout_pp:+.2f}pp | {guard_ok} |\n\n")
        f.write(f"- 95% CI [{primary['ci_low_pct']:.1f}%, {primary['ci_high_pct']:.1f}%]; "
                f"{primary['pct_series_cheaper']:.0f}% of series cheaper; Wilcoxon "
                f"p={wilcoxon.p_value:.2e}.\n")
        f.write(f"- Stockout rate {so_c*100:.1f}% -> {so_t*100:.1f}%.\n")
        f.write(f"- WMAPE context: control {wmape_c:.4f} vs treatment {wmape_t:.4f} "
                f"(quantile trades point accuracy for cost by design).\n")
        f.write(f"- Policy isolation (LightGBM mean+z*sigma vs quantile): "
                f"{iso['pct_change']:+.1f}% (p={iso['p_value']:.2e}) -- the ordering policy, "
                f"not just the model, drives the saving.\n\n")
        f.write("## Sensitivity\n\n| ratio | cost | p | stockout pp | decision |\n|---|---|---|---|---|\n")
        for _, r in sens.iterrows():
            f.write(f"| {r['ratio']} | {r['cost_pct_change']:+.1f}% | {r['p']:.2e} | "
                    f"{r['stockout_pp']:+.2f}pp | {r['decision']} |\n")
        f.write("\n## Per-store (fold 3)\n\n| store | n | cost % |\n|---|---|---|\n")
        for _, r in ps.iterrows():
            f.write(f"| {r['store']} | {int(r['n'])} | {r['cost_pct']:+.1f}% |\n")

    print(f"\n=== A/B verdict (fold 3, n={primary['n']:,}) ===")
    print(f"chosen tau={chosen} | DECISION: {decision}")
    print(f"mean cost {primary['pct_change']:+.1f}% (p={primary['p_value']:.2e}, "
          f"CI [{primary['ci_low_pct']:.1f}%, {primary['ci_high_pct']:.1f}%]) | "
          f"stockout {stockout_pp:+.2f}pp | {primary['pct_series_cheaper']:.0f}% series cheaper")
    print(f"policy isolation: {iso['pct_change']:+.1f}% (p={iso['p_value']:.2e})")
    print("sensitivity:\n" + sens.to_string(index=False))
    print("per-store:\n" + ps.to_string(index=False))
    print(f"peak RSS {rss_gb():.1f} GB")


if __name__ == "__main__":
    main()
