"""
Phase 8 -- The A/B test (champion vs best challenger), done strongly but plainly.

Control  = Phase-5 champion (best classical).
Treatment = the most ACCURATE challenger (lower WMAPE of LightGBM / LSTM) -- so the
            test asks the real question: does better accuracy buy lower inventory cost?

What makes it strong (not complicated):
  * A newsvendor business simulation turns each forecast into an ORDER and prices the
    mistakes (understock vs overstock). Money, not error.
  * PRE-REGISTERED decision rule, stated before looking at the result.
  * Stratified assignment + an unpaired test (mimics a live experiment) AND a paired
    counterfactual test (higher power, correct for an offline backtest).
  * Power analysis (MDE), a stockout-rate guardrail, and a cost-ratio sensitivity sweep
    (the verdict's biggest assumption).

Outputs: reports/phase8_ab_test/*.png + *.csv + PHASE8_SUMMARY.md
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, experiment as ex, metrics

OUT = config.REPORTS / "phase8_ab_test"

# ---- PRE-REGISTERED decision rule (fixed before seeing results) -------------
PRIMARY_CU, PRIMARY_CO = 5.0, 1.0          # stockout:holding = 5:1
MIN_COST_REDUCTION_PCT = 5.0               # ship needs >=5% cheaper
ALPHA = 0.05
MAX_STOCKOUT_INCREASE_PP = 2.0             # guardrail: service can't degrade >2pp
MAX_WMAPE_DEGRADATION = 1.05               # challenger WMAPE <= 1.05x champion
SENSITIVITY_RATIOS = [(3, 1), (5, 1), (9, 1)]


def load_models():
    champ_pred = pd.read_parquet(config.REPORTS / "phase5_baselines" / "predictions.parquet")
    lb5 = pd.read_csv(config.REPORTS / "phase5_baselines" / "leaderboard_summary.csv")
    champion = lb5.iloc[0]["model"]
    base = champ_pred[["id", "week_end_date", "fold", "y", champion]].rename(
        columns={champion: "champ_pred"})

    challengers = {}
    for name, path in [("lightgbm", "phase6_lightgbm"), ("lstm_seq2seq", "phase7_lstm")]:
        p = config.REPORTS / path / "predictions.parquet"
        if p.exists():
            cp = pd.read_parquet(p)[["id", "week_end_date", "y_pred"]].rename(
                columns={"y_pred": name})
            base = base.merge(cp, on=["id", "week_end_date"], how="left")
            challengers[name] = metrics.wmape(
                base["y"], base[name].fillna(base["champ_pred"]))
    # most accurate challenger
    challenger = min(challengers, key=challengers.get)
    return base, champion, challenger, challengers


def run_costs(tbl, sigma, cu, co):
    champ = ex.per_series(ex.simulate(tbl.rename(columns={"champ_pred": "p"}), "p", sigma, cu, co))
    chall = ex.per_series(ex.simulate(tbl.rename(columns={"chall_pred": "p"}), "p", sigma, cu, co))
    return champ, chall


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")

    w = pd.read_parquet(config.PROCESSED / "weekly_features.parquet")
    prof = pd.read_csv(config.REPORTS / "phase3_eda" / "series_profile.csv")
    prof["volume_tier"] = pd.qcut(prof["total_units"], 3, labels=["low", "mid", "high"])

    base, champion, challenger, chall_wmapes = load_models()
    tbl = base.rename(columns={challenger: "chall_pred"})[
        ["id", "week_end_date", "fold", "y", "champ_pred", "chall_pred"]]
    champ_wmape = metrics.wmape(tbl["y"], tbl["champ_pred"])
    chall_wmape = metrics.wmape(tbl["y"], tbl["chall_pred"])
    print(f"champion={champion} (WMAPE {champ_wmape:.4f}) | "
          f"challenger={challenger} (WMAPE {chall_wmape:.4f})")

    # per-series demand std from history before the first test week
    first_test = tbl["week_end_date"].min()
    sigma = w[w["week_end_date"] < first_test].groupby("id")["sales"].std().fillna(0.0)

    # ---- primary analysis at 5:1 -----------------------------------------
    champ_ps, chall_ps = run_costs(tbl, sigma, PRIMARY_CU, PRIMARY_CO)
    units = (champ_ps.rename(columns={"cost": "champ_cost", "stockout_rate": "champ_stockout"})
             [["id", "champ_cost", "champ_stockout"]]
             .merge(chall_ps.rename(columns={"cost": "chall_cost", "stockout_rate": "chall_stockout"})
                    [["id", "chall_cost", "chall_stockout"]], on="id")
             .merge(prof[["id", "cat_id", "sbc_class", "volume_tier"]], on="id"))

    # paired (correct offline design)
    paired = ex.paired_cost_test(units["champ_cost"], units["chall_cost"])

    # unpaired (mimics a live randomized experiment)
    assign = ex.stratified_assignment(units, ["volume_tier", "sbc_class"])
    u = units.merge(assign, on="id")
    ctrl = u.loc[u["arm"] == "control", "champ_cost"]
    trt = u.loc[u["arm"] == "treatment", "chall_cost"]
    unpaired = ex.unpaired_cost_test(ctrl, trt)

    # stockout guardrail (pooled over all test cells)
    champ_sim = ex.simulate(tbl.rename(columns={"champ_pred": "p"}), "p", sigma, PRIMARY_CU, PRIMARY_CO)
    chall_sim = ex.simulate(tbl.rename(columns={"chall_pred": "p"}), "p", sigma, PRIMARY_CU, PRIMARY_CO)
    so_champ = champ_sim["stockout"].mean(); so_chall = chall_sim["stockout"].mean()
    guard = ex.two_proportion_z(so_champ, len(champ_sim), so_chall, len(chall_sim))
    stockout_pp = (so_chall - so_champ) * 100

    # ---- pre-registered decision -----------------------------------------
    cost_ok = paired.pct_change <= -MIN_COST_REDUCTION_PCT
    sig_ok = paired.p_value < ALPHA
    guard_ok = stockout_pp <= MAX_STOCKOUT_INCREASE_PP
    wmape_ok = chall_wmape <= champ_wmape * MAX_WMAPE_DEGRADATION
    ship = cost_ok and sig_ok and guard_ok and wmape_ok
    decision = "SHIP" if ship else "HOLD"

    # ---- cost-ratio sensitivity ------------------------------------------
    sens = []
    for cu, co in SENSITIVITY_RATIOS:
        cps, chps = run_costs(tbl, sigma, cu, co)
        pr = ex.paired_cost_test(cps["cost"], chps["cost"])
        cs = ex.simulate(tbl.rename(columns={"champ_pred": "p"}), "p", sigma, cu, co)["stockout"].mean()
        hs = ex.simulate(tbl.rename(columns={"chall_pred": "p"}), "p", sigma, cu, co)["stockout"].mean()
        sens.append(dict(ratio=f"{cu}:{co}", cost_pct_change=pr.pct_change, p=pr.p_value,
                         ci_low=pr.ci_low, ci_high=pr.ci_high, stockout_pp=(hs - cs) * 100,
                         decision="SHIP" if (pr.pct_change <= -MIN_COST_REDUCTION_PCT
                                             and pr.p_value < ALPHA and (hs - cs) * 100
                                             <= MAX_STOCKOUT_INCREASE_PP and wmape_ok) else "HOLD"))
    sens = pd.DataFrame(sens)

    # ---- per-segment savings (paired) ------------------------------------
    seg_rows = []
    for col in ["cat_id", "sbc_class", "volume_tier"]:
        for val, g in units.groupby(col):
            d = (g["chall_cost"] - g["champ_cost"])
            seg_rows.append(dict(segment=col, value=val, n=len(g),
                                 champ_cost=g["champ_cost"].mean(), chall_cost=g["chall_cost"].mean(),
                                 cost_pct=d.sum() / g["champ_cost"].sum() * 100 if g["champ_cost"].sum() else np.nan))
    seg = pd.DataFrame(seg_rows)

    units.to_csv(OUT / "per_series_costs.csv", index=False)
    seg.to_csv(OUT / "segment_savings.csv", index=False)
    sens.to_csv(OUT / "cost_ratio_sensitivity.csv", index=False)

    # ---- figures ----------------------------------------------------------
    _figures(units, paired, unpaired, sens, seg, so_champ, so_chall,
             champion, challenger, decision)

    # ---- summary ----------------------------------------------------------
    md = OUT / "PHASE8_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 8 -- A/B Test: champion vs best challenger\n\n")
        f.write(f"**Control** = `{champion}` (champion, WMAPE {champ_wmape:.4f}).  "
                f"**Treatment** = `{challenger}` (most accurate challenger, WMAPE {chall_wmape:.4f}).\n\n")
        f.write("Newsvendor business sim, stockout:holding = 5:1. Per-series total cost over "
                f"{tbl['week_end_date'].nunique()} test weeks x {units.shape[0]} series.\n\n")
        f.write("## Pre-registered decision rule\n\n")
        f.write(f"SHIP iff: paired cost reduction >= {MIN_COST_REDUCTION_PCT:.0f}% AND Wilcoxon "
                f"p < {ALPHA} AND stockout-rate increase <= {MAX_STOCKOUT_INCREASE_PP:.0f}pp AND "
                f"challenger WMAPE <= {MAX_WMAPE_DEGRADATION:.2f}x champion.\n\n")
        f.write(f"## VERDICT: **{decision}**\n\n")
        f.write("| gate | value | pass |\n|---|---|---|\n")
        f.write(f"| cost reduction >= 5% | {-paired.pct_change:.1f}% | {cost_ok} |\n")
        f.write(f"| Wilcoxon p < 0.05 | {paired.p_value:.2e} | {sig_ok} |\n")
        f.write(f"| stockout increase <= 2pp | {stockout_pp:+.2f}pp | {guard_ok} |\n")
        f.write(f"| WMAPE not worse | {chall_wmape:.4f} vs {champ_wmape:.4f} | {wmape_ok} |\n\n")
        f.write("## Paired vs unpaired (cost)\n\n")
        f.write("| test | cost change | p-value | 95% CI (mean diff) | crosses 0? |\n|---|---|---|---|---|\n")
        f.write(f"| unpaired (live-like) | {unpaired.pct_change:+.1f}% | {unpaired.p_value:.2e} | "
                f"[{unpaired.ci_low:.1f}, {unpaired.ci_high:.1f}] | "
                f"{'yes' if unpaired.ci_low*unpaired.ci_high<0 else 'no'} |\n")
        f.write(f"| **paired (correct)** | **{paired.pct_change:+.1f}%** | **{paired.p_value:.2e}** | "
                f"[{paired.ci_low:.1f}, {paired.ci_high:.1f}] | "
                f"{'yes' if paired.ci_low*paired.ci_high<0 else 'no'} |\n\n")
        f.write(f"- Series where challenger is cheaper: {paired.extra['pct_series_cheaper']:.1f}%\n")
        f.write(f"- Unpaired power: MDE ~= {unpaired.extra['mde_pct']:.1f}% of control mean "
                f"(n/arm ~= {unpaired.n}).\n")
        f.write(f"- Stockout rate: champion {so_champ*100:.1f}% -> challenger {so_chall*100:.1f}% "
                f"({stockout_pp:+.2f}pp; two-prop p={guard['p_value']:.2e}).\n\n")
        f.write("## Cost-ratio sensitivity (the key assumption)\n\n")
        f.write("| stockout:holding | cost change | p | stockout pp | decision |\n|---|---|---|---|---|\n")
        for _, r in sens.iterrows():
            f.write(f"| {r['ratio']} | {r['cost_pct_change']:+.1f}% | {r['p']:.2e} | "
                    f"{r['stockout_pp']:+.2f}pp | {r['decision']} |\n")
        f.write("\n## Per-segment cost change (paired, negative = challenger cheaper)\n\n")
        f.write("| segment | value | n | cost % |\n|---|---|---|---|\n")
        for _, r in seg.iterrows():
            f.write(f"| {r['segment']} | {r['value']} | {int(r['n'])} | {r['cost_pct']:+.1f}% |\n")
        f.write("\n## Outputs\n\n- `per_series_costs.csv`, `segment_savings.csv`, "
                "`cost_ratio_sensitivity.csv`\n- figures 01-05\n")

    print("\n=== Phase 8 verdict ===")
    print(f"DECISION: {decision}")
    print(f"paired cost change {paired.pct_change:+.1f}% (p={paired.p_value:.2e}, "
          f"CI [{paired.ci_low:.1f},{paired.ci_high:.1f}]); "
          f"stockout {stockout_pp:+.2f}pp; {paired.extra['pct_series_cheaper']:.0f}% series cheaper")
    print("sensitivity:\n" + sens.to_string(index=False))


def _figures(units, paired, unpaired, sens, seg, so_champ, so_chall,
             champion, challenger, decision):
    # 01 cost distribution by model
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cap = np.percentile(units[["champ_cost", "chall_cost"]].values, 98)
    ax.hist(units["champ_cost"].clip(upper=cap), bins=40, alpha=0.6, label=f"champion ({champion})", color="#55A868")
    ax.hist(units["chall_cost"].clip(upper=cap), bins=40, alpha=0.6, label=f"challenger ({challenger})", color="#8172B3")
    ax.set_xlabel("per-series total cost"); ax.set_ylabel("series"); ax.legend()
    ax.set_title("Per-series cost distribution"); fig.tight_layout()
    fig.savefig(OUT / "01_cost_distribution.png", dpi=110); plt.close(fig)

    # 02 paired bootstrap CI (the money plot) -- CI bounds converted to %
    lo_pct = paired.ci_low / paired.mean_control * 100
    hi_pct = paired.ci_high / paired.mean_control * 100
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot([lo_pct, hi_pct], [1, 1], color="#C44E52", lw=3, label="95% CI")
    ax.plot([paired.pct_change], [1], "o", color="#C44E52", ms=9, label="paired point est.")
    ax.axvline(0, ls="--", c="grey"); ax.axvline(-5, ls=":", c="green", label="ship threshold -5%")
    ax.set_ylim(0.5, 1.5); ax.set_yticks([]); ax.set_xlabel("cost change % (challenger vs champion)")
    ax.set_title(f"Paired cost effect (95% CI)  ->  {decision}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "02_paired_ci.png", dpi=110); plt.close(fig)

    # 03 service guardrail: stockout rate
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(["champion", "challenger"], [so_champ * 100, so_chall * 100], color=["#55A868", "#8172B3"])
    ax.set_ylabel("stockout-week rate %"); ax.set_title("Service guardrail: stockout rate")
    for i, v in enumerate([so_champ * 100, so_chall * 100]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(OUT / "03_stockout_guardrail.png", dpi=110); plt.close(fig)

    # 04 sensitivity
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sens["ratio"], sens["cost_pct_change"],
           color=["#55A868" if d == "SHIP" else "#C44E52" for d in sens["decision"]])
    ax.axhline(-5, ls=":", c="green"); ax.axhline(0, c="grey", lw=0.7)
    ax.set_ylabel("paired cost change %"); ax.set_xlabel("stockout:holding ratio")
    ax.set_title("Cost-ratio sensitivity (green=SHIP)")
    for i, (v, d) in enumerate(zip(sens["cost_pct_change"], sens["decision"])):
        ax.text(i, v, f"{v:+.0f}%\n{d}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "04_sensitivity.png", dpi=110); plt.close(fig)

    # 05 per-segment savings
    sub = seg[seg["segment"] == "sbc_class"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sub["value"], sub["cost_pct"], color="#4C72B0")
    ax.axhline(0, c="grey", lw=0.7); ax.set_ylabel("cost change %")
    ax.set_title("Per-segment cost change by intermittency class")
    fig.tight_layout(); fig.savefig(OUT / "05_segment_savings.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
