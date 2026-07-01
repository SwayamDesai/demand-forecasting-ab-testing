"""
Phase 3 -- EDA on the weekly data.

Every figure here exists to justify a later modeling decision -- not decoration.
Questions answered, and what each one decides:

  1. Weekly demand distribution / zero-inflation -> metric choice (WMAPE, not RMSE)
  2. Volume Pareto                                -> volume-weighting + where to focus
  3. Intermittency classes (SBC: ADI x CV^2)      -> do we need Croston? segment models?
  4. Yearly seasonality (week-of-year)            -> seasonal models + calendar features
  5. Calendar effects (events, SNAP, price)       -> which exogenous features matter
  6. Category / store segment patterns            -> global model + segment features
  7. Aggregate trend over time                    -> level/trend, sanity on the series

Outputs: reports/phase3_eda/*.png + *.csv + PHASE3_SUMMARY.md
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

OUT = config.REPORTS / "phase3_eda"

# Syntetos-Boylan-Croston (SBC) cutoffs -- the textbook intermittency taxonomy
ADI_CUT = 1.32
CV2_CUT = 0.49


# ---- series-level profile (the backbone of the analysis) --------------------

def series_profile(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in w.groupby("id", sort=False):
        s = g["sales"].to_numpy()
        n = len(s)
        nz = s[s > 0]
        n_nz = len(nz)
        zero_rate = 1 - n_nz / n
        adi = n / n_nz if n_nz else np.inf
        cv2 = (nz.std() / nz.mean()) ** 2 if n_nz > 1 and nz.mean() > 0 else 0.0
        if n_nz == 0:
            cls = "dead"
        elif adi < ADI_CUT and cv2 < CV2_CUT:
            cls = "smooth"
        elif adi >= ADI_CUT and cv2 < CV2_CUT:
            cls = "intermittent"
        elif adi < ADI_CUT and cv2 >= CV2_CUT:
            cls = "erratic"
        else:
            cls = "lumpy"
        rows.append(dict(
            id=sid, cat_id=g["cat_id"].iloc[0], store_id=g["store_id"].iloc[0],
            n_weeks=n, total_units=int(s.sum()), mean_weekly=s.mean(),
            zero_rate=zero_rate, adi=adi, cv2=cv2, sbc_class=cls))
    return pd.DataFrame(rows)


# ---- figures ----------------------------------------------------------------

def fig_distribution(w, prof):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    zr = (w["sales"] == 0).mean()
    cap = np.percentile(w["sales"], 99)
    ax[0].hist(w["sales"].clip(upper=cap), bins=60, color="#4C72B0")
    ax[0].set_title(f"Weekly unit-sales distribution\nzero weeks = {zr*100:.1f}% of observations")
    ax[0].set_xlabel("units / week (capped at p99)"); ax[0].set_ylabel("week count")
    ax[1].hist(prof["zero_rate"] * 100, bins=30, color="#C44E52")
    ax[1].set_title("Per-series zero-week rate")
    ax[1].set_xlabel("% of weeks with zero sales"); ax[1].set_ylabel("series")
    fig.tight_layout(); fig.savefig(OUT / "01_weekly_distribution.png", dpi=110); plt.close(fig)


def fig_pareto(prof):
    p = prof.sort_values("total_units", ascending=False).reset_index(drop=True)
    cum = p["total_units"].cumsum() / p["total_units"].sum()
    share_series = (np.arange(len(p)) + 1) / len(p)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(share_series * 100, cum * 100, color="#55A868", lw=2)
    # where does 80% of volume come from?
    idx80 = int((cum >= 0.8).idxmax())
    pct80 = share_series[idx80] * 100
    ax.axhline(80, ls="--", c="grey"); ax.axvline(pct80, ls="--", c="grey")
    ax.set_title(f"Volume Pareto: top {pct80:.0f}% of SKUs = 80% of units")
    ax.set_xlabel("% of series (ranked by volume)"); ax.set_ylabel("cumulative % of units")
    fig.tight_layout(); fig.savefig(OUT / "02_volume_pareto.png", dpi=110); plt.close(fig)
    return pct80


def fig_sbc(prof):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"smooth": "#55A868", "intermittent": "#4C72B0",
              "erratic": "#DD8452", "lumpy": "#C44E52", "dead": "grey"}
    for cls, c in colors.items():
        sub = prof[prof["sbc_class"] == cls]
        if len(sub):
            ax[0].scatter(sub["adi"], sub["cv2"], s=12, alpha=0.5, c=c, label=cls)
    ax[0].axvline(ADI_CUT, ls="--", c="k", lw=0.8); ax[0].axhline(CV2_CUT, ls="--", c="k", lw=0.8)
    ax[0].set_xlim(0.8, min(prof["adi"].replace(np.inf, np.nan).max(), 8))
    ax[0].set_ylim(0, min(prof["cv2"].max(), 4))
    ax[0].set_xlabel("ADI (avg weeks between sales)"); ax[0].set_ylabel("CV^2 of demand size")
    ax[0].set_title("SBC intermittency map"); ax[0].legend(fontsize=8)
    counts = prof["sbc_class"].value_counts()
    ax[1].bar(counts.index, counts.values, color=[colors[c] for c in counts.index])
    ax[1].set_title("Series per intermittency class"); ax[1].set_ylabel("series")
    for i, v in enumerate(counts.values):
        ax[1].text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(OUT / "03_intermittency_sbc.png", dpi=110); plt.close(fig)
    return counts


def fig_seasonality(w):
    woy = (w.groupby(["cat_id", "weekofyear"])["sales"].mean().reset_index())
    woy["norm"] = woy.groupby("cat_id")["sales"].transform(lambda s: s / s.mean())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for cat, g in woy.groupby("cat_id"):
        ax.plot(g["weekofyear"], g["norm"], label=cat, lw=1.6)
    ax.axhline(1.0, ls=":", c="grey")
    ax.set_title("Yearly seasonality by category (mean weekly sales, indexed to 1.0)")
    ax.set_xlabel("week of year"); ax.set_ylabel("seasonal index"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "04_seasonality_woy.png", dpi=110); plt.close(fig)
    return woy


def fig_calendar_effects(w):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    # event weeks vs not, per category (normalized demand index controls for scale)
    sm = w.groupby("id")["sales"].transform("mean").replace(0, np.nan)
    w = w.assign(demand_idx=w["sales"] / sm)
    ev = w.assign(event=(w["event_days"] > 0)).groupby(["cat_id", "event"])["demand_idx"].mean().unstack()
    ev.plot(kind="bar", ax=ax[0], color=["#4C72B0", "#DD8452"])
    ax[0].set_title("Holiday-week demand lift (indexed)"); ax[0].set_ylabel("mean demand index")
    ax[0].legend(["no event", "event week"]); ax[0].set_xlabel("")
    ax[0].tick_params(axis="x", rotation=0)
    # SNAP DOSE-RESPONSE: a binary "any SNAP" flag is misleading (boundary weeks
    # with 1-5 SNAP days behave differently from weeks fully inside the SNAP
    # window). Show mean sales vs the *count* of SNAP days, FOODS vs non-FOODS.
    w2 = w.assign(grp=np.where(w["cat_id"] == "FOODS", "FOODS", "non-FOODS"))
    dose = w2.groupby(["grp", "snap_days"])["sales"].mean().unstack(0)
    for col, c in [("FOODS", "#55A868"), ("non-FOODS", "#4C72B0")]:
        if col in dose:
            ax[1].plot(dose.index, dose[col], marker="o", label=col, color=c)
    ax[1].set_title("SNAP dose-response: mean weekly sales by # SNAP days")
    ax[1].set_xlabel("SNAP days in week (0-7)"); ax[1].set_ylabel("mean weekly units")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "05_calendar_effects.png", dpi=110); plt.close(fig)
    return ev, dose


def within_series_lift(w: pd.DataFrame, hi_mask: pd.Series, lo_mask: pd.Series) -> dict:
    """
    Robust effect size: per series, mean(sales | hi) / mean(sales | lo), then take
    the median across series. Equal-weights series (so a few big SKUs can't flip the
    sign) and is immune to the scale problems of an indexed-then-averaged estimate.
    """
    d = w.assign(_hi=hi_mask.values, _lo=lo_mask.values)
    def ratio(g):
        a = g.loc[g["_hi"], "sales"].mean()
        b = g.loc[g["_lo"], "sales"].mean()
        return a / b if (b and b > 0 and not np.isnan(a)) else np.nan
    r = d.groupby("id").apply(ratio, include_groups=False).dropna()
    return {"median_pct": (r.median() - 1) * 100, "mean_pct": (r.mean() - 1) * 100, "n": len(r)}


def fig_price_effect(w):
    sm = w.groupby("id")["sales"].transform("mean").replace(0, np.nan)
    w = w.assign(demand_idx=w["sales"] / sm)
    bins = [0, 0.9, 0.97, 1.03, 1.1, np.inf]
    labels = ["<0.90 (deep promo)", "0.90-0.97", "0.97-1.03 (flat)", "1.03-1.10", ">1.10 (hiked)"]
    w = w.dropna(subset=["price_ratio_8w", "demand_idx"])
    w = w.assign(pbin=pd.cut(w["price_ratio_8w"], bins=bins, labels=labels))
    tab = w.groupby("pbin", observed=True)["demand_idx"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(len(tab)), tab.values, color="#8172B3")
    ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index, rotation=20, ha="right", fontsize=8)
    ax.axhline(1.0, ls=":", c="grey")
    ax.set_title("Price vs demand: price relative to trailing 8-week mean")
    ax.set_ylabel("mean demand index")
    fig.tight_layout(); fig.savefig(OUT / "06_price_effect.png", dpi=110); plt.close(fig)
    return tab


def fig_segments_and_trend(w, prof):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    seg = prof.groupby("cat_id").agg(mean_weekly=("mean_weekly", "mean"),
                                     zero_rate=("zero_rate", "mean"))
    x = np.arange(len(seg)); width = 0.4
    ax[0].bar(x - width/2, seg["mean_weekly"], width, label="mean weekly units", color="#4C72B0")
    ax2 = ax[0].twinx()
    ax2.bar(x + width/2, seg["zero_rate"] * 100, width, label="zero-week %", color="#C44E52")
    ax[0].set_xticks(x); ax[0].set_xticklabels(seg.index)
    ax[0].set_ylabel("mean weekly units"); ax2.set_ylabel("zero-week %")
    ax[0].set_title("Category profile: volume vs sparsity")
    ax[0].legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    # aggregate trend
    tot = w.groupby("week_end_date")["sales"].sum()
    ax[1].plot(tot.index, tot.values, color="#999", lw=0.7, label="weekly total")
    ax[1].plot(tot.index, tot.rolling(8).mean(), color="#C44E52", lw=2, label="8-week mean")
    ax[1].set_title("Total weekly demand over time"); ax[1].set_ylabel("units"); ax[1].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(OUT / "07_segments_trend.png", dpi=110); plt.close(fig)
    return seg


# ---- driver -----------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(config.PROCESSED / "weekly_features.parquet")

    print("[1/8] series profile + SBC classes...")
    prof = series_profile(w)
    prof.to_csv(OUT / "series_profile.csv", index=False)

    print("[2/8] distribution...");      fig_distribution(w, prof)
    print("[3/8] volume Pareto...");      pct80 = fig_pareto(prof)
    print("[4/8] intermittency SBC...");  counts = fig_sbc(prof)
    print("[5/8] seasonality...");        woy = fig_seasonality(w); woy.to_csv(OUT / "seasonality_woy.csv", index=False)
    print("[6/8] calendar effects...");   ev, dose = fig_calendar_effects(w)
    print("[6/8] price effect...");       ptab = fig_price_effect(w)
    print("[7/8] segments + trend...");   seg = fig_segments_and_trend(w, prof)

    # ---- robust (within-series) effect sizes for the writeup ---------------
    food = w[w["cat_id"] == "FOODS"]
    snap_food = within_series_lift(food, food["snap_days"] > 0, food["snap_days"] == 0)
    wp = w.dropna(subset=["price_ratio_8w"])
    promo = within_series_lift(wp, wp["price_ratio_8w"] < 0.90,
                               wp["price_ratio_8w"].between(0.97, 1.03))
    event = within_series_lift(w, w["event_days"] > 0, w["event_days"] == 0)
    promo_rarity = (wp["price_ratio_8w"] < 0.90).mean() * 100
    snap_full = dose["FOODS"].reindex([0, 7]).tolist() if "FOODS" in dose else [np.nan, np.nan]

    # segment summary table
    seg_full = prof.groupby(["cat_id"]).agg(
        n_series=("id", "nunique"), total_units=("total_units", "sum"),
        mean_weekly=("mean_weekly", "mean"), zero_rate=("zero_rate", "mean")).reset_index()
    seg_full.to_csv(OUT / "segment_summary.csv", index=False)

    # consistency cross-check
    assert prof["id"].nunique() == 900, "lost series in profiling"
    assert counts.sum() == 900, "SBC classes don't sum to 900"

    vw_zero = np.average(prof["zero_rate"], weights=prof["total_units"]) * 100

    print("[8/8] write summary...")
    md = OUT / "PHASE3_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 3 -- Weekly EDA\n\n")
        f.write("Each finding maps to a modeling decision in Phases 4-7.\n\n")
        f.write("## Headline findings\n\n")
        f.write(f"1. **Still zero-inflated, but workable.** {(w['sales']==0).mean()*100:.1f}% of "
                f"weekly observations are zero (vs 62% daily). Volume-weighted zero-rate is only "
                f"{vw_zero:.1f}% -- the units that matter are far less sparse. -> use **WMAPE**, "
                f"report bias; RMSE as secondary.\n")
        f.write(f"2. **Volume is concentrated.** The top {pct80:.0f}% of SKUs carry 80% of all units. "
                f"-> volume-weighted evaluation and an A/B that is stratified by volume tier.\n")
        f.write(f"3. **Intermittency taxonomy (SBC):** "
                + ", ".join(f"{k}={int(v)}" for k, v in counts.items())
                + f". -> a Croston/SBA baseline is justified for the {int(counts.get('intermittent',0))+int(counts.get('lumpy',0))} "
                  f"intermittent/lumpy series; smooth series suit ETS/ML.\n")
        f.write("4. **Clear yearly seasonality** (week-of-year), strongest in FOODS -> seasonal "
                "models (ETS season=52, seasonal-naive) and woy features earn their place.\n")
        f.write(f"5. **Exogenous signals are real but must be encoded carefully (honest read):**\n")
        f.write(f"   - **SNAP is nonlinear, not binary.** Within-series, FOODS sells "
                f"+{snap_food['median_pct']:.0f}% (median) in weeks with any SNAP day, but the effect is "
                f"a dose-response: full-SNAP weeks (7 days) average {snap_full[1]:.1f} units vs "
                f"{snap_full[0]:.1f} at 0 days (~+{(snap_full[1]/snap_full[0]-1)*100:.0f}%), while partial "
                f"boundary weeks are flat-to-lower. A naive 'any SNAP' flag is volume-weighted ~0% and "
                f"misleading. -> feed **snap_days (count)**, not a binary flag.\n")
        f.write(f"   - **Promotions matter but are rare and heterogeneous.** Deep promos (<0.90 of trailing "
                f"price) occur in only {promo_rarity:.1f}% of active weeks; within-series lift is "
                f"+{promo['median_pct']:.0f}% median / +{promo['mean_pct']:.0f}% mean (a few big responders). "
                f"-> keep price features, but don't oversell elasticity.\n")
        f.write(f"   - **Holiday/event weeks are ~neutral at the median series** "
                f"({event['median_pct']:+.0f}% within-series median): weekly aggregation dilutes "
                f"single-day holiday spikes and some holidays (e.g. Christmas closures) depress sales, "
                f"so effects roughly cancel. -> keep event features for the ML/DL models to exploit "
                f"selectively, but expect little from them in the linear baselines.\n\n")
        f.write("## Category profile\n\n| cat | series | total units | mean weekly | zero-week % |\n|---|---|---|---|---|\n")
        for _, r in seg_full.iterrows():
            f.write(f"| {r['cat_id']} | {int(r['n_series'])} | {int(r['total_units']):,} | "
                    f"{r['mean_weekly']:.2f} | {r['zero_rate']*100:.1f}% |\n")
        f.write("\n## Figures\n\n")
        for p in sorted(OUT.glob("*.png")):
            f.write(f"- `{p.name}`\n")
        f.write("\n## Tables\n\n- `series_profile.csv` (per-series ADI/CV^2/class)\n"
                "- `segment_summary.csv`\n- `seasonality_woy.csv`\n")

    print("\n=== Phase 3 summary ===")
    print(f"weekly zero-rate        : {(w['sales']==0).mean()*100:.1f}%  "
          f"(volume-weighted only {vw_zero:.1f}%)")
    print(f"volume Pareto (80% vol) : top {pct80:.0f}% of SKUs")
    print(f"SBC classes             : {dict(counts)}")
    print(f"SNAP on FOODS (within)  : +{snap_food['median_pct']:.0f}% median; "
          f"dose 0->7 days: {snap_full[0]:.1f}->{snap_full[1]:.1f} units")
    print(f"deep-promo (rare {promo_rarity:.1f}%) : +{promo['median_pct']:.0f}% median / "
          f"+{promo['mean_pct']:.0f}% mean within-series")
    print(f"event-week lift (within): +{event['median_pct']:.0f}% median")
    print(f"figures + tables in     : {OUT}")


if __name__ == "__main__":
    main()
