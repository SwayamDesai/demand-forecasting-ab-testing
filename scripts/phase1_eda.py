"""
Phase 1 EDA -- emits 8 PNGs to reports/phase1_eda/ + prints interpretations.

Lean by design: only the diagnostics that actually inform later modeling choices.
Run: `python -m scripts.phase1_eda`
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

DATA = Path("data/processed/m5_long_sample.parquet")
OUT = Path("reports/phase1_eda")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    p = OUT / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def pick_representative_skus(df: pd.DataFrame) -> dict[str, str]:
    """Pick 3 SKUs: high-volume (fast), median, low-but-active (intermittent)."""
    active = df[df["is_active"]]
    vol = active.groupby("id")["sales"].sum().sort_values()
    vol = vol[vol > 0]                                    # exclude truly dead series
    return {
        "fast":         vol.index[-1],                    # top seller
        "moderate":     vol.index[len(vol) // 2],         # median
        "intermittent": vol.index[len(vol) // 20],        # bottom 5% but nonzero
    }


def daily_series(df: pd.DataFrame, sku: str) -> pd.Series:
    """Active-only daily sales series for one SKU, indexed by date."""
    s = df[(df["id"] == sku) & df["is_active"]].set_index("date")["sales"].sort_index()
    return s.asfreq("D").fillna(0)                        # ensure no missing days


# ============================================================================
# Plot 1 -- Missingness / pre-launch effect
# ============================================================================
def plot_missingness(df: pd.DataFrame):
    print("\n[plot 1/8] missingness summary")
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]

    # also: pre-launch share over time (rows where is_active=False)
    prelaunch = (
        df.assign(prelaunch=(~df["is_active"]).astype(int))
          .groupby("date")["prelaunch"].mean()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    miss.mul(100).plot.barh(ax=ax1, color="#c44e52")
    ax1.set_xlabel("% missing"); ax1.set_title("Missing values by column")

    prelaunch.mul(100).plot(ax=ax2, color="#4c72b0")
    ax2.set_ylabel("% of SKUs not-yet-launched")
    ax2.set_title("Pre-launch share over time (decays as items roll out)")
    save(fig, "01_missingness")

    print(f"  -> sell_price missing {miss.get('sell_price', 0)*100:.1f}% (= pre-launch + temporarily-out-of-catalog)")
    print(f"  -> event cols missing ~97% (normal; events are rare)")


# ============================================================================
# Plot 2 -- Sales distribution (the zero spike)
# ============================================================================
def plot_sales_distribution(df: pd.DataFrame):
    print("\n[plot 2/8] sales distribution (intermittency check)")
    s = df.loc[df["is_active"], "sales"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    vc = s.value_counts().sort_index().head(15)
    vc.plot.bar(ax=ax1, color="#55a868")
    ax1.set_title(f"Daily-sales value counts (first 15 vals); {(s==0).mean()*100:.1f}% are zero")
    ax1.set_xlabel("units sold in a day"); ax1.set_ylabel("# rows")

    # log-scale histogram for tail
    s_pos = s[s > 0]
    ax2.hist(s_pos, bins=60, color="#8172b2")
    ax2.set_yscale("log")
    ax2.set_title("Distribution of nonzero daily sales (log y)")
    ax2.set_xlabel("units sold"); ax2.set_ylabel("count (log)")
    save(fig, "02_sales_distribution")

    print(f"  -> zero-sales day share: {(s==0).mean()*100:.1f}%   <-- intermittent demand")
    print(f"  -> mean(nonzero)={s_pos.mean():.2f}, median(nonzero)={s_pos.median():.1f}, max={s.max()}")


# ============================================================================
# Plot 3 -- SKU pareto (top sellers vs long tail)
# ============================================================================
def plot_sku_pareto(df: pd.DataFrame):
    print("\n[plot 3/8] SKU pareto (long-tail check)")
    vol = df.loc[df["is_active"]].groupby("id")["sales"].sum().sort_values(ascending=False)
    cum_share = vol.cumsum() / vol.sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(vol)+1) / len(vol) * 100, cum_share.values * 100, color="#dd8452")
    ax.axhline(80, ls="--", color="grey", lw=1); ax.axvline(20, ls="--", color="grey", lw=1)
    ax.set_xlabel("% of SKUs (sorted high -> low)"); ax.set_ylabel("cumulative % of total units sold")
    ax.set_title("SKU Pareto curve")
    save(fig, "03_sku_pareto")

    # actual top-X-pct stat
    top20 = cum_share.iloc[int(len(vol) * 0.20)] * 100
    print(f"  -> top 20% of SKUs account for {top20:.1f}% of total volume (classic long tail)")


# ============================================================================
# Plot 4 -- 3-SKU time series with 30-day rolling mean
# ============================================================================
def plot_three_sku_timeseries(df: pd.DataFrame, skus: dict[str, str]):
    print("\n[plot 4/8] time-series + 30d rolling mean (3 representative SKUs)")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, (label, sku) in zip(axes, skus.items()):
        s = daily_series(df, sku)
        ax.plot(s.index, s.values, lw=0.6, alpha=0.6, label="daily")
        ax.plot(s.index, s.rolling(30).mean(), lw=2, color="black", label="30d rolling mean")
        ax.set_title(f"[{label}] {sku}   (total units = {int(s.sum())})")
        ax.set_ylabel("units / day"); ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("date")
    save(fig, "04_three_sku_timeseries")
    print("  -> look for: trend, level shifts (launches/promo), variance change, gaps")


# ============================================================================
# Plot 5 -- STL decomposition on the fast-mover
# ============================================================================
def plot_stl(df: pd.DataFrame, sku: str):
    print("\n[plot 5/8] STL decomposition (fast-mover only; STL is unreliable on intermittent series)")
    s = daily_series(df, sku)

    # STL needs a period. Weekly seasonality dominates retail -> period=7.
    res = STL(s, period=7, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    for ax, comp, name, color in zip(
        axes,
        [s, res.trend, res.seasonal, res.resid],
        ["observed", "trend", "seasonal (weekly)", "residual"],
        ["#4c72b0", "#dd8452", "#55a868", "#8172b2"],
    ):
        ax.plot(comp.index, comp.values, color=color, lw=0.8)
        ax.set_ylabel(name)
    axes[0].set_title(f"STL decomposition — {sku}")
    save(fig, "05_stl_fast_sku")

    # also report seasonality strength F_s
    var_e = np.var(res.resid)
    var_es = np.var(res.resid + res.seasonal)
    Fs = max(0.0, 1 - var_e / var_es) if var_es > 0 else 0.0
    print(f"  -> seasonality strength F_s = {Fs:.2f}  (rule of thumb: >0.6 means clearly seasonal)")


# ============================================================================
# Plot 6 -- ACF/PACF on the fast-mover, raw vs first-differenced
# ============================================================================
def plot_acf_pacf(df: pd.DataFrame, sku: str):
    print("\n[plot 6/8] ACF + PACF (raw and first-differenced)")
    s = daily_series(df, sku)
    sd = s.diff().dropna()

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    plot_acf (s,  lags=40, ax=axes[0, 0]); axes[0, 0].set_title("ACF (raw)")
    plot_pacf(s,  lags=40, ax=axes[0, 1], method="ywm"); axes[0, 1].set_title("PACF (raw)")
    plot_acf (sd, lags=40, ax=axes[1, 0]); axes[1, 0].set_title("ACF (1st-differenced)")
    plot_pacf(sd, lags=40, ax=axes[1, 1], method="ywm"); axes[1, 1].set_title("PACF (1st-differenced)")
    save(fig, "06_acf_pacf")

    print("  -> expect: large spike at lag 7 (weekly seasonality), decay pattern dictates ARIMA p,q")


# ============================================================================
# Plot 7 -- ADF + KPSS stationarity tests on 3 SKUs, raw vs differenced
# ============================================================================
def plot_stationarity_tests(df: pd.DataFrame, skus: dict[str, str]):
    print("\n[plot 7/8] ADF + KPSS stationarity tests (raw vs 1st-difference vs seasonal-diff)")
    rows = []
    for label, sku in skus.items():
        s = daily_series(df, sku)
        variants = {"raw": s, "Δy": s.diff().dropna(), "Δ₇y": s.diff(7).dropna()}
        for vname, v in variants.items():
            try:
                adf_p = adfuller(v, autolag="AIC")[1]
            except Exception:
                adf_p = np.nan
            try:
                kpss_p = kpss(v, regression="c", nlags="auto")[1]
            except Exception:
                kpss_p = np.nan
            rows.append((label, vname, adf_p, kpss_p))
    res = pd.DataFrame(rows, columns=["sku", "variant", "ADF p", "KPSS p"])
    print(res.to_string(index=False))

    # save as heatmap so it's a *plot* per spec
    pivot = res.set_index(["sku", "variant"])[["ADF p", "KPSS p"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0.05,
                cbar_kws={"label": "p-value"}, ax=ax)
    ax.set_title("Stationarity tests  (ADF: low p = stationary;  KPSS: high p = stationary)")
    save(fig, "07_stationarity_tests")

    print("  -> read: cell where ADF<.05 AND KPSS>.05 is unambiguously stationary;")
    print("           disagreement is common on intermittent series — trust visuals + STL.")


# ============================================================================
# Plot 8 -- Calendar effects (weekday + month + SNAP)
# ============================================================================
def plot_calendar_effects(df: pd.DataFrame):
    print("\n[plot 8/8] calendar effects (weekday / month / SNAP)")
    a = df[df["is_active"]]
    by_dow = a.groupby("dow_name")["sales"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    by_month = a.groupby("month")["sales"].mean()
    by_snap = a.groupby(["cat_id", "has_snap"])["sales"].mean().unstack()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    by_dow.plot.bar(ax=axes[0], color="#4c72b0");  axes[0].set_title("Mean sales by weekday"); axes[0].set_ylabel("units")
    by_month.plot.bar(ax=axes[1], color="#dd8452"); axes[1].set_title("Mean sales by month")
    by_snap.plot.bar(ax=axes[2]);                   axes[2].set_title("SNAP day effect, by category")
    axes[2].legend(title="SNAP day", labels=["no", "yes"])
    save(fig, "08_calendar_effects")

    snap_lift = (by_snap[1] / by_snap[0] - 1) * 100
    print("  -> SNAP-day sales lift by category (%):")
    print(snap_lift.round(1).to_string())


# ============================================================================
def main():
    print(f"loading {DATA}...")
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {len(df):,} rows, {df['id'].nunique()} SKUs, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    skus = pick_representative_skus(df)
    print(f"\nrepresentative SKUs picked by activity:")
    for k, v in skus.items():
        tot = df.loc[(df["id"] == v) & df["is_active"], "sales"].sum()
        print(f"  {k:13s} = {v}  (total units = {tot})")

    plot_missingness(df)
    plot_sales_distribution(df)
    plot_sku_pareto(df)
    plot_three_sku_timeseries(df, skus)
    plot_stl(df, skus["fast"])
    plot_acf_pacf(df, skus["fast"])
    plot_stationarity_tests(df, skus)
    plot_calendar_effects(df)

    print(f"\n✓ phase 1 EDA complete — {len(list(OUT.glob('*.png')))} plots in {OUT}/")


if __name__ == "__main__":
    main()
