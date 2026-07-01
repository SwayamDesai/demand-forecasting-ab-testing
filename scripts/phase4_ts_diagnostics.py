"""
Phase 4 -- Time-series preprocessing / diagnostics.

Purpose: stop guessing model families. Use the data to answer four questions,
each of which fixes a concrete choice in Phase 5:

  Q1 Stationarity (ADF + KPSS)      -> does ARIMA need differencing (d, D)?
  Q2 Seasonality (STL strength @52) -> are seasonal terms / seasonal-naive worth it?
  Q3 Autocorrelation (ACF / PACF)   -> rough AR/MA orders; is there a lag-52 spike?
  Q4 Variance vs level              -> model raw, or log / Tweedie?

Diagnostics run on (a) the aggregate weekly demand -- a clean, smooth signal -- and
(b) the full panel of 900 series, summarised as distributions (so we don't over-read
one cherry-picked SKU). Intermittent series are flagged where classical tools strain.

Outputs: reports/phase4_ts_diagnostics/*.png + *.csv + PHASE4_SUMMARY.md
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.tsa.seasonal import STL

from src import config

OUT = config.REPORTS / "phase4_ts_diagnostics"
SEASON = config.SEASON_WEEKS          # 52
MIN_FOR_TEST = 60                     # min weeks for a meaningful stationarity test
MIN_FOR_STL = 2 * SEASON + 6          # need >2 full years for period-52 STL


# ---- helpers ----------------------------------------------------------------

def aggregate_series(w: pd.DataFrame) -> pd.Series:
    """Total weekly demand across all series -- a smooth signal for decomposition."""
    s = w.groupby("week_end_date")["sales"].sum().sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s


def adf_pvalue(s: np.ndarray) -> float:
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return np.nan


def kpss_pvalue(s: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # "p-value outside table" -> clamped
        try:
            return float(kpss(s, regression="c", nlags="auto")[1])
        except Exception:
            return np.nan


def stl_strength(s: np.ndarray) -> tuple[float, float]:
    """Hyndman trend/seasonal strength in [0,1] from an STL fit (period=52)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = STL(pd.Series(s), period=SEASON, robust=True).fit()
    r = res.resid
    f_trend = max(0.0, 1 - r.var() / (res.trend + r).var())
    f_seas = max(0.0, 1 - r.var() / (res.seasonal + r).var())
    return float(f_trend), float(f_seas)


# ---- Q1: stationarity across the whole panel --------------------------------

def stationarity_panel(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in w.groupby("id", sort=False):
        s = g.sort_values("week_end_date")["sales"].to_numpy(dtype=float)
        if len(s) < MIN_FOR_TEST or np.allclose(s, s[0]):
            continue
        ds = np.diff(s)
        rows.append(dict(
            id=sid,
            adf_p_level=adf_pvalue(s), kpss_p_level=kpss_pvalue(s),
            adf_p_diff=adf_pvalue(ds), kpss_p_diff=kpss_pvalue(ds),
        ))
    df = pd.DataFrame(rows)
    # verdicts: ADF stationary if p<0.05 (reject unit root); KPSS stationary if p>0.05
    df["adf_stat_level"] = df["adf_p_level"] < 0.05
    df["kpss_stat_level"] = df["kpss_p_level"] > 0.05
    df["adf_stat_diff"] = df["adf_p_diff"] < 0.05
    df["kpss_stat_diff"] = df["kpss_p_diff"] > 0.05
    df["both_stat_level"] = df["adf_stat_level"] & df["kpss_stat_level"]
    df["both_stat_diff"] = df["adf_stat_diff"] & df["kpss_stat_diff"]
    return df


# ---- figures ----------------------------------------------------------------

def fig_stl_aggregate(agg: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = STL(agg, period=SEASON, robust=True).fit()
    ft, fs = stl_strength(agg.to_numpy())
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(agg.index, agg.values, color="#333"); ax[0].set_ylabel("observed")
    ax[1].plot(agg.index, res.trend, color="#C44E52"); ax[1].set_ylabel("trend")
    ax[2].plot(agg.index, res.seasonal, color="#4C72B0"); ax[2].set_ylabel("seasonal")
    ax[3].plot(agg.index, res.resid, color="#999"); ax[3].set_ylabel("resid")
    ax[0].set_title(f"STL of aggregate weekly demand  (trend strength={ft:.2f}, "
                    f"seasonal strength={fs:.2f})")
    fig.tight_layout(); fig.savefig(OUT / "01_stl_aggregate.png", dpi=110); plt.close(fig)
    return ft, fs


def fig_acf_pacf(agg: pd.Series, rep: pd.Series, rep_name: str):
    nlags = 60
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for col, (s, name) in enumerate([(agg, "aggregate"), (rep, rep_name)]):
        a = acf(s, nlags=nlags, fft=True)
        p = pacf(s, nlags=nlags)
        ax[0, col].stem(range(len(a)), a); ax[0, col].set_title(f"ACF -- {name}")
        ax[1, col].stem(range(len(p)), p); ax[1, col].set_title(f"PACF -- {name}")
        ci = 1.96 / np.sqrt(len(s))
        for r in (ax[0, col], ax[1, col]):
            r.axhline(ci, ls="--", c="grey", lw=0.7); r.axhline(-ci, ls="--", c="grey", lw=0.7)
            r.axvline(SEASON, ls=":", c="red", lw=0.8)   # mark the yearly lag
            r.set_xlabel("lag (weeks)")
    fig.tight_layout(); fig.savefig(OUT / "02_acf_pacf.png", dpi=110); plt.close(fig)


def fig_stationarity_summary(sp: pd.DataFrame):
    cats = ["ADF\n(level)", "KPSS\n(level)", "ADF\n(diff)", "KPSS\n(diff)"]
    vals = [sp["adf_stat_level"].mean(), sp["kpss_stat_level"].mean(),
            sp["adf_stat_diff"].mean(), sp["kpss_stat_diff"].mean()]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(cats, [v * 100 for v in vals],
                  color=["#4C72B0", "#4C72B0", "#55A868", "#55A868"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v*100, f"{v*100:.0f}%", ha="center", va="bottom")
    ax.set_ylabel("% of series judged stationary"); ax.set_ylim(0, 105)
    ax.set_title(f"Stationarity across {len(sp)} series: level vs first-differenced")
    fig.tight_layout(); fig.savefig(OUT / "03_stationarity_summary.png", dpi=110); plt.close(fig)


def fig_strength_dist(strength: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for cat, g in strength.groupby("cat_id"):
        ax[0].scatter(g["f_trend"], g["f_seasonal"], s=12, alpha=0.5, label=cat)
    ax[0].set_xlabel("trend strength"); ax[0].set_ylabel("seasonal strength")
    ax[0].set_title("STL strength per series (eligible series)"); ax[0].legend(fontsize=8)
    ax[0].axhline(0.3, ls=":", c="grey"); ax[0].axvline(0.3, ls=":", c="grey")
    ax[1].hist(strength["f_seasonal"], bins=30, color="#4C72B0", alpha=0.8, label="seasonal")
    ax[1].hist(strength["f_trend"], bins=30, color="#C44E52", alpha=0.6, label="trend")
    ax[1].set_title("Distribution of STL strengths"); ax[1].set_xlabel("strength"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "04_strength_distribution.png", dpi=110); plt.close(fig)


def fig_mean_variance(w: pd.DataFrame):
    g = w.groupby("id")["sales"].agg(["mean", "std"]).replace(0, np.nan).dropna()
    g = g[g["mean"] > 0]
    lm, ls = np.log(g["mean"]), np.log(g["std"])
    slope, intercept = np.polyfit(lm, ls, 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(g["mean"], g["std"], s=10, alpha=0.4, color="#8172B3")
    xs = np.linspace(g["mean"].min(), g["mean"].max(), 50)
    ax.plot(xs, np.exp(intercept) * xs**slope, c="k",
            label=f"log-log slope = {slope:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("per-series mean weekly sales"); ax.set_ylabel("per-series std")
    ax.set_title("Variance grows with level -> multiplicative noise")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "05_mean_variance.png", dpi=110); plt.close(fig)
    return float(slope)


# ---- driver -----------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(config.PROCESSED / "weekly_features.parquet")
    prof = pd.read_csv(config.REPORTS / "phase3_eda" / "series_profile.csv")

    print("[1/6] aggregate signal + STL...")
    agg = aggregate_series(w)
    ft_agg, fs_agg = fig_stl_aggregate(agg)

    print("[2/6] stationarity tests across panel (ADF + KPSS)...")
    sp = stationarity_panel(w)
    sp.to_csv(OUT / "stationarity_summary.csv", index=False)
    fig_stationarity_summary(sp)

    print("[3/6] STL strength across eligible series...")
    rows = []
    for sid, g in w.groupby("id", sort=False):
        s = g.sort_values("week_end_date")["sales"].to_numpy(dtype=float)
        if len(s) < MIN_FOR_STL:
            continue
        ft, fs = stl_strength(s)
        rows.append(dict(id=sid, cat_id=g["cat_id"].iloc[0], f_trend=ft, f_seasonal=fs))
    strength = pd.DataFrame(rows)
    strength.to_csv(OUT / "decomposition_strength.csv", index=False)
    fig_strength_dist(strength)

    print("[4/6] ACF / PACF (aggregate + representative smooth series)...")
    # representative = highest-volume smooth series (classical tools behave here)
    smooth = prof[prof["sbc_class"] == "smooth"].sort_values("total_units", ascending=False)
    rep_id = smooth.iloc[0]["id"]
    rep = w[w["id"] == rep_id].sort_values("week_end_date")["sales"].reset_index(drop=True)
    fig_acf_pacf(agg, rep, f"{rep_id[:22]} (smooth)")

    print("[5/6] mean-variance (transform) diagnostic...")
    mv_slope = fig_mean_variance(w)

    # ACF numbers for the writeup
    a = acf(agg, nlags=SEASON + 2, fft=True)
    acf1 = a[1]
    acf52 = a[SEASON]

    print("[6/6] write summary...")
    md = OUT / "PHASE4_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 4 -- Time-Series Diagnostics\n\n")
        f.write("Each result fixes a Phase-5 modeling choice.\n\n")
        f.write("## Q1 -- Stationarity (ADF + KPSS, "
                f"{len(sp)} series with >={MIN_FOR_TEST} weeks)\n\n")
        f.write(f"- At **level**: ADF says stationary for {sp['adf_stat_level'].mean()*100:.0f}% of "
                f"series; KPSS for {sp['kpss_stat_level'].mean()*100:.0f}%; both agree on "
                f"{sp['both_stat_level'].mean()*100:.0f}%.\n")
        f.write(f"- After **first differencing**: ADF {sp['adf_stat_diff'].mean()*100:.0f}%, "
                f"KPSS {sp['kpss_stat_diff'].mean()*100:.0f}%, both {sp['both_stat_diff'].mean()*100:.0f}%.\n")
        f.write("- **Implication:** weekly demand is mostly level-stationary (mean-reverting around a "
                "level, little trend). ARIMA needs at most **d=1**; let AutoARIMA pick per series. "
                "First differencing helps the minority with drift.\n\n")
        f.write("## Q2 -- Seasonality (STL, period=52)\n\n")
        f.write(f"- Aggregate demand: trend strength **{ft_agg:.2f}**, seasonal strength **{fs_agg:.2f}**.\n")
        f.write(f"- Per series (eligible n={len(strength)}): median seasonal strength "
                f"{strength['f_seasonal'].median():.2f}, median trend strength "
                f"{strength['f_trend'].median():.2f}; "
                f"{(strength['f_seasonal']>0.3).mean()*100:.0f}% of series have non-trivial yearly seasonality.\n")
        f.write("- **Implication:** seasonality is real but moderate per-SKU and strong in aggregate -> "
                "keep **seasonal-naive(52)** and **ETS(season=52)** as baselines; week-of-year features "
                "for ML. Don't force a seasonal term on every SKU.\n\n")
        f.write("## Q3 -- Autocorrelation (ACF / PACF on aggregate)\n\n")
        f.write(f"- ACF(lag 1) = {acf1:.2f} (strong short memory); ACF(lag 52) = {acf52:.2f} "
                f"(yearly echo). PACF cuts off quickly -> low-order AR.\n")
        f.write("- **Implication:** short lags (1,2,4) carry most signal -> they dominate the ML feature "
                "set; AutoARIMA search bounds can stay small (max_p,q <= 2-3).\n\n")
        f.write("## Q4 -- Variance vs level (transform)\n\n")
        f.write(f"- log(std) vs log(mean) across series has slope **{mv_slope:.2f}** (≈1 would be pure "
                f"multiplicative). Variance clearly grows with level.\n")
        f.write("- **Implication:** model on a variance-stabilised target -> **log1p** for the LSTM, "
                "**Tweedie** objective for LightGBM (handles zeros + multiplicative spread). Classical "
                "models run on raw units (interpretable) but we report bias to catch scale errors.\n\n")
        f.write("## Figures\n\n")
        for p in sorted(OUT.glob("*.png")):
            f.write(f"- `{p.name}`\n")

    print("\n=== Phase 4 summary ===")
    print(f"stationarity (level): ADF {sp['adf_stat_level'].mean()*100:.0f}% | "
          f"KPSS {sp['kpss_stat_level'].mean()*100:.0f}% | both {sp['both_stat_level'].mean()*100:.0f}%  "
          f"(n={len(sp)})")
    print(f"after differencing  : ADF {sp['adf_stat_diff'].mean()*100:.0f}% | "
          f"KPSS {sp['kpss_stat_diff'].mean()*100:.0f}%")
    print(f"aggregate STL       : trend={ft_agg:.2f}  seasonal={fs_agg:.2f}")
    print(f"per-series seasonal : median {strength['f_seasonal'].median():.2f}; "
          f"{(strength['f_seasonal']>0.3).mean()*100:.0f}% have yearly seasonality (n={len(strength)})")
    print(f"ACF lag1={acf1:.2f}  lag52={acf52:.2f}")
    print(f"mean-variance slope : {mv_slope:.2f} -> log1p / Tweedie justified")


if __name__ == "__main__":
    main()
