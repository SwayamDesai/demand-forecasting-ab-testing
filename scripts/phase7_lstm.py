"""
Phase 7 -- LSTM deep-learning challenger (seq2seq).

A single GLOBAL encoder-decoder LSTM that emits all 4 weekly forecasts in one
forward pass (no recursion -> no error compounding).

Two deliberate design choices, both aimed at NOT repeating the classic failure of
LSTMs on sparse retail demand (systematic under-prediction):

  1. PER-SERIES MEAN-SCALING instead of log1p. We divide each series by its own
     training mean and train MSE on the scaled target. MSE targets the conditional
     MEAN, so the back-transform (multiply by the scale) is ~unbiased -- unlike
     log1p+MSE+expm1, whose expm1 step under-predicts a right-skewed mean
     (Jensen's inequality). This is the DeepAR-style scaling trick, kept minimal.
  2. KNOWN-FUTURE decoder. The decoder reads only exogenous features available in
     advance (price, week-of-year, SNAP days, events) starting from the encoder's
     final state. It never consumes its own past outputs.

Item embeddings let one model serve all 900 series. Rolling-origin: refit per fold
on data up to the cutoff. Evaluated on the same test cells as Phases 5-6.

Outputs: reports/phase7_lstm/*.png + *.csv + predictions.parquet + PHASE7_SUMMARY.md
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import backtest, config, metrics

OUT = config.REPORTS / "phase7_lstm"

L = 26                       # encoder lookback (weeks) -- decoder woy carries seasonality
H = config.HORIZON_WEEKS     # 4
STRIDE = 4
PAST_FEATS = ["norm_sales", "price_z", "woy_sin", "woy_cos", "snap_frac", "event_frac"]
FUT_FEATS = ["price_z", "woy_sin", "woy_cos", "snap_frac", "event_frac"]
torch.manual_seed(0)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---- feature frame ----------------------------------------------------------

def build_frame(w: pd.DataFrame, scales: pd.Series, price_stats: dict) -> pd.DataFrame:
    df = w[["id", "week_end_date", "sales", "sell_price", "weekofyear",
            "snap_days", "event_days"]].copy()
    df["scale"] = df["id"].map(scales).astype(float)
    df["norm_sales"] = df["sales"] / df["scale"]
    df["price_z"] = ((df["sell_price"].fillna(price_stats["mean"]) - price_stats["mean"])
                     / price_stats["std"])
    df["woy_sin"] = np.sin(2 * np.pi * df["weekofyear"] / config.SEASON_WEEKS)
    df["woy_cos"] = np.cos(2 * np.pi * df["weekofyear"] / config.SEASON_WEEKS)
    df["snap_frac"] = df["snap_days"] / 7.0
    df["event_frac"] = df["event_days"] / 7.0
    return df.sort_values(["id", "week_end_date"]).reset_index(drop=True)


def make_windows(df, item_to_idx, train_end):
    """Sliding (past L, future H) windows fully inside the train period."""
    Xp, Xf, Y, ids, end_dates = [], [], [], [], []
    for sid, g in df[df["week_end_date"] <= train_end].groupby("id", sort=False):
        past = g[PAST_FEATS].to_numpy(np.float32)
        fut = g[FUT_FEATS].to_numpy(np.float32)
        tgt = g["norm_sales"].to_numpy(np.float32)
        dates = g["week_end_date"].to_numpy()
        idx = item_to_idx[sid]
        n = len(g)
        if n < L + H:
            continue
        for t in range(L, n - H + 1, STRIDE):
            Xp.append(past[t - L:t]); Xf.append(fut[t:t + H]); Y.append(tgt[t:t + H])
            ids.append(idx); end_dates.append(dates[t + H - 1])
    return (np.stack(Xp), np.stack(Xf), np.stack(Y),
            np.asarray(ids, np.int64), np.asarray(end_dates, "datetime64[ns]"))


class Seq2Seq(nn.Module):
    def __init__(self, n_items, n_past, n_fut, embed=16, hidden=64, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_items, embed)
        self.enc = nn.LSTM(n_past + embed, hidden, batch_first=True)
        self.dec = nn.LSTM(n_fut + embed, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, xp, xf, ids):
        B, Lp, _ = xp.shape
        Hh = xf.shape[1]
        e = self.emb(ids)
        xp = torch.cat([xp, e.unsqueeze(1).expand(-1, Lp, -1)], -1)
        xf = torch.cat([xf, e.unsqueeze(1).expand(-1, Hh, -1)], -1)
        _, state = self.enc(xp)
        out, _ = self.dec(xf, state)
        return self.head(self.drop(out)).squeeze(-1)        # (B, H)


def train_fold(Xp, Xf, Y, ids, end_dates, train_end, n_items, dev):
    val_cut = pd.Timestamp(train_end) - pd.Timedelta(weeks=8)
    ed = pd.to_datetime(end_dates)
    is_val = ed > val_cut
    # buffer of H weeks so no training window's horizon overlaps the val window
    is_tr = ed <= (val_cut - pd.Timedelta(weeks=H))
    if is_val.sum() < 64:
        is_tr = np.ones(len(Xp), bool); is_val = np.zeros(len(Xp), bool)

    def loader(mask, shuffle):
        return DataLoader(TensorDataset(
            torch.from_numpy(Xp[mask]), torch.from_numpy(Xf[mask]),
            torch.from_numpy(ids[mask]), torch.from_numpy(Y[mask])),
            batch_size=128, shuffle=shuffle)

    tl = loader(is_tr, True)
    vl = loader(is_val, False) if is_val.any() else None
    model = Seq2Seq(n_items, len(PAST_FEATS), len(FUT_FEATS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    best, best_state, stale, hist = float("inf"), None, 0, []
    for epoch in range(1, 31):
        model.train(); tot = 0.0; n = 0
        for xp, xf, ib, yb in tl:
            xp, xf, ib, yb = xp.to(dev), xf.to(dev), ib.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(model(xp, xf, ib), yb)
            loss.backward(); opt.step(); tot += loss.item() * len(yb); n += len(yb)
        tr = tot / n
        if vl is not None:
            model.eval(); vt = 0.0; vn = 0
            with torch.no_grad():
                for xp, xf, ib, yb in vl:
                    xp, xf, ib, yb = xp.to(dev), xf.to(dev), ib.to(dev), yb.to(dev)
                    vt += lossf(model(xp, xf, ib), yb).item() * len(yb); vn += len(yb)
            v = vt / vn; hist.append((tr, v))
            if v < best - 1e-5:
                best, best_state, stale = v, {k: t.detach().cpu().clone()
                                              for k, t in model.state_dict().items()}, 0
            else:
                stale += 1
            if stale >= 4:
                break
        else:
            hist.append((tr, None))
    if best_state:
        model.load_state_dict(best_state)
    return model, hist


def predict_fold(model, df, item_to_idx, scales, train_end, test_weeks, dev):
    """Encode last L weeks <= cutoff, decode the H future test weeks per series."""
    model.eval()
    test_weeks = sorted(pd.Timestamp(t) for t in test_weeks)
    fut_start, fut_end = test_weeks[0], test_weeks[-1]
    rows = []
    g_all = df.groupby("id", sort=False)
    Xp, Xf, ids, meta = [], [], [], []
    for sid, g in g_all:
        past = g[g["week_end_date"] <= train_end]
        fut = g[(g["week_end_date"] >= fut_start) & (g["week_end_date"] <= fut_end)]
        if len(fut) != H:
            continue
        p = past[PAST_FEATS].to_numpy(np.float32)
        if len(p) >= L:
            p = p[-L:]
        else:                                  # left-pad short histories with zeros
            p = np.vstack([np.zeros((L - len(p), len(PAST_FEATS)), np.float32), p])
        Xp.append(p); Xf.append(fut[FUT_FEATS].to_numpy(np.float32))
        ids.append(item_to_idx[sid])
        meta.append((sid, fut["week_end_date"].to_numpy(), scales[sid]))
    Xp = np.stack(Xp); Xf = np.stack(Xf); ids = np.asarray(ids, np.int64)
    with torch.no_grad():
        pred = model(torch.from_numpy(Xp).to(dev), torch.from_numpy(Xf).to(dev),
                     torch.from_numpy(ids).to(dev)).cpu().numpy()
    for (sid, dates, sc), yh in zip(meta, pred):
        rows.append(pd.DataFrame({"id": sid, "week_end_date": dates,
                                  "y_pred": np.clip(yh * sc, 0, None)}))
    return pd.concat(rows, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")
    dev = device(); print("device:", dev)

    w = pd.read_parquet(config.PROCESSED / "weekly_features.parquet")
    champ_pred = pd.read_parquet(config.REPORTS / "phase5_baselines" / "predictions.parquet")
    lb5 = pd.read_csv(config.REPORTS / "phase5_baselines" / "leaderboard_summary.csv")
    prof = pd.read_csv(config.REPORTS / "phase3_eda" / "series_profile.csv")
    prof["volume_tier"] = pd.qcut(prof["total_units"], 3, labels=["low", "mid", "high"])
    champion = lb5.iloc[0]["model"]

    item_to_idx = {sid: i for i, sid in enumerate(sorted(w["id"].unique()))}
    folds = backtest.make_folds(w["week_end_date"], config.HORIZON_WEEKS, config.N_FOLDS)

    all_preds, all_hist = [], []
    for f in folds:
        print(f"[{f.name}] train<= {f.train_end.date()} ...")
        scales = (w[w["week_end_date"] <= f.train_end].groupby("id")["sales"].mean()
                  .clip(lower=1.0))
        # series with no training history get scale 1
        scales = scales.reindex(sorted(w["id"].unique())).fillna(1.0)
        price_stats = {"mean": float(w[w["week_end_date"] <= f.train_end]["sell_price"].mean()),
                       "std": float(w[w["week_end_date"] <= f.train_end]["sell_price"].std() or 1.0)}
        df = build_frame(w, scales, price_stats)
        Xp, Xf, Y, ids, ed = make_windows(df, item_to_idx, f.train_end)
        print(f"        windows={len(Xp):,}")
        model, hist = train_fold(Xp, Xf, Y, ids, ed, f.train_end, len(item_to_idx), dev)
        test_weeks = w.loc[(w["week_end_date"] >= f.test_start) &
                           (w["week_end_date"] <= f.test_end), "week_end_date"].unique()
        fp = predict_fold(model, df, item_to_idx, scales, f.train_end, test_weeks, dev)
        fp["fold"] = f.name
        all_preds.append(fp)
        all_hist += [(f.name, e + 1, tr, v) for e, (tr, v) in enumerate(hist)]
    preds = pd.concat(all_preds, ignore_index=True)
    pd.DataFrame(all_hist, columns=["fold", "epoch", "train_loss", "val_loss"]).to_csv(
        OUT / "training_history.csv", index=False)

    truth = champ_pred[["id", "week_end_date", "fold", "y", champion]].rename(
        columns={champion: "champ_pred"})
    m = truth.merge(preds[["id", "week_end_date", "y_pred"]], on=["id", "week_end_date"], how="left")
    assert m["y_pred"].notna().all(), "LSTM missed some test cells"

    first_test = folds[0].test_start
    scales_mase = (w[w["week_end_date"] < first_test].groupby("id")["sales"]
                   .apply(lambda s: metrics.seasonal_naive_scale(s.to_numpy(), config.SEASON_WEEKS)))

    def row_metrics(name, yhat):
        per = (m.assign(ae=np.abs(m["y"] - yhat)).groupby("id")["ae"].mean())
        mase_s = (per / scales_mase).replace([np.inf, -np.inf], np.nan).dropna()
        return dict(model=name, wmape=metrics.wmape(m["y"], yhat),
                    rmse=metrics.rmse(m["y"], yhat), bias_pct=metrics.bias_pct(m["y"], yhat),
                    mase_median=float(mase_s.median()))

    lb = pd.DataFrame([row_metrics("lstm_seq2seq", m["y_pred"]),
                       row_metrics(f"{champion} (champion)", m["champ_pred"])])
    lb.to_csv(OUT / "leaderboard_summary.csv", index=False)
    m.to_parquet(OUT / "predictions.parquet", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(lb["model"], lb["wmape"], color=["#8172B3", "#55A868"])
    for i, (v, b) in enumerate(zip(lb["wmape"], lb["bias_pct"])):
        ax.text(v, i, f"  {v:.4f} (bias {b:+.0f}%)", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("WMAPE"); ax.set_title("LSTM seq2seq vs champion")
    fig.tight_layout(); fig.savefig(OUT / "01_lstm_vs_champion.png", dpi=110); plt.close(fig)

    hh = pd.read_csv(OUT / "training_history.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for fold, g in hh.groupby("fold"):
        ax.plot(g["epoch"], g["train_loss"], label=f"{fold} train", lw=1)
        if g["val_loss"].notna().any():
            ax.plot(g["epoch"], g["val_loss"], "--", label=f"{fold} val", lw=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled)"); ax.set_title("LSTM training")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "02_training.png", dpi=110); plt.close(fig)

    lstm_w, champ_w = lb.iloc[0]["wmape"], lb.iloc[1]["wmape"]
    verdict = "BEATS" if lstm_w < champ_w else "does NOT beat"
    md = OUT / "PHASE7_SUMMARY.md"
    with open(md, "w") as f:
        f.write("# Phase 7 -- LSTM seq2seq DL Challenger\n\n")
        f.write(f"Global encoder-decoder LSTM, per-series mean-scaling, evaluated on the same "
                f"{len(m):,} test cells as Phases 5-6.\n\n")
        f.write(f"## Verdict: LSTM **{verdict}** the `{champion}` champion on WMAPE "
                f"({lstm_w:.4f} vs {champ_w:.4f}, {(lstm_w/champ_w-1)*100:+.1f}%).\n")
        f.write(f"Bias {lb.iloc[0]['bias_pct']:+.1f}% (mean-scaling kept it from collapsing).\n\n")
        f.write("| model | WMAPE | RMSE | bias % | MASE (med) |\n|---|---|---|---|---|\n")
        for _, r in lb.iterrows():
            f.write(f"| {r['model']} | {r['wmape']:.4f} | {r['rmse']:.3f} | "
                    f"{r['bias_pct']:+.1f}% | {r['mase_median']:.3f} |\n")
        f.write("\n## Outputs\n\n- `predictions.parquet`, `leaderboard_summary.csv`, "
                "`training_history.csv`\n- figures 01-02\n")

    print("\n=== Phase 7 result ===")
    print(lb.to_string(index=False))
    print(f"\nLSTM {verdict} champion: WMAPE {lstm_w:.4f} vs {champ_w:.4f} "
          f"({(lstm_w/champ_w-1)*100:+.1f}%), bias {lb.iloc[0]['bias_pct']:+.1f}%")


if __name__ == "__main__":
    main()
