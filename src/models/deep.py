"""
Deep-learning challenger -- a global LSTM with item embeddings.

Design choices (industry-pragmatic, kept simple):
  - GLOBAL model: one LSTM trained over all 300 series simultaneously, with an
    item_id embedding so the model can tell series apart. (Per-series LSTMs at
    300x is wasteful; the global pooling pattern is what M5 winners used.)
  - log1p target: stabilises the right-skew we found in Phase 1 EDA (TS rule
    aside, this is a justified variance-stabiliser).
  - Sequence length L=28, stride 7: each window is 4 weeks of context. Stride 7
    keeps training set tractable on CPU/MPS without throwing away coverage.
  - Per-timestep features: log1p(sales), price, calendar/SNAP signals.
  - Static feature: item_id embedding (dim=8), broadcast across the sequence.
  - Single LSTM layer, hidden=64, dropout=0.2 -- defensible default; the point
    of Phase 3 is the end-to-end pipeline, not a hyperparameter sweep.
  - Recursive multi-step prediction (same protocol as LightGBM, so the A/B test
    in Phase 4 is apples-to-apples).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---- feature config ---------------------------------------------------------

TS_FEATURE_COLS = [
    "log1p_sales", "sell_price_norm",
    "wday_sin", "wday_cos",
    "has_snap", "has_event", "is_weekend", "snap_x_food",
]
N_TS_FEATURES = len(TS_FEATURE_COLS)

# Seq2seq feature split: the decoder consumes KNOWN-FUTURE features only
# (calendar, price, SNAP/events) -- everything we have in advance.
# It never sees its own past outputs, so there's no error compounding.
PAST_FEATURE_COLS = TS_FEATURE_COLS                              # encoder input
FUTURE_FEATURE_COLS = [c for c in TS_FEATURE_COLS if c != "log1p_sales"]  # decoder input
N_PAST_FEAT = len(PAST_FEATURE_COLS)
N_FUT_FEAT  = len(FUTURE_FEATURE_COLS)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---- model ------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, n_items: int, n_ts_features: int = N_TS_FEATURES,
                 item_embed_dim: int = 8, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, item_embed_dim)
        self.lstm = nn.LSTM(
            input_size=n_ts_features + item_embed_dim,
            hidden_size=hidden, num_layers=1, batch_first=True, dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_seq: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, L, F)  item_ids: (B,)
        emb = self.item_embed(item_ids).unsqueeze(1).expand(-1, x_seq.size(1), -1)  # (B, L, D)
        x = torch.cat([x_seq, emb], dim=-1)                                          # (B, L, F+D)
        h, _ = self.lstm(x)                                                          # (B, L, H)
        last = self.dropout(h[:, -1, :])                                             # (B, H)
        return self.head(last).squeeze(-1)                                           # (B,)


# ---- seq2seq architecture (the fix for Option B) ---------------------------

class LSTMSeq2Seq(nn.Module):
    """
    Encoder-decoder LSTM producing H=28 forecasts in one forward pass.

    Encoder reads past L days of (sales + exogenous) features.
    Decoder reads future H days of EXOGENOUS-ONLY features (calendar, price,
    SNAP) -- known in advance, NOT sales -- starting from the encoder's final
    state. Output is H sales predictions, one per future day. No recursion,
    so day-28 prediction is NOT a function of day-27 prediction.
    """
    def __init__(self, n_items: int,
                 n_past_feat: int = N_PAST_FEAT,
                 n_fut_feat: int = N_FUT_FEAT,
                 item_embed_dim: int = 8,
                 hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, item_embed_dim)
        self.encoder = nn.LSTM(input_size=n_past_feat + item_embed_dim,
                               hidden_size=hidden, batch_first=True)
        self.decoder = nn.LSTM(input_size=n_fut_feat + item_embed_dim,
                               hidden_size=hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_past: torch.Tensor, x_fut: torch.Tensor,
                item_ids: torch.Tensor) -> torch.Tensor:
        # x_past: (B, L, F_past)   x_fut: (B, H, F_fut)   item_ids: (B,)
        B, L, _ = x_past.shape
        _, H, _ = x_fut.shape
        emb = self.item_embed(item_ids)                            # (B, D)
        emb_p = emb.unsqueeze(1).expand(-1, L, -1)                 # (B, L, D)
        emb_f = emb.unsqueeze(1).expand(-1, H, -1)                 # (B, H, D)
        xp = torch.cat([x_past, emb_p], dim=-1)
        xf = torch.cat([x_fut,  emb_f], dim=-1)
        _, (h, c) = self.encoder(xp)                               # final state
        dec_out, _ = self.decoder(xf, (h, c))                      # (B, H, hidden)
        out = self.head(self.dropout(dec_out)).squeeze(-1)         # (B, H)
        return out


# ---- feature prep -----------------------------------------------------------

def _add_lstm_features(df: pd.DataFrame, price_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Adds: log1p_sales, sell_price_norm, wday_sin/cos, snap_x_food.
    `price_stats` is fit on TRAIN ONLY (TS rule #4) and reused at predict time.
    """
    df = df.copy()
    df["log1p_sales"] = np.log1p(df["sales"].clip(lower=0))

    if price_stats is None:
        price_stats = {"mean": float(df["sell_price"].mean()),
                       "std":  float(df["sell_price"].std() or 1.0)}
    df["sell_price_norm"] = (df["sell_price"].fillna(price_stats["mean"])
                             - price_stats["mean"]) / price_stats["std"]
    df["wday_sin"] = np.sin(2 * np.pi * df["wday"] / 7)
    df["wday_cos"] = np.cos(2 * np.pi * df["wday"] / 7)
    df["snap_x_food"] = (df["has_snap"] * (df["cat_id"] == "FOODS").astype(int)).astype("int8")
    for c in ("has_snap", "has_event", "is_weekend", "snap_x_food"):
        df[c] = df[c].astype("float32")
    return df, price_stats


def _make_windows(df: pd.DataFrame, item_to_idx: dict, L: int, stride: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding training windows. Returns (X, y, item_ids):
      X: (N, L, F)  -- sequence inputs
      y: (N,)       -- log1p(sales) for the day AFTER the window
      item_ids: (N,)
    """
    X_list, y_list, id_list = [], [], []
    for sid, g in df.sort_values(["id", "date"]).groupby("id", sort=False):
        arr = g[TS_FEATURE_COLS].to_numpy(dtype=np.float32)
        target = g["log1p_sales"].to_numpy(dtype=np.float32)
        idx = item_to_idx.get(sid)
        if idx is None or len(arr) <= L:
            continue
        # we predict day t given features for days [t-L .. t-1]
        for t in range(L, len(arr), stride):
            X_list.append(arr[t - L:t])
            y_list.append(target[t])
            id_list.append(idx)
    return (np.stack(X_list),
            np.asarray(y_list, dtype=np.float32),
            np.asarray(id_list, dtype=np.int64))


# ---- one-shot run (train+predict on one fold) -------------------------------

@dataclass
class LSTMConfig:
    L: int = 28
    stride: int = 7
    hidden: int = 64
    item_embed_dim: int = 8
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 5


@dataclass
class Seq2SeqConfig:
    L: int = 28               # encoder lookback length
    H: int = 28               # forecast horizon (must match backtest)
    stride: int = 7
    hidden: int = 64
    item_embed_dim: int = 8
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 30          # MAX epochs; early stopping usually halts sooner
    # If set (0,1), use pinball/quantile loss at this tau instead of MSE.
    # tau=0.80 -> punish under-prediction 4x more than over-prediction.
    # Newsvendor-optimal tau* = c_stockout / (c_stockout + c_holding).
    quantile: float | None = None
    # ---- validation + early stopping (TS-correct: val is the most-recent
    #      H-day inner fold of the TRAIN window, never a random split) -------
    val_horizon: int = 28     # hold out the last `val_horizon` days of train for val
    patience: int = 4         # stop if val loss doesn't improve for this many epochs
    min_delta: float = 1e-4   # minimum val-loss improvement to count as progress


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """L_tau(y, yhat) = max(tau*(y-yhat), (tau-1)*(y-yhat)). Element-wise mean."""
    diff = target - pred
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


def train_and_predict(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    full_raw: pd.DataFrame,
    config: LSTMConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Train an LSTM on `train_raw`, recursively forecast the dates in `test_raw`,
    return (predictions df, training info dict).

    `full_raw` is the entire active long frame -- we walk forward over it during
    recursive prediction so future calendar/price features are available.
    """
    cfg = config or LSTMConfig()
    device = get_device()

    # --- 1) feature prep on TRAIN; reuse stats at predict time -------------
    train, price_stats = _add_lstm_features(train_raw)
    item_to_idx = {sid: i for i, sid in enumerate(sorted(train["id"].unique()))}

    X, y, ids = _make_windows(train, item_to_idx, cfg.L, cfg.stride)
    print(f"    training windows: {len(X):,}  (L={cfg.L}, stride={cfg.stride})")

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(ids), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # --- 2) model + train loop --------------------------------------------
    model = LSTMForecaster(n_items=len(item_to_idx),
                           hidden=cfg.hidden,
                           item_embed_dim=cfg.item_embed_dim,
                           dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    epoch_losses = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, ib, yb in loader:
            xb = xb.to(device); ib = ib.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb, ib)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * yb.size(0); n += yb.size(0)
        avg = running / n
        epoch_losses.append(avg)
        print(f"    epoch {epoch}/{cfg.epochs}  loss={avg:.4f}")

    # --- 3) recursive forecast --------------------------------------------
    preds_df = _recursive_predict(model, full_raw, train_raw, test_raw,
                                  item_to_idx, price_stats, cfg, device)

    info = {"epoch_losses": epoch_losses,
            "n_windows": int(len(X)),
            "n_items": len(item_to_idx),
            "device": str(device)}
    return preds_df, info


# ---- seq2seq window builder + train loop -----------------------------------

def _make_seq2seq_windows(
    df: pd.DataFrame, item_to_idx: dict, L: int, H: int, stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X_past, X_fut, Y, item_ids, target_end_date) tuples where:
      X_past : (N, L, F_past)  past features incl. sales
      X_fut  : (N, H, F_fut)   future EXOGENOUS-only features
      Y      : (N, H)          log1p(sales) for the H future days
      target_end_date : (N,)   the LAST date each window forecasts (for a
                               time-based train/val split -- TS rule #1)
    """
    Xp, Xf, Y, ids, tgt_end = [], [], [], [], []
    for sid, g in df.sort_values(["id", "date"]).groupby("id", sort=False):
        past  = g[PAST_FEATURE_COLS].to_numpy(dtype=np.float32)
        fut   = g[FUTURE_FEATURE_COLS].to_numpy(dtype=np.float32)
        target = g["log1p_sales"].to_numpy(dtype=np.float32)
        dates  = g["date"].to_numpy()
        idx = item_to_idx.get(sid)
        if idx is None or len(past) < L + H:
            continue
        for t in range(L, len(past) - H + 1, stride):
            Xp.append(past[t - L:t])
            Xf.append(fut[t:t + H])
            Y.append(target[t:t + H])
            ids.append(idx)
            tgt_end.append(dates[t + H - 1])
    return (np.stack(Xp), np.stack(Xf), np.stack(Y),
            np.asarray(ids, dtype=np.int64),
            np.asarray(tgt_end, dtype="datetime64[ns]"))


def train_and_predict_seq2seq(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    full_raw: pd.DataFrame,
    config: Seq2SeqConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Train a seq2seq LSTM and predict the H=test_horizon days for each series
    in ONE forward pass (no recursion).

    Returns (predictions df, info dict).
    """
    cfg = config or Seq2SeqConfig()
    device = get_device()

    # --- 1) feature prep on TRAIN; reuse stats at predict time -------------
    train, price_stats = _add_lstm_features(train_raw)
    item_to_idx = {sid: i for i, sid in enumerate(sorted(train["id"].unique()))}

    Xp, Xf, Y, ids, tgt_end = _make_seq2seq_windows(
        train, item_to_idx, cfg.L, cfg.H, cfg.stride)

    # --- 1b) TIME-BASED train/val split (inner fold) ----------------------
    # Validation = windows whose forecast falls in the LAST `val_horizon` days
    # of the train period; training = windows whose forecast ends BEFORE that.
    # Windows straddling the boundary are dropped so val never leaks into train.
    train_end = pd.Timestamp(train_raw["date"].max())
    val_cutoff = train_end - pd.Timedelta(days=cfg.val_horizon)
    tgt_end_ts = pd.to_datetime(tgt_end)
    is_val   = tgt_end_ts >  val_cutoff
    is_train = tgt_end_ts <= (val_cutoff - pd.Timedelta(days=cfg.H))  # full buffer
    n_val = int(is_val.sum())
    use_val = n_val >= 32                       # need a meaningful val set
    if not use_val:
        is_train = np.ones(len(Xp), dtype=bool)  # fall back: train on everything
    print(f"    windows: {len(Xp):,} total -> train={int(is_train.sum()):,}  "
          f"val={n_val if use_val else 0:,}  (val = last {cfg.val_horizon}d of train)")

    def _loader(mask, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(Xp[mask]), torch.from_numpy(Xf[mask]),
                          torch.from_numpy(ids[mask]), torch.from_numpy(Y[mask])),
            batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False)

    train_loader = _loader(is_train, shuffle=True)
    val_loader = _loader(is_val, shuffle=False) if use_val else None

    # --- 2) model + loss --------------------------------------------------
    model = LSTMSeq2Seq(n_items=len(item_to_idx),
                        hidden=cfg.hidden,
                        item_embed_dim=cfg.item_embed_dim,
                        dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.quantile is not None:
        tau = float(cfg.quantile)
        loss_fn = lambda p, t: pinball_loss(p, t, tau=tau)
        print(f"    loss: pinball (quantile={tau})")
    else:
        loss_fn = nn.MSELoss()
        print("    loss: MSE")

    # --- 3) train loop with early stopping --------------------------------
    epoch_losses, val_losses = [], []
    best_val = float("inf"); best_state = None; best_epoch = 0; stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xp, xf, ib, yb in train_loader:
            xp, xf, ib, yb = xp.to(device), xf.to(device), ib.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xp, xf, ib), yb)
            loss.backward()
            opt.step()
            running += loss.item() * yb.size(0); n += yb.size(0)
        train_loss = running / max(n, 1)
        epoch_losses.append(train_loss)

        if use_val:
            val_loss = _eval_loss(model, val_loader, loss_fn, device)
            val_losses.append(val_loss)
            improved = val_loss < best_val - cfg.min_delta
            if improved:
                best_val = val_loss; best_epoch = epoch; stale = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
            print(f"    epoch {epoch:2d}/{cfg.epochs}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  {'*' if improved else f'(stale {stale})'}")
            if stale >= cfg.patience:
                print(f"    early stop at epoch {epoch} (best val={best_val:.4f} "
                      f"@ epoch {best_epoch})")
                break
        else:
            print(f"    epoch {epoch:2d}/{cfg.epochs}  train={train_loss:.4f}")

    # restore best weights (the whole point of early stopping)
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 4) single-forward-pass forecast ----------------------------------
    preds_df = _seq2seq_predict(model, full_raw, train_raw, test_raw,
                                item_to_idx, price_stats, cfg, device)
    info = {"epoch_losses": epoch_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": best_val if use_val else None,
            "n_windows": int(is_train.sum()),
            "n_val_windows": n_val if use_val else 0,
            "n_items": len(item_to_idx),
            "device": str(device)}
    return preds_df, info


def _eval_loss(model, loader, loss_fn, device) -> float:
    """Mean loss over a loader, no grad. Used for val monitoring / early stopping."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xp, xf, ib, yb in loader:
            xp, xf, ib, yb = xp.to(device), xf.to(device), ib.to(device), yb.to(device)
            loss = loss_fn(model(xp, xf, ib), yb)
            total += loss.item() * yb.size(0); n += yb.size(0)
    return total / max(n, 1)


def _seq2seq_predict(model, full_raw, train_raw, test_raw,
                     item_to_idx, price_stats, cfg, device) -> pd.DataFrame:
    """One forward pass per series: encode past L days, decode H future days."""
    model.eval()
    train_end = train_raw["date"].max()

    # build a feature frame for both encoder and decoder using the train-fit stats
    work = full_raw[full_raw["id"].isin(item_to_idx.keys())].copy()
    work, _ = _add_lstm_features(work, price_stats=price_stats)
    work = work.sort_values(["id", "date"]).reset_index(drop=True)

    # past window: last L days <= train_end ; future window: H days after train_end
    past_start = train_end - pd.Timedelta(days=cfg.L - 1)
    fut_end    = train_end + pd.Timedelta(days=cfg.H)

    past_rows = work[(work["date"] >= past_start) & (work["date"] <= train_end)]
    fut_rows  = work[(work["date"] >  train_end) & (work["date"] <= fut_end)]

    xp_list, xf_list, id_list, out_meta = [], [], [], []
    for sid, idx in item_to_idx.items():
        pg = past_rows[past_rows["id"] == sid]
        fg = fut_rows[fut_rows["id"] == sid]
        if len(pg) != cfg.L or len(fg) != cfg.H:
            continue                                    # warm-up edge case
        xp_list.append(pg[PAST_FEATURE_COLS].to_numpy(dtype=np.float32))
        xf_list.append(fg[FUTURE_FEATURE_COLS].to_numpy(dtype=np.float32))
        id_list.append(idx)
        out_meta.append((sid, fg["date"].to_numpy()))

    Xp = np.stack(xp_list); Xf = np.stack(xf_list); ids = np.asarray(id_list, dtype=np.int64)
    with torch.no_grad():
        log_pred = model(
            torch.from_numpy(Xp).to(device),
            torch.from_numpy(Xf).to(device),
            torch.from_numpy(ids).to(device),
        ).cpu().numpy()                                  # (n_series, H)
    yhat = np.clip(np.expm1(log_pred), a_min=0, a_max=None)

    out_rows = []
    for (sid, dates), yh in zip(out_meta, yhat):
        out_rows.append(pd.DataFrame({"id": sid, "date": dates, "y_pred": yh}))
    return pd.concat(out_rows, ignore_index=True)


def _recursive_predict(model, full_raw, train_raw, test_raw,
                       item_to_idx, price_stats, cfg, device) -> pd.DataFrame:
    """
    Walk forward over test dates, predicting all series simultaneously each day,
    feeding predictions back as 'sales' for the next iteration.
    """
    model.eval()
    train_end = train_raw["date"].max()

    # Build the working frame: train sales = actual, test sales = NaN-then-filled
    work = full_raw[full_raw["id"].isin(item_to_idx.keys())].copy()
    work = work.sort_values(["id", "date"]).reset_index(drop=True)
    work["sales_work"] = np.where(work["date"] <= train_end, work["sales"], np.nan)

    # Pre-compute static (non-sales) parts of features once
    static, _ = _add_lstm_features(work.assign(sales=work["sales_work"].fillna(0)),
                                   price_stats=price_stats)
    # We'll overwrite log1p_sales each iteration to reflect updated sales_work
    static["log1p_sales"] = np.nan

    test_dates = sorted(test_raw["date"].unique())
    out_rows = []
    for d in test_dates:
        d = pd.Timestamp(d)
        # refresh log1p_sales for the previous day (using sales_work)
        static["log1p_sales"] = np.log1p(work["sales_work"].clip(lower=0))

        # Build a batch: for each series, the window of length L ending at day d-1
        win_end_mask = static["date"] == d - pd.Timedelta(days=1)
        end_idx = static.index[win_end_mask]
        # safety: only include series with full L-window of history available
        starts = end_idx - cfg.L + 1
        valid = starts >= 0
        end_idx = end_idx[valid]; starts = starts[valid]

        # gather batch tensors
        feats = static[TS_FEATURE_COLS].to_numpy(dtype=np.float32)
        win = np.stack([feats[s:e + 1] for s, e in zip(starts, end_idx)])  # (B, L, F)
        ids = static.loc[end_idx, "id"].map(item_to_idx).to_numpy(dtype=np.int64)

        with torch.no_grad():
            xb = torch.from_numpy(win).to(device)
            ib = torch.from_numpy(ids).to(device)
            log_pred = model(xb, ib).cpu().numpy()

        yhat = np.clip(np.expm1(log_pred), a_min=0, a_max=None)

        # store + feed back
        series_ids = static.loc[end_idx, "id"].to_numpy()
        out_rows.append(pd.DataFrame({"id": series_ids, "date": d, "y_pred": yhat}))
        mask = work["date"] == d
        pred_map = dict(zip(series_ids, yhat))
        work.loc[mask, "sales_work"] = work.loc[mask, "id"].map(pred_map)

    return pd.concat(out_rows, ignore_index=True)
