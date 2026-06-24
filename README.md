# Demand Forecasting Platform with Champion–Challenger A/B Testing

A retail demand-forecasting **decision system**: it serves SKU-level forecasts, then runs a
controlled champion–challenger experiment to prove a deep-learning model beats the incumbent
baseline *before* rollout. Forecasts are surfaced in **Tableau**; the experiment readout in **Power BI**.

> The point isn't "I trained an LSTM." The point is the full chain a real DS team uses to gate a launch:
> **offline metrics → experimental validation → business go/no-go decision.**

---

## Status

| Phase | What | State |
|------|------|-------|
| 0 | Repo + env + Kaggle data pipeline | ✅ done |
| 1 | Data loading + cleaning + EDA | ⬜ next |
| 2 | Features + baselines (the *champion*) | ⬜ |
| 3 | Deep-learning model (the *challenger*) | ⬜ |
| 4 | Champion–challenger A/B experiment | ⬜ |
| 5 | Dashboards (Tableau + Power BI) | ⬜ |
| 6 | Tests + polish + v1 | ⬜ |

## Dataset

[M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (Walmart):
~30K store–SKU daily series, 5+ years, with price + calendar/event features. We use three files:
`sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`.

## Time-series ground rules (the spine of this project)

1. **Never shuffle time** — split by date only (train past → test future), never random rows.
2. **No future leakage** — a feature for day *t* uses only info available at/before *t* (lags are `.shift`ed).
3. **Rolling-origin backtesting** — slide the cutoff forward over several folds, don't trust one split.
4. **Fit scalers on train only** — apply train stats to val/test; never fit on the whole series.
5. **Retail-correct metrics** — primary **WMAPE**, secondary **MASE** (vs seasonal-naive), not just RMSE.

## How to run

```bash
# one-time: create env + install (we install lazily per phase; this does it all)
python3.12 -m venv .venv
make install

# download the raw M5 files into data/ (requires Kaggle token at ~/.kaggle/access_token)
make data

# later phases
make sample      # carve fast dev subset
make features    # build features
make train       # baselines + deep model
make experiment  # A/B test + stats
make test        # pytest
```

## Results

_TBD — filled in as phases complete (WMAPE leaderboard, A/B lift + significance, dashboard screenshots)._

## What I'd do next

Sequential testing / always-valid p-values, multi-armed bandits for adaptive rollout,
hierarchical forecast reconciliation across the store/category hierarchy.
