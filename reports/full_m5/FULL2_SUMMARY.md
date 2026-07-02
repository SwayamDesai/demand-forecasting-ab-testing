# Full M5 -- Baselines + per-store LightGBM

Panel: 30,490 series, 365,880 test cells (3 folds x 4-week horizon).

| model | WMAPE | RMSE | bias % | MASE (med) |
|---|---|---|---|---|
| lightgbm **<- champion** | 0.3706 | 8.500 | -1.2% | 0.635 |
| ma4 | 0.3875 | 9.263 | -1.3% | 0.650 |
| ma8 | 0.3975 | 9.493 | -2.8% | 0.648 |
| naive_last | 0.4283 | 9.627 | -5.3% | 0.733 |
| snaive_52 | 0.6218 | 14.890 | -12.9% | 0.916 |

Per-store WMAPE in `leaderboard_per_store.csv`. ETS/ARIMA/LSTM deliberately excluded at this scale (see script docstring).
