# Phase 7 -- LSTM seq2seq DL Challenger

Global encoder-decoder LSTM, per-series mean-scaling, evaluated on the same 10,800 test cells as Phases 5-6.

## Verdict: LSTM **does NOT beat** the `ets` champion on WMAPE (0.4118 vs 0.4054, +1.6%).
Bias -1.6% (mean-scaling kept it from collapsing).

| model | WMAPE | RMSE | bias % | MASE (med) |
|---|---|---|---|---|
| lstm_seq2seq | 0.4118 | 5.834 | -1.6% | 0.654 |
| ets (champion) | 0.4054 | 5.777 | -3.7% | 0.665 |

## Outputs

- `predictions.parquet`, `leaderboard_summary.csv`, `training_history.csv`
- figures 01-02
