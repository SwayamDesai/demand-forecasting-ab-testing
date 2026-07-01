# Phase 5 -- Time-Series Baselines (champion bake-off)

Rolling-origin backtest: 3 folds x 4-week horizon, last 12 weeks. Primary metric WMAPE (volume-weighted, pooled). Predictions clipped to >=0.

## Champion: `ets`  (WMAPE 0.4054)

This is the bar Phases 6-7 must beat and the control arm in Phase 8.

## Leaderboard

| model | WMAPE | RMSE | bias % | MASE (med) |
|---|---|---|---|---|
| ets **<- champion** | 0.4054 | 5.777 | -3.7% | 0.665 |
| arima | 0.4087 | 5.775 | -3.8% | 0.665 |
| moving_average_4 | 0.4164 | 5.938 | -2.5% | 0.672 |
| moving_average_8 | 0.4234 | 6.116 | -3.4% | 0.663 |
| croston_sba | 0.4389 | 6.181 | -8.6% | 0.679 |
| naive_last | 0.4726 | 6.621 | -4.5% | 0.771 |
| seasonal_naive_52 | 0.6335 | 9.600 | -3.5% | 0.942 |

## WMAPE by intermittency class (champion vs specialists)

| class | ets | seasonal_naive_52 | croston_sba |
|---|---|---|---|
| erratic | 0.548 | 0.797 | 0.544 |
| intermittent | 0.514 | 0.828 | 0.625 |
| lumpy | 0.637 | 0.955 | 0.641 |
| smooth | 0.358 | 0.559 | 0.377 |

## Outputs

- `leaderboard_summary.csv`, `_per_fold.csv`, `_by_segment.csv`
- `predictions.parquet` (canonical test cells for Phases 6-8)
- figures 01-03
