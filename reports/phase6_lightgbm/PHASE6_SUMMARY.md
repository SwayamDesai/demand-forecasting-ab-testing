# Phase 6 -- LightGBM ML Challenger

Global Tweedie LightGBM, recursive 4-week forecast, evaluated on the same 10,800 test cells as Phase 5.

## Verdict: LightGBM **BEATS** the `ets` champion on WMAPE (0.4026 vs 0.4054, -0.7%).

| model | WMAPE | RMSE | bias % | MASE (med) |
|---|---|---|---|---|
| lightgbm | 0.4026 | 5.664 | -0.9% | 0.658 |
| ets (champion) | 0.4054 | 5.777 | -3.7% | 0.665 |

## WMAPE by segment

| segment | value | lightgbm | champion |
|---|---|---|---|
| cat_id | FOODS | 0.368 | 0.373 |
| cat_id | HOBBIES | 0.491 | 0.497 |
| cat_id | HOUSEHOLD | 0.400 | 0.396 |
| sbc_class | erratic | 0.545 | 0.548 |
| sbc_class | intermittent | 0.513 | 0.514 |
| sbc_class | lumpy | 0.646 | 0.637 |
| sbc_class | smooth | 0.354 | 0.358 |
| volume_tier | low | 0.700 | 0.685 |
| volume_tier | mid | 0.492 | 0.499 |
| volume_tier | high | 0.345 | 0.348 |

## Outputs

- `predictions.parquet` (aligned to champion cells)
- `leaderboard_summary.csv`, `wmape_by_segment.csv`
- figures 01-03
