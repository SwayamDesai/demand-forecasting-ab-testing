# Phase 8 -- A/B Test: champion vs best challenger

**Control** = `ets` (champion, WMAPE 0.4054).  **Treatment** = `lightgbm` (most accurate challenger, WMAPE 0.4026).

Newsvendor business sim, stockout:holding = 5:1. Per-series total cost over 12 test weeks x 900 series.

## Pre-registered decision rule

SHIP iff: paired cost reduction >= 5% AND Wilcoxon p < 0.05 AND stockout-rate increase <= 2pp AND challenger WMAPE <= 1.05x champion.

## VERDICT: **HOLD**

| gate | value | pass |
|---|---|---|
| cost reduction >= 5% | -1.2% | False |
| Wilcoxon p < 0.05 | 4.50e-02 | True |
| stockout increase <= 2pp | -0.41pp | True |
| WMAPE not worse | 0.4026 vs 0.4054 | True |

## Paired vs unpaired (cost)

| test | cost change | p-value | 95% CI (mean diff) | crosses 0? |
|---|---|---|---|---|
| unpaired (live-like) | +5.3% | 8.58e-01 | [-7.8, 16.3] | yes |
| **paired (correct)** | **+1.2%** | **4.50e-02** | [-0.5, 2.4] | yes |

- Series where challenger is cheaper: 38.7%
- Unpaired power: MDE ~= 20.5% of control mean (n/arm ~= 447).
- Stockout rate: champion 11.0% -> challenger 10.6% (-0.41pp; two-prop p=3.35e-01).

## Cost-ratio sensitivity (the key assumption)

| stockout:holding | cost change | p | stockout pp | decision |
|---|---|---|---|---|
| 3:1 | +1.5% | 2.98e-01 | -1.04pp | HOLD |
| 5:1 | +1.2% | 4.50e-02 | -0.41pp | HOLD |
| 9:1 | +0.5% | 1.19e-02 | -0.46pp | HOLD |

## Per-segment cost change (paired, negative = challenger cheaper)

| segment | value | n | cost % |
|---|---|---|---|
| cat_id | FOODS | 300 | +0.1% |
| cat_id | HOBBIES | 300 | +1.8% |
| cat_id | HOUSEHOLD | 300 | +2.8% |
| sbc_class | erratic | 37 | +2.5% |
| sbc_class | intermittent | 305 | -1.2% |
| sbc_class | lumpy | 72 | +6.5% |
| sbc_class | smooth | 486 | +1.4% |
| volume_tier | low | 303 | +0.8% |
| volume_tier | mid | 297 | -1.7% |
| volume_tier | high | 300 | +2.5% |

## Outputs

- `per_series_costs.csv`, `segment_savings.csv`, `cost_ratio_sensitivity.csv`
- figures 01-05
