# Full M5 -- Cost-Aware A/B at 30K series

Control = ma4 + z(0.833)*sigma. Treatment = per-store LightGBM quantile orders. Panel = 30,474 paired series.

## Pre-registered rule

SHIP iff mean cost reduction >= 5% AND paired-t p < 0.05 AND stockout increase <= +2pp. At this n, the 5% practical gate carries the decision; p is merely necessary.

## Selection (folds 1-2)

| tau | cost | stockout pp | guard |
|---|---|---|---|
| q0.85 | -13.7% | +2.47pp | FAIL |
| q0.866 **<- chosen** | -13.5% | +1.28pp | pass |
| q0.9 | -11.7% | -1.33pp | pass |

## VERDICT (fold 3): **SHIP**

| gate | value | pass |
|---|---|---|
| mean cost reduction >= 5% | 14.1% | True |
| paired-t p < 0.05 | 2.32e-110 | True |
| stockout <= +2pp | +1.57pp | True |

- 95% CI [-15.3%, -12.9%]; 50% of series cheaper; Wilcoxon p=0.00e+00.
- Stockout rate 11.5% -> 13.1%.
- WMAPE context: control 0.3795 vs treatment 0.5219 (quantile trades point accuracy for cost by design).
- Policy isolation (LightGBM mean+z*sigma vs quantile): -9.4% (p=4.14e-51) -- the ordering policy, not just the model, drives the saving.

## Sensitivity

| ratio | cost | p | stockout pp | decision |
|---|---|---|---|---|
| 5:1 | -14.1% | 2.32e-110 | +1.57pp | SHIP |
| 3:1 | -9.0% | 2.96e-74 | -4.00pp | SHIP |
| 9:1 | -9.4% | 5.64e-26 | +5.56pp | HOLD |

## Per-store (fold 3)

| store | n | cost % |
|---|---|---|
| CA_3 | 3046 | -23.0% |
| WI_3 | 3049 | -17.6% |
| TX_2 | 3048 | -14.7% |
| WI_2 | 3046 | -14.7% |
| CA_1 | 3047 | -14.3% |
| WI_1 | 3048 | -13.4% |
| TX_1 | 3049 | -12.0% |
| CA_4 | 3046 | -10.1% |
| TX_3 | 3049 | -9.0% |
| CA_2 | 3046 | -5.9% |
