# Phase 8c -- Experiment 2: the pre-registered retest

Runs the follow-up documented in PHASE8B_SUMMARY.md *before* this data was touched: mean-based (total-dollars) primary test, tau in {0.85, 0.866, 0.90}, on 24 fresh weeks never previously evaluated.

## Why this is honest, not test-shopping

- Evaluation data: the 24 weeks ending where fold 1 began -- no prior phase ever scored a model there. Models refit per window with train < cutoff.
- The mean-based rule was declared in the 8b post-mortem, before this experiment.
- tau chosen on windows 1-3; confirmed once on windows 4-6.
- Panel: 896 of 900 series (4 excluded from BOTH arms as too new to have history in this window range).

## Pre-registered rule

SHIP iff: paired MEAN cost reduction >= 5% AND paired-t p < 0.05 AND stockout increase <= +2pp.

## Selection (windows 1-3)

| tau | cost vs champion | stockout pp | guardrail |
|---|---|---|---|
| q0.85 | -9.7% | +2.01pp | FAIL |
| q0.866 **<- chosen** | -9.8% | +0.92pp | pass |
| q0.9 | -7.7% | -1.42pp | pass |

## VERDICT (confirmation windows 4-6): **SHIP**

| gate | value | pass |
|---|---|---|
| mean cost reduction >= 5% | 6.9% | True |
| paired-t p < 0.05 | 4.88e-04 | True |
| stockout increase <= +2pp | +0.65pp (two-prop p=1.25e-01) | True |

- 95% CI on mean cost change: [-11.1%, -3.3%]; 48% of series cheaper.
- Secondary (context): Wilcoxon p = 1.11e-03.
- Stockout rate: 10.5% -> 11.2%.
- WMAPE (context, not gated): champion 0.4354 vs treatment 0.6680.

## Sensitivity

| ratio | mean cost change | p | stockout pp | decision |
|---|---|---|---|---|
| 3:1 | -0.8% | 6.09e-01 | -4.47pp | HOLD |
| 5:1 | -6.9% | 4.88e-04 | +0.65pp | SHIP |
| 9:1 | -3.1% | 2.98e-01 | +4.41pp | HOLD |

## Outputs

- `selection_windows123.csv`, `confirmation_sensitivity.csv`, `per_series_costs_confirmation.csv`, `predictions.parquet`
- figures 01-02
