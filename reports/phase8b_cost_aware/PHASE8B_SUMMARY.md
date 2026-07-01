# Phase 8b -- Cost-Aware Challenger (order the quantile directly)

**Control** = `ets` + Gaussian newsvendor policy (z(0.833)*sigma).  **Treatment** = LightGBM pinball loss predicting the demand quantile as the order.

Both arms target the SAME 5:1 critical ratio tau*=0.833; they differ only in how the quantile is estimated (Gaussian assumption vs learned per-SKU).

## Protocol (pre-registered)

- tau swept on folds 1-2 only; winner confirmed ONCE on untouched fold 3.
- SHIP iff: paired cost reduction >= 5% AND Wilcoxon p < 0.05 AND stockout increase <= +2pp.
- WMAPE reported as context, NOT gated (quantile forecasts trade point accuracy for cost by design).

## Selection (folds 1-2)

| tau | cost vs champion | stockout pp | guardrail |
|---|---|---|---|
| q0.75 | +0.4% | +9.38pp | FAIL |
| q0.833 | -3.9% | +3.11pp | FAIL |
| q0.9 **<- chosen** | -1.5% | -1.53pp | pass |

## VERDICT (confirmation fold 3): **HOLD**

| gate | value | pass |
|---|---|---|
| cost reduction >= 5% | 5.6% | True |
| Wilcoxon p < 0.05 | 1.30e-01 | False |
| stockout increase <= +2pp | -0.75pp (two-prop p=3.12e-01) | True |

- Paired 95% CI on cost change: [-9.7%, -1.7%]; 43% of series cheaper under treatment.
- Stockout rate: 11.5% -> 10.8%.
- WMAPE (context only): champion 0.3935 vs treatment 0.6173 -- expected to be worse; the treatment optimises cost, not point error.

## Sensitivity: reprice the same orders

| ratio | cost change | p | stockout pp | decision |
|---|---|---|---|---|
| 3:1 | +4.6% | 4.64e-17 | -6.42pp | HOLD |
| 5:1 | -5.6% | 1.30e-01 | -0.75pp | HOLD |
| 9:1 | -6.1% | 3.39e-09 | +2.94pp | HOLD |

## Outputs

- `selection_folds12.csv`, `confirmation_sensitivity.csv`, `per_series_costs_fold3.csv`, `predictions.parquet`
- figures 01-03

## Post-hoc diagnosis (clearly labeled: computed AFTER the verdict, not part of the gate)

The -5.6% saving is real money but concentrated: the top-10% costliest series contribute
**all** of it (-1,346 cost units vs +26 for the other 90%); the median series is unchanged
(median paired diff = 0.00). This splits the two tests:

- Wilcoxon signed-rank (pre-registered primary) asks "does the *typical* series improve?"
  -> p = 0.13, no.
- A paired t-test on the *mean* (what total dollars follow) -> p = 0.007, yes.

**We keep the HOLD** -- switching tests after seeing data is p-hacking. But the lesson is
recorded: for a volume-weighted business question, the primary test should have been
mean-based (paired t / bootstrap on the mean). Next confirmation (new data or full M5)
should pre-register exactly that, plus a tau ~ 0.85-0.87 to trade a little of q0.9's
service headroom (-0.75pp) for more cost reduction.
