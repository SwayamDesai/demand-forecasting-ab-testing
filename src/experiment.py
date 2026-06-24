"""
Champion–challenger A/B experiment (Phase 4).

Each store–SKU is an experimental unit.

Planned:
  - assign(units, seed, stratify_by="demand_volume")  stratified random control/treatment
  - power_analysis(effect_size, alpha=0.05, power=0.8) required sample size
  - simulate_decision(forecast, actual, safety_stock)  order policy -> stockout/holding cost
  - analyze(control, treatment)  t-test / Mann-Whitney / two-proportion z-test,
                                 effect size, CIs, p-values, guardrail check
  - recommend()  go / no-go with the numbers behind it

CLI: `python -m src.experiment`
"""
