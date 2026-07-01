# Phase 3 -- Weekly EDA

Each finding maps to a modeling decision in Phases 4-7.

## Headline findings

1. **Still zero-inflated, but workable.** 23.9% of weekly observations are zero (vs 62% daily). Volume-weighted zero-rate is only 12.4% -- the units that matter are far less sparse. -> use **WMAPE**, report bias; RMSE as secondary.
2. **Volume is concentrated.** The top 34% of SKUs carry 80% of all units. -> volume-weighted evaluation and an A/B that is stratified by volume tier.
3. **Intermittency taxonomy (SBC):** smooth=486, intermittent=305, lumpy=72, erratic=37. -> a Croston/SBA baseline is justified for the 377 intermittent/lumpy series; smooth series suit ETS/ML.
4. **Clear yearly seasonality** (week-of-year), strongest in FOODS -> seasonal models (ETS season=52, seasonal-naive) and woy features earn their place.
5. **Exogenous signals are real but must be encoded carefully (honest read):**
   - **SNAP is nonlinear, not binary.** Within-series, FOODS sells +10% (median) in weeks with any SNAP day, but the effect is a dose-response: full-SNAP weeks (7 days) average 20.2 units vs 12.1 at 0 days (~+67%), while partial boundary weeks are flat-to-lower. A naive 'any SNAP' flag is volume-weighted ~0% and misleading. -> feed **snap_days (count)**, not a binary flag.
   - **Promotions matter but are rare and heterogeneous.** Deep promos (<0.90 of trailing price) occur in only 0.4% of active weeks; within-series lift is +3% median / +20% mean (a few big responders). -> keep price features, but don't oversell elasticity.
   - **Holiday/event weeks are ~neutral at the median series** (-0% within-series median): weekly aggregation dilutes single-day holiday spikes and some holidays (e.g. Christmas closures) depress sales, so effects roughly cancel. -> keep event features for the ML/DL models to exploit selectively, but expect little from them in the linear baselines.

## Category profile

| cat | series | total units | mean weekly | zero-week % |
|---|---|---|---|---|
| FOODS | 300 | 754,715 | 10.80 | 18.6% |
| HOBBIES | 300 | 310,500 | 4.41 | 28.8% |
| HOUSEHOLD | 300 | 393,951 | 6.07 | 23.6% |

## Figures

- `01_weekly_distribution.png`
- `02_volume_pareto.png`
- `03_intermittency_sbc.png`
- `04_seasonality_woy.png`
- `05_calendar_effects.png`
- `06_price_effect.png`
- `07_segments_trend.png`

## Tables

- `series_profile.csv` (per-series ADI/CV^2/class)
- `segment_summary.csv`
- `seasonality_woy.csv`
