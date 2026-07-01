# Phase 2 -- Preprocessing (weekly aggregation + features)

## The payoff: noise collapses at the weekly grain

- Daily zero-sales rate (active): **62.2%**
- Weekly zero-sales rate (active full weeks): **23.9%**

## Result

- Quality checks passed: **7/7**
- Model-ready weekly rows: 194,937  (series=900)
- Weeks per series: median 241, min 14, max 277
- Week range: 2011-02-04 -> 2016-05-20
- Feature columns: 11 lag/rolling + price + calendar + intermittency

## Checks

| check | passed | detail |
|---|---|---|
| weekly_reconciles_daily | PASS | daily=1,461,695 weekly=1,461,695 |
| no_duplicate_id_week | PASS | 0 dup rows |
| kept_weeks_full_active | PASS | n_days in [np.int64(7)], n_active in [np.int64(7)] |
| weeks_contiguous_7d | PASS | max within-series gap = 7 days (expect 7) |
| kept_weeks_have_price | PASS | 0.000% null price |
| features_leakage_free | PASS | 0 mismatches over 25 sampled series |
| lag1_not_current_week | PASS | 25.5% rows where lag_1==sales (coincidental zeros ok) |

## Outputs

- `data/processed/weekly_features.parquet`
- `reports/phase2_preprocessing/data_quality.csv`
