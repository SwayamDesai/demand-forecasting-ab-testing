# Phase 1 -- Data Cleaning

Raw M5 -> clean daily tidy table, gated by explicit quality checks. No weekly aggregation / features / modeling yet.

## Result

- Quality checks passed: **12/12**
- Series: 900 (by category {'FOODS': 300, 'HOBBIES': 300, 'HOUSEHOLD': 300}; by store {'CA_1': 300, 'TX_1': 300, 'WI_1': 300})
- Daily rows: 1,746,900 (1,366,359 active = 78.2%)
- Date range: 2011-01-29 -> 2016-05-22
- Daily zero-sales rate (active rows): **62.2%**  <- the noise weekly will smooth

## Checks

| check | passed | detail |
|---|---|---|
| series_count_900 | PASS | 900 series |
| balanced_by_category | PASS | {'FOODS': 300, 'HOBBIES': 300, 'HOUSEHOLD': 300} |
| balanced_by_store | PASS | {'CA_1': 300, 'TX_1': 300, 'WI_1': 300} |
| no_duplicate_id_date | PASS | 0 dup rows |
| full_calendar_per_series | PASS | expected 1941/series; min=1941, max=1941 |
| sales_non_null | PASS | 0 nulls |
| sales_non_negative | PASS | min=0 |
| date_is_datetime | PASS | datetime64[ns] |
| active_rows_have_price | PASS | 0.000% of active rows missing price |
| prelaunch_has_no_price | PASS | 0 pre-launch rows with a price |
| is_active_monotonic | PASS | 0 series toggle off after launch |
| calendar_fully_joined | PASS | nulls: {'date': 0, 'wday': 0, 'wm_yr_wk': 0} |

## Outputs

- `data/processed/daily_clean.parquet`
- `reports/phase1_cleaning/data_quality.csv`
