# Tableau Build Recipe — Forecasting Accuracy Dashboard

Recreate `mock_tableau_forecasts.png` as a real interactive `.twbx`. Audience: retail ops —
actuals vs predicted with drill-down, on the **latest** results (5 models incl. the deployed
LSTM **τ=0.90** challenger).

## Data source: `forecasts_long.csv`

| column | type | notes |
|---|---|---|
| `id` | string | series id |
| `item_id`, `dept_id`, `cat_id` | string | hierarchy for drill-down |
| `store_id`, `state_id` | string | location |
| `date` | date | test-window date |
| `fold` | string | `fold_1` / `fold_2` / `fold_3` |
| `model` | string | `seasonal_naive` / `ets` / `lightgbm` / `lstm_seq2seq` / `lstm_q90` |
| `actual` | number | actual units sold |
| `y_pred` | number | model forecast |
| `error`, `abs_error` | number | y_pred − actual, |error| |

> `forecasts_long.csv` is regenerable and gitignored (large). Build it with
> `python -m scripts.phase5_dashboards`.

## Build steps (Tableau Public / Desktop)

1. **Connect → Text File** → `forecasts_long.csv`. Set `date` = Date; `actual`/`y_pred`/`error`/
   `abs_error` = Number; rest = String.
2. **Calculated fields:**
   - `WMAPE = SUM([abs_error]) / SUM([actual])`
   - `Volume = SUM([actual])`
3. **Sheet 1 — Actuals vs predicted (top SKUs):** Columns `date`; Rows `actual` + `y_pred`
   (dual-axis, synchronized); Color = `model`; filter `id` to Top 3 by `Volume`; expose `fold`
   as a filter.
4. **Sheet 2 — Leaderboard:** Rows `model`, Columns `WMAPE`, sort ascending, reference line at 1.0
   (seasonal-naive floor).
5. **Sheet 3 — WMAPE by category:** Columns `cat_id`, Rows `WMAPE`, Color `model`.
6. **Sheet 4 — WMAPE heatmap:** Columns `store_id`, Rows `model`, Color & label `WMAPE`
   (red-green diverging centered at 1.0).
7. **Dashboard:** title band + filter cards (category / store / model / fold) on top; Sheet 1 across
   the middle; Sheets 2–4 across the bottom. Add a parameter `Champion vs Challenger`
   (`ets` ↔ `lstm_q90`) wired to the model filter for a one-click toggle.
8. **Export As Packaged Workbook** → `reports/tableau/forecasts.twbx`; screenshot for the README.
