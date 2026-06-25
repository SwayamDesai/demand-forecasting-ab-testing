# Tableau Build Recipe — Forecasting Dashboard

**Goal:** an interactive forecasting dashboard for retail ops. Audience: someone who orders inventory for a store and needs to see actuals vs predicted with drill-down.

**Data source:** `reports/phase5_dashboards/forecasts_long.csv`

| Column | Type | Notes |
|---|---|---|
| `id` | string | series id, e.g. `FOODS_2_197_CA_1_evaluation` |
| `item_id`, `dept_id`, `cat_id` | string | hierarchy for drill-down |
| `store_id`, `state_id` | string | location |
| `date` | date | test-window date |
| `fold` | string | `fold_1` / `fold_2` / `fold_3` |
| `model` | string | `seasonal_naive` / `ets` / `lightgbm` / `lstm_seq2seq` / `lstm_seq2seq_q80` |
| `actual` | int | actual units sold |
| `y_pred` | float | model forecast (units) |
| `error`, `abs_error` | float | y_pred − actual, |error| |

---

## Build steps (Tableau Public or Desktop)

### 1. Connect
- Open Tableau → **Connect** → **Text File** → pick `forecasts_long.csv`.
- In the data pane, set `date` as Date. Set `actual`, `y_pred`, `error`, `abs_error` as Number (decimal). Set everything else as String.

### 2. Calculated fields
Create these in **Analysis → Create Calculated Field**:

| Field | Formula |
|---|---|
| `WMAPE` | `SUM([abs_error]) / SUM([actual])` |
| `Volume` | `SUM([actual])` |
| `Model_isLSTM` | `CONTAINS([model], "lstm")` |

### 3. Sheet 1 — Actuals vs Predictions (line chart, top-N SKUs)
- **Columns:** `date` (continuous, day)
- **Rows:** `MEASURE(actual)`, `MEASURE(y_pred)` (dual axis, synchronize)
- **Color:** `model`
- **Filter:** `id` → Top 3 by Sum(`actual`)
- **Filter:** `fold` → use as filter shelf (so user picks the test window)

### 4. Sheet 2 — WMAPE heatmap (store × model)
- **Columns:** `store_id`
- **Rows:** `model`
- **Color & Label:** `WMAPE` (use Red-Green-diverging, centered at 1.0)
- Sort rows by total WMAPE ascending so the best model is on top.

### 5. Sheet 3 — Leaderboard
- **Rows:** `model`
- **Columns:** `WMAPE`
- Sort ascending. Color bars green if `Model_isLSTM`, else blue. Add a vertical reference line at `WMAPE = 1.0` (seasonal-naive level).

### 6. Sheet 4 — WMAPE by category (grouped bars)
- **Columns:** `cat_id`
- **Rows:** `WMAPE`
- **Color:** `model`

### 7. Dashboard
- New Dashboard → 1600 × 1000 px.
- Top row: title text + filter cards (cat_id, store_id, fold, model).
- Middle row: Sheet 1 (3 SKU charts side-by-side, or one big chart with Action → Filter so the user clicks an SKU in a list).
- Bottom row: Sheet 2 + Sheet 3 + Sheet 4.

### 8. Polish
- Add a parameter `Champion vs Challenger` (string list: `champion` = "ets", `challenger` = "lstm_seq2seq"). Use it to filter the model field with a calculated field.
- Add tooltips: hover should show actual, predicted, error, and the SKU's category/store.

### 9. Export
- File → Export As Packaged Workbook → save as `reports/tableau/forecasts.twbx`.
- Take screenshots of the dashboard for the README.
