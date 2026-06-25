# Power BI Build Recipe — A/B Experiment Readout

**Goal:** a single-page A/B experiment readout for DS / eng / exec audiences. Mirrors the structure of a real experimentation-platform dashboard.

**Data sources:**
- `reports/phase5_dashboards/ab_readout.csv` — per-series-day outcomes for both versions
- `reports/phase5_dashboards/ab_summary.csv` — 3 metrics × 2 versions (one row per metric × version)

---

## Schema

### ab_readout.csv (151k rows)

| Column | Type | Notes |
|---|---|---|
| `id` | text | series id |
| `date` | date | test-window date |
| `fold` | text | `fold_1` / `fold_2` / `fold_3` |
| `arm` | text | `control` / `treatment` |
| `y_pred` | decimal | model forecast |
| `sales` | int | actual units sold |
| `safety_stock` | int | extra units ordered on top of forecast |
| `order` | int | total units ordered = round(y_pred + safety_stock) |
| `stockouts`, `holding`, `cost` | decimal | newsvendor outcomes |
| `model` | text | `champion_ets` / `challenger_lstm_seq2seq` / `challenger_lstm_q80` |
| `version` | text | `v1_lstm_mse` / `v2_lstm_q80` |

### ab_summary.csv (6 rows)

| Column | Notes |
|---|---|
| `metric` | `total_cost_per_series` / `stockout_day_rate` / `wmape_guardrail` |
| `mean_control`, `mean_treatment` | per-series averages |
| `diff`, `pct_change` | treatment − control |
| `ci95_low`, `ci95_high` | 95% CI on the difference (cost only) |
| `cohens_d`, `p_value`, `test` | effect size, p, test name |
| `version` | `v1_lstm_mse` / `v2_lstm_q80` |

---

## Build steps (Power BI Desktop)

### 1. Connect
- Home → Get Data → Text/CSV → pick both files.
- In the model view, leave them un-related (both summary and long are filtered by `version`).

### 2. Measures (DAX)
Create these on `ab_summary`:

```dax
Cost % Change      = SELECTEDVALUE(ab_summary[pct_change])
Cost p-value       = SELECTEDVALUE(ab_summary[p_value])
Stockout pp Change = SELECTEDVALUE(ab_summary[diff]) * 100
CI Low %           = SELECTEDVALUE(ab_summary[ci95_low])  / SELECTEDVALUE(ab_summary[mean_control]) * 100
CI High %          = SELECTEDVALUE(ab_summary[ci95_high]) / SELECTEDVALUE(ab_summary[mean_control]) * 100

Decision =
VAR cost_ok  = [Cost % Change] <= -5
VAR sig_ok   = [Cost p-value] < 0.05
VAR guard_ok = [Stockout pp Change] <= 2
RETURN IF(cost_ok && sig_ok && guard_ok, "SHIP", "HOLD")

Decision Color = IF([Decision] = "SHIP", "#55a868", "#c44e52")
```

### 3. Page layout (1920 × 1080)

**Top strip — title + slicer**
- Text box: "A/B Experiment Readout — Champion (ETS) vs LSTM"
- Slicer on `ab_summary[version]` (Single Select). Default: `v2_lstm_q80`.
- Card showing decision rule text (static).

**Hero card — DECISION**
- Card visual; field: `[Decision]`.
- Conditional formatting on Font Color → use `[Decision Color]`.
- Font size 96 pt.

**Below the hero — 3 KPI tiles**
- Card 1: `[Cost % Change]` (format as % with sign, conditional color).
- Card 2: `[Cost p-value]` (4-decimal).
- Card 3: `[Stockout pp Change]` (format with sign + " pp").

**Cost lift chart**
- Clustered bar; X = `version` filter only; Y = pct_change for `metric = total_cost_per_series`.
- Add error bars using `ci95_low` / `ci95_high` (Visual format → Error bars).
- Reference line at 0; reference line at −5 (the SHIP threshold) labeled "SHIP threshold".

**Stockout vs guardrail chart**
- Bar chart of stockout pp change; reference line at +2 pp labeled "guardrail".

**Drill-through page**
- Use `ab_readout` for a detail view: per-day actual vs order vs stockouts for any selected `id`.

### 4. Polish
- Bookmark: "Compare v1 vs v2" — show both bars side-by-side (drop the slicer filter).
- Theme: match company palette (defaults are fine for portfolio).
- Title each visual and add a small data-as-of date.

### 5. Export
- File → Save as → `reports/powerbi/ab_readout.pbix`.
- Screenshots for the README.
