# Power BI Build Recipe — A/B Experiment Readout

Recreate `mock_powerbi_ab.png` as a real interactive `.pbix`. Single-page readout for a
DS / eng / exec audience. Built on the **latest** results: champion = ETS (+ safety stock),
challenger = LSTM quantile **τ=0.90**, evaluated with the **paired** A/B test.

## Data sources (in this folder)

**`ab_summary.csv`** — 2 rows (one per scope), drives the KPI cards.

| column | meaning |
|---|---|
| `scope` | `fold_3 (confirmatory)` ← headline, or `all folds (context)` |
| `n_skus` | paired SKUs in scope |
| `champion_mean_cost`, `challenger_mean_cost` | mean cost / SKU per arm |
| `cost_pct_change` | challenger vs champion (negative = cheaper) |
| `wilcoxon_p` | paired significance (Wilcoxon signed-rank) |
| `ci95_low`, `ci95_high` | 95% CI on the mean cost difference |
| `pct_skus_cheaper` | % of SKUs cheaper under challenger |
| `stockout_champion_pct`, `stockout_challenger_pct`, `stockout_change_pp` | service |
| `decision` | `SHIP` / `HOLD` from the pre-registered rule |

**`ab_paired_readout.csv`** — one row per (SKU, fold); drives charts + drill-down.

| column | meaning |
|---|---|
| `id`, `fold` | SKU + backtest fold |
| `cat_id`, `store_id`, `volume_tertile` | segments (slicers) |
| `champion_cost`, `challenger_cost`, `cost_diff` | per-SKU costs; diff < 0 = cheaper |
| `champion_stockout_rate`, `challenger_stockout_rate` | per-SKU service |
| `challenger_cheaper` | 1 if challenger cheaper |
| `is_confirmatory_fold` | 1 for fold_3 (the held-out fold) |

## Build steps (Power BI Desktop)

1. **Get Data → Text/CSV** → load both files. Mark `wilcoxon_p` as Decimal.
2. **Slicers:** `scope` (default `fold_3 (confirmatory)`) on the summary; `cat_id`, `store_id`,
   `volume_tertile` on the readout. Add a page filter `is_confirmatory_fold = 1` so charts default
   to the held-out fold.
3. **KPI cards (one row of 5):** Card visuals on `ab_summary` for `cost_pct_change`,
   `wilcoxon_p`, `stockout_change_pp`, `pct_skus_cheaper`, and `decision`. Conditional formatting:
   - cost green if `≤ -5`, else red
   - stockout green if `≤ 2`, else red
   - decision green = SHIP, red = HOLD
4. **Cost-difference distribution:** histogram of `cost_diff` (readout, confirmatory fold). Add
   reference lines at 0 and at the mean; shade the CI band from `ab_summary`.
5. **Mean cost by category:** clustered column, axis `cat_id`, values `champion_cost` &
   `challenger_cost` (Average).
6. **Stockout vs guardrail:** column of `stockout_champion_pct` / `stockout_challenger_pct`;
   dashed reference line at champion + 2 pp.
7. **Drill-through page:** table on `ab_paired_readout` (id, costs, stockout rates) for any SKU.
8. **Title + rule banner:** "Champion–Challenger A/B — Experiment Readout" and the pre-registered
   decision rule as a text box.
9. **Save As** → `reports/powerbi/ab_readout.pbix`; screenshot for the README.

> DAX for the decision (if you prefer to compute it live):
> ```DAX
> Decision =
> VAR c = SELECTEDVALUE(ab_summary[cost_pct_change])
> VAR p = SELECTEDVALUE(ab_summary[wilcoxon_p])
> VAR s = SELECTEDVALUE(ab_summary[stockout_change_pp])
> RETURN IF(c <= -5 && p < 0.05 && s <= 2, "SHIP", "HOLD")
> ```
