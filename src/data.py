"""
Data loading + cleaning for M5 (pandas only).

Phase 1 implements:
  - load_raw()      read calendar.csv, sales_train_evaluation.csv, sell_prices.csv
  - wide_to_long()  reshape sales from wide (one column per day) to tidy long
                    (one row per date x store x item)  [respects TS rule: no shuffle]
  - join_calendar() attach date, weekday, month, events, SNAP flags
  - join_prices()   attach weekly sell_price; flag "item not launched yet" rows
  - clean()         handle missing prices, intermittent/zero demand, dtypes
  - make_sample()   carve a fast dev subset (a few categories / stores)

CLI: `python -m src.data --make-sample`
"""
