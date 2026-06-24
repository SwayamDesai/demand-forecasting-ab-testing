"""
Rolling-origin (time-based) backtesting (Phase 2; TS rule #3).

Slides the train/test cutoff forward over several folds so evaluation reflects
"retrain weekly, forecast the next H days" rather than one lucky split.

Planned:
  - rolling_origin_splits(dates, n_folds, horizon, step)  -> list of (train_idx, test_idx)
  - run_backtest(model, data, splits, metrics)            -> per-fold + aggregate scores
"""
