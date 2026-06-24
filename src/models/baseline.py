"""
Baseline forecasters — the *champion* candidates (Phase 2).

Planned (same backtest + metrics for apples-to-apples):
  - SeasonalNaive (weekly)   the "do nothing smart" reference
  - ETS / ARIMA via statsforecast
  - Prophet
  - LightGBM on lag/rolling/price/calendar features

Best performer by WMAPE becomes the champion the challenger must beat.
CLI: `python -m src.models.baseline`
"""
