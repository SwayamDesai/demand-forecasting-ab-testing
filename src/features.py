"""
Feature engineering (Phase 2).

All time-derived features obey TS rule #2 (no future leakage): lags/rolling
stats are computed per series and `.shift(1)`-ed so day t never sees day t.

Planned:
  - lag features (7, 28, 365)
  - rolling mean/std (shifted) over 7/28-day windows
  - price features (price ratio vs rolling avg, price-change flag)
  - calendar encodings (weekday, month, event type, SNAP)
"""
