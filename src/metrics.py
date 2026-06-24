"""
Forecast + business metrics (Phase 2; unit-tested in tests/test_metrics.py).

Forecast accuracy:
  - wmape(y_true, y_pred)   primary, retail-standard
  - rmse(y_true, y_pred)
  - mase(y_true, y_pred, y_train, season)   vs seasonal-naive (>1 = worse than naive)

Business (Phase 4):
  - simulated stockout rate and holding cost from order = forecast + safety_stock
"""
