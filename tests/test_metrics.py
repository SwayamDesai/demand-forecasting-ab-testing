"""Tests for src.metrics. Real WMAPE/RMSE/MASE + business-metric cases land in Phase 2.

For now: a smoke test that the module imports (catches packaging/syntax breakage early).
"""


def test_metrics_module_imports():
    import src.metrics  # noqa: F401
