"""Tests for src.features. Real lag/rolling no-leakage cases land in Phase 2.

The key test to come: assert a lag/rolling feature for day t never uses data from
day >= t (the leakage guard, TS rule #2).

For now: a smoke test that the module imports.
"""


def test_features_module_imports():
    import src.features  # noqa: F401
