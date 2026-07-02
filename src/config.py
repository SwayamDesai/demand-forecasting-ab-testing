"""
Project-wide configuration: paths and the backtest setup, fixed in one place so
every step evaluates on the identical folds.
"""
from __future__ import annotations

from pathlib import Path

# ---- paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"                 # symlink to the M5 raw files
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# ---- sample definition ------------------------------------------------------
# One store per state -> geographic spread + a genuine 3-state SNAP signal.
# All three categories, balanced, so segment analysis is meaningful.
# 100 items per (store, category) -> 3 stores x 3 cats x 100 = 900 series.
SAMPLE_STORES = ["CA_1", "TX_1", "WI_1"]
ITEMS_PER_STORE_CAT = 100
SAMPLE_SEED = 20240601                       # reproducible item pick

# ---- forecast / backtest setup (weekly) -------------------------------------
HORIZON_WEEKS = 4                            # forecast 4 weeks ahead
N_FOLDS = 3                                  # rolling-origin folds
SEASON_WEEKS = 52                            # weekly yearly seasonality

# M5 weekday encoding: wday 1=Saturday ... 7=Friday
WEEKEND_WDAYS = (1, 2)                        # Sat, Sun
SNAP_BY_STATE = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


def ensure_dirs() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
