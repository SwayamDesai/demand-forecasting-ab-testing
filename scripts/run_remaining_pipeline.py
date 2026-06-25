"""
Run everything after Phase 2 baselines, in order:
  1. Phase 3a: LSTM seq2seq MSE        -- the accurate-but-biased model
  2. Phase 3b: LSTM seq2seq q80        -- pinball-loss variant (cost-aware bias)
  3. Phase 4a: A/B v1 (LightGBM vs LSTM MSE)
  4. Phase 4b: A/B v2 (LightGBM vs LSTM q80, no extra safety stock)

Single tracked process so we get one completion notification.
Run: python -m scripts.run_remaining_pipeline
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("Phase 2: baselines (sn/ets/arima/lgbm)", ["scripts.phase2_baselines"]),
    ("Phase 3a: LSTM seq2seq MSE",  ["scripts.phase3_lstm_seq2seq"]),
    ("Phase 3b: LSTM seq2seq q80",  ["scripts.phase3_lstm_quantile"]),
    ("Phase 4a: A/B v1",            ["scripts.phase4_experiment"]),
    ("Phase 4b: A/B v2",            ["scripts.phase4_experiment_v2"]),
]


def run(label: str, module: str) -> tuple[bool, float]:
    t0 = time.time()
    print(f"\n{'='*70}\n>>> {label}\n{'='*70}", flush=True)
    res = subprocess.run([sys.executable, "-m", module], check=False)
    dt = time.time() - t0
    ok = res.returncode == 0
    print(f">>> {label}: {'OK' if ok else 'FAILED'} ({dt/60:.1f} min)", flush=True)
    return ok, dt


def main() -> int:
    total_t0 = time.time()
    summary = []
    for label, args in STEPS:
        ok, dt = run(label, args[0])
        summary.append({"step": label, "ok": ok, "minutes": round(dt / 60, 1)})
        if not ok:
            print(f"\n!!! pipeline halted at: {label}\n", flush=True)
            break

    total = (time.time() - total_t0) / 60
    print(f"\n{'='*70}\nPIPELINE DONE in {total:.1f} min\n{'='*70}")
    for row in summary:
        print(f"  {row['step']:40s}  {'OK' if row['ok'] else 'FAIL':6s}  "
              f"{row['minutes']:>6.1f} min")
    return 0 if all(r["ok"] for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
