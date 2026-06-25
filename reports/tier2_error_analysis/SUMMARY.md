# Tier 2a -- Error Analysis (champion = lstm_seq2seq)

## Headline findings

- **Intermittency drives error.** Per-series WMAPE vs zero-rate correlation r = 0.71. The worst-20 series average a 83% zero-rate vs 63% overall.
- **Low-volume movers are hardest:** WMAPE low=1.31 vs high=0.67.
- **By category:** FOODS=0.70, HOBBIES=0.94, HOUSEHOLD=0.86.
- **Worst-20 concentration by category:** HOBBIES:12, FOODS:6, HOUSEHOLD:2.
- **The champion systematically UNDER-predicts, and the bias grows with volume:** mean signed error low=-0.07, mid=-0.16, high=-0.53 units/day (vs ETS high=+0.01, ~centered).

## What this implies

- The model is good on fast/steady items and weak on sparse, low-volume SKUs -- exactly where *any* point forecast struggles. A two-stage (zero / nonzero) or Croston-style model is the right next step for the long tail.
- **This bias is the mechanistic cause of the A/B stockout result.** The LSTM under-orders high-volume SKUs, so when its (low) forecast drives the order policy it stocks out more than the well-centered ETS champion. It also explains why the quantile-loss fix (which deliberately shifts predictions UP) was the right lever -- and why a high enough quantile is needed to neutralise this under-bias.
