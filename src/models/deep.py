"""
Deep-learning forecaster — the *challenger* (Phase 3).

Plan: raw PyTorch LSTM first (interview-defensible, you understand the internals),
then optionally a TFT via pytorch-forecasting for the modern angle.

Same rolling-origin protocol and metrics as the baselines; scalers fit on the
train fold only (TS rule #4).
"""
