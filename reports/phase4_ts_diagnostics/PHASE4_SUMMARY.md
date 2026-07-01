# Phase 4 -- Time-Series Diagnostics

Each result fixes a Phase-5 modeling choice.

## Q1 -- Stationarity (ADF + KPSS, 892 series with >=60 weeks)

- At **level**: ADF says stationary for 77% of series; KPSS for 52%; both agree on 47%.
- After **first differencing**: ADF 100%, KPSS 96%, both 95%.
- **Implication:** weekly demand is mostly level-stationary (mean-reverting around a level, little trend). ARIMA needs at most **d=1**; let AutoARIMA pick per series. First differencing helps the minority with drift.

## Q2 -- Seasonality (STL, period=52)

- Aggregate demand: trend strength **0.92**, seasonal strength **0.68**.
- Per series (eligible n=812): median seasonal strength 0.30, median trend strength 0.14; 50% of series have non-trivial yearly seasonality.
- **Implication:** seasonality is real but moderate per-SKU and strong in aggregate -> keep **seasonal-naive(52)** and **ETS(season=52)** as baselines; week-of-year features for ML. Don't force a seasonal term on every SKU.

## Q3 -- Autocorrelation (ACF / PACF on aggregate)

- ACF(lag 1) = 0.89 (strong short memory); ACF(lag 52) = 0.44 (yearly echo). PACF cuts off quickly -> low-order AR.
- **Implication:** short lags (1,2,4) carry most signal -> they dominate the ML feature set; AutoARIMA search bounds can stay small (max_p,q <= 2-3).

## Q4 -- Variance vs level (transform)

- log(std) vs log(mean) across series has slope **0.74** (≈1 would be pure multiplicative). Variance clearly grows with level.
- **Implication:** model on a variance-stabilised target -> **log1p** for the LSTM, **Tweedie** objective for LightGBM (handles zeros + multiplicative spread). Classical models run on raw units (interpretable) but we report bias to catch scale errors.

## Figures

- `01_stl_aggregate.png`
- `02_acf_pacf.png`
- `03_stationarity_summary.png`
- `04_strength_distribution.png`
- `05_mean_variance.png`
