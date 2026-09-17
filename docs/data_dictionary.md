# Data Dictionary

## 1. Source dataset

| Property | Value |
|---|---|
| Name | Gold Futures (GC=F) historical data |
| URL | https://finance.yahoo.com/quote/GC%3DF/history/ |
| Source | Yahoo Finance, downloaded with `yfinance` via `python scripts/download_data.py` |
| Accessed | 2026-09-17 |
| Frequency | Daily (trading days) |
| Coverage | 2015-01-02 – 2026-04-29, 2,847 rows × 6 columns |
| Format | Single CSV |
| Licence | Research and educational use only (Yahoo Finance terms); do not redistribute the raw file |

> **Spot vs futures.** Yahoo's gold series is the front-month COMEX futures contract (`GC=F`), which tracks spot XAU/USD
> closely but not exactly (a small carry premium and roll effects). The report uses it as a proxy for the XAU/USD gold
> price and states this explicitly.

Profile of the file used in this project (from the executed notebook): no missing values, no duplicate dates, no
non-positive prices, no inconsistent OHLC bars. Calendar gaps between rows: 1 day ×2,229, 2 days ×26, 3 days ×510,
4 days ×81. `Close` ranges from USD 1,050.80 to 5,318.40 (mean 1,863.88, std 837.29).

## 2. Raw variables

| Column | Type | Unit | Description |
|---|---|---|---|
| `Date` | date | — | Trading day |
| `Open` | float | USD / troy oz | Opening price |
| `High` | float | USD / troy oz | Session high |
| `Low` | float | USD / troy oz | Session low |
| `Close` | float | USD / troy oz | Closing price; the next-day value is the forecasting target |
| `Volume` | int | contracts | Traded volume. **Not used as a feature**: 35 zero-volume days, and the median jumps from ~56–195 (2015–2019) to ~160k–208k (2020 onward), which points to a change in how the source reports volume |

## 3. Derived variables

| Column | Formula | Used by |
|---|---|---|
| `log_return` | `ln(Close[t] / Close[t-1])` | EDA, LSTM input |
| `ret_lag_k` (k = 0..9) | `log_return[t-k]` | XGBoost |
| `ret_mean_w` (w = 5, 10, 20) | trailing mean of `log_return` | XGBoost |
| `ret_std_w` | trailing standard deviation of `log_return` | XGBoost |
| `close_sma_ratio_w` | `Close[t] / SMA_w[t] − 1` | XGBoost |
| `hl_range` | `(High[t] − Low[t]) / Close[t]` | XGBoost |
| `oc_change` | `Close[t] / Open[t] − 1` | XGBoost |
| `day_of_week` | 0 = Monday … 4 = Friday | XGBoost |
| `close` | `Close[t]` | all (price reconstruction, naive) |
| `target_return` | `ln(Close[t+1] / Close[t])` | XGBoost, LSTM target |
| `target_close` | `Close[t+1]` | evaluation target |
