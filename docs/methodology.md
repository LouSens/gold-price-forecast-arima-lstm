# Methodology

## 1. Problem framing

- **Task:** univariate, one-step-ahead point forecast of the next trading day's gold closing price.
- **Forecast origin:** close of day `t`; **target:** `Close[t+1]`.
- **Why one-step-ahead:** it matches daily decision cycles (hedging, rebalancing) and is the standard setting in the
  comparison studies this project builds on (Siami-Namini et al., 2018; Xiao et al., 2022).

## 2. Data preparation (in execution order)

1. **Schema normalisation.** Map alias column names to `Date, Open, High, Low, Close, Volume`; parse dates (timezone
   removed, normalised to midnight); strip thousands separators.
2. **Cleaning.** Drop unparseable dates → sort → drop duplicate dates (keep last) → drop missing or non-positive `Close`
   → fill missing `Open/High/Low` with `Close`. Trading-day gaps are kept as they are.
3. **Date filter.** 2015-01-01 to 2026-04-30.
4. **Log returns.** `r[t] = ln(Close[t]/Close[t-1])`.
5. **Feature engineering.** 22 features at origin `t` (10 lagged returns, rolling mean/std of returns and price-to-SMA
   ratio over 5/10/20 days, high-low range, open-to-close change, weekday) plus the targets.
6. **Chronological split.** 80 % train / 10 % validation / 10 % test.
7. **Design checks without test data.** Returns-vs-levels, differencing order, seasonality and autocorrelation are
   re-verified on training and validation data only (91.8 % of validation closes exceed the training maximum; train ADF
   p = 0.79 for log price and 0.00 for returns; largest monthly mean effect = 0.14 daily standard deviations; Ljung-Box
   p = 0.32 at lag 10).
8. **Standardisation (LSTM).** z-score with training-period mean and std.
9. **Sliding windows (LSTM).** `W ∈ {30, 60}` past standardised returns → next standardised return.

## 3. Models

| Model | Input | Target | Selection procedure |
|---|---|---|---|
| Naive | `Close[t]` | `Close[t+1]` | none |
| ARIMA(p,d,q) | `log Close` history | `log Close[t+1]` | `d` by ADF (α = 0.05); `(p,q) ∈ {0..3}²` by AIC on train |
| XGBoost | 22 engineered features | `r[t+1]` | 12-point grid on validation price RMSE; early stopping (50 rounds) |
| LSTM | window of `W` standardised returns | standardised `r[t+1]` | 8-point grid (`W`, hidden, layers) on validation price RMSE; early stopping (patience 10) |

Return forecasts are converted back to prices with `Close[t] · exp(r̂[t+1])`.

## 4. Evaluation protocol

- Every model is scored on the **same** test origins with the **same** targets.
- **RMSE** (primary), **MAE**, **MAPE**, **directional accuracy**, and fit+predict **time**.
- **Diebold-Mariano test** (squared-error loss, h = 1, Harvey-Leybourne-Newbold correction) of each model against the
  naive forecast.
- **Selection:** the best model is the one with the lowest **validation** RMSE (the test set never influences the choice).
  It is recommended over the naive/ARIMA baseline only if its test improvement is statistically significant
  (p < 0.05); otherwise the cheaper model is preferred.

## 5. Expected behaviour and how to read the results

Daily gold returns are close to unpredictable from their own history (weak-form efficiency, Fama 1970). It is therefore
common for all models to land within a fraction of a percent of the naive RMSE, and for ARIMA to select a near-random-walk
order such as (0,1,0). This is a valid, reportable finding that answers the problem statements: it quantifies how much
(or how little) the extra complexity of XGBoost and LSTM buys, in line with Makridakis et al. (2018).

### Observed on the real dataset

| Expectation | Observed |
|---|---|
| ARIMA close to a random walk | AIC selected ARIMA(0,1,0); Ljung-Box p-values 0.39 / 0.32 / 0.13 at lags 5 / 10 / 20 |
| ML models within a fraction of a percent of Naive | XGBoost −0.35 %, LSTM −0.20 % RMSE; DM p-values 0.781 and 0.338 |
| Little learnable signal | XGBoost early-stopped at 4–43 boosting rounds (`best_iteration` is zero-based, so rounds = `best_iteration` + 1) and its feature importances are nearly uniform; LSTM validation loss bottomed at epoch 2 and its test forecasts are always "up" |
| Test period harder than validation | Annualised volatility 27.5 % (test) vs 15.4 % (validation); std of daily USD change 75.81 vs 24.48 |

Selected by validation RMSE: LSTM (test RMSE 75.65, −0.20 % vs Naive, p = 0.338, not significant).
Recommended model: Naive / ARIMA(0,1,0). See `reports/laporan.md` for the full discussion.

## 6. Known limitations and extensions

- Univariate only; candidate exogenous drivers include the US dollar index, real yields (10-year TIPS), VIX, oil and
  central-bank purchase data.
- Single test period, which may coincide with an unusual regime; rolling-origin cross-validation would give a more robust estimate.
- Point forecasts only; quantile or GARCH-type volatility forecasts would add risk information.
- ARIMA has no drift term under `d = 1`; a drift (`trend="t"`) variant could be added to the search.
