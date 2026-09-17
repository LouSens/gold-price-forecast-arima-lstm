import numpy as np

from src.config import ArimaConfig, LSTMConfig, XGBConfig
from src.features import build_feature_frame, feature_columns
from src.models import ArimaForecaster, LSTMForecaster, NaiveForecaster, XGBReturnForecaster
from src.preprocessing import add_returns, chronological_split, clean_prices


def _prepared(ohlc_df):
    df = add_returns(clean_prices(ohlc_df)[0])
    frame = build_feature_frame(df, n_lags=5, windows=(5, 10))
    return df, frame, *chronological_split(frame, 0.7, 0.15)


def test_naive_predicts_current_close(ohlc_df):
    _, _, _, _, test = _prepared(ohlc_df)
    assert np.array_equal(NaiveForecaster().predict_close(test), test["close"].to_numpy())


def test_arima_one_step_forecasts_are_close_to_actual(ohlc_df):
    df, _, train, _, test = _prepared(ohlc_df)
    model = ArimaForecaster(ArimaConfig(p_values=(0, 1), q_values=(0, 1))).fit(df["Close"].loc[: train.index[-1]])
    assert model.order_[1] == 1
    pred = model.predict_close(df["Close"], test.index)
    assert pred.shape == (len(test),) and np.isfinite(pred).all()
    assert np.mean(np.abs(pred / test["target_close"] - 1)) < 0.05


def test_xgboost_tunes_and_predicts(ohlc_df):
    _, frame, train, val, test = _prepared(ohlc_df)
    feats = feature_columns(frame)
    cfg = XGBConfig(param_grid={"max_depth": [2, 3], "learning_rate": [0.1]}, n_estimators=50, early_stopping_rounds=5)
    model = XGBReturnForecaster(cfg).fit(
        train[feats], train["target_return"], val[feats], val["target_return"], val["close"], val["target_close"]
    )
    assert len(model.tuning_results_) == 2
    pred = model.predict_close(test[feats], test["close"])
    assert np.isfinite(pred).all()


def test_lstm_trains_on_cpu(ohlc_df):
    df, _, train, val, test = _prepared(ohlc_df)
    cfg = LSTMConfig(windows=(10,), hidden_sizes=(8,), num_layers=(1,), max_epochs=2, patience=1)
    model = LSTMForecaster(cfg, device="cpu").fit(df["log_return"], train, val)
    pred = model.predict_close(test)
    assert np.isfinite(pred).all()
    assert model.best_params_ == {"window": 10, "hidden_size": 8, "num_layers": 1}
