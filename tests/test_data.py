import numpy as np
import pandas as pd
import pytest

from src.data_loader import read_price_csv, resolve_data_path
from src.features import build_feature_frame, feature_columns, make_sequences, returns_to_price
from src.preprocessing import add_returns, chronological_split, clean_prices, select_differencing_order, split_index


def test_reads_plain_csv_with_aliases(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text('date,price,open,high,low,vol.\n2024-01-03,"2,050.5",2040,2060,2030,10\n2024-01-02,2045,2040,2050,2035,12\n')
    df = read_price_csv(path)
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert df.loc[df["Date"] == "2024-01-03", "Close"].item() == pytest.approx(2050.5)


def test_reads_yfinance_multirow_header(tmp_path):
    path = tmp_path / "yf.csv"
    path.write_text(
        "Price,Close,High,Low,Open,Volume\nTicker,GC=F,GC=F,GC=F,GC=F,GC=F\nDate,,,,,\n"
        "2024-01-02,2064.4,2070.0,2050.0,2060.0,100\n2024-01-03,2034.2,2060.0,2030.0,2055.0,90\n"
    )
    df = read_price_csv(path)
    assert len(df) == 2
    assert df["Close"].tolist() == [2064.4, 2034.2]
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])


def test_timezone_aware_dates_keep_calendar_day(tmp_path):
    path = tmp_path / "tz.csv"
    path.write_text("Date,Close\n2024-01-02 00:00:00-05:00,2000\n")
    assert read_price_csv(path)["Date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_missing_close_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date,Open\n2024-01-02,1\n")
    with pytest.raises(ValueError):
        read_price_csv(path)


def test_resolve_data_path_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        resolve_data_path(tmp_path / "xauusd_daily.csv")
    (tmp_path / "other.csv").write_text("Date,Close\n")
    assert resolve_data_path(tmp_path / "xauusd_daily.csv").name == "other.csv"


def test_clean_prices_handles_duplicates_missing_and_order(ohlc_df):
    messy = pd.concat([ohlc_df.iloc[[5]], ohlc_df]).sample(frac=1, random_state=0)
    messy.loc[messy.index[0], "Close"] = np.nan
    messy.loc[messy.index[1], "High"] = np.nan
    clean, report = clean_prices(messy)
    assert clean.index.is_monotonic_increasing and clean.index.is_unique
    assert report.duplicate_dates == 1
    assert clean["Close"].notna().all() and clean["High"].notna().all()


def test_chronological_split_is_ordered_and_disjoint(ohlc_df):
    frame = ohlc_df.set_index("Date")
    train, val, test = chronological_split(frame, 0.8, 0.1)
    assert len(train) + len(val) + len(test) == len(frame)
    assert train.index.max() < val.index.min() and val.index.max() < test.index.min()
    with pytest.raises(ValueError):
        split_index(100, 0.9, 0.2)


def test_random_walk_needs_one_difference(ohlc_df):
    log_close = np.log(ohlc_df["Close"])
    assert select_differencing_order(log_close) == 1


def test_feature_frame_has_no_lookahead(ohlc_df):
    df, _ = clean_prices(ohlc_df)
    frame = build_feature_frame(df, n_lags=5, windows=(5, 10))
    next_close = df["Close"].shift(-1).loc[frame.index]
    assert np.allclose(frame["target_close"], next_close)
    assert np.allclose(frame["target_return"], np.log(next_close / frame["close"]))

    # Perturbing a future price must not change any feature at earlier origins.
    origin = frame.index[100]
    bumped = df.copy()
    bumped.loc[bumped.index > origin, ["Open", "High", "Low", "Close"]] *= 1.5
    frame_bumped = build_feature_frame(bumped, n_lags=5, windows=(5, 10))
    cols = feature_columns(frame)
    pd.testing.assert_frame_equal(frame.loc[:origin, cols], frame_bumped.loc[:origin, cols])


def test_make_sequences_windows_end_at_origin(ohlc_df):
    df = add_returns(clean_prices(ohlc_df)[0])
    origins = df.index[[0, 4, 5, 50]]
    X, mask = make_sequences(df["log_return"], origins, window=5)
    assert mask.tolist() == [False, False, True, True]  # window at position 4 would include the NaN first return
    assert np.allclose(X[1, :, 0], df["log_return"].iloc[46:51].to_numpy())


def test_returns_to_price_roundtrip():
    close = np.array([100.0, 200.0])
    assert np.allclose(returns_to_price(close, np.log([1.1, 0.9])), [110.0, 180.0])
