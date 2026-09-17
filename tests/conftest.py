import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlc_df() -> pd.DataFrame:
    """~600 business days of geometric-random-walk OHLCV with a Date column."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=600)
    close = 1500 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates))))
    return pd.DataFrame({
        "Date": dates,
        "Open": close * (1 + rng.normal(0, 0.002, len(dates))),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(100, 1000, len(dates)),
    })
