"""Data cleaning, quality reporting, stationarity testing and chronological splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    rows_in: int
    rows_out: int
    duplicate_dates: int
    missing_close: int
    non_positive_close: int
    filled_ohl: int
    ohlc_inconsistent: int
    flat_bars: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def clean_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Return a Date-indexed, sorted, de-duplicated OHLC frame plus a quality report.

    Steps (order matters and mirrors the report):
    1. drop rows with an unparseable date;
    2. sort chronologically and drop duplicate dates (keep the last record);
    3. drop rows whose Close is missing or non-positive (the target cannot be imputed);
    4. fill missing Open/High/Low with that day's Close (a flat bar);
    5. count, but keep, bars where High/Low do not bracket Open/Close.

    Non-trading days are *not* inserted: forecasting is done on the trading-day
    sequence, so reindexing to calendar days would fabricate prices.
    """
    rows_in = len(df)
    out = df.dropna(subset=["Date"]).sort_values("Date")

    duplicate_dates = int(out["Date"].duplicated(keep="last").sum())
    out = out.drop_duplicates(subset="Date", keep="last")

    missing_close = int(out["Close"].isna().sum())
    out = out.dropna(subset=["Close"])
    non_positive = int((out["Close"] <= 0).sum())
    out = out[out["Close"] > 0]

    filled = 0
    for col in ("Open", "High", "Low"):
        if col in out.columns:
            mask = out[col].isna() | (out[col] <= 0)
            filled += int(mask.sum())
            out.loc[mask, col] = out.loc[mask, "Close"]

    inconsistent = flat = 0
    if {"Open", "High", "Low"} <= set(out.columns):
        # O=H=L=C bars (settlement-only days); kept, but reported
        flat = int(((out["Open"] == out["High"]) & (out["High"] == out["Low"]) & (out["Low"] == out["Close"])).sum())
        hi = out[["Open", "Close"]].max(axis=1)
        lo = out[["Open", "Close"]].min(axis=1)
        inconsistent = int(((out["High"] < hi) | (out["Low"] > lo)).sum())

    out = out.set_index("Date")
    out.index.name = "Date"
    report = QualityReport(
        rows_in=rows_in,
        rows_out=len(out),
        duplicate_dates=duplicate_dates,
        missing_close=missing_close,
        non_positive_close=non_positive,
        filled_ohl=filled,
        ohlc_inconsistent=inconsistent,
        flat_bars=flat,
        date_min=out.index.min(),
        date_max=out.index.max(),
    )
    return out, report


def filter_date_range(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    return df.loc[slice(start, end)]


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["Close"]).diff()
    return out


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test (H0: unit root / non-stationary)."""
    from statsmodels.tsa.stattools import adfuller

    stat, pvalue, lags, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    return {"adf_stat": stat, "p_value": pvalue, "lags": lags, "nobs": nobs, **{f"crit_{k}": v for k, v in crit.items()}}


def select_differencing_order(series: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    """Smallest ``d`` for which the ADF test rejects a unit root."""
    current = series.dropna()
    for d in range(max_d + 1):
        if adf_test(current)["p_value"] < alpha:
            return d
        current = current.diff().dropna()
    return max_d


def split_index(n: int, train_frac: float, val_frac: float) -> tuple[slice, slice, slice]:
    """Positional, non-overlapping, chronological train/val/test slices."""
    if not (0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Require 0 < train_frac, 0 <= val_frac and train_frac + val_frac < 1.")
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def chronological_split(frame: pd.DataFrame, train_frac: float, val_frac: float):
    tr, va, te = split_index(len(frame), train_frac, val_frac)
    return frame.iloc[tr], frame.iloc[va], frame.iloc[te]
