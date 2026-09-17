"""Leakage-safe supervised framing.

Each row is a *forecast origin* ``t`` (a trading day). Features use information
available at the close of ``t`` only; the targets describe day ``t+1``:

* ``target_return`` = ``log(Close[t+1] / Close[t])``
* ``target_close``  = ``Close[t+1]``

Tree ensembles and neural nets are trained on the return target because gold
reached all-time highs in the test period: a model trained on raw price levels
cannot extrapolate above the training maximum. Prices are reconstructed with
``Close[t] * exp(predicted_return)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NON_FEATURE_COLUMNS = ("close", "target_return", "target_close")


def build_feature_frame(df: pd.DataFrame, n_lags: int = 10, windows: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    close = df["Close"]
    ret = np.log(close).diff()
    out = pd.DataFrame(index=df.index)

    for k in range(n_lags):
        out[f"ret_lag_{k}"] = ret.shift(k)  # lag 0 = today's return, known at the close
    for w in windows:
        out[f"ret_mean_{w}"] = ret.rolling(w).mean()
        out[f"ret_std_{w}"] = ret.rolling(w).std()
        out[f"close_sma_ratio_{w}"] = close / close.rolling(w).mean() - 1.0
    if {"High", "Low"} <= set(df.columns):
        out["hl_range"] = (df["High"] - df["Low"]) / close
    if "Open" in df.columns:
        out["oc_change"] = close / df["Open"] - 1.0
    out["day_of_week"] = df.index.dayofweek

    out["close"] = close
    out["target_return"] = np.log(close.shift(-1) / close)
    out["target_close"] = close.shift(-1)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def returns_to_price(current_close: np.ndarray | pd.Series, predicted_log_return: np.ndarray) -> np.ndarray:
    return np.asarray(current_close, dtype=float) * np.exp(np.asarray(predicted_log_return, dtype=float))


def make_sequences(log_returns: pd.Series, origins: pd.DatetimeIndex, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of past log returns ending at each origin (inclusive).

    Returns ``X`` with shape ``(n_valid, window, 1)`` and a boolean mask over
    ``origins`` marking which origins had a full window of history.
    """
    values = log_returns.to_numpy(dtype=float)
    positions = log_returns.index.get_indexer(origins)
    if (positions < 0).any():
        raise KeyError("Some origins are not present in the return series index.")
    mask = np.zeros(len(origins), dtype=bool)
    windows = []
    for i, p in enumerate(positions):
        start = p - window + 1
        if start < 0:
            continue
        seg = values[start : p + 1]
        if np.isnan(seg).any():
            continue
        mask[i] = True
        windows.append(seg)
    X = np.asarray(windows, dtype=np.float32).reshape(-1, window, 1)
    return X, mask
