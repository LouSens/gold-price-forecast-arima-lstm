"""Forecast accuracy metrics and a Diebold-Mariano significance test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent. Safe here because gold prices are strictly positive."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def directional_accuracy(current, y_true, y_pred) -> float:
    """Share of days (in percent) where the predicted up/down move matches the realised one.

    Days with zero realised change are excluded. Returns NaN for a flat forecaster
    (e.g. the naive random walk), which never predicts a direction.
    """
    current, y_true, y_pred = (np.asarray(a, float) for a in (current, y_true, y_pred))
    tol = 1e-9 * np.abs(current)
    true_dir = np.sign(y_true - current)
    pred_dir = np.where(np.abs(y_pred - current) <= tol, 0.0, np.sign(y_pred - current))
    valid = true_dir != 0
    if not valid.any() or not pred_dir.any():
        return float("nan")
    return float(np.mean(true_dir[valid] == pred_dir[valid]) * 100.0)


def evaluate(current, y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
        "Directional Acc. (%)": directional_accuracy(current, y_true, y_pred),
    }


def compare_models(current, y_true, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = {name: evaluate(current, y_true, pred) for name, pred in predictions.items()}
    return pd.DataFrame(rows).T.sort_values("RMSE")


def diebold_mariano(y_true, pred_a, pred_b, h: int = 1, power: int = 2) -> dict:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    H0: both forecasts are equally accurate. A negative statistic means
    ``pred_a`` has the lower loss.
    """
    y_true, pred_a, pred_b = (np.asarray(a, float) for a in (y_true, pred_a, pred_b))
    d = np.abs(y_true - pred_a) ** power - np.abs(y_true - pred_b) ** power
    n = len(d)
    d_mean = d.mean()
    gamma = [np.sum((d[k:] - d_mean) * (d[: n - k] - d_mean)) / n for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    dm = d_mean / np.sqrt(var_d)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    p_value = 2 * stats.t.sf(np.abs(dm_hln), df=n - 1)
    return {"dm_stat": float(dm_hln), "p_value": float(p_value)}
