"""Matplotlib figures used by the pipeline, the notebook and the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    return fig


def plot_price_history(df: pd.DataFrame, path=None) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(df.index, df["Close"], color="#B8860B", lw=1)
    axes[0].plot(df.index, df["Close"].rolling(200).mean(), color="#444", lw=1, ls="--", label="SMA 200")
    axes[0].set_ylabel("Close (USD/oz)")
    axes[0].set_title("Gold daily close price")
    axes[0].legend()
    if "Volume" in df.columns:
        axes[1].bar(df.index, df["Volume"], color="#888", width=1.0)
    axes[1].set_ylabel("Volume")
    return _save(fig, path)


def plot_returns(log_returns: pd.Series, path=None) -> plt.Figure:
    r = log_returns.dropna()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(r.index, r, lw=0.6, color="#4C72B0")
    axes[0].set_title("Daily log return")
    axes[1].hist(r, bins=80, density=True, color="#4C72B0", alpha=0.7)
    x = np.linspace(r.min(), r.max(), 300)
    axes[1].plot(x, np.exp(-((x - r.mean()) ** 2) / (2 * r.var())) / np.sqrt(2 * np.pi * r.var()), color="crimson", label="Normal fit")
    axes[1].set_title(f"Distribution (kurtosis={r.kurt():.2f})")
    axes[1].legend()
    vol = r.rolling(21).std() * np.sqrt(252) * 100
    axes[2].plot(vol.index, vol, color="#DD8452", lw=0.8)
    axes[2].set_title("21-day annualised volatility (%)")
    fig.tight_layout()
    return _save(fig, path)


def plot_acf_pacf(series: pd.Series, lags: int = 40, title: str = "", path=None) -> plt.Figure:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF {title}")
    axes[1].set_title(f"PACF {title}")
    fig.tight_layout()
    return _save(fig, path)


def plot_seasonal_profile(df: pd.DataFrame, path=None) -> plt.Figure:
    r = np.log(df["Close"]).diff().dropna() * 100
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    yearly = df["Close"].resample("YE").last().pct_change().dropna() * 100
    axes[0].bar(yearly.index.year, yearly.values, color=np.where(yearly.values >= 0, "#55A868", "#C44E52"))
    axes[0].set_title("Annual return (%)")
    r.groupby(r.index.month).mean().plot.bar(ax=axes[1], color="#4C72B0")
    axes[1].set_title("Mean daily log return by month (%)")
    r.groupby(r.index.dayofweek).mean().plot.bar(ax=axes[2], color="#8172B2")
    axes[2].set_title("Mean daily log return by weekday (%)")
    fig.tight_layout()
    return _save(fig, path)


def plot_split(close: pd.Series, train_end, val_end, path=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(close.loc[:train_end], label="Train", color="#4C72B0", lw=1)
    ax.plot(close.loc[train_end:val_end], label="Validation", color="#DD8452", lw=1)
    ax.plot(close.loc[val_end:], label="Test", color="#55A868", lw=1)
    ax.set_title("Chronological train / validation / test split")
    ax.set_ylabel("Close (USD/oz)")
    ax.legend()
    return _save(fig, path)


def plot_predictions(dates, y_true, predictions: dict[str, np.ndarray], last_n: int | None = None, path=None) -> plt.Figure:
    dates = pd.DatetimeIndex(dates)
    sl = slice(-last_n, None) if last_n else slice(None)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates[sl], np.asarray(y_true)[sl], color="black", lw=1.6, label="Actual")
    for name, pred in predictions.items():
        ax.plot(dates[sl], np.asarray(pred)[sl], lw=1, alpha=0.85, label=name)
    ax.set_title("Test set: actual vs one-step-ahead forecasts" + (f" (last {last_n} days)" if last_n else ""))
    ax.set_ylabel("Close (USD/oz)")
    ax.legend(ncol=5)
    return _save(fig, path)


def plot_residuals(dates, y_true, predictions: dict[str, np.ndarray], path=None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for name, pred in predictions.items():
        err = np.asarray(y_true) - np.asarray(pred)
        axes[0].plot(pd.DatetimeIndex(dates), err, lw=0.7, alpha=0.8, label=name)
        axes[1].hist(err, bins=50, alpha=0.45, label=name)
    axes[0].set_title("Forecast error over time")
    axes[1].set_title("Error distribution")
    axes[0].legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_learning_curve(history: pd.DataFrame, path=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardised returns)")
    ax.set_title("LSTM learning curve (best configuration)")
    ax.legend()
    return _save(fig, path)


def plot_feature_importance(importance: pd.Series, top_n: int = 15, path=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.head(top_n)[::-1].plot.barh(ax=ax, color="#55A868")
    ax.set_title(f"XGBoost feature importance (top {top_n})")
    return _save(fig, path)


def plot_metric_comparison(metrics: pd.DataFrame, path=None) -> plt.Figure:
    cols = [c for c in ("RMSE", "MAE", "MAPE (%)") if c in metrics.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    for ax, col in zip(np.atleast_1d(axes), cols):
        metrics[col].plot.bar(ax=ax, color="#4C72B0")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return _save(fig, path)
