"""Load XAU/USD daily OHLCV data from CSV (Kaggle / Yahoo export) or Yahoo Finance.

The loader normalises the many CSV layouts found in the wild:

* plain ``Date,Open,High,Low,Close,Volume`` files (Kaggle),
* the multi-row header written by ``yfinance>=0.2.51``
  (``Price`` / ``Ticker`` / ``Date`` rows),
* lower-case or alias column names (``price``, ``adj_close``, ``vol.``),
* thousands separators (``1,234.50``) and timezone-aware timestamps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

COLUMN_ALIASES = {
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "time": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "price": "Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "adjclose": "Adj Close",
    "volume": "Volume",
    "vol": "Volume",
    "vol.": "Volume",
}


def _is_yfinance_multiheader(path: Path) -> bool:
    head = pd.read_csv(path, nrows=2, header=None, dtype=str)
    return (
        len(head) >= 2
        and str(head.iloc[0, 0]).strip().lower() == "price"
        and str(head.iloc[1, 0]).strip().lower() == "ticker"
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename alias columns to canonical names and keep Date + OHLCV only."""
    renamed = {c: COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip()) for c in df.columns}
    df = df.rename(columns=renamed)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    missing = {"Date", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}; found {list(df.columns)}")

    keep = ["Date"] + [c for c in OHLCV if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    for col in keep[1:]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return df


def read_price_csv(path: str | Path) -> pd.DataFrame:
    """Read a daily OHLCV CSV into a canonical ``Date, Open, High, Low, Close[, Volume]`` frame."""
    path = Path(path)
    if _is_yfinance_multiheader(path):
        raw = pd.read_csv(path, header=0, skiprows=[1, 2])
        raw = raw.rename(columns={raw.columns[0]: "Date"})
    else:
        raw = pd.read_csv(path)
    return standardize_columns(raw)


def resolve_data_path(preferred: str | Path) -> Path:
    """Return ``preferred`` if it exists, else the single CSV found next to it."""
    preferred = Path(preferred)
    if preferred.exists():
        return preferred
    candidates = sorted(preferred.parent.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    hint = (
        f"No dataset at {preferred}. Download the Kaggle 'Gold Price Master Dataset (2015-2026) XAU/USD' "
        f"CSV into {preferred.parent}, or run `python scripts/download_data.py`."
    )
    if candidates:
        hint += f" Multiple CSVs found ({[c.name for c in candidates]}); pass the path explicitly."
    raise FileNotFoundError(hint)


def download_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance via ``yfinance`` (research/education use only)."""
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {ticker} between {start} and {end}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return standardize_columns(data.reset_index())
