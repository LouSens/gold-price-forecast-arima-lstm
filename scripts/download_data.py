"""Download daily gold futures (GC=F) OHLCV from Yahoo Finance into data/raw/.

This is the data source of the project (page: https://finance.yahoo.com/quote/GC%3DF/history/).
Yahoo data is for research and education only and must not be redistributed.

    python scripts/download_data.py
    python scripts/download_data.py --ticker GC=F --start 2015-01-01 --end 2026-04-30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.data_loader import download_yahoo  # noqa: E402


def main() -> None:
    cfg = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default=cfg.data.ticker)
    parser.add_argument("--start", default=cfg.data.start_date)
    parser.add_argument("--end", default=cfg.data.end_date, help="Exclusive end date")
    parser.add_argument("--out", type=Path, default=cfg.paths.raw_data)
    args = parser.parse_args()

    df = download_yahoo(args.ticker, args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows ({df['Date'].min().date()} -> {df['Date'].max().date()}) to {args.out}")


if __name__ == "__main__":
    main()
