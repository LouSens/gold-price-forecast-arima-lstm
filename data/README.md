# Data

Raw and processed data are **not committed**. The source is Yahoo Finance data, which is licensed for research and
educational use only.

```
data/
├── raw/
│   └── xauusd_daily.csv      <- written by scripts/download_data.py
└── processed/
    └── xauusd_clean.csv      <- written by the notebook / pipeline
```

## Source

Daily COMEX gold futures (`GC=F`) from Yahoo Finance: https://finance.yahoo.com/quote/GC%3DF/history/

```bash
python scripts/download_data.py            # GC=F, 2015-01-01 to 2026-04-30 (end date exclusive)
```

The file used in this project was downloaded on 2026-09-17 and contains 2,847 rows (2015-01-02 to 2026-04-29).
Yahoo may revise historical values, so a later download can differ slightly.

The loader also accepts other daily OHLCV CSVs: plain headers, the multi-row header written by recent `yfinance`
versions, lower-case or alias column names (`price`, `vol.`, `adj_close`) and thousands separators. See
`docs/data_dictionary.md` for variable definitions.
