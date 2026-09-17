# Data

Raw and processed data are **not committed**. The source is Yahoo Finance data, which is licensed for research and
educational use only.

```
data/
├── raw/
│   └── xauusd_daily.csv      <- place the Kaggle CSV here (rename it), or run scripts/download_data.py
└── processed/
    └── xauusd_clean.csv      <- written by the notebook / pipeline
```

## Option A: Kaggle (primary)

1. Download *Gold Price Master Dataset (2015–2026) XAU/USD* from
   https://www.kaggle.com/datasets/aminasalamt/gold-price-master-dataset-2015-2026-lxauusd
2. Unzip and save the daily CSV as `data/raw/xauusd_daily.csv`. If it is the only CSV in `data/raw/`, any file name works.

## Option B: Yahoo Finance (fallback)

```bash
python scripts/download_data.py            # GC=F, 2015-01-01 to 2026-04-30
```

The loader accepts plain OHLCV headers, the multi-row header written by recent `yfinance` versions, lower-case or alias
column names (`price`, `vol.`, `adj_close`) and thousands separators. See `docs/data_dictionary.md` for variable definitions.
