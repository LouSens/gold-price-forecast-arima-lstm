# Reproducibility

How to re-run the project from scratch and what is checked along the way.

## 1. Fresh-environment guarantees

| Concern | How it is handled |
|---|---|
| Missing folders | The notebook creates `data/raw`, `data/processed`, `reports/figures`, `results` and `models` before writing anything. `src/config.py` (`PathConfig.ensure`) does the same for the CLI pipeline. |
| Missing data | If `data/raw/xauusd_daily.csv` is absent, the notebook downloads Yahoo Finance `GC=F` (2015-01-01 to 2026-04-30, end date exclusive) with `yfinance`. `scripts/download_data.py` is an optional command-line equivalent and is not required. |
| Randomness | Python, NumPy and PyTorch seeds are fixed and cuDNN runs deterministically, so accuracy metrics are identical across runs. Runtimes vary slightly. |
| Data revisions | Yahoo can revise history. The file used for the reported results was downloaded on 2026-09-17 and has 2,847 rows (2015-01-02 to 2026-04-29). |
| Notebook vs script drift | `notebooks/gold_price_forecasting.py` is the single source; `scripts/build_notebook.py` generates the `.ipynb` from it. |

## 2. Run everything

```bash
pip install -r requirements.txt
python scripts/build_notebook.py --execute --kernel current
python -m pytest
python scripts/package_report.py
```

- `--kernel current` executes the notebook with the Python interpreter that runs the command, without registering a
  Jupyter kernel.
- `package_report.py` writes `dist/gold_forecast_report.zip` (report, executed notebook, script, figures). It refuses to
  build the ZIP if the notebook has unexecuted cells, figures are missing, or `[ISI` placeholders remain in the report.
- The modular pipeline is an alternative entry point: `python -m src.pipeline` (writes figures to
  `reports/figures/pipeline/`).

## 3. Conventions that affect how results are read

- **Origin vs target dates.** Each supervised row is a forecast origin `t`; the predicted value is the close on the next
  trading day `t+1`. Test origins are 2025-03-13 to 2026-04-28 and test targets are 2025-03-14 to 2026-04-29.
  `results/test_predictions.csv` stores both `origin_date` and `target_date`, and forecast plots use target dates.
- **Model selection.** The model is chosen by validation RMSE; the test set is used once for final evaluation.
- **XGBoost boosting rounds.** `best_iteration` is zero-based; the model predicts with `best_iteration + 1` rounds
  (31 for the reported run). Both values are recorded in `results/run_summary.json`.

## 4. Checks

| Check | Command or location |
|---|---|
| Unit tests (loader, look-ahead safety, metrics, models) | `python -m pytest` |
| Every code cell has a text cell; notebook executed without errors | enforced when building the report bundle; see `scripts/package_report.py` |
| Report numbers match results | compare `reports/laporan.md` with `results/metrics.csv` and `results/run_summary.json` |
