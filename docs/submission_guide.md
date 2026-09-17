# Dicoding Submission Guide

A checklist that maps the Dicoding *Predictive Analytics* criteria to this repository.

## 1. Mandatory criteria (a missing item means rejection)

| Requirement | Where it is met | Status |
|---|---|---|
| Own work, not previously submitted or **published anywhere** | Keep the GitHub repository **private** until the submission has been reviewed | ☐ |
| Quantitative dataset with at least 500 samples | 2,847 daily rows × 6 columns | ✅ |
| Notebook text cells document **every** stage | `notebooks/gold_price_forecasting.py` (markdown cells per stage) | ✅ |
| Classification, regression or **time-series forecasting** | Next-day price forecasting | ✅ |
| Report in **Markdown** following the template | `reports/laporan.md` (all values filled from the executed notebook) | ✅ |
| **.ipynb and .py** included | `notebooks/gold_price_forecasting.ipynb`, `.py` | ✅ |
| Notebook **has been run** (outputs visible) | All 31 code cells executed, no errors | ✅ |
| Every code cell documented by a text cell | Each of the 31 code cells is directly preceded by a markdown cell (checked automatically) | ✅ |
| Submitted as **.zip** | `python scripts/make_submission.py` → `dist/submission_gold_forecast.zip` (built, 1.8 MB) | ✅ |

## 2. Report rubric: mandatory and additional (for 4–5 stars)

| Section | Mandatory | Additional (stars) | Report location |
|---|---|---|---|
| Project Domain | Background | Why and how to solve it; credible references | `## Domain Proyek` |
| Business Understanding | Problem statements, goals | Solution statements (≥ 2, measurable) | `## Business Understanding` |
| Data Understanding | Size, condition, link, all variables | EDA / visualisation with insight | `## Data Understanding` |
| Data Preparation | Techniques, same order as notebook | Process explained, reasons given | `## Data Preparation` |
| Modeling | Model, stages and parameters | Pros/cons per algorithm; tuning or best-model justification | `## Modeling` |
| Evaluation | Metrics, results | Metric formulas and how they work | `## Evaluation` |
| Report structure | Template structure, sensible code snippets, images load | — | whole report |

Five stars requires **all six** additional criteria; the report template in this repository includes all of them.

## 3. Step-by-step

```bash
# 1) Get the data (Kaggle CSV -> data/raw/xauusd_daily.csv), or:
python scripts/download_data.py

# 2) Build and execute the notebook (outputs saved into the .ipynb, figures into reports/figures/)
python scripts/build_notebook.py --execute --kernel current
#    or open notebooks/gold_price_forecasting.ipynb in Jupyter/VS Code and "Run All", then save

# 3) Copy the numbers printed by the notebook into reports/laporan.md (replace every [ISI ...])

# 4) Package
python scripts/make_submission.py
```

`make_submission.py` refuses to build the ZIP if the notebook is unexecuted, figures are missing or `[ISI ...]`
placeholders remain.

## 4. About the ZIP contents

The checklist asks for a ZIP with 3 files (`.md`, `.py`, executed `.ipynb`). The ZIP contains exactly those three, plus a
`figures/` folder. The folder is required by the *Struktur Laporan* criterion ("resources such as images must load when
the Markdown is read"); without it, every image link in `laporan.md` would be broken.

## 5. Reviewer-proofing tips

- Open `laporan.md` in a Markdown previewer from inside the unzipped folder to confirm every image renders.
- Keep numbers in the report identical to the executed notebook output (reviewers cross-check).
- Do not claim the LSTM "wins" unless the Diebold-Mariano p-value supports it; an honest "no significant improvement
  over the random walk" is a legitimate and well-supported result.
- Do not include the raw CSV in the ZIP unless required; cite and link the dataset instead.
