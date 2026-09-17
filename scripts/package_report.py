"""Package the project report bundle (report, notebook, script and figures) as a ZIP.

Contents (flat layout so relative image links in the report keep working):

    gold_forecast_report.zip
    ├── laporan.md                     (reports/laporan.md)
    ├── figures/*.png                  (reports/figures/*.png, excluding the pipeline/ subfolder)
    ├── gold_price_forecasting.ipynb   (must already be executed)
    └── gold_price_forecasting.py

    python scripts/package_report.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "laporan.md"
FIGURES = ROOT / "reports" / "figures"
NOTEBOOK = ROOT / "notebooks" / "gold_price_forecasting.ipynb"
SCRIPT = ROOT / "notebooks" / "gold_price_forecasting.py"
OUT = ROOT / "dist" / "gold_forecast_report.zip"


def check() -> list[str]:
    problems = [f"missing {p.relative_to(ROOT)}" for p in (REPORT, NOTEBOOK, SCRIPT) if not p.exists()]
    if NOTEBOOK.exists():
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        if not code or any(c.get("execution_count") is None for c in code):
            problems.append("notebook has unexecuted cells (run: python scripts/build_notebook.py --execute)")
        if any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])):
            problems.append("notebook contains error outputs")
        cells = nb["cells"]
        undocumented = [i for i, c in enumerate(cells)
                        if c["cell_type"] == "code" and (i == 0 or cells[i - 1]["cell_type"] != "markdown")]
        if undocumented:
            problems.append(f"code cells without a preceding text cell: {undocumented}")
    if REPORT.exists() and "[ISI" in REPORT.read_text(encoding="utf-8"):
        problems.append("reports/laporan.md still contains [ISI ...] placeholders")
    if not list(FIGURES.glob("*.png")):
        problems.append("no figures in reports/figures (execute the notebook first)")
    return problems


def main() -> None:
    problems = check()
    if problems:
        print("Cannot package report bundle:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(REPORT, "laporan.md")
        zf.write(NOTEBOOK, NOTEBOOK.name)
        zf.write(SCRIPT, SCRIPT.name)
        for fig in sorted(FIGURES.glob("*.png")):
            zf.write(fig, f"figures/{fig.name}")
    print(f"Created {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
