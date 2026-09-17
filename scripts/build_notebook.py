"""Convert the percent-format notebook source (.py) into a Jupyter notebook (.ipynb), optionally executing it.

The .py file is the single source of truth: it is both the plain-script version of the notebook and the
input for the .ipynb, so the two can never drift apart.

    python scripts/build_notebook.py              # build notebooks/gold_price_forecasting.ipynb
    python scripts/build_notebook.py --execute --kernel current   # build and run with this Python (outputs saved)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "notebooks" / "gold_price_forecasting.py"
CELL_MARKER = re.compile(r"^# %%(?P<md>\s*\[markdown\])?.*$")


def parse_percent_script(text: str) -> list:
    cells, kind, buf = [], None, []

    def flush():
        body = "\n".join(buf).strip("\n")
        if kind is None or not body.strip():
            return
        if kind == "markdown":
            lines = [re.sub(r"^# ?", "", line) if line.startswith("#") else line for line in body.splitlines()]
            cells.append(new_markdown_cell("\n".join(lines)))
        else:
            cells.append(new_code_cell(body))

    for line in text.splitlines():
        match = CELL_MARKER.match(line)
        if match:
            flush()
            kind, buf = ("markdown" if match.group("md") else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def _register_temporary_kernel() -> str:
    """Expose the running interpreter as a kernel for this process only (no change to the user's Jupyter config)."""
    spec_root = Path(tempfile.mkdtemp(prefix="gold-kernel-"))
    spec_dir = spec_root / "kernels" / "gold-forecast-current"
    spec_dir.mkdir(parents=True)
    (spec_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (current interpreter)",
        "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = os.pathsep.join(filter(None, [str(spec_root), os.environ.get("JUPYTER_PATH")]))
    return "gold-forecast-current"


def build(src: Path, dst: Path, execute: bool, timeout: int, kernel: str = "python3") -> None:
    nb = new_notebook(cells=parse_percent_script(src.read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    if execute:
        if kernel == "current":
            kernel = _register_temporary_kernel()
        from nbconvert.preprocessors import ExecutePreprocessor

        ExecutePreprocessor(timeout=timeout, kernel_name=kernel).preprocess(nb, {"metadata": {"path": str(dst.parent)}})
    nbformat.write(nb, dst)
    print(f"Wrote {dst} ({len(nb.cells)} cells{', executed' if execute else ''})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-cell timeout in seconds")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name (see `jupyter kernelspec list`), or 'current' for this Python")
    args = parser.parse_args()
    build(args.src, args.dst or args.src.with_suffix(".ipynb"), args.execute, args.timeout, args.kernel)


if __name__ == "__main__":
    main()
