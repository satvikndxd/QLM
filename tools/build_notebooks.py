"""Build unexecuted Colab .ipynb files from plain cell-source files.

Each source file under notebooks/src/ is a text file of code cells
separated by lines containing exactly:

    # --- CELL ---

This builder ONLY serializes those cells into nbformat-4 JSON with empty
outputs and null execution counts. It never executes any cell code — the
notebooks are written for the user to run later in Google Colab.

A light static syntax check (ast.parse with shell/magic lines stripped)
catches authoring errors without executing anything.

Usage:
    python tools/build_notebooks.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "notebooks" / "src"
OUT_DIR = REPO_ROOT / "notebooks"

CELL_SEP = "# --- CELL ---"


def split_cells(text: str) -> list[str]:
    cells, current = [], []
    for line in text.splitlines():
        if line.strip() == CELL_SEP:
            if current:
                cells.append("\n".join(current).strip("\n"))
            current = []
        else:
            current.append(line)
    if current:
        chunk = "\n".join(current).strip("\n")
        if chunk:
            cells.append(chunk)
    return [c for c in cells if c.strip()]


def static_syntax_check(cell_src: str, where: str) -> None:
    """ast-parse the cell with Colab shell/magic lines blanked out.

    This is a static check only — nothing is executed.
    """
    cleaned_lines = []
    for line in cell_src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            indent = line[: len(line) - len(stripped)]
            cleaned_lines.append(f"{indent}pass  # colab shell/magic line")
        else:
            cleaned_lines.append(line)
    try:
        ast.parse("\n".join(cleaned_lines))
    except SyntaxError as exc:
        raise SyntaxError(f"{where}: {exc}") from exc


def build_notebook(src_path: Path) -> Path:
    cells = split_cells(src_path.read_text(encoding="utf-8"))
    if not cells:
        raise ValueError(f"{src_path}: no cells found")
    nb_cells = []
    for i, cell in enumerate(cells):
        static_syntax_check(cell, f"{src_path.name} cell {i + 1}")
        # nbformat stores source as a list of lines with trailing newlines.
        lines = cell.splitlines()
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        nb_cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],  # never executed here, by design
                "source": source,
            }
        )
    name = src_path.name.replace(".cells.py", "")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": f"{name}.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": nb_cells,
    }
    out_path = OUT_DIR / f"{name}.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_files = sorted(SRC_DIR.glob("*.cells.py"))
    if not src_files:
        raise SystemExit(f"no cell-source files in {SRC_DIR}")
    for src in src_files:
        out = build_notebook(src)
        n_cells = len(split_cells(src.read_text(encoding="utf-8")))
        print(f"built {out.relative_to(REPO_ROOT)} ({n_cells} cells)")


if __name__ == "__main__":
    main()
