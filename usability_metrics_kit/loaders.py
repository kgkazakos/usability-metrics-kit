"""CSV loading helpers.

Files are expected to have one row per respondent/observation. A header row is
auto-detected (any row whose first cell is non-numeric is treated as a header
and dropped). Empty cells are skipped.
"""

from __future__ import annotations

import csv
from pathlib import Path


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_matrix(path: str | Path) -> list[list[float]]:
    """Load a 2-D numeric matrix (rows = respondents, cols = items)."""
    rows: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for raw in reader:
            cells = [c.strip() for c in raw if c.strip() != ""]
            if not cells:
                continue
            if not _is_number(cells[0]):  # header row
                continue
            rows.append([float(c) for c in cells])
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return rows


def load_values(path: str | Path) -> list[float]:
    """Load a single column / flat list of numbers."""
    matrix = load_matrix(path)
    # Flatten: take the first column if multi-column, else the single value.
    return [row[0] for row in matrix]


def load_int_values(path: str | Path) -> list[int]:
    return [int(round(v)) for v in load_values(path)]
