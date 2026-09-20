# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Helpers for CSV files that are only ever appended to, never rewritten.

Append-only files are safe for long-running programs (a crash loses at most the
last row) and keep history, because nothing is silently overwritten.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path


def append_csv_row(
    path: Path, fieldnames: Sequence[str], row: Mapping[str, object]
) -> None:
    """Add one row to the end of a CSV, writing the header first if it's
    new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read every row of a CSV as a {column name: value} dictionary."""
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
