# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""An append-only label log.

Every labeling decision is a new row: which image, which label, who, when.
Nothing is overwritten. If the same image appears more than once, the newest
row wins. That gives undo, corrections, and a full audit trail, and the file
diffs cleanly in git, so label changes can be reviewed like code changes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.cv_tools.csv_log import append_csv_row, read_csv_rows

LABEL_FIELDS = ("image_path", "label", "labeled_by", "labeled_at")


def append_label(
    labels_file: Path, image_path: str, label: str, labeled_by: str
) -> None:
    append_csv_row(
        labels_file,
        LABEL_FIELDS,
        {
            "image_path": image_path,
            "label": label,
            "labeled_by": labeled_by,
            "labeled_at": datetime.now().astimezone().isoformat(),
        },
    )


def latest_labels(labels_file: Path) -> dict[str, str]:
    """Map each image path to its most recent label. Empty if no labels yet."""
    if not labels_file.exists():
        return {}
    labels: dict[str, str] = {}
    for row in read_csv_rows(labels_file):
        labels[row["image_path"]] = row["label"]  # later rows replace earlier
    return labels
