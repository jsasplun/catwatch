# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Joins raw captures with labels and assigns each image to train/val/test."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from catwatch.settings import project_path
from core.cv_tools.capture_store import read_capture_log
from core.cv_tools.labels import latest_labels
from core.cv_tools.splits import assign_split


@dataclass(frozen=True)
class LabeledImage:
    path: Path
    label: str
    split: str


def group_for(captured_at: str) -> str:
    """Images from the same clock hour share a group, and therefore a split.

    An hour is a compromise. Finer groups (single minutes) would let frames of
    one drinking visit leak across splits. Coarser groups (whole days) leave
    too few groups when you only have a week or two of data.
    """
    return datetime.fromisoformat(captured_at).strftime("%Y-%m-%d_%H")


def load_labeled_images(config: dict[str, Any]) -> list[LabeledImage]:
    """Every captured image whose latest label is one of the model's
    classes."""
    raw_dir = project_path(config["paths"]["raw_dir"])
    labels = latest_labels(project_path(config["paths"]["labels_file"]))
    class_names = config["classes"]
    fractions = config["training"]["split_fractions"]
    images: list[LabeledImage] = []
    for capture in read_capture_log(raw_dir):
        label = labels.get(capture["image_path"])
        if label is None or label not in class_names:
            continue  # not labeled yet, or labeled "unusable"
        split = assign_split(group_for(capture["captured_at"]), fractions)
        images.append(LabeledImage(
            raw_dir / capture["image_path"],
            label,
            split
        ))
    return images


def items_for_split(
    images: Sequence[LabeledImage], split: str, class_names: Sequence[str]
) -> list[tuple[Path, int]]:
    """(path, class index) pairs for one split, in the format training
    expects."""
    index_of = {name: index for index, name in enumerate(class_names)}
    return [
        (image.path, index_of[image.label])
        for image in images
        if image.split == split
    ]


def report_split_counts(
    images: Sequence[LabeledImage],
    class_names: Sequence[str],
    split_names: Sequence[str],
) -> None:
    """Print a class-by-split table. Zeros here explain most surprising
    metrics."""
    counts = Counter((image.split, image.label) for image in images)
    print(f"{'class':<20}" + "".join(f"{name:>8}" for name in split_names))
    for class_name in class_names:
        row = "".join(
            f"{counts[(split, class_name)]:>8}" for split in split_names
        )
        print(f"{class_name:<20}{row}")
