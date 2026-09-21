# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Tests for catwatch.data: joining captures with labels and assigning
splits."""

from __future__ import annotations
import sys
from pathlib import Path
# Adds the parent directory of this file to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from catwatch.data import (
    LabeledImage,
    group_for,
    items_for_split,
    load_labeled_images,
    report_split_counts,
)
from core.cv_tools.labels import append_label
from core.cv_tools.splits import assign_split

CLASSES = ["empty", "black_patch_cat", "orange_patch_cat", "both_cats"]


def test_group_for_is_the_clock_hour() -> None:
    assert group_for("2026-09-19T14:03:59-05:00") == "2026-09-19_14"


def test_group_for_puts_the_same_hour_together_and_other_hours_apart() -> None:
    assert group_for("2026-09-19T14:00:00-05:00") == group_for(
        "2026-09-19T14:59:59-05:00"
    )
    assert group_for("2026-09-19T14:59:59-05:00") != group_for(
        "2026-09-19T15:00:00-05:00"
    )
    assert group_for("2026-09-19T14:00:00-05:00") != group_for(
        "2026-09-20T14:00:00-05:00"
    )


def test_group_for_output_format_is_stable() -> None:
    # Split assignment hashes this exact string, so changing its format would
    # silently move images between train, val, and test.
    assert group_for("2026-01-02T03:04:05+00:00") == "2026-01-02_03"


@pytest.fixture
def labeled_dataset(
    config: dict[str, Any], add_capture: Callable[..., str]
) -> dict[str, str]:
    """Five captures and a labels file covering every case load_labeled_images
    must handle. Returns {short name: image path relative to raw_dir}."""
    raw_dir = Path(config["paths"]["raw_dir"])
    labels_file = Path(config["paths"]["labels_file"])
    names = {
        "plain": add_capture(raw_dir, "2026-09-19/a.jpg", "2026-09-19T14:05:00-05:00"),
        "sibling": add_capture(
            raw_dir, "2026-09-19/b.jpg", "2026-09-19T14:40:00-05:00"
        ),
        "unusable": add_capture(
            raw_dir, "2026-09-19/c.jpg", "2026-09-19T15:00:00-05:00"
        ),
        "unlabeled": add_capture(
            raw_dir, "2026-09-19/d.jpg", "2026-09-19T16:00:00-05:00"
        ),
        "relabeled": add_capture(
            raw_dir, "2026-09-20/e.jpg", "2026-09-20T09:00:00-05:00"
        ),
    }
    append_label(labels_file, names["plain"], "empty", "tester")
    append_label(labels_file, names["sibling"], "black_patch_cat", "tester")
    append_label(labels_file, names["unusable"], "unusable", "tester")
    append_label(labels_file, names["relabeled"], "black_patch_cat", "tester")
    append_label(labels_file, names["relabeled"], "both_cats", "tester")
    return names


def test_load_labeled_images_keeps_only_images_with_a_class_label(
    config: dict[str, Any], labeled_dataset: dict[str, str]
) -> None:
    images = load_labeled_images(config)
    raw_dir = Path(config["paths"]["raw_dir"])
    got = {image.path.relative_to(raw_dir).as_posix(): image.label for image in images}
    # "unusable" is recorded but is not a class; the unlabeled image has no row.
    assert got == {
        labeled_dataset["plain"]: "empty",
        labeled_dataset["sibling"]: "black_patch_cat",
        labeled_dataset["relabeled"]: "both_cats",  # newest label wins
    }


def test_load_labeled_images_returns_full_paths_that_exist(
    config: dict[str, Any], labeled_dataset: dict[str, str]
) -> None:
    images = load_labeled_images(config)
    assert images and all(image.path.is_file() for image in images)


def test_load_labeled_images_assigns_split_from_the_hour_group(
    config: dict[str, Any], labeled_dataset: dict[str, str]
) -> None:
    fractions = config["training"]["split_fractions"]
    by_path = {image.path.name: image.split for image in load_labeled_images(config)}
    assert by_path["a.jpg"] == assign_split("2026-09-19_14", fractions)
    assert by_path["e.jpg"] == assign_split("2026-09-20_09", fractions)
    # a.jpg and b.jpg were taken within the same hour, so they must share a
    # split. Otherwise near-identical frames would leak between train and test.
    assert by_path["a.jpg"] == by_path["b.jpg"]


def test_load_labeled_images_with_no_labels_file_returns_nothing(
    config: dict[str, Any], add_capture: Callable[..., str]
) -> None:
    add_capture(Path(config["paths"]["raw_dir"]), "2026-09-19/a.jpg")
    assert load_labeled_images(config) == []


def test_load_labeled_images_ignores_labels_for_uncaptured_images(
    config: dict[str, Any], add_capture: Callable[..., str]
) -> None:
    raw_dir = Path(config["paths"]["raw_dir"])
    add_capture(raw_dir, "2026-09-19/a.jpg")
    append_label(Path(config["paths"]["labels_file"]), "ghost.jpg", "empty", "t")
    assert load_labeled_images(config) == []


def test_load_labeled_images_does_not_modify_labels_or_captures(
    config: dict[str, Any], labeled_dataset: dict[str, str]
) -> None:
    files = [
        Path(config["paths"]["labels_file"]),
        Path(config["paths"]["raw_dir"]) / "captures.csv",
    ]
    before = [file.read_bytes() for file in files]
    load_labeled_images(config)
    assert [file.read_bytes() for file in files] == before


def _images() -> list[LabeledImage]:
    return [
        LabeledImage(Path("a.jpg"), "empty", "train"),
        LabeledImage(Path("b.jpg"), "both_cats", "train"),
        LabeledImage(Path("c.jpg"), "black_patch_cat", "val"),
        LabeledImage(Path("d.jpg"), "orange_patch_cat", "test"),
    ]


def test_items_for_split_returns_path_and_class_index_pairs() -> None:
    # Indices follow the order of CLASSES, which is what the model outputs.
    assert items_for_split(_images(), "train", CLASSES) == [
        (Path("a.jpg"), 0),
        (Path("b.jpg"), 3),
    ]
    assert items_for_split(_images(), "val", CLASSES) == [(Path("c.jpg"), 1)]
    assert items_for_split(_images(), "test", CLASSES) == [(Path("d.jpg"), 2)]


def test_items_for_split_class_order_changes_indices() -> None:
    reordered = ["both_cats", "empty", "black_patch_cat", "orange_patch_cat"]
    assert items_for_split(_images(), "train", reordered)[0] == (Path("a.jpg"), 1)


def test_items_for_split_unknown_split_is_empty() -> None:
    assert items_for_split(_images(), "nonsense", CLASSES) == []


def test_report_split_counts_prints_a_class_by_split_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    images = _images() + [LabeledImage(Path("e.jpg"), "empty", "train")]
    report_split_counts(images, CLASSES, ["train", "val", "test"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["class", "train", "val", "test"]
    rows = {line.split()[0]: line.split()[1:] for line in lines[1:]}
    assert rows == {
        "empty": ["2", "0", "0"],
        "black_patch_cat": ["0", "1", "0"],
        "orange_patch_cat": ["0", "0", "1"],
        "both_cats": ["1", "0", "0"],
    }
