# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Automated tests for tests/label_harness.py, the human labeling harness.

The harness grades a real person, so its own grading must be trustworthy. A
scripted "perfect person" must pass every check, and each kind of human slip
(quitting late, mislabeling, skipping the back-and-redo step, answering "no"
to a visual question) must fail exactly the check meant to catch it.

These tests never open a window. The real, interactive run is
`python -m tests.label_harness`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from core.cv_tools.capture_store import read_capture_log
from tests import label_harness
from tests.label_harness import (
    CORRECT_LABELS,
    SESSION_ONE_IMAGE_COUNT,
    UNASSIGNED_KEY,
    VISUAL_QUESTIONS,
    Check,
    black_pixel_mask,
    build_sandbox,
    draw_scene,
    orange_pixel_mask,
    print_report,
    run_harness,
)

# --- The sandbox -------------------------------------------------------------


def test_sandbox_writes_ten_decodable_images_and_a_capture_log(
    tmp_path: Path,
) -> None:
    sandbox = build_sandbox(tmp_path)
    assert len(sandbox.image_paths) == len(CORRECT_LABELS) == 10
    assert [row["image_path"] for row in read_capture_log(sandbox.raw_dir)] == (
        sandbox.image_paths
    )
    for path in sandbox.image_paths:
        image = cv2.imread(str(sandbox.raw_dir / path))
        assert image is not None and image.shape == (480, 640, 3)


def test_sandbox_lives_entirely_inside_the_given_folder(tmp_path: Path) -> None:
    sandbox = build_sandbox(tmp_path)
    for path in sandbox.config["paths"].values():
        assert Path(path).is_relative_to(tmp_path), path
    assert sandbox.labels_file.is_relative_to(tmp_path)


def test_sandbox_correct_labels_are_all_real_label_keys() -> None:
    config = label_harness.load_real_config_without_secrets()
    assert set(CORRECT_LABELS) <= set(config["label_keys"].values())
    assert UNASSIGNED_KEY not in config["label_keys"]


def test_every_picture_is_different(tmp_path: Path) -> None:
    sandbox = build_sandbox(tmp_path)
    images = [cv2.imread(str(sandbox.raw_dir / path)) for path in sandbox.image_paths]
    for i, first in enumerate(images):
        for second in images[i + 1 :]:
            assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    ("label_name", "expects_orange", "expects_black"),
    [
        ("empty", False, False),
        ("black_patch_cat", False, True),
        ("orange_patch_cat", True, False),
        ("both_cats", True, True),
        ("unusable", False, False),
    ],
)
def test_pictures_show_the_patch_colors_their_label_promises(
    label_name: str, expects_orange: bool, expects_black: bool
) -> None:
    image = draw_scene(label_name, seed=3)
    assert (orange_pixel_mask(image).sum() > 300) == expects_orange
    assert (black_pixel_mask(image).sum() > 300) == expects_black


def test_config_without_a_crop_gets_a_fallback_so_the_outline_can_be_checked(
    tmp_path: Path,
) -> None:
    sandbox = build_sandbox(tmp_path)
    assert sandbox.config["bowl_crop"] is not None


# --- A scripted perfect person -----------------------------------------------


def key_for(config: dict[str, Any], label_name: str) -> str:
    return next(k for k, name in config["label_keys"].items() if name == label_name)


def perfect_keystrokes(config: dict[str, Any], wrong_label: str = "both_cats") -> str:
    """The exact keys the on-screen instructions ask a person to press.

    Session 1: image 1; the ignored key then image 2; z; wrong label for
    image 2; z; correct label for image 2; images 3 and 4; q on image 5.
    Session 2: the remaining six images.
    """
    key = lambda name: key_for(config, name)  # noqa: E731
    right = CORRECT_LABELS
    session_one = (
        key(right[0])
        + UNASSIGNED_KEY
        + key(right[1])
        + "z"
        + key(wrong_label)
        + "z"
        + key(right[1])
        + key(right[2])
        + key(right[3])
        + "q"
    )
    session_two = "".join(key(name) for name in right[SESSION_ONE_IMAGE_COUNT:])
    return session_one + session_two


def run_scripted(
    tmp_path: Path,
    install_window: Callable[[str], Any],
    keys_for: Callable[[dict[str, Any]], str] = perfect_keystrokes,
    answers: str = "y",
) -> list[Check]:
    config = label_harness.load_real_config_without_secrets()
    install_window(keys_for(config))
    prompts_seen: list[str] = []

    def scripted_input(prompt: str) -> str:
        prompts_seen.append(prompt)
        return answers if "[y/n]" in prompt else ""

    return run_harness(tmp_path, input_fn=scripted_input, output_fn=lambda _: None)


def failed_names(checks: list[Check]) -> list[str]:
    return [check.name for check in checks if not check.passed]


def test_a_perfect_person_passes_every_check(
    tmp_path: Path, install_window: Callable[[str], Any]
) -> None:
    checks = run_scripted(tmp_path, install_window)
    assert failed_names(checks) == []
    assert len(checks) == 3 + 4 + len(VISUAL_QUESTIONS)


def test_the_harness_labels_are_written_by_the_real_label_command(
    tmp_path: Path, install_window: Callable[[str], Any]
) -> None:
    run_scripted(tmp_path, install_window)
    rows = label_harness.read_rows(tmp_path / "labels.csv")
    assert len(rows) == 6 + 6  # session 1 wrote 6 rows (image 2 three times)
    assert {row["labeled_by"] for row in rows} == {label_harness.HARNESS_USER}


def test_running_the_harness_never_touches_the_real_labels_file(
    tmp_path: Path, install_window: Callable[[str], Any]
) -> None:
    real_labels = label_harness.load_real_config_without_secrets()["paths"][
        "labels_file"
    ]
    real_path = label_harness.CONFIG_PATH.parent / real_labels
    before = real_path.read_bytes() if real_path.exists() else None
    run_scripted(tmp_path, install_window)
    after = real_path.read_bytes() if real_path.exists() else None
    assert before == after


# --- Human slips must be caught ----------------------------------------------


def test_quitting_a_picture_too_late_is_caught(
    tmp_path: Path, install_window: Callable[[str], Any]
) -> None:
    def labels_image_five_too(config: dict[str, Any]) -> str:
        keys = perfect_keystrokes(config)
        # Session 1 ends "...<image 4 key>q". Label image 5 before quitting;
        # image 5 is "unusable", and the quit then lands on image 6.
        marker = "q"
        first_q = keys.index(marker)
        return keys[:first_q] + key_for(config, "unusable") + "q" + keys[first_q + 1 :]

    # The extra "q" is consumed by session 1, so session 2 gets its own keys.
    checks = run_scripted(tmp_path, install_window, labels_image_five_too)
    failed = failed_names(checks)
    assert "Session 1 labeled images 1-4 and nothing else" in failed


def test_a_mislabeled_image_is_named_in_the_failure(
    tmp_path: Path, install_window: Callable[[str], Any]
) -> None:
    def mislabels_image_seven(config: dict[str, Any]) -> str:
        keys = list(perfect_keystrokes(config))
        session_two_start = keys.index("q") + 1
        seventh_from_start = session_two_start + (7 - 1 - SESSION_ONE_IMAGE_COUNT)
        keys[seventh_from_start] = key_for(config, "both_cats")  # should be empty
        return "".join(keys)

    checks = run_scripted(tmp_path, install_window, mislabels_image_seven)
    (mislabel_check,) = [
        c
        for c in checks
        if c.name == "The newest label of every image is the correct one"
    ]
    assert not mislabel_check.passed
    assert "img_07.jpg" in mislabel_check.detail
    assert "expected empty, got both_cats" in mislabel_check.detail


def test_skipping_the_back_and_redo_step_is_caught(
    tmp_path: Path, install_window: Callable[[str], Any]
) -> None:
    def no_back_step(config: dict[str, Any]) -> str:
        right = CORRECT_LABELS
        key = lambda name: key_for(config, name)  # noqa: E731
        session_one = "".join(key(name) for name in right[:4]) + "q"
        session_two = "".join(key(name) for name in right[4:])
        return session_one + session_two

    failed = failed_names(run_scripted(tmp_path, install_window, no_back_step))
    assert any(name.startswith("Image 2 has 3 rows") for name in failed)


def test_a_no_answer_to_a_visual_question_fails_that_check(
    tmp_path: Path, install_window: Callable[[str], Any]
) -> None:
    checks = run_scripted(tmp_path, install_window, answers="n")
    assert failed_names(checks) == VISUAL_QUESTIONS


# --- Reporting ---------------------------------------------------------------


def test_report_lists_pass_and_fail_lines_and_returns_overall_result() -> None:
    lines: list[str] = []
    ok = print_report(
        [Check("good thing", True), Check("bad thing", False, "why it failed")],
        output_fn=lines.append,
    )
    text = "\n".join(lines)
    assert not ok
    assert "[PASS] good thing" in text
    assert "[FAIL] bad thing" in text
    assert "why it failed" in text
    assert "1 of 2 checks FAILED" in text


def test_report_says_all_passed_when_nothing_failed() -> None:
    lines: list[str] = []
    assert print_report([Check("a", True)], output_fn=lines.append)
    assert "All 1 checks passed." in "\n".join(lines)
