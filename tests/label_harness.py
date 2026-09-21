# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Human test harness for the labeling tool. A person runs this; pytest does not.

    python -m tests.label_harness        (run from the repository root)

Automated tests can press keys for a pretend person (tests/test_label.py), but
they can't tell whether a real window opens, whether the image is readable,
whether colors look right, or whether the keys feel right. This harness checks
those things with a real person at a real window.

What it does:

1. Builds a throwaway sandbox in a temporary folder: 10 synthetic pictures of
   a bowl (empty, or with cats that have black or orange patches) whose correct
   labels are known.
2. Opens the real labeling command, `catwatch.label`, on that sandbox. It uses
   the real key bindings from config.yaml. Your real data/raw and
   data/labels.csv are never opened, read, or written.
3. Walks you through two labeling sessions, telling you exactly which keys to
   press, including going back, pressing a key that means nothing, quitting,
   and resuming.
4. Grades what ended up in the sandbox's labels.csv, asks you a few yes/no
   questions about what you saw, and prints PASS or FAIL for each check.

Exit code is 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations
import sys
from pathlib import Path
# Adds the parent directory of this file to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import cv2
import numpy as np
import yaml

from catwatch import label
from catwatch.settings import CONFIG_PATH
from core.cv_tools.capture_store import CAPTURE_FIELDS, CAPTURE_LOG_NAME
from core.cv_tools.csv_log import append_csv_row, read_csv_rows
from core.cv_tools.labels import latest_labels

HARNESS_USER = "label-harness"  # stored in the labeled_by column of every row

# The correct label for each sandbox picture, in the order they are shown.
# "unusable" is not a class; it is the label for a picture too washed out to
# judge. The order is deliberately not 0, 1, 2, 3, ... so you have to look at
# each picture instead of pressing keys by habit.
CORRECT_LABELS = [
    "orange_patch_cat",  # image 1
    "empty",  # image 2
    "both_cats",  # image 3
    "black_patch_cat",  # image 4
    "unusable",  # image 5
    "black_patch_cat",  # image 6
    "empty",  # image 7
    "orange_patch_cat",  # image 8
    "both_cats",  # image 9
    "unusable",  # image 10
]
SESSION_ONE_IMAGE_COUNT = 4  # session 1 labels images 1-4, then quits
UNASSIGNED_KEY = "w"  # must not be in config.yaml's label_keys

IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
# Used when config.yaml has no bowl_crop yet, so the yellow outline can still
# be checked. It encloses the bowl and the cats around it.
FALLBACK_CROP = {"x": 60, "y": 80, "width": 540, "height": 320}

# Colors are (blue, green, red), the order OpenCV uses. Orange in BGR is
# (0, 140, 255): if a viewer swapped the channels it would turn blue, which is
# why the harness asks you whether the orange looks orange.
FLOOR = (140, 150, 160)
BOWL_RIM = (175, 175, 180)
BOWL_WATER = (205, 195, 160)
CREAM_FUR = (220, 230, 240)
BLACK_PATCH = (25, 25, 25)
ORANGE_PATCH = (0, 140, 255)
BOWL_CENTER = (320, 240)


@dataclass(frozen=True)
class Sandbox:
    """A temporary copy of the labeling setup, seeded with known pictures."""

    config: dict[str, Any]  # config.yaml with every path moved into the sandbox
    raw_dir: Path
    labels_file: Path
    image_paths: list[str]  # relative to raw_dir, in the order shown
    correct_labels: list[str]  # same order as image_paths


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


# --- Building the sandbox ----------------------------------------------------


def draw_bowl(image: np.ndarray) -> None:
    cv2.circle(image, BOWL_CENTER, 95, BOWL_RIM, -1, cv2.LINE_AA)
    cv2.circle(image, BOWL_CENTER, 75, BOWL_WATER, -1, cv2.LINE_AA)


def draw_cat(
    image: np.ndarray, center: tuple[int, int], patch_color: tuple[int, int, int]
) -> None:
    """A cream-colored cat (body and head) with two patches of one coat color.

    Cats in this project are told apart only by patch color, so the patches
    are the one thing that differs between the "black" and "orange" pictures.
    """
    x, y = center
    cv2.ellipse(image, center, (110, 55), 0, 0, 360, CREAM_FUR, -1, cv2.LINE_AA)
    cv2.circle(image, (x + 105, y - 10), 38, CREAM_FUR, -1, cv2.LINE_AA)  # head
    cv2.circle(image, (x - 40, y - 10), 26, patch_color, -1, cv2.LINE_AA)
    cv2.circle(image, (x + 30, y + 15), 20, patch_color, -1, cv2.LINE_AA)


def draw_scene(correct_label: str, seed: int) -> np.ndarray:
    """One synthetic picture. `seed` makes brightness and noise differ between
    pictures so no two are identical."""
    rng = np.random.default_rng(seed)
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), FLOOR, dtype=np.uint8)
    if correct_label == "unusable":
        # Overexposed and blurry: you can't tell what is in the bowl.
        image[:] = 235
        image = cv2.GaussianBlur(image, (0, 0), 25)
    else:
        draw_bowl(image)
        if correct_label == "black_patch_cat":
            draw_cat(image, (300, 310), BLACK_PATCH)
        elif correct_label == "orange_patch_cat":
            draw_cat(image, (300, 310), ORANGE_PATCH)
        elif correct_label == "both_cats":
            draw_cat(image, (200, 320), BLACK_PATCH)
            draw_cat(image, (440, 320), ORANGE_PATCH)
    brightness = int(rng.integers(-20, 21))
    noise = rng.integers(-6, 7, image.shape)
    return np.clip(image.astype(int) + brightness + noise, 0, 255).astype(np.uint8)


def orange_pixel_mask(image: np.ndarray) -> np.ndarray:
    """True where a BGR pixel is close to the orange patch color."""
    blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    return (blue < 60) & (green > 100) & (green < 180) & (red > 200)


def black_pixel_mask(image: np.ndarray) -> np.ndarray:
    """True where a BGR pixel is close to the black patch color."""
    return (image < 50).all(axis=2)


def load_real_config_without_secrets() -> dict[str, Any]:
    """Reads config.yaml directly. catwatch.settings.load_config() would also
    load .env, which holds secrets this harness has no need for."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def build_sandbox(root: Path) -> Sandbox:
    config = load_real_config_without_secrets()
    known_labels = set(config["label_keys"].values())
    unknown = set(CORRECT_LABELS) - known_labels
    if unknown:
        raise ValueError(
            f"config.yaml label_keys no longer has {sorted(unknown)}. Update "
            "CORRECT_LABELS in tests/label_harness.py to match."
        )
    if UNASSIGNED_KEY in config["label_keys"]:
        raise ValueError(f"'{UNASSIGNED_KEY}' is now a label key; pick another")

    raw_dir = root / "raw"
    config["paths"] = {
        "raw_dir": str(raw_dir),
        "labels_file": str(root / "labels.csv"),
        "events_file": str(root / "events.csv"),
        "runs_dir": str(root / "runs"),
        "models_dir": str(root / "models"),
    }
    if config["bowl_crop"] is None:
        config["bowl_crop"] = dict(FALLBACK_CROP)

    image_paths: list[str] = []
    for number, correct in enumerate(CORRECT_LABELS, start=1):
        relative = f"harness/img_{number:02d}.jpg"
        (raw_dir / "harness").mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(raw_dir / relative), draw_scene(correct, number)):
            raise OSError(f"Could not write {raw_dir / relative}")
        append_csv_row(
            raw_dir / CAPTURE_LOG_NAME,
            CAPTURE_FIELDS,
            {
                "image_path": relative,
                "captured_at": f"2026-01-01T00:{number:02d}:00+00:00",
                "reason": "harness",
            },
        )
        image_paths.append(relative)
    return Sandbox(
        config=config,
        raw_dir=raw_dir,
        labels_file=root / "labels.csv",
        image_paths=image_paths,
        correct_labels=list(CORRECT_LABELS),
    )


# --- Running the real labeling command ---------------------------------------


def run_labeling_session(sandbox: Sandbox) -> None:
    """Runs `python -m catwatch.label --labeled-by label-harness`, pointed at
    the sandbox instead of the real data. Blocks until the window closes."""
    with (
        mock.patch.object(label, "load_config", lambda: sandbox.config),
        mock.patch.object(sys, "argv", ["label", "--labeled-by", HARNESS_USER]),
    ):
        label.main()


def read_rows(labels_file: Path) -> list[dict[str, str]]:
    """All rows of the labels file; empty if nothing was ever labeled."""
    return read_csv_rows(labels_file) if labels_file.exists() else []


# --- Grading -----------------------------------------------------------------


def _rows_for(rows: list[dict[str, str]], image_path: str) -> list[str]:
    return [row["label"] for row in rows if row["image_path"] == image_path]


def grade_session_one(sandbox: Sandbox, rows: list[dict[str, str]]) -> list[Check]:
    """Grade the labels file as it stands after session 1."""
    first, second, third, fourth = sandbox.image_paths[:SESSION_ONE_IMAGE_COUNT]
    right = dict(zip(sandbox.image_paths, sandbox.correct_labels))
    labeled = {row["image_path"] for row in rows}
    second_rows = _rows_for(rows, second)
    return [
        Check(
            "Session 1 labeled images 1-4 and nothing else",
            labeled == set(sandbox.image_paths[:SESSION_ONE_IMAGE_COUNT]),
            f"labeled: {sorted(labeled)}",
        ),
        Check(
            "Images 1, 3 and 4 each got exactly one correct label",
            all(
                _rows_for(rows, path) == [right[path]]
                for path in (first, third, fourth)
            ),
            ", ".join(f"{p}: {_rows_for(rows, p)}" for p in (first, third, fourth)),
        ),
        Check(
            "Image 2 has 3 rows: correct, wrong-on-purpose, then correct again "
            "(the unassigned key wrote nothing and 'back' allowed a redo)",
            len(second_rows) == 3
            and second_rows[0] == right[second]
            and second_rows[1] != right[second]
            and second_rows[2] == right[second],
            f"rows for image 2, oldest first: {second_rows}",
        ),
    ]


def grade_session_two(
    sandbox: Sandbox,
    session_one_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
) -> list[Check]:
    """Grade the labels file after session 2 finished."""
    new_rows = final_rows[len(session_one_rows) :]
    newest = latest_labels(sandbox.labels_file)
    wrong = [
        f"{path}: expected {want}, got {newest.get(path)}"
        for path, want in zip(sandbox.image_paths, sandbox.correct_labels)
        if newest.get(path) != want
    ]
    return [
        Check(
            "Session 2 did not rewrite or remove any session 1 row (append-only)",
            final_rows[: len(session_one_rows)] == session_one_rows,
        ),
        Check(
            "Session 2 resumed at image 5 and labeled images 5-10 once each, "
            "in order (already-labeled images were skipped)",
            [row["image_path"] for row in new_rows]
            == sandbox.image_paths[SESSION_ONE_IMAGE_COUNT:],
            f"session 2 labeled: {[row['image_path'] for row in new_rows]}",
        ),
        Check(
            "The newest label of every image is the correct one",
            not wrong,
            "; ".join(wrong),
        ),
        Check(
            f"Every row records labeled_by = {HARNESS_USER!r}",
            bool(final_rows)
            and all(r["labeled_by"] == HARNESS_USER for r in final_rows),
        ),
    ]


# --- The guided walk-through -------------------------------------------------

# Yes/no questions about things only a person looking at the screen can judge.
VISUAL_QUESTIONS = [
    "A yellow rectangle was drawn on every picture?",
    "The orange patches looked ORANGE (not blue), and the black patches black?",
    "The top line showed the counter and the hints '[z] back' and '[q] quit' as "
    "plain text (no stray braces or placeholder names), and the list of label "
    "keys below it was readable over the picture?",
    "In session 2 the counter started at 1/6 (only unlabeled images shown)?",
    "The window closed by itself at the end of each session?",
]


def _key_for(label_keys: dict[str, str], label_name: str) -> str:
    return next(key for key, name in label_keys.items() if name == label_name)


def describe_keys(sandbox: Sandbox) -> str:
    return "\n".join(
        f"      press {key!r}  ->  {name}"
        for key, name in sandbox.config["label_keys"].items()
    )


IMAGE_GUIDE = """\
How to tell the pictures apart:
  empty             the water bowl and floor only
  black_patch_cat   one cream cat with BLACK patches
  orange_patch_cat  one cream cat with ORANGE patches
  both_cats         two cats, one black-patched and one orange-patched
  unusable          washed-out white blur where nothing can be seen
"""


def run_harness(
    root: Path,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> list[Check]:
    """Runs the whole guided test and returns every check, passed or not."""
    sandbox = build_sandbox(root)
    output_fn(f"Sandbox created at {root} (your real data is not used).\n")
    output_fn(IMAGE_GUIDE)
    output_fn("Label keys from config.yaml:\n" + describe_keys(sandbox))
    output_fn(
        "      press 'z'  ->  go back one image\n"
        "      press 'q'  ->  quit (everything so far is saved)\n"
        f"      press {UNASSIGNED_KEY!r}  ->  not assigned to anything (used below)\n"
    )
    output_fn(
        "SESSION 1. Follow these steps in order:\n"
        "  1. Image 1: label it normally.\n"
        f"  2. Image 2: first press {UNASSIGNED_KEY!r}. Nothing should happen.\n"
        "     Then label image 2 correctly.\n"
        "  3. Image 3 appears: press 'z'. You should be shown image 2 again.\n"
        "  4. On image 2, deliberately press a WRONG label key.\n"
        "  5. Image 3 appears again: press 'z' again, and this time label\n"
        "     image 2 correctly.\n"
        "  6. Label image 3 and image 4 normally.\n"
        "  7. When image 5 appears, press 'q' WITHOUT labeling it.\n"
    )
    input_fn("Press Enter to open the labeling window... ")
    run_labeling_session(sandbox)
    session_one_rows = read_rows(sandbox.labels_file)

    output_fn(
        "\nSESSION 2. The window reopens. It should skip the images you already\n"
        "labeled, so the counter should read 1/6. Label all 6 remaining images\n"
        "correctly. The window closes by itself after the last one.\n"
    )
    input_fn("Press Enter to reopen the labeling window... ")
    run_labeling_session(sandbox)
    final_rows = read_rows(sandbox.labels_file)

    checks = grade_session_one(sandbox, session_one_rows)
    checks += grade_session_two(sandbox, session_one_rows, final_rows)

    output_fn("\nA few questions about what you saw:")
    for question in VISUAL_QUESTIONS:
        answer = input_fn(f"  {question} [y/n] ").strip().lower()
        checks.append(Check(question, answer.startswith("y")))
    return checks


def print_report(checks: list[Check], output_fn: Callable[[str], None] = print) -> bool:
    """Prints one PASS/FAIL line per check. Returns True if all passed."""
    output_fn("\nRESULTS")
    for check in checks:
        output_fn(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}")
        if not check.passed and check.detail:
            output_fn(f"         {check.detail}")
    failed = [check for check in checks if not check.passed]
    if failed:
        output_fn(f"\n{len(failed)} of {len(checks)} checks FAILED.")
    else:
        output_fn(f"\nAll {len(checks)} checks passed.")
    return not failed


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="label_harness_") as folder:
        checks = run_harness(Path(folder))
    sys.exit(0 if print_report(checks) else 1)


if __name__ == "__main__":
    main()
