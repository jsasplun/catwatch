# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Creates the test fixtures in data/test_fixtures/.

    python -m tests.make_fixtures                  (from the repository root)
    python -m tests.make_fixtures --output some/other/folder

The fixtures are SYNTHETIC: drawn by code, not photographed. They exist so
tests can run the real camera, motion, labeling, and monitoring code on real
files (JPEGs and a video) whose correct answers are known exactly. They are
kept apart from the real dataset on purpose: nothing here is under data/raw/,
and its labels live in data/test_fixtures/labels.csv, never data/labels.csv.

Everything is generated from fixed random seeds, so running this again gives
the same pictures. It refuses to write into a folder that already exists,
because tests treat the committed fixtures as read-only.

Layout of the output folder:

    raw/captures.csv            one row per still image, same columns as real captures
    raw/stills/*.jpg            14 still images (640x480)
    labels.csv                  the correct label for every still, real labels format
    bowl_visits.avi             a 20 second video of a bowl (320x240, 5 frames/second)
    bowl_visits.json            what happens in the video, frame by frame
    MANIFEST.json               SHA-256 fingerprint of every file above
    README.md                   plain-language description
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from catwatch.data import group_for
from catwatch.settings import PROJECT_ROOT
from core.cv_tools.capture_store import CAPTURE_FIELDS, CAPTURE_LOG_NAME
from core.cv_tools.csv_log import append_csv_row
from core.cv_tools.labels import LABEL_FIELDS
from core.cv_tools.run_records import file_sha256, write_json
from core.cv_tools.splits import assign_split
from tests.label_harness import (
    BLACK_PATCH,
    FLOOR,
    ORANGE_PATCH,
    draw_bowl,
    draw_cat,
    draw_scene,
    load_real_config_without_secrets,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test_fixtures"
FIXTURE_LABELER = "fixture-generator"
FIXTURE_TIMESTAMP = "2026-08-01T00:00:00+00:00"

STILL_LABELS = ["empty", "black_patch_cat", "orange_patch_cat", "both_cats"]
STILL_SPLITS = ["train", "val", "test"]

VIDEO_NAME = "bowl_visits.avi"
VIDEO_INFO_NAME = "bowl_visits.json"
VIDEO_FPS = 5
VIDEO_SIZE = (320, 240)  # (width, height); the cats are drawn at twice this
# size and shrunk, so they look like real frames.
VIDEO_FRAME_COUNT = 100

# The story of the video, in frame numbers. A cat slides in from the right
# for SLIDE_FRAMES frames, sits (with a slight bobbing) for the "settled"
# frames, then slides out again. "Settled" is the stretch a classifier should
# reliably name; the slides are ambiguous by design.
SLIDE_FRAMES = 5
VISITS = [
    {"label": "black_patch_cat", "settled_first": 30, "settled_last": 54},
    {"label": "orange_patch_cat", "settled_first": 70, "settled_last": 84},
]
PATCH_COLORS = {"black_patch_cat": BLACK_PATCH, "orange_patch_cat": ORANGE_PATCH}


def find_hour_in_split(split: str, fractions: dict[str, float]) -> str:
    """The first clock hour, as "2026-08-DDTHH", whose group hashes into
    `split`. Splits are decided by a hash of the hour, so to put an image in
    a chosen split we search for an hour that lands there."""
    for day in range(1, 29):
        for hour in range(24):
            hour_text = f"2026-08-{day:02d}T{hour:02d}"
            if assign_split(group_for(f"{hour_text}:00:00"), fractions) == split:
                return hour_text
    raise AssertionError(f"No hour found for split {split!r}")


def write_stills(output: Path, config: dict) -> None:
    """One image per (class, split), plus two "unusable" washed-out images.

    Every split therefore contains every class exactly once, which is what
    tests of training and evaluation need at minimum.
    """
    fractions = config["training"]["split_fractions"]
    raw_dir = output / "raw"
    (raw_dir / "stills").mkdir(parents=True)
    labels_file = output / "labels.csv"
    hours = {split: find_hour_in_split(split, fractions) for split in STILL_SPLITS}

    entries: list[tuple[str, str, str]] = []  # (relative path, label, captured_at)
    seed = 100
    for split in STILL_SPLITS:
        for minute, label in enumerate(STILL_LABELS):
            entries.append(
                (
                    f"stills/{split}_{label}.jpg",
                    label,
                    f"{hours[split]}:{minute:02d}:00+00:00",
                )
            )
    for number in (1, 2):
        entries.append(
            (
                f"stills/unusable_{number}.jpg",
                "unusable",
                f"{hours['train']}:{30 + number:02d}:00+00:00",
            )
        )

    for relative, label, captured_at in entries:
        seed += 1
        image = draw_scene(label, seed)
        if not cv2.imwrite(str(raw_dir / relative), image):
            raise OSError(f"Could not write {raw_dir / relative}")
        append_csv_row(
            raw_dir / CAPTURE_LOG_NAME,
            CAPTURE_FIELDS,
            {"image_path": relative, "captured_at": captured_at, "reason": "fixture"},
        )
        # Written directly (not through append_label) so the timestamp is fixed
        # and regenerating gives an identical file.
        append_csv_row(
            labels_file,
            LABEL_FIELDS,
            {
                "image_path": relative,
                "label": label,
                "labeled_by": FIXTURE_LABELER,
                "labeled_at": FIXTURE_TIMESTAMP,
            },
        )


def video_frame(frame_number: int, rng: np.random.Generator) -> np.ndarray:
    """One frame of the video: the bowl, plus a cat if one is visiting."""
    width, height = VIDEO_SIZE
    scene = np.full((height * 2, width * 2, 3), FLOOR, dtype=np.uint8)
    draw_bowl(scene)
    for visit in VISITS:
        first = visit["settled_first"]
        last = visit["settled_last"]
        if not first - SLIDE_FRAMES <= frame_number < last + 1 + SLIDE_FRAMES:
            continue
        if frame_number < first:  # sliding in from off the right edge
            offset = round(400 * (first - frame_number) / SLIDE_FRAMES)
        elif frame_number > last:  # sliding back out
            offset = round(400 * (frame_number - last) / SLIDE_FRAMES)
        else:
            offset = 0
        bob = round(3 * np.sin(frame_number / 2))  # a cat is never perfectly still
        draw_cat(scene, (300 + offset, 310 + bob), PATCH_COLORS[str(visit["label"])])
    small = cv2.resize(scene, VIDEO_SIZE, interpolation=cv2.INTER_AREA)
    noise = rng.integers(-3, 4, small.shape)  # sensor noise
    return np.clip(small.astype(int) + noise, 0, 255).astype(np.uint8)


def write_video(output: Path) -> None:
    writer = cv2.VideoWriter(
        str(output / VIDEO_NAME),
        cv2.VideoWriter.fourcc(*"MJPG"),
        float(VIDEO_FPS),
        VIDEO_SIZE,
    )
    if not writer.isOpened():
        raise OSError("This OpenCV build cannot write MJPG video")
    rng = np.random.default_rng(7)
    for frame_number in range(VIDEO_FRAME_COUNT):
        writer.write(video_frame(frame_number, rng))
    writer.release()
    write_json(
        output / VIDEO_INFO_NAME,
        {
            "description": "Synthetic bowl video. A cat slides in, sits, slides out.",
            "fps": VIDEO_FPS,
            "width": VIDEO_SIZE[0],
            "height": VIDEO_SIZE[1],
            "frame_count": VIDEO_FRAME_COUNT,
            "slide_frames": SLIDE_FRAMES,
            "visits": VISITS,
        },
    )


README_TEXT = """\
# Test fixtures

Synthetic (drawn by code, not photographed) images and video for the tests.
They are NOT training data and are kept apart from data/raw/ and
data/labels.csv on purpose. Do not label, edit, or add to them: tests check
them against MANIFEST.json.

- raw/stills/*.jpg + raw/captures.csv + labels.csv: 14 images in the same
  formats as the real dataset. Each split (train/val/test) holds one image of
  each class, plus two `unusable` images.
- bowl_visits.avi + bowl_visits.json: a 20 second video. Black-patched cat
  visits, then the orange-patched cat. The json lists the exact frames.

Regenerate into a NEW folder with `python -m tests.make_fixtures --output <dir>`.
"""


def write_manifest(output: Path) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in ("MANIFEST.json", "README.md")
    )
    write_json(
        output / "MANIFEST.json",
        {path.relative_to(output).as_posix(): file_sha256(path) for path in files},
    )


def generate(output: Path) -> None:
    """Write the whole fixture set into `output`, which must not exist yet."""
    output.mkdir(parents=True, exist_ok=False)
    write_stills(output, load_real_config_without_secrets())
    write_video(output)
    (output / "README.md").write_text(README_TEXT, encoding="utf-8")
    write_manifest(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        generate(args.output)
    except FileExistsError:
        raise SystemExit(
            f"{args.output} already exists. Fixtures are read-only once made; "
            "use --output with a new folder name to generate another copy."
        ) from None
    print(f"Wrote fixtures to {args.output}")


if __name__ == "__main__":
    main()
