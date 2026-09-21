# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Tests that run the real code on the files in data/test_fixtures/.

Most other tests use scripted fakes for the camera and the model. These use
actual files: JPEG stills with known labels, and a 20 second video with a known
story (empty bowl, black-patched cat visits, empty, orange-patched cat visits,
empty). The real OpenCV video reader, the real motion detector, and the real
collect, monitor, check_crop, and label commands all run on them.

The fixtures are synthetic (see data/test_fixtures/README.md) and read-only:
every test that writes anything writes into pytest's tmp_path, and
test_fixture_files_match_their_manifest fails if a fixture is ever changed.
"""

from __future__ import annotations
import sys
from pathlib import Path
# Adds the parent directory of this file to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from catwatch import check_crop, collect, label, monitor
from catwatch.data import load_labeled_images, report_split_counts
from catwatch.settings import PROJECT_ROOT
from core.cv_tools.camera import OpenCVSource
from core.cv_tools.capture_store import read_capture_log
from core.cv_tools.csv_log import read_csv_rows
from core.cv_tools.labels import latest_labels
from core.cv_tools.motion import MotionDetector
from core.cv_tools.onnx_classifier import Prediction
from core.cv_tools.run_records import file_sha256, read_json, write_json
from core.cv_tools.splits import assign_split
from tests import make_fixtures
from tests.label_harness import black_pixel_mask, orange_pixel_mask

FIXTURES = PROJECT_ROOT / "data" / "test_fixtures"
STILLS_ROOT = FIXTURES / "raw"
VIDEO = FIXTURES / make_fixtures.VIDEO_NAME
VIDEO_INFO = read_json(FIXTURES / make_fixtures.VIDEO_INFO_NAME)
MIN_PATCH_PIXELS = 300  # patch pixels expected in a still that shows that patch
MIN_VIDEO_PATCH_PIXELS = 100  # the video is half the size, so a quarter the area


@pytest.fixture(autouse=True)
def fixtures_exist() -> None:
    assert (FIXTURES / "MANIFEST.json").is_file(), (
        f"{FIXTURES} is missing. Restore it from git, or run "
        "`python -m tests.make_fixtures`."
    )


def read_video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def motion_frames_in_video() -> list[int]:
    """Frame numbers where the real MotionDetector (default 2% threshold)
    reports motion."""
    detector = MotionDetector(min_changed_fraction=0.02)
    return [
        number
        for number, frame in enumerate(read_video_frames(VIDEO))
        if detector.update(frame)
    ]


class ColorRuleClassifier:
    """A stand-in for the neural network that answers from patch colors.

    Where the real model would look at the picture and guess, this counts
    black and orange patch pixels. That gives the monitor tests a classifier
    that is right about the fixture video every time, so any wrong event
    comes from the code under test and not from a weak model.

    `mistakes` maps a frame number to the wrong label to give on that frame,
    imitating the odd blurry frame a real model misreads.
    """

    def __init__(self, mistakes: dict[int, str] | None = None) -> None:
        self._mistakes = mistakes or {}
        self._frames_seen = 0

    def factory(self, *_args: Any) -> ColorRuleClassifier:
        """Stands in for the OnnxImageClassifier class itself."""
        return self

    def predict(self, bgr_image: np.ndarray) -> Prediction:
        frame_number = self._frames_seen
        self._frames_seen += 1
        if frame_number in self._mistakes:
            return Prediction(self._mistakes[frame_number], 0.99)
        has_black = black_pixel_mask(bgr_image).sum() > MIN_VIDEO_PATCH_PIXELS
        has_orange = orange_pixel_mask(bgr_image).sum() > MIN_VIDEO_PATCH_PIXELS
        if has_black and has_orange:
            return Prediction("both_cats", 0.99)
        if has_black:
            return Prediction("black_patch_cat", 0.99)
        if has_orange:
            return Prediction("orange_patch_cat", 0.99)
        return Prediction("empty", 0.99)


# --- The fixtures themselves --------------------------------------------------


def test_fixture_files_match_their_manifest() -> None:
    manifest = read_json(FIXTURES / "MANIFEST.json")
    on_disk = {
        path.relative_to(FIXTURES).as_posix()
        for path in FIXTURES.rglob("*")
        if path.is_file() and path.name not in ("MANIFEST.json", "README.md")
    }
    assert on_disk == set(manifest), "files were added to or removed from the fixtures"
    changed = [
        name
        for name, digest in manifest.items()
        if file_sha256(FIXTURES / name) != digest
    ]
    assert changed == [], f"fixtures were modified: {changed}"


def test_fixtures_are_kept_apart_from_the_real_dataset() -> None:
    data_dir = PROJECT_ROOT / "data"
    assert FIXTURES.parent == data_dir
    assert not FIXTURES.is_relative_to(data_dir / "raw")
    assert (FIXTURES / "labels.csv") != data_dir / "labels.csv"


def test_generator_reproduces_the_committed_fixtures(tmp_path: Path) -> None:
    again = tmp_path / "again"
    make_fixtures.generate(again)
    for name in ("labels.csv", "raw/captures.csv", make_fixtures.VIDEO_INFO_NAME):
        assert (again / name).read_bytes() == (FIXTURES / name).read_bytes(), name
    # Pictures are compared after decoding, since JPEG bytes can differ
    # slightly between library versions.
    for still in sorted((FIXTURES / "raw" / "stills").glob("*.jpg")):
        new = cv2.imread(str(again / "raw" / "stills" / still.name)).astype(int)
        old = cv2.imread(str(still)).astype(int)
        assert np.abs(new - old).max() <= 3, still.name
    assert len(read_video_frames(again / make_fixtures.VIDEO_NAME)) == len(
        read_video_frames(VIDEO)
    )


def test_generator_refuses_to_overwrite_an_existing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        make_fixtures.generate(tmp_path)


# --- Still images ------------------------------------------------------------


def fixture_label_rows() -> list[dict[str, str]]:
    return read_csv_rows(FIXTURES / "labels.csv")


def test_every_still_is_logged_labeled_and_decodable() -> None:
    captures = read_capture_log(STILLS_ROOT)
    assert len(captures) == 14
    assert {row["image_path"] for row in captures} == {
        row["image_path"] for row in fixture_label_rows()
    }
    for capture in captures:
        image = cv2.imread(str(STILLS_ROOT / capture["image_path"]))
        assert image is not None and image.shape == (480, 640, 3)


def test_each_stills_pixels_agree_with_its_label() -> None:
    """Guards against a mislabeled or corrupted fixture: the black and orange
    patches must be present exactly when the label says a cat with those
    patches is there."""
    expected = {
        "empty": (False, False),
        "black_patch_cat": (True, False),
        "orange_patch_cat": (False, True),
        "both_cats": (True, True),
        "unusable": (False, False),
    }
    for row in fixture_label_rows():
        image = cv2.imread(str(STILLS_ROOT / row["image_path"]))
        seen = (
            black_pixel_mask(image).sum() > MIN_PATCH_PIXELS,
            orange_pixel_mask(image).sum() > MIN_PATCH_PIXELS,
        )
        assert seen == expected[row["label"]], row["image_path"]


@pytest.fixture
def fixture_config(config: dict[str, Any]) -> dict[str, Any]:
    """The temp config, but reading the fixture stills and their labels.
    Outputs (events, snapshots, new labels) still go to tmp_path."""
    config["paths"]["raw_dir"] = str(STILLS_ROOT)
    config["paths"]["labels_file"] = str(FIXTURES / "labels.csv")
    return config


def test_load_labeled_images_reads_the_fixture_dataset(
    fixture_config: dict[str, Any],
) -> None:
    images = load_labeled_images(fixture_config)
    assert len(images) == 12  # 4 classes x 3 splits; the 2 unusable are left out
    assert all(image.path.is_file() for image in images)
    fractions = fixture_config["training"]["split_fractions"]
    captured_at = {
        row["image_path"]: row["captured_at"] for row in read_capture_log(STILLS_ROOT)
    }
    for image in images:
        relative = image.path.relative_to(STILLS_ROOT).as_posix()
        hour = make_fixtures.group_for(captured_at[relative])
        assert image.split == assign_split(hour, fractions)


def test_every_split_holds_every_class_exactly_once(
    fixture_config: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    # If this fails after editing split_fractions, the fixtures' hours no
    # longer land in their intended splits. Regenerate them into a new folder.
    images = load_labeled_images(fixture_config)
    report_split_counts(images, fixture_config["classes"], ["train", "val", "test"])
    lines = capsys.readouterr().out.splitlines()[1:]
    assert [line.split()[1:] for line in lines] == [["1", "1", "1"]] * 4


def test_the_labeler_reproduces_the_fixture_labels_from_the_fixture_images(
    fixture_config: dict[str, Any],
    tmp_path: Path,
    install_window: Callable[[str], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scripted person labels the real fixture JPEGs, writing to a new
    labels file. The result must equal the fixture's own labels."""
    fixture_config["paths"]["labels_file"] = str(tmp_path / "labels.csv")
    truth = latest_labels(FIXTURES / "labels.csv")
    key_of = {name: key for key, name in fixture_config["label_keys"].items()}
    keys = "".join(
        key_of[truth[row["image_path"]]] for row in read_capture_log(STILLS_ROOT)
    )
    window = install_window(keys)
    monkeypatch.setattr(label, "load_config", lambda: fixture_config)
    monkeypatch.setattr(sys, "argv", ["label", "--labeled-by", "tester"])
    label.main()

    assert len(window.shown) == 14
    assert latest_labels(tmp_path / "labels.csv") == truth


# --- The video ---------------------------------------------------------------


def test_video_matches_its_description_file() -> None:
    frames = read_video_frames(VIDEO)
    assert len(frames) == VIDEO_INFO["frame_count"] == 100
    assert frames[0].shape == (VIDEO_INFO["height"], VIDEO_INFO["width"], 3)
    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.get(cv2.CAP_PROP_FPS) == VIDEO_INFO["fps"]
    capture.release()


def test_video_shows_each_cat_exactly_when_the_description_says() -> None:
    frames = read_video_frames(VIDEO)
    settled = {}
    for visit in VIDEO_INFO["visits"]:
        first, last = visit["settled_first"], visit["settled_last"]
        settled[visit["label"]] = range(first, last + 1)
    for number, frame in enumerate(frames):
        has_black = black_pixel_mask(frame).sum() > MIN_VIDEO_PATCH_PIXELS
        has_orange = orange_pixel_mask(frame).sum() > MIN_VIDEO_PATCH_PIXELS
        if number in settled["black_patch_cat"]:
            assert has_black and not has_orange, number
        if number in settled["orange_patch_cat"]:
            assert has_orange and not has_black, number
    # Before the first visit and after the last, the bowl is empty.
    for number in list(range(0, 24)) + list(range(92, 100)):
        assert not black_pixel_mask(frames[number]).any(), number
        assert not orange_pixel_mask(frames[number]).any(), number


def test_opencv_source_plays_the_whole_video_then_reports_the_end() -> None:
    camera = OpenCVSource(str(VIDEO))
    try:
        for _ in range(VIDEO_INFO["frame_count"]):
            assert camera.read().shape == (240, 320, 3)
        with pytest.raises(RuntimeError, match="No frame returned"):
            camera.read()
    finally:
        camera.close()


def test_motion_detector_fires_when_cats_arrive_and_leave_not_while_they_sit() -> None:
    moved = set(motion_frames_in_video())
    assert not moved & set(range(1, 25))  # empty bowl, nothing changes
    assert moved & set(range(25, 36))  # black cat arrives
    assert not moved & set(range(45, 55))  # black cat sits still
    assert moved & set(range(55, 60))  # black cat leaves
    assert moved & set(range(65, 73))  # orange cat arrives
    assert moved & set(range(85, 90))  # orange cat leaves
    assert not moved & set(range(92, 100))  # empty again


# --- Programs run on the video -----------------------------------------------


@pytest.fixture
def video_config(config: dict[str, Any]) -> dict[str, Any]:
    """Temp config whose camera is the fixture video file, exactly as a
    developer would run the programs on a laptop without a camera."""
    config["camera"]["source"] = str(VIDEO)
    return config


def test_collect_saves_a_frame_for_every_motion_frame_and_only_those(
    video_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    video_config["capture"].update(
        {
            "min_seconds_between_motion_saves": 0,  # the video plays instantly
            "periodic_save_every_seconds": 10**9,
            "motion_min_changed_fraction": 0.02,
        }
    )
    monkeypatch.setattr(collect, "load_config", lambda: video_config)
    # collect only stops on Ctrl+C, so the end of the video surfaces as the
    # camera's "no frame" error.
    with pytest.raises(RuntimeError, match="No frame returned"):
        collect.main()

    raw_dir = Path(video_config["paths"]["raw_dir"])
    captures = read_capture_log(raw_dir)
    assert len(captures) == len(motion_frames_in_video())
    assert {row["reason"] for row in captures} == {"motion"}
    saved = cv2.imread(str(raw_dir / captures[0]["image_path"]))
    assert saved.shape == (240, 320, 3)
    assert len(captures) < VIDEO_INFO["frame_count"] / 2  # still frames skipped


def run_monitor_on_video(
    video_config: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mistakes: dict[int, str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Runs `python -m catwatch.monitor` on the fixture video.

    Returns (rows of events.csv, rows of the snapshot capture log).
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_json(
        model_dir / "model_card.json",
        {"class_names": video_config["classes"], "image_size": 32, "bowl_crop": None},
    )
    video_config["monitor"].update(
        {
            "model_dir": str(model_dir),
            "smoothing_window": 3,
            "min_confidence": 0.6,
            "min_event_seconds": 0,  # the video plays instantly, so visits last ~0 s
        }
    )
    classifier = ColorRuleClassifier(mistakes)
    monkeypatch.setattr(monitor, "load_config", lambda: video_config)
    monkeypatch.setattr(monitor, "OnnxImageClassifier", classifier.factory)
    monkeypatch.setattr(
        monitor, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _: None)
    )
    monkeypatch.setattr(sys, "argv", ["monitor"])
    original_handler = monitor.signal.getsignal(monitor.signal.SIGTERM)
    try:
        # monitor only stops on Ctrl+C, so the end of the video surfaces as
        # the camera's "no frame" error. Its `finally` block still runs.
        with pytest.raises(RuntimeError, match="No frame returned"):
            monitor.main()
    finally:
        monitor.signal.signal(monitor.signal.SIGTERM, original_handler)

    with Path(video_config["paths"]["events_file"]).open(newline="") as events_file:
        events = list(csv.DictReader(events_file))
    return events, read_capture_log(Path(video_config["paths"]["raw_dir"]))


def test_monitor_logs_one_event_per_cat_visit_in_the_video(
    video_config: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, snapshots = run_monitor_on_video(video_config, tmp_path, monkeypatch)
    expected = [visit["label"] for visit in VIDEO_INFO["visits"]]
    assert [event["cat"] for event in events] == expected
    assert [row["reason"] for row in snapshots] == [
        f"event-{name}" for name in expected
    ]

    # Each snapshot was taken as its visit began, so it must show that cat.
    raw_dir = Path(video_config["paths"]["raw_dir"])
    first, second = (cv2.imread(str(raw_dir / row["image_path"])) for row in snapshots)
    assert black_pixel_mask(first).sum() > MIN_VIDEO_PATCH_PIXELS
    assert orange_pixel_mask(second).sum() > MIN_VIDEO_PATCH_PIXELS


def test_monitor_smoothing_absorbs_one_frame_misreads_in_the_video(
    video_config: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A misread in the middle of each visit, and a false "both cats" alarm on
    # the empty bowl. Majority voting over 3 frames must ignore all of them.
    mistakes = {
        40: "empty",
        75: "black_patch_cat",
        12: "both_cats",
        95: "orange_patch_cat",
    }
    events, snapshots = run_monitor_on_video(
        video_config, tmp_path, monkeypatch, mistakes
    )
    expected = [visit["label"] for visit in VIDEO_INFO["visits"]]
    assert [event["cat"] for event in events] == expected
    assert len(snapshots) == 2


def test_check_crop_on_the_video_saves_the_frame_and_the_cropped_region(
    video_config: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_config["bowl_crop"] = {"x": 20, "y": 10, "width": 120, "height": 100}
    monkeypatch.setattr(check_crop, "load_config", lambda: video_config)
    monkeypatch.setattr(
        check_crop, "project_path", lambda relative: tmp_path / relative
    )
    check_crop.main()
    full = cv2.imread(str(tmp_path / "data" / "crop_check_full.jpg"))
    crop = cv2.imread(str(tmp_path / "data" / "crop_check_crop.jpg"))
    assert full.shape == (240, 320, 3)
    assert crop.shape == (100, 120, 3)
