# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Tests for catwatch.collect: the Pi program that saves training images.

Time and the motion detector are scripted, so the test decides exactly when
motion "happens" and how many seconds pass between frames.
"""

from __future__ import annotations
import sys
from pathlib import Path
# Adds the parent directory of this file to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from catwatch import collect
from core.cv_tools.capture_store import read_capture_log
from core.cv_tools.motion import MotionDetector
from tests.fakes import FakeCamera, make_frame


class ScriptedDetector:
    """Says "motion" or "no motion" from a prepared list, one answer per
    frame."""

    def __init__(self, answers: list[bool]) -> None:
        self._answers: Iterator[bool] = iter(answers)

    def update(self, _frame: np.ndarray) -> bool:
        return next(self._answers)


def run_collect(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    clock_readings: list[float],
    motion_answers: list[bool],
) -> FakeCamera:
    """Runs collect.main() over len(motion_answers) frames.

    clock_readings[0] is the moment collect starts. The rest are the times at
    which each frame is read, so it must be one longer than motion_answers.
    """
    assert len(clock_readings) == len(motion_answers) + 1
    readings = iter(clock_readings)
    monkeypatch.setattr(
        collect, "time", SimpleNamespace(monotonic=lambda: next(readings))
    )
    monkeypatch.setattr(collect, "load_config", lambda: config)
    monkeypatch.setattr(
        collect, "MotionDetector", lambda **_kwargs: ScriptedDetector(motion_answers)
    )
    # Distinct colors let the test confirm the right frame reached the disk.
    frames = [
        make_frame(64, 48, (index * 20, 0, 0)) for index in range(len(motion_answers))
    ]
    camera = FakeCamera(frames)
    monkeypatch.setattr(collect, "open_camera", lambda _config: camera)
    collect.main()
    return camera


@pytest.fixture
def capture_config(config: dict[str, Any]) -> dict[str, Any]:
    config["capture"].update(
        {
            "min_seconds_between_motion_saves": 2,
            "periodic_save_every_seconds": 10,
            "motion_min_changed_fraction": 0.02,
        }
    )
    return config


def saved_reasons(config: dict[str, Any]) -> list[str]:
    raw_dir = Path(config["paths"]["raw_dir"])
    if not (raw_dir / "captures.csv").exists():
        return []
    return [row["reason"] for row in read_capture_log(raw_dir)]


def test_motion_saves_are_spaced_at_least_the_minimum_seconds_apart(
    monkeypatch: pytest.MonkeyPatch,
    capture_config: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Motion on every frame, one second apart. With a 2 s minimum gap only
    # every second frame may be saved (t=2 and t=4, not t=1 or t=3).
    run_collect(
        monkeypatch,
        capture_config,
        clock_readings=[0, 1, 2, 3, 4],
        motion_answers=[True, True, True, True],
    )
    assert saved_reasons(capture_config) == ["motion", "motion"]
    assert "Saved 2 images this session." in capsys.readouterr().out


def test_a_periodic_frame_is_saved_when_nothing_moves(
    monkeypatch: pytest.MonkeyPatch, capture_config: dict[str, Any]
) -> None:
    run_collect(
        monkeypatch,
        capture_config,
        clock_readings=[0, 5, 9, 10, 11],
        motion_answers=[False, False, False, False],
    )
    # Only the reading at t=10 reaches the 10 s period; t=11 is 1 s later.
    assert saved_reasons(capture_config) == ["periodic"]


def test_motion_and_periodic_saves_mix(
    monkeypatch: pytest.MonkeyPatch, capture_config: dict[str, Any]
) -> None:
    run_collect(
        monkeypatch,
        capture_config,
        clock_readings=[0, 1, 2, 3, 4, 12, 13],
        motion_answers=[True, True, True, True, False, False],
    )
    assert saved_reasons(capture_config) == ["motion", "motion", "periodic"]


def test_no_motion_and_no_time_passing_saves_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capture_config: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_collect(
        monkeypatch,
        capture_config,
        clock_readings=[0, 1, 2, 3],
        motion_answers=[False, False, False],
    )
    assert saved_reasons(capture_config) == []
    assert "Saved 0 images this session." in capsys.readouterr().out


def test_saved_files_are_real_jpegs_of_the_camera_frame(
    monkeypatch: pytest.MonkeyPatch, capture_config: dict[str, Any]
) -> None:
    run_collect(
        monkeypatch,
        capture_config,
        clock_readings=[0, 2],
        motion_answers=[True],
    )
    raw_dir = Path(capture_config["paths"]["raw_dir"])
    (capture,) = read_capture_log(raw_dir)
    image = cv2.imread(str(raw_dir / capture["image_path"]))
    assert image is not None and image.shape == (48, 64, 3)


def test_the_camera_is_released_when_collection_stops(
    monkeypatch: pytest.MonkeyPatch, capture_config: dict[str, Any]
) -> None:
    camera = run_collect(
        monkeypatch, capture_config, clock_readings=[0, 1], motion_answers=[False]
    )
    assert camera.closed


def test_the_motion_threshold_comes_from_config(
    monkeypatch: pytest.MonkeyPatch, capture_config: dict[str, Any]
) -> None:
    capture_config["capture"]["motion_min_changed_fraction"] = 0.07
    received: dict[str, Any] = {}

    def make_detector(**kwargs: Any) -> ScriptedDetector:
        received.update(kwargs)
        return ScriptedDetector([False])

    monkeypatch.setattr(
        collect, "time", SimpleNamespace(monotonic=iter([0, 1]).__next__)
    )
    monkeypatch.setattr(collect, "load_config", lambda: capture_config)
    monkeypatch.setattr(collect, "MotionDetector", make_detector)
    monkeypatch.setattr(
        collect, "open_camera", lambda _config: FakeCamera([make_frame()])
    )
    collect.main()
    assert received == {"min_changed_fraction": 0.07}


def test_the_real_motion_detector_ignores_a_still_scene_and_sees_a_change() -> None:
    """Not a scripted fake: a quick check that MotionDetector behaves as the
    collector assumes, using synthetic frames."""
    detector = MotionDetector(min_changed_fraction=0.02)
    still_scene = make_frame(320, 240, (100, 100, 100))
    for _ in range(30):  # let the background model learn the still scene
        detector.update(still_scene)
    assert not detector.update(still_scene)

    with_visitor = still_scene.copy()
    with_visitor[60:180, 80:240] = (240, 240, 240)  # a big bright object appears
    assert detector.update(with_visitor)
