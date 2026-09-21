# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Tests for catwatch.hardware: choosing which camera config.yaml asks for."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from catwatch import hardware
from core.cv_tools.camera import OpenCVSource


class RecordingSource:
    """Remembers how it was constructed, instead of touching a camera."""

    def __init__(self, *args: Any) -> None:
        self.args = args


def test_picamera2_source_gets_size_and_lens_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "Picamera2Source", RecordingSource)
    config = {
        "camera": {
            "source": "picamera2",
            "width": 1280,
            "height": 960,
            "lens_position": 2.0,
        }
    }
    camera = hardware.open_camera(config)
    assert isinstance(camera, RecordingSource)
    assert camera.args == (1280, 960, 2.0)


def test_webcam_index_goes_to_the_opencv_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "OpenCVSource", RecordingSource)
    camera = hardware.open_camera({"camera": {"source": 0}})
    assert isinstance(camera, RecordingSource)
    assert camera.args == (0,)


def test_any_other_string_is_treated_as_a_video_file_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "OpenCVSource", RecordingSource)
    camera = hardware.open_camera({"camera": {"source": "clips/bowl.mp4"}})
    assert isinstance(camera, RecordingSource)
    assert camera.args == ("clips/bowl.mp4",)


def test_video_file_source_yields_bgr_frames(tmp_path: Path) -> None:
    """Uses a real (tiny) video file, so this exercises OpenCV for real."""
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter.fourcc(*"MJPG"), 5.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG video")
    for _ in range(3):
        writer.write(np.full((48, 64, 3), (255, 0, 0), dtype=np.uint8))  # blue
    writer.release()

    camera = hardware.open_camera({"camera": {"source": str(video_path)}})
    assert isinstance(camera, OpenCVSource)
    try:
        frame = camera.read()
    finally:
        camera.close()
    assert frame.shape == (48, 64, 3)
    blue, green, red = frame.reshape(-1, 3).mean(axis=0)
    assert blue > 200 and green < 60 and red < 60  # BGR order preserved


def test_missing_video_file_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Could not open video source"):
        hardware.open_camera({"camera": {"source": str(tmp_path / "nope.mp4")}})
