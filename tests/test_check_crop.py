# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Tests for catwatch.check_crop: the helper that saves a picture showing where
the bowl crop sits, so the user can tune config.yaml."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import cv2
import pytest

from catwatch import check_crop
from tests.fakes import FakeCamera, make_frame


@pytest.fixture
def run_check_crop(
    config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """run(frame) runs the command against a camera that always shows `frame`
    and returns (camera, folder the images were written to)."""
    output_dir = tmp_path / "data"
    monkeypatch.setattr(check_crop, "load_config", lambda: config)
    monkeypatch.setattr(
        check_crop, "project_path", lambda relative: tmp_path / relative
    )

    def run(frame: Any) -> tuple[FakeCamera, Path]:
        camera = FakeCamera(itertools.repeat(frame))
        monkeypatch.setattr(check_crop, "open_camera", lambda _config: camera)
        check_crop.main()
        return camera, output_dir

    return run


def test_with_no_crop_only_the_full_frame_image_is_saved(
    run_check_crop: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    camera, output_dir = run_check_crop(make_frame(64, 48, (30, 60, 90)))
    assert (output_dir / "crop_check_full.jpg").is_file()
    assert not (output_dir / "crop_check_crop.jpg").exists()
    assert "Frame size: 64 x 48 pixels" in capsys.readouterr().out
    assert camera.closed


def test_with_a_crop_the_crop_and_an_outlined_full_frame_are_saved(
    config: dict[str, Any], run_check_crop: Any
) -> None:
    config["bowl_crop"] = {"x": 20, "y": 10, "width": 30, "height": 25}
    _, output_dir = run_check_crop(make_frame(100, 80, (30, 60, 90)))

    crop = cv2.imread(str(output_dir / "crop_check_crop.jpg"))
    assert crop.shape == (25, 30, 3)  # height 25, width 30

    full = cv2.imread(str(output_dir / "crop_check_full.jpg"))
    assert full.shape == (80, 100, 3)
    # Outline: the box's left edge is the column x=20. Yellow survives JPEG
    # compression only approximately, so compare with a tolerance.
    blue, green, red = (int(v) for v in full[22, 20])
    assert blue < 90 and green > 200 and red > 200


def test_camera_warms_up_before_the_frame_is_taken(
    run_check_crop: Any,
) -> None:
    # Auto-exposure needs about 30 frames to settle, so the saved picture is
    # not the very first (often too dark or too bright) frame.
    camera, _ = run_check_crop(make_frame())
    assert camera.reads == 1 + check_crop.WARMUP_FRAMES


def test_output_folder_is_created_when_missing(
    tmp_path: Path, run_check_crop: Any
) -> None:
    assert not (tmp_path / "data").exists()
    run_check_crop(make_frame())
    assert (tmp_path / "data").is_dir()
