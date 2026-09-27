# Author: John Asplund
# Date Created: 9/27/26
# AI tool: Claude Sonnet 5

"""Tests for core.cv_tools.camera: correcting a camera's mounting angle."""

from __future__ import annotations

import sys
from pathlib import Path

# Adds the parent directory of this file to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from core.cv_tools.camera import RotatedSource, rotate_frame
from tests.fakes import FakeCamera, make_frame


def test_zero_degrees_returns_the_frame_unchanged() -> None:
    frame = make_frame(64, 48, (10, 20, 30))
    assert np.array_equal(rotate_frame(frame, 0), frame)


def test_180_degrees_flips_a_non_symmetric_frame() -> None:
    # A frame with a bright corner top-left should have that same bright
    # corner bottom-right after a 180 degree turn.
    frame = make_frame(4, 4, (0, 0, 0))
    frame[0, 0] = (255, 255, 255)
    rotated = rotate_frame(frame, 180)
    assert tuple(rotated[3, 3]) == (255, 255, 255)
    assert tuple(rotated[0, 0]) == (0, 0, 0)


def test_90_and_270_degrees_swap_width_and_height() -> None:
    frame = make_frame(64, 48)  # width 64, height 48
    assert rotate_frame(frame, 90).shape == (64, 48, 3)
    assert rotate_frame(frame, 270).shape == (64, 48, 3)


def test_invalid_rotation_is_rejected() -> None:
    with pytest.raises(ValueError, match="0, 90, 180, or 270"):
        rotate_frame(make_frame(), 45)


def test_rotated_source_rotates_every_frame_it_reads() -> None:
    frame = make_frame(4, 4, (0, 0, 0))
    frame[0, 0] = (255, 255, 255)
    camera = FakeCamera([frame])
    source = RotatedSource(camera, 180)
    rotated = source.read()
    assert tuple(rotated[3, 3]) == (255, 255, 255)


def test_rotated_source_closes_the_wrapped_source() -> None:
    camera = FakeCamera([])
    RotatedSource(camera, 180).close()
    assert camera.closed
