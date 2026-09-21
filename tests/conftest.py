# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Shared fixtures: a fake OpenCV window, and a config whose folders all point
into a temporary directory.

None of the tests may touch the real data/ folder. Raw images and labels are
irreplaceable, so every test works on copies in pytest's tmp_path instead.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from catwatch.settings import CONFIG_PATH
from core.cv_tools.capture_store import CAPTURE_FIELDS, CAPTURE_LOG_NAME
from core.cv_tools.csv_log import append_csv_row
from tests.fakes import FakeWindow, make_frame


@pytest.fixture
def install_window(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], FakeWindow]:
    """Call install_window("12q") to script the keys a person will press."""

    def install(keys: str) -> FakeWindow:
        window = FakeWindow(keys)
        monkeypatch.setattr(cv2, "imshow", window.imshow)
        monkeypatch.setattr(cv2, "waitKey", window.wait_key)
        monkeypatch.setattr(cv2, "destroyAllWindows", window.destroy_all_windows)
        return window

    return install


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    """The real config.yaml, but with every folder moved into tmp_path.

    Starting from the real file keeps tests honest about the real class names
    and label keys. project_path() joins PROJECT_ROOT with a path, and joining
    with an absolute path just yields that absolute path, so absolute temp
    paths here redirect all reads and writes without patching anything.
    """
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        loaded: dict[str, Any] = yaml.safe_load(config_file)
    loaded["paths"] = {
        "raw_dir": str(tmp_path / "raw"),
        "labels_file": str(tmp_path / "labels.csv"),
        "events_file": str(tmp_path / "events.csv"),
        "runs_dir": str(tmp_path / "runs"),
        "models_dir": str(tmp_path / "models"),
    }
    loaded["bowl_crop"] = None
    return loaded


@pytest.fixture
def add_capture() -> Callable[..., str]:
    """Pretend the camera saved an image: writes the JPEG and its
    captures.csv row, with a timestamp of the test's choosing.

    Returns the image path relative to raw_dir, the form labels.csv uses.
    """

    def add(
        raw_dir: Path,
        relative_path: str,
        captured_at: str = "2026-09-19T14:00:00-05:00",
        reason: str = "motion",
        frame: np.ndarray | None = None,
    ) -> str:
        full_path = raw_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        image = make_frame(96, 72, (90, 90, 90)) if frame is None else frame
        assert cv2.imwrite(str(full_path), image)
        append_csv_row(
            raw_dir / CAPTURE_LOG_NAME,
            CAPTURE_FIELDS,
            {
                "image_path": relative_path,
                "captured_at": captured_at,
                "reason": reason,
            },
        )
        return relative_path

    return add
