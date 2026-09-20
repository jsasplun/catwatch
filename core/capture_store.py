# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Saving camera frames to disk, with a log describing each one.

Folder layout (root is any directory you choose):

    root/captures.csv                        one row per saved image
    root/2026-09-19/143015_123456_motion.jpg

Saved images are treated as read-only from then on ("raw data is immutable").
Labels, crops, and train/test splits all refer to images by their path relative
to root, so the raw folder stays the single source of truth and every later
step can be redone from scratch.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.csv_log import append_csv_row, read_csv_rows

CAPTURE_LOG_NAME = "captures.csv"
CAPTURE_FIELDS = ("image_path", "captured_at", "reason")


class CaptureWriter:
    """Writes JPEG images under a root folder and logs each in captures.csv."""

    def __init__(self, root: Path, jpeg_quality: int = 95) -> None:
        self._root = root
        self._jpeg_quality = jpeg_quality

    def save(self, frame: np.ndarray, reason: str) -> Path:
        """Save one frame. `reason` is a short tag stored with it, e.g.
        "motion"."""

        # astimezone() attaches the local UTC offset, so timestamps stay
        # unambiguous across machines and daylight-saving changes.
        now = datetime.now().astimezone()
        file_name = f"{now.strftime('%H%M%S_%f')}_{reason}.jpg"
        relative_path = Path(now.strftime("%Y-%m-%d"), file_name)
        full_path = self._root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        quality_setting = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        if not cv2.imwrite(str(full_path), frame, quality_setting):
            raise OSError(f"OpenCV could not write {full_path}")
        append_csv_row(
            self._root / CAPTURE_LOG_NAME,
            CAPTURE_FIELDS,
            {
                "image_path": relative_path.as_posix(),
                "captured_at": now.isoformat(),
                "reason": reason,
            },
        )
        return full_path


def read_capture_log(root: Path) -> list[dict[str, str]]:
    """All rows of root/captures.csv, in the order images were captured."""
    return read_csv_rows(root / CAPTURE_LOG_NAME)
