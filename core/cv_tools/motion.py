# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Cheap motion detection, used to decide when a frame is worth saving."""

from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    """Reports whether a noticeable part of the image changed.

    OpenCV's MOG2 background subtractor keeps a statistical model of what each
    pixel normally looks like and marks pixels that deviate from it as
    "foreground". Frames are shrunk first, because motion doesn't need detail
    and small images are much faster on a Raspberry Pi. It counts as motion
    when the foreground fraction of the image is at least min_changed_fraction
    (0.02 = 2% of pixels).

    Something that stops moving slowly fades into the background model, over
    roughly `history` frames. That is fine for deciding when to save pictures.
    """

    def __init__(
        self,
        min_changed_fraction: float = 0.02,
        history: int = 500,
        work_width: int = 160,
    ) -> None:
        self._min_changed_fraction = min_changed_fraction
        self._work_width = work_width
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, detectShadows=False
        )

    def update(self, frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        work_height = round(height * self._work_width / width)
        small = cv2.resize(frame, (self._work_width, work_height))
        foreground_mask = self._subtractor.apply(small)
        changed_fraction = np.count_nonzero(foreground_mask) \
            / foreground_mask.size
        return changed_fraction >= self._min_changed_fraction
