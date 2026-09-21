# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Camera access behind one small interface.

Anything that can hand back "the latest image" counts as a frame source.
Keeping the rest of the code behind this interface means the same scripts run
on a Raspberry Pi camera, a USB webcam, or a recorded video file on a laptop.

Every frame is a NumPy array shaped (height, width, 3) with color channels in
BGR order (blue, green, red), which is the order OpenCV expects.

Use sources with contextlib.closing so the camera is released even on errors:

    with closing(OpenCVSource(0)) as camera:
        frame = camera.read()
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    """Anything with read() -> image and close()."""

    def read(self) -> np.ndarray: ...

    def close(self) -> None: ...


class OpenCVSource:
    """Frames from a USB webcam (pass an integer index) or a video file (a
    path)."""

    def __init__(self, device: int | str = 0) -> None:
        self._capture = cv2.VideoCapture(device)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video source {device!r}")

    def read(self) -> np.ndarray:
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("No frame returned: end of video or camera" +
                               "unplugged")
        return frame

    def close(self) -> None:
        self._capture.release()


class Picamera2Source:
    """Frames from a Raspberry Pi ribbon-cable camera via the Picamera2
    library.

    lens_position only matters for autofocus cameras such as Camera Module 3.
    Autofocus tends to "hunt" when an animal walks into view, which blurs
    frames and makes pictures inconsistent. Passing a number switches to fixed
    manual focus. The unit is dioptres, which is 1 divided by the distance in
    metres: an object 0.5 m below the lens is 2.0. Leave it as None to keep
    autofocus.
    """

    def __init__(
        self, width: int, height: int, lens_position: float | None = None
    ) -> None:
        # Imported here instead of at the top of the file because these
        # libraries only exist on a Raspberry Pi. Importing lazily lets this
        # module load on a laptop for testing.
        from libcamera import controls  # pyright: ignore[reportMissingImports]

        # pyright: ignore[reportMissingImports]
        from picamera2 import Picamera2

        self._camera = Picamera2()
        # "RGB888" is libcamera's name for a layout whose bytes are stored
        # blue, green, red. That is exactly OpenCV's BGR order, so frames can
        # be used directly without converting colors.
        video_config = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._camera.configure(video_config)
        self._camera.start()
        if lens_position is not None:
            self._camera.set_controls(
                {
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": lens_position
                }
            )

    def read(self) -> np.ndarray:
        return self._camera.capture_array()

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()
