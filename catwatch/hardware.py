# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Opens whichever camera config.yaml asks for."""

from __future__ import annotations

from typing import Any

from core.camera import FrameSource, OpenCVSource, Picamera2Source


def open_camera(config: dict[str, Any]) -> FrameSource:
    camera = config["camera"]
    if camera["source"] == "picamera2":
        return Picamera2Source(
            camera["width"],
            camera["height"],
            camera["lens_position"]
        )
    return OpenCVSource(camera["source"])  # webcam index or video file path
