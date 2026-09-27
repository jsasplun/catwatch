# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Opens whichever camera config.yaml asks for."""

from __future__ import annotations

from typing import Any

from core.cv_tools.camera import (
    FrameSource,
    OpenCVSource,
    Picamera2Source,
    RotatedSource,
)


def open_camera(config: dict[str, Any]) -> FrameSource:
    camera = config["camera"]
    if camera["source"] == "picamera2":
        source: FrameSource = Picamera2Source(
            camera["width"], camera["height"], camera["lens_position"]
        )
    else:
        source = OpenCVSource(camera["source"])  # webcam index or video file path

    # Older config.yaml files (and tests) may not have this key at all, which
    # means "mounted right side up": no rotation.
    degrees = camera.get("rotate_degrees", 0)
    return source if degrees == 0 else RotatedSource(source, degrees)
