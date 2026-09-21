# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Saves one camera frame with the configured crop drawn on it, for tuning
bowl_crop in config.yaml.

    python -m catwatch.check_crop

Writes data/crop_check_full.jpg and, if a crop is set,
data/crop_check_crop.jpg. Open both and look. The crop should contain the bowl
AND enough area to show the back of a cat that is drinking, because coat
patches are often on the back.
"""

from __future__ import annotations

from contextlib import closing

import cv2

from catwatch.hardware import open_camera
from catwatch.settings import bowl_crop_box, load_config, project_path
from core.cv_tools.preprocessing import crop_to_box

WARMUP_FRAMES = 30  # let auto-exposure and white balance settle first


def main() -> None:
    config = load_config()
    with closing(open_camera(config)) as camera:
        frame = camera.read()
        for _ in range(WARMUP_FRAMES):
            frame = camera.read()

    height, width = frame.shape[:2]
    print(
        f"Frame size: {width} x {height} pixels (x grows right, y grows down)"
    )
    output_dir = project_path("data")
    output_dir.mkdir(exist_ok=True)
    box = bowl_crop_box(config)
    annotated = frame.copy()
    if box is not None:
        x, y, box_width, box_height = box
        cv2.rectangle(
            annotated,
            (x, y),
            (x + box_width, y + box_height),
            (0, 255, 255),
            3
        )
        cv2.imwrite(
            str(output_dir / "crop_check_crop.jpg"),
            crop_to_box(frame, box)
        )
    cv2.imwrite(str(output_dir / "crop_check_full.jpg"), annotated)
    print(f"Saved check images to {output_dir}")


if __name__ == "__main__":
    main()
