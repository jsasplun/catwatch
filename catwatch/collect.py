# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Runs on the Raspberry Pi to collect training images.

Saves a frame whenever motion is detected (at most one every
min_seconds_between_motion_saves), plus one frame every
periodic_save_every_seconds no matter what. The periodic frames give the
dataset empty-bowl pictures across all lighting conditions. Stop with Ctrl+C.

    python -m catwatch.collect
"""

from __future__ import annotations

import time
from contextlib import closing

from catwatch.hardware import open_camera
from catwatch.settings import load_config, project_path
from core.capture_store import CaptureWriter
from core.motion import MotionDetector


def main() -> None:
    config = load_config()
    capture = config["capture"]
    writer = CaptureWriter(project_path(config["paths"]["raw_dir"]))
    detector = MotionDetector(min_changed_fraction=capture[
        "motion_min_changed_fraction"
    ])
    # time.monotonic() measures elapsed time and never jumps backwards, unlike
    # the wall clock, which can jump when the Pi syncs its time over the
    # network.
    last_motion_save = last_periodic_save = time.monotonic()
    saved_count = 0

    print("Collecting images. Press Ctrl+C to stop.")
    with closing(open_camera(config)) as camera:
        try:
            while True:
                frame = camera.read()
                now = time.monotonic()
                moved = detector.update(frame)
                since_motion_save = now - last_motion_save
                if moved and since_motion_save >= capture[
                    "min_seconds_between_motion_saves"
                ]:
                    writer.save(frame, "motion")
                    last_motion_save = now
                    saved_count += 1
                elif now - last_periodic_save >= capture[
                    "periodic_save_every_seconds"
                ]:
                    writer.save(frame, "periodic")
                    last_periodic_save = now
                    saved_count += 1
        except KeyboardInterrupt:
            print(f"\nStopped. Saved {saved_count} images this session.")


if __name__ == "__main__":
    main()
