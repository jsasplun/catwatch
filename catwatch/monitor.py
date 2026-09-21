# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Runs on the Raspberry Pi: watches the bowl and logs which cat visits.

    python -m catwatch.monitor           # headless (for the systemd service)
    python -m catwatch.monitor --show    # also draw predictions on the screen

Each visit is appended to data/events.csv. The frame at the start of each visit
is saved into the raw data folder (reason "event-<label>"), so you can check
the model's calls by eye and later label those frames as new training data.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from catwatch.hardware import open_camera
from catwatch.settings import bowl_crop_box, load_config, project_path
from core.cv_tools.capture_store import CaptureWriter
from core.cv_tools.csv_log import append_csv_row
from core.cv_tools.events import Event, EventTracker
from core.cv_tools.onnx_classifier import OnnxImageClassifier, Prediction
from core.cv_tools.preprocessing import Box, crop_to_box
from core.cv_tools.run_records import read_json

EVENT_FIELDS = (
    "cat",
    "display_name",
    "started_at",
    "ended_at",
    "duration_seconds"
)


def log_event(
        events_file: Path,
        event: Event,
        display_names: dict[str, str]
) -> None:
    name = display_names.get(event.label, event.label)
    append_csv_row(
        events_file,
        EVENT_FIELDS,
        {
            "cat": event.label,
            "display_name": name,
            "started_at": event.started_at.isoformat(),
            "ended_at": event.ended_at.isoformat(),
            "duration_seconds": round(event.duration_seconds, 1),
        },
    )
    print(f"{event.started_at:%H:%M:%S}  {name} at bowl for" +
          f" {event.duration_seconds:.0f}s")


def show_prediction(
        frame: np.ndarray,
        prediction: Prediction,
        crop: Box | None
) -> None:
    shown = frame.copy()
    if crop is not None:
        x, y, width, height = crop
        cv2.rectangle(shown, (x, y), (x + width, y + height), (0, 255, 255), 3)
    text = f"{prediction.label} {prediction.confidence:.2f}"
    cv2.putText(
        shown,
        text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        6
    )
    cv2.putText(
        shown,
        text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2
    )
    cv2.imshow(
        "catwatch",
        cv2.resize(shown, (800, 800 * frame.shape[0] // frame.shape[1]))
    )
    cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live predictions."
    )
    args = parser.parse_args()

    config = load_config()
    settings = config["monitor"]
    model_dir = project_path(settings["model_dir"])
    model_card = read_json(model_dir / "model_card.json")
    classifier = OnnxImageClassifier(
        model_dir / "model.onnx",
        model_card["class_names"],
        model_card["image_size"]
    )
    crop = bowl_crop_box(model_card)  # the crop this model was trained with
    tracker = EventTracker(
        background_label=settings["background_label"],
        window_size=settings["smoothing_window"],
        min_duration_seconds=settings["min_event_seconds"],
    )
    snapshot_writer = CaptureWriter(project_path(config["paths"]["raw_dir"]))
    events_file = project_path(config["paths"]["events_file"])
    display_names = config["display_names"]
    seconds_per_prediction = 1 / settings["predictions_per_second"]

    # systemd stops services with SIGTERM. Turning it into a normal exit lets
    # the `finally` block record a visit that is still in progress.
    signal.signal(signal.SIGTERM, lambda _signum, _stack: sys.exit(0))

    print(f"Monitoring with model {model_dir.name}. Ctrl+C to stop.")
    with closing(open_camera(config)) as camera:
        try:
            while True:
                loop_started = time.monotonic()
                frame = camera.read()
                region = frame if crop is None else crop_to_box(frame, crop)
                prediction = classifier.predict(region)
                if prediction.confidence >= settings["min_confidence"]:
                    previously_active = tracker.active_label
                    finished = tracker.update(
                        prediction.label,
                        datetime.now().astimezone()
                    )
                    if finished is not None:
                        log_event(events_file, finished, display_names)
                    if tracker.active_label not in (None, previously_active):
                        snapshot_writer.save(
                            frame,
                            f"event-{tracker.active_label}"
                        )
                if args.show:
                    show_prediction(frame, prediction, crop)
                elapsed = time.monotonic() - loop_started
                time.sleep(max(0.0, seconds_per_prediction - elapsed))
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            final_event = tracker.finish(datetime.now().astimezone())
            if final_event is not None:
                log_event(events_file, final_event, display_names)


if __name__ == "__main__":
    main()
