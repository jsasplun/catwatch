# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

from datetime import datetime, timedelta
import sys
from pathlib import Path
# Adds the parent directory of this file to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cv_tools.events import EventTracker

START = datetime(2026, 9, 19, 12, 0, 0)


def feed(tracker: EventTracker, labels: list[str]) -> list:
    events = []
    for second, label in enumerate(labels):
        event = tracker.update(label, START + timedelta(seconds=second))
        if event is not None:
            events.append(event)
    return events


def test_sustained_visit_becomes_one_event() -> None:
    tracker = EventTracker("empty", window_size=3, min_duration_seconds=3)
    events = feed(tracker, ["empty"] * 5 + ["cat"] * 10 + ["empty"] * 5)
    assert len(events) == 1
    assert events[0].label == "cat"
    assert events[0].duration_seconds == 10


def test_single_flicker_is_ignored() -> None:
    tracker = EventTracker("empty", window_size=3, min_duration_seconds=3)
    assert feed(tracker, ["empty"] * 5 + ["cat"] + ["empty"] * 5) == []


def test_finish_closes_event_in_progress() -> None:
    tracker = EventTracker("empty", window_size=3, min_duration_seconds=3)
    feed(tracker, ["cat"] * 10)
    event = tracker.finish(START + timedelta(seconds=10))
    assert event is not None and event.label == "cat"
