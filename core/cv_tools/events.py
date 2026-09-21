# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Turning a noisy stream of per-frame labels into clean start/stop events."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Event:
    label: str
    started_at: datetime
    ended_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()


class EventTracker:
    """Groups consecutive frames with the same label into one event.

    A classifier that judges single frames will occasionally flicker, for
    example a blurry frame read as the wrong class. Two mechanisms clean that
    up:

    1. Majority vote: the current label is whichever label appears most often
       among the last `window_size` frames, so one odd frame can't flip it.
       Use an odd window size so the vote can't tie.
    2. Minimum duration: events shorter than min_duration_seconds are dropped,
       which filters out brief walk-bys.

    background_label is the label meaning "nothing happening". No events are
    reported for it.
    """

    def __init__(
        self,
        background_label: str,
        window_size: int,
        min_duration_seconds: float
    ) -> None:
        self._background_label = background_label
        self._min_duration_seconds = min_duration_seconds
        self._recent_labels: deque[str] = deque(maxlen=window_size)
        self._active_label: str | None = None
        self._active_since: datetime | None = None

    @property
    def active_label(self) -> str | None:
        """The label of the event in progress, or None if nothing is
        happening."""
        return self._active_label

    def update(self, label: str, timestamp: datetime) -> Event | None:
        """Feed one frame's label. Returns an Event if one just ended, else
        None."""
        self._recent_labels.append(label)
        majority_label = Counter(self._recent_labels).most_common(1)[0][0]
        new_active: Counter | None = None \
            if majority_label == self._background_label \
            else majority_label
        if new_active == self._active_label:
            return None
        finished: Event | None = self.finish(timestamp)
        if new_active is not None:
            self._active_label, self._active_since = new_active, timestamp
        return finished

    def finish(self, timestamp: datetime) -> Event | None:
        """End any event in progress (also call this at shutdown).

        Returns the event if it was long enough to count, otherwise None.
        """
        if self._active_label is None or self._active_since is None:
            return None
        event: Event = Event(self._active_label, self._active_since, timestamp)
        self._active_label, self._active_since = None, None
        if event.duration_seconds < self._min_duration_seconds:
            return None
        return event
