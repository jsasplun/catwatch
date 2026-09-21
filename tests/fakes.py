# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Stand-ins for hardware and windows, so tests run without a camera, a
screen, or a person."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np


def make_frame(
    width: int = 64, height: int = 48, color: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """A solid-color image shaped like a camera frame: (height, width, 3),
    channels in BGR order (blue, green, red), values 0-255."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class FakeCamera:
    """Stands in for a real camera.

    The programs under test loop forever until a person presses Ctrl+C. A test
    can't press keys, so this camera does it: after it has handed out all of
    its frames, the next read() raises `end_with` (KeyboardInterrupt by
    default), which is exactly what Ctrl+C does inside a running program.
    """

    def __init__(
        self,
        frames: Iterable[np.ndarray],
        end_with: BaseException | None = None,
    ) -> None:
        self._frames: Iterator[np.ndarray] = iter(frames)
        self._end_with = KeyboardInterrupt() if end_with is None else end_with
        self.reads = 0
        self.closed = False

    def read(self) -> np.ndarray:
        try:
            frame = next(self._frames)
        except StopIteration:
            raise self._end_with from None
        self.reads += 1
        return frame

    def close(self) -> None:
        self.closed = True


class FakeWindow:
    """Replaces OpenCV's window functions so no window ever opens.

    `keys` is the sequence of keypresses the pretend person types, one
    character per waitKey() call. Every image handed to imshow() is kept in
    `shown`, so tests can inspect what would have been on screen.
    """

    def __init__(self, keys: str) -> None:
        self._keys = iter(keys)
        self.shown: list[np.ndarray] = []
        self.windows_destroyed = False

    def imshow(self, _window_name: str, image: np.ndarray) -> None:
        self.shown.append(image)

    def wait_key(self, _delay_ms: int = 0) -> int:
        try:
            return ord(next(self._keys))
        except StopIteration:
            raise AssertionError(
                "The program asked for more keypresses than the test scripted"
            ) from None

    def destroy_all_windows(self) -> None:
        self.windows_destroyed = True
