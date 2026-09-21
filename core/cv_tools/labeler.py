# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""A minimal keyboard labeling tool: one image on screen, one keypress per
label."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.cv_tools.labels import append_label, latest_labels

WINDOW_NAME: str = "labeler"
BACK_KEY: str = "z"
QUIT_KEY: str = "q"


class KeyboardLabeler:
    """Shows unlabeled images one at a time; you press a key to label each.

    key_to_label maps single characters to label names, e.g. {"1": "cat_a"}.
    Press z to go back one image. Labeling it again adds a newer row, which
    wins. Press q to quit. Every keypress is saved immediately, so you can
    quit at any time and resume later; images that already have a label are
    skipped.

    highlight_box (x, y, width, height), if given, is drawn on every image so
    the person labeling judges the same region the model will see.
    """

    def __init__(
        self,
        image_root: Path,
        labels_file: Path,
        key_to_label: dict[str, str],
        labeled_by: str,
        highlight_box: tuple[int, int, int, int] | None = None,
        display_width: int = 960,
    ) -> None:
        reserved = {BACK_KEY, QUIT_KEY} & set(key_to_label)
        if reserved:
            raise ValueError(f"Keys {reserved} are reserved for back/quit")
        self._image_root = image_root
        self._labels_file = labels_file
        self._key_to_label = key_to_label
        self._labeled_by = labeled_by
        self._highlight_box = highlight_box
        self._display_width = display_width

    def run(self, image_paths: list[str]) -> int:
        """Label every image in image_paths that has no label yet.

        Returns how many labels were written this session.
        """
        already_labeled = latest_labels(self._labels_file)
        to_label = [
            path for path in image_paths
            if path not in already_labeled
        ]
        position = 0
        labels_written = 0
        while position < len(to_label):
            image_path = to_label[position]
            image = cv2.imread(str(self._image_root / image_path))
            if image is None:
                raise FileNotFoundError(self._image_root / image_path)
            cv2.imshow(WINDOW_NAME, self._decorate(
                image, position,
                len(to_label)
            ))
            key = chr(cv2.waitKey(0) & 0xFF)
            if key == QUIT_KEY:
                break
            if key == BACK_KEY:
                position = max(0, position - 1)
                continue
            label = self._key_to_label.get(key)
            if label is None:
                continue  # unknown key: ignore it and keep showing this image
            append_label(
                self._labels_file,
                image_path,
                label,
                self._labeled_by
            )
            labels_written += 1
            position += 1
        cv2.destroyAllWindows()
        return labels_written

    def _decorate(
            self,
            image: np.ndarray,
            position: int,
            total: int
    ) -> np.ndarray:
        shown = image.copy()
        if self._highlight_box is not None:
            x, y, width, height = self._highlight_box
            cv2.rectangle(shown, (x, y), (x + width, y + height),
                          (0, 255, 255), 3)
        scale = self._display_width / shown.shape[1]
        shown = cv2.resize(shown, None, fx=scale, fy=scale)
        legend = "  ".join(f"[{k}] {v}" for k, v in self._key_to_label.items())
        status = f"{position + 1}/{total}   [{BACK_KEY}] back " \
            + "  [{QUIT_KEY}] quit"
        _put_outlined_text(shown, status, 30)
        _put_outlined_text(shown, legend, 60)
        return shown


def _put_outlined_text(image: np.ndarray, text: str, y: int) -> None:
    # Black outline under white text keeps it readable on white fur and
    # on dark floors alike.
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (10, y), font, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (10, y), font, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
