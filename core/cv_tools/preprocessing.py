# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Turning image files into the exact numbers a network expects.

to_model_input() is shared by evaluation and deployed inference, so the model
is tested on precisely the same preprocessing it will see in production.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Per-channel average and spread of the ImageNet photo collection that the
# pretrained network originally learned from. Inputs must be normalized the
# same way or the pretrained features stop working well.
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

Box = tuple[int, int, int, int]  # (x, y, width, height) in pixels


def crop_to_box(image: np.ndarray, box: Box) -> np.ndarray:
    x, y, width, height = box
    cropped = image[y: y + height, x: x + width]
    if cropped.size == 0:
        raise ValueError(f"Crop box {box} is outside image of shape " +
                         "{image.shape}")
    return cropped


def load_image(path: Path, crop_box: Box | None = None) -> np.ndarray:
    """Read an image file as BGR and optionally crop it."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image if crop_box is None else crop_to_box(image, crop_box)


def to_model_input(bgr_image: np.ndarray, size: int) -> np.ndarray:
    """Resize, convert to RGB, scale to 0-1, and normalize.

    Returns float32 of shape (3, size, size): channels first, the layout
    PyTorch and ONNX models use.
    """
    resized = cv2.resize(bgr_image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1),
                                dtype=np.float32)
