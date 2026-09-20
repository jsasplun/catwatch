# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Running an exported classifier with ONNX Runtime. No PyTorch needed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from core.preprocessing import to_model_input


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float  # softmax probability of `label`, between 0 and 1


class OnnxImageClassifier:
    """Classifies single images using a model file exported to ONNX.

    ONNX is a portable file format for trained networks. ONNX Runtime executes
    it efficiently on a Raspberry Pi CPU, where installing PyTorch would be
    large and slow.

    Treat `confidence` as a rough signal, not a guaranteed probability: neural
    networks are often overconfident.
    """

    def __init__(
        self, model_path: Path, class_names: Sequence[str], image_size: int
    ) -> None:
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._class_names = list(class_names)
        self._image_size = image_size

    def predict(self, bgr_image: np.ndarray) -> Prediction:
        batch = to_model_input(bgr_image, self._image_size)[np.newaxis]
        logits = self._session.run(None, {self._input_name: batch})[0][0]
        # Softmax converts raw scores into probabilities that sum to 1. It's
        # written out here because the usual library for it (SciPy) is a large
        # install on the Pi for a two-line formula. Subtracting the max first
        # prevents overflow in exp() without changing the result.
        exponentials = np.exp(logits - logits.max())
        probabilities = exponentials / exponentials.sum()
        best_index = int(np.argmax(probabilities))
        return Prediction(
            self._class_names[best_index],
            float(probabilities[best_index])
        )
