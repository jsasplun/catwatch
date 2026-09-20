# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Running a trained classifier over a dataset and summarizing how it did."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_classes(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[list[int], list[int]]:
    """Return (true class indices, predicted class indices) in loader order."""
    model.eval()
    true_classes: list[int] = []
    predicted_classes: list[int] = []
    for images, targets in loader:
        logits = model(images.to(device))
        predicted_classes.extend(logits.argmax(dim=1).cpu().tolist())
        true_classes.extend(targets.tolist())
    return true_classes, predicted_classes


def classification_summary(
    true_classes: Sequence[int],
    predicted_classes: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Per-class precision, recall, F1, and the confusion matrix, JSON-ready.

    precision: of the frames the model called class X, the fraction truly X.
    recall:    of the frames that truly were X, the fraction the model caught.
    F1:        one number balancing the two (their harmonic mean).
    confusion matrix: row i, column j counts frames of true class i that were
    predicted as class j. Everything off the diagonal is a mistake.
    """
    labels: list[int] = list(range(len(class_names)))
    report = classification_report(
        true_classes,
        predicted_classes,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_classes, predicted_classes, labels=labels)
    return {
        "class_names": list(class_names),
        "report": report,
        "confusion_matrix": matrix.tolist(),
    }
