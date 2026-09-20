# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Fine-tuning a small pretrained image classifier."""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

from core.evaluation import predict_classes
from core.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Box,
    load_image,
    to_model_input,
)


def set_seed(seed: int) -> None:
    """Make random choices repeatable across Python, NumPy, and PyTorch.

    GPU arithmetic can still differ very slightly between runs, which is one
    reason model comparisons should use several seeds, not one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ImageClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    """(image tensor, class index) pairs for PyTorch training and evaluation.

    items: list of (image file path, class index).
    crop_box: region cut out of every image first, or None for the full image.
    augment: random transform for training images. When None (evaluation), the
    image goes through to_model_input(), the same function deployed inference
    uses, so evaluation measures what deployment will actually see.
    """

    def __init__(
        self,
        items: Sequence[tuple[Path, int]],
        image_size: int,
        crop_box: Box | None,
        augment: v2.Compose | None = None,
    ) -> None:
        self._items: list[Sequence[tuple[Path, int]]] = list(items)
        self._image_size: int = image_size
        self._crop_box: Box | None = crop_box
        self._augment: v2.Compose | None = augment

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, class_index = self._items[index]
        image = load_image(path, self._crop_box)
        if self._augment is None:
            return (
                torch.from_numpy(to_model_input(image, self._image_size)),
                class_index
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # (3, H, W) uint8
        channels_first = torch.from_numpy(rgb).permute(2, 0, 1)
        return self._augment(channels_first), class_index


def build_training_augmentation(image_size: int) -> v2.Compose:
    """Random changes to training images so the model learns the subject,
    not the particular photo.

    Rotation and flips: for a camera looking straight down, a subject facing
    any direction is still the same subject.
    Random crop: small shifts in position and scale.
    Brightness and contrast: indoor lighting changes over the day.
    Deliberately no hue shift: when classes differ by color, shifting hue
    would blur exactly the signal the model needs.
    """
    return v2.Compose(
        [
            v2.RandomRotation(degrees=180),
            v2.RandomResizedCrop(image_size, scale=(0.6, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=IMAGENET_MEAN.tolist(),
                std=IMAGENET_STD.tolist()
            ),
        ]
    )


def build_mobilenet_v3_small(
        num_classes: int,
        pretrained: bool = True
) -> nn.Module:
    """MobileNetV3-Small with its last layer replaced for our classes.

    "Pretrained" means it already learned general visual features from about
    1.2 million ImageNet photos. We only teach it our few classes ("transfer
    learning"), which is why a few hundred labeled images can be enough. This
    network was designed to be fast on phone-class CPUs like a Raspberry Pi's.
    Use pretrained=False when you're about to load your own saved weights.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    last_layer = model.classifier[-1]
    assert isinstance(last_layer, nn.Linear)
    model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
    return model


def inverse_frequency_weights(
    class_indices: Sequence[int], num_classes: int
) -> torch.Tensor:
    """Loss weights that make every class count equally during training.

    If 80% of training images are one class, an unweighted model can score
    well by leaning toward it. Weighting each class by
    total / (num_classes * count) makes the total cost of mistakes on a rare
    class equal to that on a common one.
    """
    counts = np.bincount(np.asarray(class_indices), minlength=num_classes)
    missing = [index for index in range(num_classes) if counts[index] == 0]
    if missing:
        raise ValueError(f"No training images for class index(es) {missing}")
    weights = len(class_indices) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    train_loss: float
    val_macro_f1: float


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    class_weights: torch.Tensor,
    device: str,
) -> tuple[nn.Module, list[EpochSummary]]:
    """Train, then return the version of the model that did best on validation.

    One "epoch" is one pass over all training images. After each epoch the
    model is scored on the validation set with macro F1 (the plain average of
    per-class F1). Accuracy is not used for this choice because it can look
    excellent while the model is poor at rare classes.
    """
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_score = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    history: list[EpochSummary] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, image_count = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            image_count += len(targets)

        true_classes, predicted_classes = predict_classes(
            model,
            val_loader,
            device
        )
        score = float(
            f1_score(
                true_classes,
                predicted_classes,
                average="macro",
                zero_division=0
            )
        )
        summary = EpochSummary(epoch, loss_sum / image_count, score)
        history.append(summary)
        print(
            f"epoch {epoch:>3}  train loss {summary.train_loss:.4f}  "
            f"val macro-F1 {score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, history
