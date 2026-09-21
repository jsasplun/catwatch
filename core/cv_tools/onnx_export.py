# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Exporting a PyTorch model to ONNX and checking the export on real inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

INPUT_NAME = "image"
OUTPUT_NAME = "logits"


def export_to_onnx(
        model: nn.Module,
        output_path: Path,
        image_size: int
) -> None:
    """Save the model as ONNX and check that the file is well-formed.

    Input "image": (batch, 3, image_size, image_size). Output "logits":
    (batch, number of classes). Logits are raw scores; softmax turns them into
    probabilities.
    """
    model = model.cpu().eval()
    example_input = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        (example_input,),
        str(output_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(str(output_path)))


def compare_to_onnx(
    model: nn.Module, onnx_path: Path, inputs: np.ndarray
) -> dict[str, float | bool]:
    """Run identical inputs through PyTorch and ONNX Runtime and measure the
    gap.

    An export is supposed to match the original; this checks it on real data
    instead of assuming. Expect differences around 1e-5 and identical
    predicted classes.
    """
    with torch.no_grad():
        torch_logits = model.cpu().eval()(torch.from_numpy(inputs)).numpy()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )
    onnx_logits = session.run(None, {INPUT_NAME: inputs})[0]
    return {
        "max_abs_logit_difference": float(
            np.abs(torch_logits - onnx_logits).max()
        ),
        "same_predictions": bool(
            np.array_equal(
                torch_logits.argmax(axis=1),
                onnx_logits.argmax(axis=1)
            )
        ),
    }
