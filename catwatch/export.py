# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Export a trained run to ONNX for the Pi, and prove the export matches.

    python -m catwatch.export --run runs/20260919_143015_seed0

Creates models/<run name>/ containing model.onnx and model_card.json. Copy that
whole folder to the Pi. Each deployed model is its own folder, so rolling back
means pointing monitor.model_dir at the previous one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from catwatch.data import items_for_split, load_labeled_images
from catwatch.settings import bowl_crop_box, project_path
from core.onnx_export import compare_to_onnx, export_to_onnx
from core.preprocessing import load_image, to_model_input
from core.run_records import read_json, write_json
from core.training import build_mobilenet_v3_small

PARITY_SAMPLE_SIZE = 32
MAX_ALLOWED_LOGIT_DIFFERENCE = 1e-3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    run_info = read_json(args.run / "run_info.json")
    config = run_info["config"]
    class_names = run_info["class_names"]
    image_size = config["training"]["image_size"]
    crop = bowl_crop_box(config)

    model = build_mobilenet_v3_small(len(class_names), pretrained=False)
    model.load_state_dict(
        torch.load(
            args.run / "model.pt",
            map_location="cpu",
            weights_only=True
        )
    )
    output_dir = project_path(config["paths"]["models_dir"]) / args.run.name
    output_dir.mkdir(parents=True, exist_ok=False)
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(model, onnx_path, image_size)

    # Check the export on real validation images, not just a blank test input.
    sample = items_for_split(load_labeled_images(config), "val", class_names)
    sample = sample[:PARITY_SAMPLE_SIZE]
    if not sample:
        raise SystemExit("No validation images available" +
                         "for the parity check.")
    batch = np.stack([
        to_model_input(load_image(path, crop), image_size)
        for path, _ in sample
    ])
    parity = compare_to_onnx(model, onnx_path, batch)
    print(f"ONNX parity check: {parity}")
    if (
        not parity["same_predictions"]
        or parity["max_abs_logit_difference"] > MAX_ALLOWED_LOGIT_DIFFERENCE
    ):
        raise SystemExit("ONNX output does not match PyTorch. Do not deploy.")

    val_metrics_path = args.run / "metrics_val.json"
    write_json(
        output_dir / "model_card.json",
        {
            "source_run": args.run.name,
            "class_names": class_names,
            "image_size": image_size,
            "bowl_crop": config["bowl_crop"],
            "git_commit": run_info["git_commit"],
            "labels_sha256": run_info["labels_sha256"],
            "onnx_parity_check": parity,
            "val_metrics": read_json(val_metrics_path)
            if val_metrics_path.exists()
            else None,
        },
    )
    print(f"Exported to {output_dir}")


if __name__ == "__main__":
    main()
