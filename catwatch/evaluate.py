# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Score a trained run on the validation or test split.

    python -m catwatch.evaluate --run runs/20260919_143015_seed0
    python -m catwatch.evaluate --run runs/... --split test  # final check only

Writes metrics_<split>.json and errors_<split>.csv into the run folder. Open
the images listed in errors_<split>.csv and look at them.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from catwatch.data import items_for_split, load_labeled_images
from catwatch.settings import bowl_crop_box, project_path
from core.evaluation import classification_summary, predict_classes
from core.run_records import file_sha256, read_json, write_json
from core.training import ImageClassificationDataset, build_mobilenet_v3_small


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    run_info = read_json(args.run / "run_info.json")
    # Use the settings the model was trained with, not today's config.yaml.
    config = run_info["config"]
    class_names = run_info["class_names"]
    labels_file = project_path(config["paths"]["labels_file"])
    if file_sha256(labels_file) != run_info["labels_sha256"]:
        print(
            "WARNING: labels.csv changed after this model was trained. Scores "
            "may not be comparable with earlier evaluations of this run."
        )

    items = items_for_split(
        load_labeled_images(config),
        args.split,
        class_names
    )
    loader = DataLoader(
        ImageClassificationDataset(
            items, config["training"]["image_size"], bowl_crop_box(config)
        ),
        batch_size=config["training"]["batch_size"],
    )
    model = build_mobilenet_v3_small(len(class_names), pretrained=False)
    model.load_state_dict(
        torch.load(
            args.run / "model.pt",
            map_location="cpu",
            weights_only=True
        )
    )
    true_classes, predicted_classes = predict_classes(model, loader, "cpu")

    summary = classification_summary(
        true_classes,
        predicted_classes,
        class_names
    )
    write_json(args.run / f"metrics_{args.split}.json", summary)
    with (args.run / f"errors_{args.split}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_label", "predicted_label"])
        for (path, _), true_index, predicted_index in zip(
            items, true_classes, predicted_classes
        ):
            if true_index != predicted_index:
                writer.writerow([
                    path, class_names[true_index],
                    class_names[predicted_index]
                ])

    report = summary["report"]
    print(f"{'class':<20}{'precision':>10}{'recall':>8}{'f1':>8}{'count':>8}")
    for name in class_names:
        row = report[name]
        print(
            f"{name:<20}{row['precision']:>10.3f}{row['recall']:>8.3f}"
            f"{row['f1-score']:>8.3f}{int(row['support']):>8}"
        )
    print(f"\nmacro F1: {report['macro avg']['f1-score']:.4f}")
    print("confusion matrix (rows = true, columns = predicted):")
    for name, matrix_row in zip(class_names, summary["confusion_matrix"]):
        print(f"{name:<20}{matrix_row}")


if __name__ == "__main__":
    main()
