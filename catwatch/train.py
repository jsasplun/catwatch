# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Train a classifier and save it, with a full record, under runs/.

    python -m catwatch.train              # seed from config.yaml
    python -m catwatch.train --seed 3
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from catwatch.data import items_for_split, load_labeled_images, report_split_counts
from catwatch.settings import bowl_crop_box, load_config, project_path
from core.cv_tools.capture_store import CAPTURE_LOG_NAME
from core.cv_tools.run_records import (
    current_git_commit,
    file_sha256,
    new_run_directory,
    write_json,
)
from core.cv_tools.training import (
    ImageClassificationDataset,
    build_mobilenet_v3_small,
    build_training_augmentation,
    inverse_frequency_weights,
    set_seed,
    train_classifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    settings = config["training"]
    seed = settings["seed"] if args.seed is None else args.seed
    set_seed(seed)

    class_names = config["classes"]
    image_size = settings["image_size"]
    crop = bowl_crop_box(config)
    split_names = list(settings["split_fractions"])
    images = load_labeled_images(config)
    report_split_counts(images, class_names, split_names)
    train_items = items_for_split(images, "train", class_names)
    val_items = items_for_split(images, "val", class_names)

    train_loader = DataLoader(
        ImageClassificationDataset(
            train_items,
            image_size,
            crop,
            build_training_augmentation(image_size)
        ),
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        ImageClassificationDataset(val_items, image_size, crop),
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    model, history = train_classifier(
        build_mobilenet_v3_small(len(class_names)),
        train_loader,
        val_loader,
        epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        class_weights=inverse_frequency_weights(
            [class_index for _, class_index in train_items], len(class_names)
        ),
        device=device,
    )

    run_dir = new_run_directory(
        project_path(config["paths"]["runs_dir"]),
        f"seed{seed}"
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    raw_dir = project_path(config["paths"]["raw_dir"])
    write_json(
        run_dir / "run_info.json",
        {
            "seed": seed,
            "class_names": class_names,
            "git_commit": current_git_commit(),
            "labels_sha256": file_sha256(
                project_path(config["paths"]["labels_file"])
            ),
            "captures_sha256": file_sha256(raw_dir / CAPTURE_LOG_NAME),
            "image_counts": {
                split: len(items_for_split(images, split, class_names))
                for split in split_names
            },
            "best_val_macro_f1": max(epoch.val_macro_f1 for epoch in history),
            "history": [asdict(epoch) for epoch in history],
            "config": config,  # contains no secrets; those stay in .env
        },
    )
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
