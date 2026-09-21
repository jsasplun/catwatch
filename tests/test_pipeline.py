# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""End-to-end test of the desktop pipeline on tiny synthetic data:

    labeled images -> train -> evaluate -> export to ONNX -> monitor

Each step's real output file is opened and checked (run_info.json, metrics,
model_card.json, model.onnx), and the exported model is then used by the real
monitor loop. The images are small colored squares, so the model quality is
irrelevant; what matters is that every step's inputs and outputs line up.

It trains for two short epochs on 64x64 images, which takes a few seconds on
CPU. It is skipped when PyTorch isn't installed.
"""

from __future__ import annotations
import sys
from pathlib import Path
# Adds the parent directory of this file to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import shutil
import signal
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip("torch")

from catwatch import evaluate, export, monitor, train  # noqa: E402
from catwatch.data import group_for  # noqa: E402
from catwatch.settings import CONFIG_PATH  # noqa: E402
from core.cv_tools.capture_store import CAPTURE_FIELDS, CAPTURE_LOG_NAME  # noqa: E402
from core.cv_tools.csv_log import append_csv_row  # noqa: E402
from core.cv_tools.labels import append_label  # noqa: E402
from core.cv_tools.onnx_classifier import OnnxImageClassifier  # noqa: E402
from core.cv_tools.preprocessing import load_image  # noqa: E402
from core.cv_tools.run_records import file_sha256, read_json  # noqa: E402
from core.cv_tools.splits import assign_split  # noqa: E402
from core.cv_tools.training import build_mobilenet_v3_small  # noqa: E402
from tests.fakes import FakeCamera  # noqa: E402

IMAGE_SIZE = 64
# One distinctive BGR color per class, in config.yaml's class order.
CLASS_COLORS = [(90, 90, 90), (20, 20, 20), (0, 140, 255), (200, 120, 40)]
TRAIN_IMAGES_PER_CLASS = 6
VAL_IMAGES_PER_CLASS = 3


def find_hour_in_split(fractions: dict[str, float], split: str, first_day: int) -> str:
    """Search for a clock-hour group that hashes into the wanted split.

    Splits are decided by a hash, so a test can't pick a split directly. It
    finds a timestamp whose hour lands where the test needs it.
    """
    for day in range(first_day, first_day + 60):
        for hour in range(24):
            timestamp = f"2026-08-{day % 28 + 1:02d}T{hour:02d}:00:00"
            if assign_split(group_for(timestamp), fractions) == split:
                return timestamp[:13]  # "2026-08-05T09": date, "T", hour
    raise AssertionError(f"No hour found for split {split!r}")


@pytest.fixture(scope="module")
def pipeline(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[SimpleNamespace]:
    """Builds the labeled dataset and runs train -> evaluate -> export once.

    Module scope: training is the slow part, and the tests below only read
    its outputs.
    """
    root = tmp_path_factory.mktemp("pipeline")
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)
    config["paths"] = {
        "raw_dir": str(root / "raw"),
        "labels_file": str(root / "labels.csv"),
        "events_file": str(root / "events.csv"),
        "runs_dir": str(root / "runs"),
        "models_dir": str(root / "models"),
    }
    config["bowl_crop"] = None
    config["training"].update(
        {
            "image_size": IMAGE_SIZE,
            "epochs": 2,
            "batch_size": 8,
            "num_workers": 0,
            "seed": 0,
        }
    )
    config["monitor"].update(
        {"smoothing_window": 3, "min_confidence": 0.0, "min_event_seconds": 0}
    )

    raw_dir = Path(config["paths"]["raw_dir"])
    labels_file = Path(config["paths"]["labels_file"])
    fractions = config["training"]["split_fractions"]
    hours = {
        "train": find_hour_in_split(fractions, "train", 0),
        "val": find_hour_in_split(fractions, "val", 30),
    }
    rng = np.random.default_rng(0)
    for class_index, class_name in enumerate(config["classes"]):
        counts = {"train": TRAIN_IMAGES_PER_CLASS, "val": VAL_IMAGES_PER_CLASS}
        for split, count in counts.items():
            for number in range(count):
                image = np.full((80, 100, 3), CLASS_COLORS[class_index], np.uint8)
                noise = rng.integers(-8, 9, image.shape)
                image = np.clip(image + noise, 0, 255).astype(np.uint8)
                relative = f"{split}/{class_name}_{number}.jpg"
                (raw_dir / split).mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(raw_dir / relative), image)
                append_csv_row(
                    raw_dir / CAPTURE_LOG_NAME,
                    CAPTURE_FIELDS,
                    {
                        "image_path": relative,
                        "captured_at": f"{hours[split]}:{number:02d}:00",
                        "reason": "test",
                    },
                )
                append_label(labels_file, relative, class_name, "test")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "load_config", lambda: config)
        patch.setattr(monitor, "load_config", lambda: config)
        # The real function downloads pretrained weights from the internet.
        # Tests must run offline, so build the same network with random
        # weights instead.
        real_builder = build_mobilenet_v3_small
        patch.setattr(
            train,
            "build_mobilenet_v3_small",
            lambda num_classes: real_builder(num_classes, pretrained=False),
        )

        patch.setattr(sys, "argv", ["train", "--seed", "1"])
        train.main()
        (run_dir,) = Path(config["paths"]["runs_dir"]).iterdir()

        patch.setattr(sys, "argv", ["evaluate", "--run", str(run_dir)])
        evaluate.main()

        patch.setattr(sys, "argv", ["export", "--run", str(run_dir)])
        export.main()
        (model_dir,) = Path(config["paths"]["models_dir"]).iterdir()

        yield SimpleNamespace(
            config=config, run_dir=run_dir, model_dir=model_dir, patch=patch
        )


def test_train_saves_weights_and_a_complete_run_record(
    pipeline: SimpleNamespace,
) -> None:
    config = pipeline.config
    assert (pipeline.run_dir / "model.pt").is_file()
    info = read_json(pipeline.run_dir / "run_info.json")
    assert info["seed"] == 1
    assert info["class_names"] == config["classes"]
    assert info["image_counts"] == {
        "train": TRAIN_IMAGES_PER_CLASS * 4,
        "val": VAL_IMAGES_PER_CLASS * 4,
        "test": 0,
    }
    assert info["labels_sha256"] == file_sha256(Path(config["paths"]["labels_file"]))
    assert info["captures_sha256"] == file_sha256(
        Path(config["paths"]["raw_dir"]) / CAPTURE_LOG_NAME
    )
    assert len(info["history"]) == 2
    assert 0.0 <= info["best_val_macro_f1"] <= 1.0
    assert info["config"]["classes"] == config["classes"]


def test_evaluate_writes_metrics_and_an_errors_file_for_the_val_split(
    pipeline: SimpleNamespace,
) -> None:
    metrics = read_json(pipeline.run_dir / "metrics_val.json")
    assert metrics["class_names"] == pipeline.config["classes"]
    matrix = np.array(metrics["confusion_matrix"])
    assert matrix.shape == (4, 4)
    assert matrix.sum() == VAL_IMAGES_PER_CLASS * 4  # every val image scored once

    with (pipeline.run_dir / "errors_val.csv").open(newline="") as errors_file:
        errors = list(csv.DictReader(errors_file))
    off_diagonal = matrix.sum() - np.trace(matrix)
    assert len(errors) == off_diagonal
    assert all(row["true_label"] != row["predicted_label"] for row in errors)


@pytest.fixture
def run_copy(pipeline: SimpleNamespace, tmp_path: Path) -> Path:
    """A scratch copy of the run folder. Re-running evaluate rewrites the
    metrics files, and the export test needs the originals unchanged."""
    copied = tmp_path / "run_copy"
    shutil.copytree(pipeline.run_dir, copied)
    return copied


def test_evaluate_prints_a_per_class_table(
    pipeline: SimpleNamespace,
    run_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipeline.patch.setattr(sys, "argv", ["evaluate", "--run", str(run_copy)])
    evaluate.main()
    output = capsys.readouterr().out
    assert "macro F1:" in output
    assert all(name in output for name in pipeline.config["classes"])
    assert "WARNING" not in output  # labels.csv is unchanged since training


def test_evaluate_warns_when_labels_changed_after_training(
    pipeline: SimpleNamespace,
    run_copy: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels_file = Path(pipeline.config["paths"]["labels_file"])
    original = labels_file.read_bytes()
    try:
        append_label(labels_file, "val/empty_0.jpg", "unusable", "late-relabel")
        pipeline.patch.setattr(sys, "argv", ["evaluate", "--run", str(run_copy)])
        evaluate.main()
    finally:
        labels_file.write_bytes(original)  # restore for the tests that follow
    assert "labels.csv changed after this model was trained" in (
        capsys.readouterr().out
    )


def test_export_produces_a_model_and_a_model_card(pipeline: SimpleNamespace) -> None:
    assert pipeline.model_dir.name == pipeline.run_dir.name
    assert (pipeline.model_dir / "model.onnx").stat().st_size > 0
    card = read_json(pipeline.model_dir / "model_card.json")
    assert card["source_run"] == pipeline.run_dir.name
    assert card["class_names"] == pipeline.config["classes"]
    assert card["image_size"] == IMAGE_SIZE
    assert card["bowl_crop"] is None
    assert (
        card["labels_sha256"]
        == read_json(pipeline.run_dir / "run_info.json")["labels_sha256"]
    )
    assert card["val_metrics"] == read_json(pipeline.run_dir / "metrics_val.json")


def test_export_parity_check_passed_and_is_recorded(
    pipeline: SimpleNamespace,
) -> None:
    parity = read_json(pipeline.model_dir / "model_card.json")["onnx_parity_check"]
    assert parity["same_predictions"] is True
    assert parity["max_abs_logit_difference"] < export.MAX_ALLOWED_LOGIT_DIFFERENCE


def test_export_refuses_to_overwrite_an_existing_model_folder(
    pipeline: SimpleNamespace,
) -> None:
    # Each deployed model is its own folder, so rolling back stays possible.
    pipeline.patch.setattr(sys, "argv", ["export", "--run", str(pipeline.run_dir)])
    with pytest.raises(FileExistsError):
        export.main()


def test_exported_model_classifies_an_image_like_the_deployed_pi_would(
    pipeline: SimpleNamespace,
) -> None:
    card = read_json(pipeline.model_dir / "model_card.json")
    classifier = OnnxImageClassifier(
        pipeline.model_dir / "model.onnx", card["class_names"], card["image_size"]
    )
    image = load_image(
        Path(pipeline.config["paths"]["raw_dir"]) / "val" / "empty_0.jpg"
    )
    prediction = classifier.predict(image)
    assert prediction.label in pipeline.config["classes"]
    assert 0.25 <= prediction.confidence <= 1.0  # softmax over 4 classes


def test_monitor_runs_with_the_exported_model_folder(
    pipeline: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The last hop: monitor reads what export wrote, using the real ONNX
    file, on frames from a fake camera."""
    config = pipeline.config
    config["monitor"]["model_dir"] = str(pipeline.model_dir)
    class_names = config["classes"]
    raw_dir = Path(config["paths"]["raw_dir"])
    frames = [
        load_image(raw_dir / "val" / f"{class_names[index % 4]}_0.jpg")
        for index in range(12)
    ]
    camera = FakeCamera(frames)
    pipeline.patch.setattr(monitor, "open_camera", lambda _config: camera)
    pipeline.patch.setattr(monitor.time, "sleep", lambda _seconds: None)
    pipeline.patch.setattr(sys, "argv", ["monitor"])

    original_handler = signal.getsignal(signal.SIGTERM)
    try:
        monitor.main()
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    assert camera.reads == 12
    assert camera.closed
    assert "Stopping." in capsys.readouterr().out
