# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Tests for catwatch.monitor: the Pi program that turns camera frames into
logged visits.

The camera and the neural network are replaced by scripted fakes so each test
controls exactly what "the model saw" on every frame. Everything else (the
smoothing, the CSV log, the saved snapshots, the model card) is the real code.
"""

from __future__ import annotations

import csv
import signal
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from catwatch import monitor
from core.cv_tools.capture_store import read_capture_log
from core.cv_tools.events import Event
from core.cv_tools.onnx_classifier import Prediction
from core.cv_tools.run_records import write_json
from tests.fakes import FakeCamera, make_frame

EMPTY = Prediction("empty", 0.95)
BLACK = Prediction("black_patch_cat", 0.9)
UNSURE_BOTH = Prediction("both_cats", 0.3)  # below the 0.6 confidence bar


class ScriptedClassifier:
    """Answers with prepared predictions, one per frame, and remembers the
    shape of every image it was asked about."""

    def __init__(self, predictions: list[Prediction]) -> None:
        self._predictions: Iterator[Prediction] = iter(predictions)
        self.seen_shapes: list[tuple[int, ...]] = []
        self.constructor_args: tuple[Any, ...] = ()

    def factory(self, *args: Any) -> ScriptedClassifier:
        """Stands in for the OnnxImageClassifier class itself."""
        self.constructor_args = args
        return self

    def predict(self, bgr_image: np.ndarray) -> Prediction:
        self.seen_shapes.append(bgr_image.shape)
        return next(self._predictions)


@pytest.fixture
def restore_sigterm_handler() -> Iterator[None]:
    """monitor.main() installs a SIGTERM handler for the whole process."""
    original = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, original)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Skips the real waiting between predictions and records the requested
    delays instead."""
    requested: list[float] = []
    fake_time = SimpleNamespace(monotonic=time.monotonic, sleep=requested.append)
    monkeypatch.setattr(monitor, "time", fake_time)
    return requested


@pytest.fixture
def deployment(
    config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    """A model folder with a model card, plus config tweaks that make events
    easy to trigger in a handful of frames."""
    model_dir = tmp_path / "models" / "run1"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "model_card.json",
        {
            "class_names": config["classes"],
            "image_size": 32,
            "bowl_crop": None,
        },
    )
    config["monitor"].update(
        {
            "model_dir": str(model_dir),
            "smoothing_window": 3,
            "min_confidence": 0.6,
            "min_event_seconds": 0,  # frames arrive instantly in a test
            "predictions_per_second": 2,
        }
    )
    monkeypatch.setattr(monitor, "load_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["monitor"])
    return SimpleNamespace(config=config, model_dir=model_dir)


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    predictions: list[Prediction],
    end_with: BaseException | None = None,
    frame_size: tuple[int, int] = (64, 48),
) -> tuple[FakeCamera, ScriptedClassifier]:
    classifier = ScriptedClassifier(predictions)
    camera = FakeCamera(
        [make_frame(*frame_size, (10, 20, 30)) for _ in predictions],
        end_with=end_with,
    )
    monkeypatch.setattr(monitor, "OnnxImageClassifier", classifier.factory)
    monkeypatch.setattr(monitor, "open_camera", lambda _config: camera)
    return camera, classifier


def event_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    events_file = Path(config["paths"]["events_file"])
    if not events_file.exists():
        return []
    with events_file.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# --- log_event ---------------------------------------------------------------


def test_log_event_writes_one_row_with_a_friendly_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    started = datetime(2026, 9, 19, 12, 0, 0)
    event = Event("black_patch_cat", started, started + timedelta(seconds=12.34))
    events_file = tmp_path / "events.csv"
    monitor.log_event(events_file, event, {"black_patch_cat": "Cat with black patches"})

    with events_file.open(newline="", encoding="utf-8") as file:
        (row,) = list(csv.DictReader(file))
    assert row == {
        "cat": "black_patch_cat",
        "display_name": "Cat with black patches",
        "started_at": "2026-09-19T12:00:00",
        "ended_at": "2026-09-19T12:00:12.340000",
        "duration_seconds": "12.3",
    }
    assert "Cat with black patches at bowl for 12s" in capsys.readouterr().out


def test_log_event_falls_back_to_the_raw_label_without_a_display_name(
    tmp_path: Path,
) -> None:
    started = datetime(2026, 9, 19, 12, 0, 0)
    event = Event("mystery", started, started + timedelta(seconds=5))
    monitor.log_event(tmp_path / "events.csv", event, {})
    text = (tmp_path / "events.csv").read_text(encoding="utf-8")
    assert "mystery,mystery," in text


def test_log_event_appends_and_writes_the_header_only_once(tmp_path: Path) -> None:
    started = datetime(2026, 9, 19, 12, 0, 0)
    event = Event("both_cats", started, started + timedelta(seconds=4))
    events_file = tmp_path / "events.csv"
    monitor.log_event(events_file, event, {})
    monitor.log_event(events_file, event, {})
    lines = events_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(monitor.EVENT_FIELDS)
    assert len(lines) == 3


# --- show_prediction ---------------------------------------------------------


def test_show_prediction_draws_on_a_copy_scaled_to_800_wide(
    install_window: Callable[[str], Any],
) -> None:
    window = install_window("a")  # waitKey(1) here only lets the window redraw
    frame = make_frame(1600, 1200, (50, 50, 50))
    original = frame.copy()
    monitor.show_prediction(frame, Prediction("empty", 0.5), None)
    (shown,) = window.shown
    assert shown.shape == (600, 800, 3)
    assert (frame == original).all()  # the caller's frame is left untouched
    assert (shown != 50).any()  # label text was drawn on it


def test_show_prediction_outlines_the_crop_in_yellow(
    install_window: Callable[[str], Any],
) -> None:
    window = install_window("a")
    frame = make_frame(800, 600, (50, 50, 50))
    monitor.show_prediction(frame, Prediction("empty", 0.5), (200, 200, 300, 300))
    left_edge_middle = window.shown[0][350, 200]  # 800 wide: no scaling
    assert tuple(int(v) for v in left_edge_middle) == (0, 255, 255)


# --- main --------------------------------------------------------------------


def test_a_sustained_visit_is_logged_once_and_snapshotted_once(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    camera, classifier = install_fakes(
        monkeypatch,
        [EMPTY] * 3 + [BLACK] * 4 + [EMPTY] * 4,
    )
    monitor.main()

    (row,) = event_rows(deployment.config)
    assert row["cat"] == "black_patch_cat"
    assert row["display_name"] == "Cat with black patches"
    assert float(row["duration_seconds"]) >= 0
    # One snapshot, taken the moment the visit began, tagged with its class.
    captures = read_capture_log(Path(deployment.config["paths"]["raw_dir"]))
    assert [capture["reason"] for capture in captures] == ["event-black_patch_cat"]
    assert (
        Path(deployment.config["paths"]["raw_dir"]) / captures[0]["image_path"]
    ).is_file()
    assert camera.closed
    assert "Stopping." in capsys.readouterr().out


def test_the_model_is_built_from_the_model_card(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    _, classifier = install_fakes(monkeypatch, [EMPTY])
    monitor.main()
    model_path, class_names, image_size = classifier.constructor_args
    assert model_path == deployment.model_dir / "model.onnx"
    assert class_names == deployment.config["classes"]
    assert image_size == 32


def test_frames_are_cropped_to_the_box_in_the_model_card_not_config_yaml(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    # The model was trained on the card's crop. Today's config.yaml may say
    # something else, and must lose.
    write_json(
        deployment.model_dir / "model_card.json",
        {
            "class_names": deployment.config["classes"],
            "image_size": 32,
            "bowl_crop": {"x": 4, "y": 2, "width": 20, "height": 10},
        },
    )
    deployment.config["bowl_crop"] = {"x": 0, "y": 0, "width": 50, "height": 40}
    _, classifier = install_fakes(monkeypatch, [EMPTY, EMPTY])
    monitor.main()
    assert classifier.seen_shapes == [(10, 20, 3), (10, 20, 3)]


def test_without_a_crop_the_whole_frame_is_classified(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    _, classifier = install_fakes(monkeypatch, [EMPTY])
    monitor.main()
    assert classifier.seen_shapes == [(48, 64, 3)]


def test_low_confidence_predictions_are_ignored(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    # Ten "both cats" frames the model is unsure about must not become a visit.
    install_fakes(monkeypatch, [EMPTY] * 3 + [UNSURE_BOTH] * 10 + [EMPTY] * 3)
    monitor.main()
    assert event_rows(deployment.config) == []
    assert not Path(deployment.config["paths"]["raw_dir"]).exists()  # no snapshot


def test_a_single_flicker_frame_makes_no_event(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    install_fakes(monkeypatch, [EMPTY] * 4 + [BLACK] + [EMPTY] * 4)
    monitor.main()
    assert event_rows(deployment.config) == []


def test_a_visit_still_in_progress_is_logged_on_ctrl_c(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    install_fakes(monkeypatch, [EMPTY] * 3 + [BLACK] * 6)  # ends mid-visit
    monitor.main()
    assert [row["cat"] for row in event_rows(deployment.config)] == ["black_patch_cat"]


def test_a_visit_still_in_progress_is_logged_when_systemd_stops_the_service(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    # systemd sends SIGTERM, which monitor turns into SystemExit. Simulate the
    # process receiving it in the middle of the visit.
    camera, _ = install_fakes(
        monkeypatch, [EMPTY] * 3 + [BLACK] * 6, end_with=SystemExit(0)
    )
    with pytest.raises(SystemExit):
        monitor.main()
    assert [row["cat"] for row in event_rows(deployment.config)] == ["black_patch_cat"]
    assert camera.closed


def test_sigterm_handler_exits_normally_so_cleanup_code_runs(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    install_fakes(monkeypatch, [EMPTY])
    monitor.main()
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)
    with pytest.raises(SystemExit) as exit_info:
        handler(signal.SIGTERM, None)
    assert exit_info.value.code == 0


def test_the_loop_waits_out_the_rest_of_each_prediction_interval(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    install_fakes(monkeypatch, [EMPTY] * 3)
    monitor.main()
    # 2 predictions per second means at most 0.5 s per frame. Fake frames are
    # instant, so almost the whole interval is spent sleeping.
    assert len(sleeps) == 3
    assert all(0.4 < delay <= 0.5 for delay in sleeps)


def test_show_flag_draws_every_frame(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    install_fakes(monkeypatch, [EMPTY, BLACK])
    shown: list[Prediction] = []
    monkeypatch.setattr(
        monitor,
        "show_prediction",
        lambda _frame, prediction, _crop: shown.append(prediction),
    )
    monkeypatch.setattr(sys, "argv", ["monitor", "--show"])
    monitor.main()
    assert shown == [EMPTY, BLACK]


def test_headless_mode_never_draws(
    deployment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    restore_sigterm_handler: None,
) -> None:
    install_fakes(monkeypatch, [EMPTY, BLACK])

    def fail(*_args: Any) -> None:
        raise AssertionError("show_prediction must not run without --show")

    monkeypatch.setattr(monitor, "show_prediction", fail)
    monitor.main()


def test_a_missing_model_card_fails_before_the_camera_is_opened(
    deployment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    (deployment.model_dir / "model_card.json").unlink()
    opened: list[bool] = []
    monkeypatch.setattr(monitor, "open_camera", lambda _config: opened.append(True))
    with pytest.raises(FileNotFoundError):
        monitor.main()
    assert opened == []
