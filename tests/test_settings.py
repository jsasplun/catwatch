# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Tests for catwatch.settings, plus sanity checks on the real config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from catwatch import settings
from catwatch.settings import CONFIG_PATH, bowl_crop_box, project_path


def load_real_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_bowl_crop_box_none_means_whole_frame() -> None:
    assert bowl_crop_box({"bowl_crop": None}) is None


def test_bowl_crop_box_returns_x_y_width_height_tuple() -> None:
    crop = {"x": 240, "y": 80, "width": 800, "height": 700}
    assert bowl_crop_box({"bowl_crop": crop}) == (240, 80, 800, 700)


def test_bowl_crop_box_reads_model_card_shaped_settings() -> None:
    # A model card stores the crop under the same key as config.yaml does,
    # next to unrelated keys the function must ignore.
    card = {
        "class_names": ["a"],
        "bowl_crop": {"x": 1, "y": 2, "width": 3, "height": 4},
    }
    assert bowl_crop_box(card) == (1, 2, 3, 4)


def test_project_path_is_inside_the_repository() -> None:
    assert project_path("data/raw") == settings.PROJECT_ROOT / "data" / "raw"
    assert (settings.PROJECT_ROOT / "config.yaml").exists()


def test_project_path_passes_absolute_paths_through(tmp_path: Path) -> None:
    # The other tests rely on this to redirect all file access to a temp dir.
    assert project_path(str(tmp_path / "x")) == tmp_path / "x"


def test_load_config_returns_config_yaml_and_loads_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # load_dotenv is replaced so the test never reads the real .env, which
    # holds secrets.
    loaded_paths: list[Path] = []
    monkeypatch.setattr(settings, "load_dotenv", loaded_paths.append)
    config = settings.load_config()
    assert loaded_paths == [settings.PROJECT_ROOT / ".env"]
    assert config == load_real_config()


class TestRealConfigIsConsistent:
    """Mistakes in config.yaml (a typo in a class name, say) would otherwise
    only show up hours later in training or on the Pi."""

    def test_classes_are_unique_and_non_empty(self) -> None:
        classes = load_real_config()["classes"]
        assert classes and len(classes) == len(set(classes))

    def test_every_class_has_a_display_name(self) -> None:
        config = load_real_config()
        assert set(config["display_names"]) == set(config["classes"])

    def test_every_class_has_a_labeling_key(self) -> None:
        config = load_real_config()
        assert set(config["label_keys"].values()) >= set(config["classes"])

    def test_labeling_keys_only_name_classes_or_unusable(self) -> None:
        config = load_real_config()
        allowed = set(config["classes"]) | {"unusable"}
        assert set(config["label_keys"].values()) <= allowed

    def test_labeling_keys_are_single_characters_that_avoid_back_and_quit(
        self,
    ) -> None:
        keys = load_real_config()["label_keys"]
        assert all(len(str(key)) == 1 for key in keys)
        assert not {"z", "q"} & {str(key) for key in keys}

    def test_split_fractions_add_up_to_one(self) -> None:
        fractions = load_real_config()["training"]["split_fractions"]
        assert sum(fractions.values()) == pytest.approx(1.0)
        assert set(fractions) == {"train", "val", "test"}

    def test_background_label_is_a_class(self) -> None:
        config = load_real_config()
        assert config["monitor"]["background_label"] in config["classes"]

    def test_smoothing_window_is_odd_so_votes_cannot_tie(self) -> None:
        assert load_real_config()["monitor"]["smoothing_window"] % 2 == 1

    def test_bowl_crop_is_null_or_a_complete_box(self) -> None:
        crop = load_real_config()["bowl_crop"]
        assert crop is None or set(crop) == {"x", "y", "width", "height"}

    def test_config_contains_no_env_style_secrets(self) -> None:
        # config.yaml is copied verbatim into every runs/*/run_info.json.
        text = CONFIG_PATH.read_text(encoding="utf-8").lower()
        assert not any(word in text for word in ("password", "api_key", "token"))
