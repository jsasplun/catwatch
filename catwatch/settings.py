# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Project settings: reads config.yaml, loads secrets from .env, resolves
paths.

All project-specific names and numbers live in config.yaml, so they can change
without code edits and every training run can save an exact copy of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from core.preprocessing import Box

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict[str, Any]:
    # Secrets (API keys, passwords) live in .env, which git ignores.
    # load_dotenv copies them into environment variables; no-op if absent.
    load_dotenv(PROJECT_ROOT / ".env")
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def bowl_crop_box(settings: Mapping[str, Any]) -> Box | None:
    """Read the "bowl_crop" entry from config.yaml or from a model card.

    Both store it under the same key, which lets deployment use the crop the
    model was trained with rather than whatever config.yaml says today.
    """
    box = settings["bowl_crop"]
    if box is None:
        return None
    return (box["x"], box["y"], box["width"], box["height"])
