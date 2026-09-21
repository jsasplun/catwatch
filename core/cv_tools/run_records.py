# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Recording exactly what produced each experiment, so results are
reproducible."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def new_run_directory(root: Path, tag: str) -> Path:
    """Create a fresh folder like root/20260919_143015_seed0 for one
    experiment."""
    path = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    """A fingerprint of a file's exact contents.

    Changing even one byte changes the fingerprint, so storing it with a model
    proves which version of, say, the labels that model was trained on.
    """
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def current_git_commit() -> str:
    """The git commit the code is at, plus "-dirty" if there are uncommitted
    edits."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git repository)"
    return f"{commit}-dirty" if changes else commit
