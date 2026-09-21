# Author: John Asplund
# Date Created: 9/21/26
# AI tool: Claude Sonnet 5

"""Enforces the Pi-side import constraint from AGENTS.md.

The Raspberry Pi has no PyTorch, torchvision, scikit-learn, or onnx package.
If a Pi-side script imports one of them, directly or through another module,
it works on the desktop and crashes on the Pi. That can't be noticed here
unless we look at what actually gets loaded.

Each check runs in a fresh subprocess. Inside pytest, torch is often already
imported by another test, which would hide (or fake) a violation.
"""

from __future__ import annotations
import sys
from pathlib import Path
# Adds the parent directory of this file to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import subprocess

import pytest

from catwatch.settings import PROJECT_ROOT

# Not installed on the Pi. "onnx" is the model-building package; onnxruntime,
# which the Pi does have, is a different module and is allowed.
DESKTOP_ONLY_MODULES = ["torch", "torchvision", "sklearn", "onnx"]

# picamera2/libcamera exist only on the Pi, so importing them at load time
# would break every desktop test run. They must be imported lazily.
PI_ONLY_MODULES = ["picamera2", "libcamera"]

PI_SIDE_MODULES = [
    "catwatch.monitor",
    "catwatch.collect",
    "catwatch.check_crop",
    "catwatch.hardware",
    "catwatch.data",
    "catwatch.settings",
    "core.cv_tools.camera",
    "core.cv_tools.motion",
    "core.cv_tools.csv_log",
    "core.cv_tools.capture_store",
    "core.cv_tools.labels",
    "core.cv_tools.splits",
    "core.cv_tools.preprocessing",
    "core.cv_tools.onnx_classifier",
    "core.cv_tools.events",
    "core.cv_tools.run_records",
]


def modules_loaded_by(module_name: str) -> set[str]:
    """Import one module in a clean interpreter; report every top-level
    package that ended up loaded."""
    code = (
        "import json, sys\n"
        f"import {module_name}\n"
        "print(json.dumps(sorted({name.split('.')[0] for name in sys.modules})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout))


@pytest.mark.parametrize("module_name", PI_SIDE_MODULES)
def test_pi_side_module_imports_nothing_desktop_only(module_name: str) -> None:
    loaded = modules_loaded_by(module_name)
    assert not loaded & set(
        DESKTOP_ONLY_MODULES
    ), f"{module_name} pulls in {sorted(loaded & set(DESKTOP_ONLY_MODULES))}"


@pytest.mark.parametrize("module_name", PI_SIDE_MODULES)
def test_pi_side_module_does_not_load_pi_camera_libraries_eagerly(
    module_name: str,
) -> None:
    loaded = modules_loaded_by(module_name)
    assert not loaded & set(PI_ONLY_MODULES)


def test_the_check_itself_can_detect_a_violation() -> None:
    # core.cv_tools.evaluation is desktop-only by design. If this fails, the
    # subprocess check above is blind and its passes mean nothing.
    assert "torch" in modules_loaded_by("core.cv_tools.evaluation")
