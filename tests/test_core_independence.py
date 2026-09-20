# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Enforces the rule that core/ never depends on project-specific code."""

import ast
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent.parent / "core"
FORBIDDEN_PACKAGES = {"catwatch"}


def test_core_never_imports_project_code() -> None:
    for source_file in CORE_DIR.rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            for module in imported:
                top_level = module.split(".")[0]
                assert top_level not in FORBIDDEN_PACKAGES, (
                    f"{source_file.name} imports {module}"
                )
