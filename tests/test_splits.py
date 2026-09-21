# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

import sys
from collections import Counter
from pathlib import Path
# Adds the parent directory of this file to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cv_tools.splits import assign_split

FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_same_group_always_same_split() -> None:
    first = assign_split("2026-09-19_14", FRACTIONS)
    assert all(
        assign_split("2026-09-19_14", FRACTIONS) == first for _ in range(10)
    )


def test_split_proportions_match_fractions() -> None:
    group_count = 10_000
    counts = Counter(
        assign_split(f"group-{i}", FRACTIONS) for i in range(group_count)
    )
    for name, fraction in FRACTIONS.items():
        assert abs(counts[name] / group_count - fraction) < 0.02
