# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Deterministic train/validation/test assignment."""

from __future__ import annotations

import hashlib
import math


def assign_split(group_id: str, fractions: dict[str, float]) -> str:
    """Put a group of related images into a split such as train, val, or test.

    Why groups instead of single images: photos taken seconds apart are nearly
    identical. If one lands in training and its near-twin in testing, the test
    checks memory rather than skill. Keeping a whole group together prevents
    that leak.

    Why hashing instead of random shuffling: the same group always lands in the
    same split, on any computer, and adding new data never moves old images
    between splits. With a reshuffle, a test image today could be a training
    image tomorrow, and test scores would quietly stop meaning anything.

    The group name is turned into a SHA-256 hash (a fixed pseudo-random number
    derived from the text), scaled into [0, 1), and compared against the
    cumulative fractions, e.g. train below 0.70, val below 0.85, test above.
    """
    if not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"Split fractions must add up to 1, got {fractions}")
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for split_name, fraction in fractions.items():
        cumulative += fraction
        if position < cumulative:
            return split_name
    return list(fractions)[-1]  # guards a floating-point rounding edge case
