from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rustfuzz import _rustfuzz


def sorensen_dice(
    s1: Any,
    s2: Any,
    *,
    n: int = 2,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> float:
    """Calculates the Sorensen-Dice coefficient between two strings based on n-grams."""
    return _rustfuzz.sorensen_dice(
        s1, s2, n=n, processor=processor, score_cutoff=score_cutoff
    )

def jaccard(
    s1: Any,
    s2: Any,
    *,
    n: int = 2,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> float:
    """Calculates the Jaccard similarity between two strings based on n-grams."""
    return _rustfuzz.jaccard(
        s1, s2, n=n, processor=processor, score_cutoff=score_cutoff
    )

__all__ = ["sorensen_dice", "jaccard"]
