from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rustfuzz import _rustfuzz


def distance(
    s1: Any,
    s2: Any,
    *,
    open_penalty: float = 1.0,
    extend_penalty: float = 0.5,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> float:
    """Calculates the Gotoh distance between two strings with affine gap penalties."""
    return _rustfuzz.gotoh_distance(
        s1, s2, open_penalty=open_penalty, extend_penalty=extend_penalty,
        processor=processor, score_cutoff=score_cutoff
    )

__all__ = ["distance"]
