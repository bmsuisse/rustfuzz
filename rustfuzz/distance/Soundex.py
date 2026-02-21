from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rustfuzz import _rustfuzz


def distance(
    s1: Any,
    s2: Any,
    *,
    processor: Callable[..., Any] | None = None,
    score_cutoff: int | None = None,
) -> int:
    """
    Calculates the Soundex distance between two strings.
    """
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    dist = _rustfuzz.soundex_distance(s1, s2)
    return dist if score_cutoff is None or dist <= score_cutoff else score_cutoff + 1


def similarity(
    s1: Any,
    s2: Any,
    *,
    processor: Callable[..., Any] | None = None,
    score_cutoff: int | None = None,
) -> int:
    """
    Calculates the Soundex similarity between two strings.
    """
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    sim = _rustfuzz.soundex_similarity(s1, s2)
    return sim if score_cutoff is None or sim >= score_cutoff else 0


def normalized_distance(
    s1: Any,
    s2: Any,
    *,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> float:
    """
    Calculates the normalized Soundex distance between two strings.
    """
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    dist = _rustfuzz.soundex_normalized_distance(s1, s2)
    return dist if score_cutoff is None or dist <= score_cutoff else 1.0


def normalized_similarity(
    s1: Any,
    s2: Any,
    *,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> float:
    """
    Calculates the normalized Soundex similarity between two strings.
    """
    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    sim = _rustfuzz.soundex_normalized_similarity(s1, s2)
    return sim if score_cutoff is None or sim >= score_cutoff else 0.0


def encode(s: Any, *, processor: Callable[..., Any] | None = None) -> str:
    """
    Encodes a string using the Soundex algorithm.
    """
    if processor is not None:
        s = processor(s)

    # Needs a small rust helper if we want to expose this directly, or we can just
    # compute it here if we add `soundex_encode` to `_rustfuzz`
    return _rustfuzz.soundex_encode(s)


__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
    "encode",
]
