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
        s1,
        s2,
        open_penalty=open_penalty,
        extend_penalty=extend_penalty,
        processor=processor,
        score_cutoff=score_cutoff,
    )


def similarity(
    s1: Any,
    s2: Any,
    *,
    open_penalty: float = 1.0,
    extend_penalty: float = 0.5,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> float:
    """Calculates the Gotoh similarity between two strings with affine gap penalties."""
    return _rustfuzz.gotoh_similarity(
        s1,
        s2,
        open_penalty=open_penalty,
        extend_penalty=extend_penalty,
        processor=processor,
        score_cutoff=score_cutoff,
    )


def normalized_distance(
    s1: Any,
    s2: Any,
    *,
    open_penalty: float = 1.0,
    extend_penalty: float = 0.5,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> float:
    """Calculates the normalized Gotoh distance (0.0–1.0) between two strings."""
    return _rustfuzz.gotoh_normalized_distance(
        s1,
        s2,
        open_penalty=open_penalty,
        extend_penalty=extend_penalty,
        processor=processor,
        score_cutoff=score_cutoff,
    )


def normalized_similarity(
    s1: Any,
    s2: Any,
    *,
    open_penalty: float = 1.0,
    extend_penalty: float = 0.5,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> float:
    """Calculates the normalized Gotoh similarity (0.0–1.0) between two strings."""
    return _rustfuzz.gotoh_normalized_similarity(
        s1,
        s2,
        open_penalty=open_penalty,
        extend_penalty=extend_penalty,
        processor=processor,
        score_cutoff=score_cutoff,
    )


__all__ = ["distance", "similarity", "normalized_distance", "normalized_similarity"]
