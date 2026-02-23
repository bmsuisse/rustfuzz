"""
rustfuzz.process — batch matching and extraction utilities.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

# Scorers implemented natively in Rust — pass scorer_obj=None to activate the native fast path.
_NATIVE_SCORER_NAMES = {
    "ratio",
    "qratio",
    "wratio",
    "partial_ratio",
    "token_sort_ratio",
    "partial_token_sort_ratio",
    "token_set_ratio",
    "partial_token_set_ratio",
    "token_ratio",
    "partial_token_ratio",
}


def extract(
    query: Any,
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    limit: int | None = 5,
    score_cutoff: float | None = None,
) -> list[tuple[Any, float, int]]:
    from . import _rustfuzz, fuzz

    _scorer = scorer if scorer is not None else fuzz.WRatio
    scorer_name = getattr(_scorer, "__name__", "unknown").lower()
    # For built-in scorers pass None so Rust can use the native zero-overhead path
    scorer_obj = None if scorer_name in _NATIVE_SCORER_NAMES else _scorer

    return _rustfuzz.extract(
        query,
        choices,
        scorer_name,
        scorer_obj,
        processor,
        limit,
        score_cutoff,
    )


def extractBests(
    query: Any,
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    limit: int | None = None,
    score_cutoff: float | None = 0.0,
) -> list[tuple[Any, float, int]]:
    """Returns all matches with score >= score_cutoff, sorted by score."""
    return extract(
        query,
        choices,
        scorer=scorer,
        processor=processor,
        limit=limit,
        score_cutoff=score_cutoff,
    )


def extractOne(
    query: Any,
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> tuple[Any, float, int] | None:
    from . import _rustfuzz, fuzz

    _scorer = scorer if scorer is not None else fuzz.WRatio
    scorer_name = getattr(_scorer, "__name__", "unknown").lower()
    scorer_obj = None if scorer_name in _NATIVE_SCORER_NAMES else _scorer

    return _rustfuzz.extract_one(
        query,
        choices,
        scorer_name,
        scorer_obj,
        processor,
        score_cutoff,
    )


def extract_iter(
    query: Any,
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> Iterator[tuple[Any, float, int]]:
    from . import _rustfuzz, fuzz

    _scorer = scorer if scorer is not None else fuzz.WRatio
    scorer_name = getattr(_scorer, "__name__", "unknown").lower()
    scorer_obj = None if scorer_name in _NATIVE_SCORER_NAMES else _scorer

    yield from _rustfuzz.extract_iter(
        query,
        choices,
        scorer_name,
        scorer_obj,
        processor,
        score_cutoff,
    )


def cdist(
    queries: Iterable[Any],
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
    dtype: Any = None,
    workers: int = 1,
) -> Any:
    """Compute a pairwise distance matrix. Requires numpy."""
    try:
        import numpy as np
    except ImportError as e:
        msg = "cdist requires numpy: pip install rustfuzz[all]"
        raise ImportError(msg) from e

    from . import _rustfuzz, fuzz

    _scorer = scorer if scorer is not None else fuzz.WRatio
    scorer_name = getattr(_scorer, "__name__", "unknown").lower()
    scorer_obj = None if scorer_name in _NATIVE_SCORER_NAMES else _scorer

    flat_array, rows, cols = _rustfuzz.cdist(
        queries,
        choices,
        scorer_name,
        scorer_obj,
        processor,
        score_cutoff,
    )

    # Reshape the flat array returned from Rust into a 2D numpy matrix
    matrix = np.array(flat_array, dtype=dtype if dtype is not None else np.float32)
    return matrix.reshape((rows, cols))


def dedupe(
    choices: Iterable[Any],
    *,
    threshold: int = 2,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
) -> list[Any]:
    """
    Deduplicate a list of choices using a BK-Tree.
    Note: Threshold is absolute Levenshtein distance (max allowed edits),
    default 2. Not a percentage score.
    """
    from . import _rustfuzz
    bktree = _rustfuzz.BKTree()
    return bktree.dedupe(list(choices), threshold)

__all__ = ["extract", "extractBests", "extractOne", "extract_iter", "cdist", "dedupe"]
