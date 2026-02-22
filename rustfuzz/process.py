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


def _resolve_scorer(
    scorer: Callable[..., float] | None,
) -> tuple[str, Callable[..., float] | None]:
    """Resolve a scorer callable to ``(scorer_name, scorer_obj)``.

    When the scorer is one of the built-in Rust implementations the returned
    ``scorer_obj`` is ``None`` so Rust can take the zero-overhead native path.
    For any other callable the object itself is returned so Python can invoke it.
    """
    from . import fuzz

    _scorer = scorer if scorer is not None else fuzz.WRatio
    scorer_name = getattr(_scorer, "__name__", "unknown").lower()
    scorer_obj: Callable[..., float] | None = (
        None if scorer_name in _NATIVE_SCORER_NAMES else _scorer
    )
    return scorer_name, scorer_obj


def extract(
    query: Any,
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    limit: int | None = 5,
    score_cutoff: float | None = None,
) -> list[tuple[Any, float, int]]:
    """Return the best matches from *choices* for *query*.

    Note
    ----
    The Rayon-parallel fast path only activates for ASCII strings.  Non-ASCII
    input falls back to a single-threaded Python iterator path automatically.
    """
    from . import _rustfuzz

    scorer_name, scorer_obj = _resolve_scorer(scorer)

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
    limit: int | None = 5,
    score_cutoff: float | None = 0.0,
) -> list[tuple[Any, float, int]]:
    """Return all matches with score >= score_cutoff, sorted by score descending.

    Defaults match rapidfuzz: limit=5, score_cutoff=0.0.
    """
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
    from . import _rustfuzz

    scorer_name, scorer_obj = _resolve_scorer(scorer)

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
    from . import _rustfuzz

    scorer_name, scorer_obj = _resolve_scorer(scorer)

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
    """Compute a pairwise distance matrix. Requires numpy.

    Note
    ----
    *workers* is accepted for API compatibility with rapidfuzz but is currently
    ignored — Rust always uses the full Rayon thread pool.  Pass ``workers=1``
    to signal single-threaded intent (no effect yet).
    """
    try:
        import numpy as np
    except ImportError as e:
        msg = "cdist requires numpy: pip install rustfuzz[all]"
        raise ImportError(msg) from e

    from . import _rustfuzz

    scorer_name, scorer_obj = _resolve_scorer(scorer)

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
    max_edits: int = 2,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
) -> list[Any]:
    """Deduplicate *choices* using a BK-Tree.

    Parameters
    ----------
    max_edits:
        Maximum allowed Levenshtein edit distance between two strings for them
        to be considered duplicates.  **This is an absolute edit count, not a
        percentage score.**  Default is 2.

    Note
    ----
    The *scorer* and *processor* parameters are accepted for API surface
    compatibility but are not used — deduplication always uses Levenshtein
    distance.
    """
    from . import _rustfuzz

    bktree = _rustfuzz.BKTree()
    return bktree.dedupe(list(choices), max_edits)


__all__ = ["extract", "extractBests", "extractOne", "extract_iter", "cdist", "dedupe"]
