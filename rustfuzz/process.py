"""
rustfuzz.process — batch matching and extraction utilities.

extract, extractOne, extract_iter, cdist and cpdist will be implemented
in Rust (process.rs) in a follow-up. Stubs are provided for API compatibility.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator


# Scorers implemented natively in Rust — pass scorer_obj=None to activate the native fast path.
_NATIVE_SCORER_NAMES = {
    "ratio", "qratio", "wratio", "partial_ratio",
    "token_sort_ratio", "partial_token_sort_ratio",
    "token_set_ratio", "partial_token_set_ratio",
    "token_ratio", "partial_token_ratio",
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
    from . import _rustfuzz
    from . import fuzz

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


def extractOne(
    query: Any,
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> tuple[Any, float, int] | None:
    from . import _rustfuzz
    from . import fuzz

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
    from . import _rustfuzz
    from . import fuzz

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

    from . import fuzz

    _scorer = scorer if scorer is not None else fuzz.WRatio
    _proc = processor
    q_list = list(queries)
    c_list = list(choices)
    if _proc:
        q_list = [_proc(q) for q in q_list]
        c_list = [_proc(c) for c in c_list]
    matrix = np.zeros((len(q_list), len(c_list)), dtype=np.float32)
    for i, q in enumerate(q_list):
        for j, c in enumerate(c_list):
            score = _scorer(q, c)
            matrix[i, j] = score if score_cutoff is None or score >= score_cutoff else 0.0
    return matrix


__all__ = ["extract", "extractOne", "extract_iter", "cdist"]
