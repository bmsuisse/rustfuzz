"""
rapidfuzz.process — batch matching and extraction utilities.

extract, extractOne, extract_iter, cdist and cpdist will be implemented
in Rust (process.rs) in a follow-up. Stubs are provided for API compatibility.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator


def extract(
    query: Any,
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    limit: int | None = 5,
    score_cutoff: float | None = None,
) -> list[tuple[Any, float, int]]:
    """Return the top `limit` matches from `choices` for `query`."""
    from . import fuzz  # local import to avoid circular

    _scorer = scorer if scorer is not None else fuzz.WRatio
    _proc = processor
    results: list[tuple[Any, float, int]] = []
    for idx, choice in enumerate(choices):
        c = _proc(choice) if _proc else choice
        q = _proc(query) if _proc else query
        score = _scorer(q, c)
        if score_cutoff is None or score >= score_cutoff:
            results.append((choice, score, idx))
    results.sort(key=lambda x: x[1], reverse=True)
    if limit is not None:
        results = results[:limit]
    return results


def extractOne(
    query: Any,
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> tuple[Any, float, int] | None:
    """Return the single best match or None."""
    results = extract(
        query,
        choices,
        scorer=scorer,
        processor=processor,
        limit=1,
        score_cutoff=score_cutoff,
    )
    return results[0] if results else None


def extract_iter(
    query: Any,
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[..., Any] | None = None,
    score_cutoff: float | None = None,
) -> Iterator[tuple[Any, float, int]]:
    """Iterate over all matches meeting `score_cutoff`."""
    from . import fuzz

    _scorer = scorer if scorer is not None else fuzz.WRatio
    _proc = processor
    for idx, choice in enumerate(choices):
        c = _proc(choice) if _proc else choice
        q = _proc(query) if _proc else query
        score = _scorer(q, c)
        if score_cutoff is None or score >= score_cutoff:
            yield (choice, score, idx)


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
        msg = "cdist requires numpy: pip install rapidfuzz[all]"
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
