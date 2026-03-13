"""
rustfuzz._types — Shared type aliases and Protocol definitions.

Centralises type definitions and shared helpers to avoid duplication
across search, engine, query, and process modules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

# Result tuples returned by BM25 / HybridSearch / Retriever search methods
Result = tuple[str, float]
MetaResult = tuple[str, float, Any]


# ── Protocol types ────────────────────────────────────────────────────


@runtime_checkable
class ScorerProtocol(Protocol):
    """A scorer callable: (query, choice, **kwargs) → float."""

    def __call__(self, s1: Any, s2: Any, **kwargs: Any) -> float: ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """A reranker callable: (query, texts) → list[float]."""

    def __call__(self, query: str, texts: list[str]) -> list[float]: ...


@runtime_checkable
class EmbeddingCallback(Protocol):
    """An embedding callback: (texts) → list[list[float]]."""

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


# ── Shared helpers ────────────────────────────────────────────────────


def _search_query(owner: Any) -> Any:
    """Lazy import to avoid circular dependency."""
    from .search.query import SearchQuery

    return SearchQuery(owner)


# ── Scorer resolution (used by process.py) ────────────────────────────

_NATIVE_SCORER_NAMES = frozenset(
    {
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
)


def resolve_scorer(
    scorer: Callable[..., float] | None,
) -> tuple[str, Callable[..., float] | None]:
    """
    Resolve a scorer to a (name, scorer_obj) pair.

    For native Rust scorers, returns ``(name, None)`` so the Rust layer
    uses its zero-overhead fast path. For custom scorers, returns
    ``(name, scorer)`` so the Rust layer calls back into Python.

    Parameters
    ----------
    scorer : callable or None
        If None, defaults to ``fuzz.WRatio``.

    Returns
    -------
    (scorer_name, scorer_obj)
        scorer_obj is None when the scorer has a native Rust implementation.
    """
    from . import fuzz

    resolved = scorer if scorer is not None else fuzz.WRatio
    name = getattr(resolved, "__name__", "unknown").lower()
    obj = None if name in _NATIVE_SCORER_NAMES else resolved
    return name, obj
