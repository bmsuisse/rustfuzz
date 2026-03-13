"""Shared helpers for the search sub-package."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .._types import MetaResult, Result


def _enrich(
    results: list[Result],
    corpus: list[str],
    metadata: list[Any] | None,
    corpus_index: dict[str, int] | None,
) -> list[Result] | list[MetaResult]:
    """Attach metadata to result tuples when metadata is available."""
    if metadata is None or corpus_index is None:
        return results
    enriched: list[MetaResult] = []
    for text, score in results:
        idx = corpus_index.get(text)
        meta = metadata[idx] if idx is not None else None
        enriched.append((text, score, meta))
    return enriched


def _build_corpus_index(corpus: list[str]) -> dict[str, int]:
    """Build a reverse lookup from document text → index."""
    index: dict[str, int] = {}
    for i, doc in enumerate(corpus):
        index[doc] = i
    return index


def _validate_metadata(
    metadata: Iterable[Any] | None, corpus_len: int
) -> list[Any] | None:
    """Validate and convert metadata to a list, checking length."""
    if metadata is None:
        return None
    meta_list = list(metadata)
    if len(meta_list) != corpus_len:
        raise ValueError(
            f"metadata length ({len(meta_list)}) must match corpus length ({corpus_len})"
        )
    return meta_list


def _blend_reranked_scores(
    docs: list[str],
    candidates: list[Result],
    rerank_scores: list[float],
    blend_alpha: float,
) -> list[float]:
    """Blend BM25 rank-based scores with reranker scores."""
    bm25_scores = {text: 1.0 / (rank + 1) for rank, (text, _) in enumerate(candidates)}
    min_s = min(rerank_scores)
    max_s = max(rerank_scores)
    rng = max_s - min_s + 1e-10
    blended: list[float] = []
    for doc, r_s in zip(docs, rerank_scores, strict=False):
        b_s = bm25_scores.get(doc, 0.0)
        norm_r = (r_s - min_s) / rng
        blended.append(blend_alpha * b_s + (1.0 - blend_alpha) * norm_r)
    return blended
