"""
rustfuzz.search — Full-text IR, BM25, and Hybrid Search capabilities.

Provides BM25 variants backed by Rust for fast indexing and scoring,
and the HybridSearch class which fuses text search with vector embeddings
using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

# Re-export Document for backward compat (was re-exported from old search.py)
from ..document import Document  # noqa: F401

# Re-export all public symbols from submodules so that
# `from rustfuzz.search import BM25` continues to work.
from ._bm25 import (
    BM25,
    BM25L,
    BM25T,
    BM25Algorithm,
    BM25Plus,
    _BaseBM25,
    _MetaResult,
    _Result,
)
from ._helpers import _blend_reranked_scores, _build_corpus_index, _enrich
from ._hybrid import HybridSearch, _coerce_corpus
from ._reranker import Reranker

__all__ = [
    "BM25",
    "BM25Algorithm",
    "BM25L",
    "BM25Plus",
    "BM25T",
    "Document",
    "HybridSearch",
    "Reranker",
    "_BaseBM25",
    "_MetaResult",
    "_Result",
    "_blend_reranked_scores",
    "_build_corpus_index",
    "_coerce_corpus",
    "_enrich",
]
