"""HybridSearch — 3-way RRF combining BM25, fuzzy, and dense vectors."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

from .. import _rustfuzz
from .._types import MetaResult, Result, _search_query
from ..compat import _coerce_to_strings
from ..document import Document
from ._bm25 import BM25Algorithm
from ._helpers import _build_corpus_index, _enrich, _validate_metadata


def _coerce_corpus(
    corpus: Iterable[str] | Iterable[Any] | Any,
) -> tuple[list[str], list[Any] | None]:
    """
    Accept multiple input formats and return (texts, metadata_or_None).

    Supported inputs:
    - list[str]                      → texts, None
    - list[Document]                 → texts, metadata list
    - list[LangChainDocument]        → texts (from page_content), metadata list
    - Polars/Pandas Series, PyArrow  → texts (via _coerce_to_strings), None
    """
    items = list(corpus)
    if not items:
        return [], None

    first = items[0]

    # Our own Document
    if isinstance(first, Document):
        texts = [d.content for d in items]
        meta = [d.metadata for d in items]
        return texts, meta

    # LangChain Document (duck-type: has page_content attribute)
    if hasattr(first, "page_content"):
        texts = [d.page_content for d in items]  # type: ignore[union-attr]
        meta = [getattr(d, "metadata", {}) for d in items]
        return texts, meta

    # Plain strings or Series/arrays → delegate to existing coercion
    return _coerce_to_strings(items), None


class HybridSearch:
    """
    Tier-3 Semantic Hybrid Search framework — 3-way RRF in Rust.

    Fuses BM25 text retrieval, fuzzy string matching, and dense vector
    cosine similarity via Reciprocal Rank Fusion. All heavy computation
    runs in Rust outside the Python GIL — designed for million-scale corpora.

    Parameters
    ----------
    corpus : Iterable[str] | Iterable[Document] | Any
        Text documents.
    embeddings : matrix-like | Callable[[list[str]], list[list[float]]] | None
        Dense vectors for the corpus.
    k1 : float, default 1.5
        BM25 parameter
    b : float, default 0.75
        BM25 parameter
    metadata : Iterable[Any] | None, default None
        Optional per-document metadata.
    """

    def __init__(
        self,
        corpus: Iterable[str] | Iterable[Any] | Any,
        embeddings: Any | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        algorithm: BM25Algorithm = "bm25",
        delta: float | None = None,
        normalize_scores: bool = False,
        metadata: Iterable[Any] | None = None,
    ) -> None:
        # ── Coerce corpus (handles str, Document, LangChain) ──
        texts, auto_metadata = _coerce_corpus(corpus)
        self._corpus = texts
        self._embeddings = embeddings
        self._normalize_scores = normalize_scores
        self._k1 = k1
        self._b = b
        self._algorithm = algorithm
        self._delta = delta

        # Explicit metadata overrides auto-extracted metadata
        if metadata is not None:
            self._metadata = _validate_metadata(metadata, len(self._corpus))
        elif auto_metadata is not None:
            self._metadata = auto_metadata
        else:
            self._metadata = None

        self._corpus_index: dict[str, int] | None = (
            _build_corpus_index(self._corpus) if self._metadata is not None else None
        )

        # ── Handle embeddings: callback or static matrix ──
        self._embed_fn: Callable[[list[str]], list[list[float]]] | None = None
        emb_list: list[list[float]] | None = None

        if callable(embeddings):
            embed_fn = cast(Callable[[list[str]], list[list[float]]], embeddings)
            self._embed_fn = embed_fn
            if self._corpus:
                emb_list = embed_fn(self._corpus)
                if len(emb_list) != len(self._corpus):
                    raise ValueError(
                        f"Embedding callback returned {len(emb_list)} vectors "
                        f"for {len(self._corpus)} documents"
                    )
        elif embeddings is not None:
            try:
                if hasattr(embeddings, "shape") and hasattr(embeddings, "tolist"):
                    emb_list = embeddings.tolist()
                else:
                    emb_list = [list(row) for row in embeddings]
            except Exception as e:
                raise ValueError(
                    "embeddings must be convertible to a list of lists of floats"
                ) from e
            if len(emb_list) != len(self._corpus):
                raise ValueError(
                    f"Length mismatch: {len(self._corpus)} documents, "
                    f"{len(emb_list)} embeddings"
                )

        # Keep a Python-side reference for pickle support
        self._embeddings = emb_list

        # ── Build Rust index ──
        self._index = _rustfuzz.HybridSearchIndex(
            self._corpus, emb_list, k1, b, algorithm, delta
        )

        # ── Push metadata to Rust for blazing-fast filter evaluation ──
        if self._metadata is not None:
            import json

            json_strings = [
                json.dumps(m if m is not None else {}) for m in self._metadata
            ]
            self._index.set_metadata_json(json_strings)

    @property
    def has_vectors(self) -> bool:
        """Whether dense embeddings are available."""
        return self._index.has_vectors

    @property
    def num_docs(self) -> int:
        """Number of documents in the index."""
        return self._index.num_docs

    def search(
        self,
        query: str,
        query_embedding: Any = None,
        n: int = 5,
        rrf_k: int = 60,
        bm25_candidates: int = 100,
    ) -> list[Result] | list[MetaResult]:
        """Run 3-way hybrid search: BM25 + Fuzzy + Dense via RRF."""
        q_emb: list[float] | None = None
        if query_embedding is not None:
            if hasattr(query_embedding, "tolist"):
                q_emb = query_embedding.tolist()
            else:
                q_emb = list(query_embedding)
        elif self._embed_fn is not None:
            q_emb = list(self._embed_fn([query])[0])

        results = self._index.search(query, q_emb, n, rrf_k, bm25_candidates)
        return _enrich(results, self._corpus, self._metadata, self._corpus_index)

    # ── Fluent query builder ──────────────────────────────

    def filter(self, expression: str) -> Any:
        """Start a filtered query chain."""
        return _search_query(self).filter(expression)

    def sort(self, expression: list[str] | str) -> Any:
        """Start a sorted query chain."""
        return _search_query(self).sort(expression)

    def match(self, query: str, **kwargs: Any) -> Any:
        """Execute a text search with the query builder."""
        return _search_query(self).match(query, **kwargs)

    def rerank(
        self,
        model_or_callable: Any,
        top_k: int = 10,
        blend_alpha: float = 0.0,
        adaptive_blend: bool = False,
    ) -> Any:
        """Add a Reranker and start a query builder chain."""
        return _search_query(self).rerank(
            model_or_callable,
            top_k=top_k,
            blend_alpha=blend_alpha,
            adaptive_blend=adaptive_blend,
        )

    def __reduce__(
        self,
    ) -> tuple[
        type,
        tuple[list[str], Any, float, float, str, float | None, bool, list[Any] | None],
    ]:
        return (
            HybridSearch,
            (
                self._corpus,
                self._embeddings,
                self._k1,
                self._b,
                self._algorithm,
                self._delta,
                self._normalize_scores,
                self._metadata,
            ),
        )
