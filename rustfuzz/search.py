"""
rustfuzz.search — Full-text IR, BM25, and Hybrid Search capabilities.

Provides the BM25 class backed by Rust for fast indexing and scoring,
and the HybridSearch class which fuses text search with vector embeddings
using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

from . import _rustfuzz
from .compat import _coerce_to_strings, _extract_column, _extract_metadata
from .document import Document  # noqa: F401 — re-export for backward compat

# Type aliases for result tuples
_Result = tuple[str, float]
_MetaResult = tuple[str, float, Any]


# Lazy import to avoid circular dependency
def _search_query(owner: Any) -> Any:
    from .query import SearchQuery

    return SearchQuery(owner)


def _enrich(
    results: list[_Result],
    corpus: list[str],
    metadata: list[Any] | None,
    corpus_index: dict[str, int] | None,
) -> list[_Result] | list[_MetaResult]:
    """Attach metadata to result tuples when metadata is available."""
    if metadata is None or corpus_index is None:
        return results
    enriched: list[_MetaResult] = []
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


class BM25:
    """
    BM25Okapi full-text search index.

    The index is built eagerly during initialization in Rust.
    Term frequencies and inverse document frequencies are pre-calculated for speed.

    Parameters
    ----------
    corpus : Iterable[str]
        Documents to index.
    k1 : float, default 1.5
        Term frequency saturation parameter.
    b : float, default 0.75
        Length normalisation factor.
    metadata : Iterable[Any] | None, default None
        Optional per-document metadata. When provided, search results become
        ``(text, score, metadata)`` triples instead of ``(text, score)`` pairs.
    normalize : bool, default False
        Whether to normalize the BM25 scores.
    """

    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
    ):
        corpus_list = _coerce_to_strings(corpus)
        self._corpus = corpus_list
        self._k1 = k1
        self._b = b
        self._normalize = normalize
        self._metadata = _validate_metadata(metadata, len(corpus_list))
        self._corpus_index: dict[str, int] | None = (
            _build_corpus_index(corpus_list) if self._metadata is not None else None
        )
        self._index = _rustfuzz.BM25Index(corpus_list, k1, b, normalize)

    @classmethod
    def from_column(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> BM25:
        """Build a BM25 index from a DataFrame column (Spark, Polars, Pandas).

        Parameters
        ----------
        df : DataFrame
            Source DataFrame.
        column : str
            Name of the text column to index.
        metadata_columns : list[str] | str | None, default None
            Column(s) to extract as per-row metadata dicts.
        """
        meta = _extract_metadata(df, metadata_columns) if metadata_columns else None
        return cls(_extract_column(df, column), metadata=meta, **kwargs)

    @property
    def num_docs(self) -> int:
        """Number of documents in the index."""
        return self._index.num_docs

    def get_scores(self, query: str) -> list[float]:
        """
        Return the BM25 score for the query against every document in the corpus.
        Returns a flat list of scores in document order.
        """
        return self._index.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> list[_Result] | list[_MetaResult]:
        """
        Return the top N matching documents and their BM25 scores.
        Only documents with score > 0.0 are returned (up to n).
        """
        return _enrich(
            self._index.get_top_n(query, n),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def get_batch_scores(self, queries: Iterable[str]) -> list[list[float]]:
        """
        Return BM25 scores for a batch of queries against all documents.
        This is parallelised over queries via Rayon.
        """
        return self._index.get_batch_scores(list(queries))

    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[_Result] | list[_MetaResult]:
        """
        Hybrid search combining BM25 Okapi with Levenshtein-based fuzzy matching.

        This drastically improves retrieval for misspellings relative to pure BM25.
        It runs BM25 first (very fast), takes top `bm25_candidates`, then recalculates
        a combined score: (1 - fuzzy_weight) * norm(bm25) + fuzzy_weight * fuzzy_ratio

        Parameters
        ----------
        query : str
        n : int, default 5
            Number of documents to return.
        bm25_candidates : int, default 50
            Number of candidates to extract from the BM25 stage to pass to the fuzzy stage.
        fuzzy_weight : float, default 0.3
            Weight applied to the Levenshtein fuzzy string similarity ratio [0.0 - 1.0].
        """
        return _enrich(
            self._index.get_top_n_fuzzy(query, n, bm25_candidates, fuzzy_weight),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def get_top_n_rrf(
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[_Result] | list[_MetaResult]:
        """
        Hybrid search combining BM25 and Levenshtein distance through Reciprocal Rank Fusion (RRF).

        This is generally more robust than `get_top_n_fuzzy` because it uses RRF,
        shielding the combined metric from score-scale variances.
        """
        return _enrich(
            self._index.get_top_n_rrf(query, n, bm25_candidates, rrf_k),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def fuzzy_only(
        self,
        query: str,
        n: int = 5,
    ) -> list[_Result] | list[_MetaResult]:
        """
        Pure fuzzy string ranking over the indexed corpus (no BM25 scores).

        Ranks all documents by WRatio fuzzy similarity to `query` and returns the
        top `n` results. Useful when the query is very short or misspelled.

        Parameters
        ----------
        query : str
        n : int, default 5
            Number of documents to return.
        """
        return _enrich(
            self._index.fuzzy_only(query, n),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    # ── New features ──────────────────────────────────────────

    def explain(self, query: str, doc: str | int) -> dict[str, Any]:
        """
        Per-term BM25 score breakdown for a query against a specific document.

        Parameters
        ----------
        query : str
            Query string.
        doc : str | int
            Either a document index (int) or a document string.

        Returns
        -------
        dict
            Keys: ``terms`` (list of per-term dicts with idf/tf/score),
            ``total_score``, ``doc_idx``, ``doc_text``.
        """
        if isinstance(doc, str):
            idx = self._corpus.index(doc)
        else:
            idx = doc
        breakdown = self._index.explain(query, idx)
        terms = [
            {"term": t, "idf": idf, "tf_norm": tf, "score": s}
            for t, idf, tf, s in breakdown
        ]
        total = sum(float(d["score"]) for d in terms)
        return {
            "terms": terms,
            "total_score": total,
            "doc_idx": idx,
            "doc_text": self._corpus[idx],
        }

    def get_idf(self) -> dict[str, float]:
        """Return the entire IDF map: term → IDF value."""
        return dict(self._index.get_idf_map())

    def get_document_vector(self, doc_idx: int) -> dict[str, float]:
        """Return the normalised TF vector for a specific document."""
        return dict(self._index.get_document_vector(doc_idx))

    def add_documents(
        self,
        docs: Iterable[str],
        metadata: Iterable[Any] | None = None,
    ) -> None:
        """
        Add documents to the index (rebuilds internally).

        Parameters
        ----------
        docs : Iterable[str]
            New documents to add.
        metadata : Iterable[Any] | None
            Optional metadata for the new documents.
        """
        new_docs = list(docs)
        self._corpus.extend(new_docs)
        if metadata is not None:
            new_meta = list(metadata)
            if len(new_meta) != len(new_docs):
                raise ValueError("metadata length must match docs length")
            if self._metadata is None:
                self._metadata = [None] * (len(self._corpus) - len(new_docs))
            self._metadata.extend(new_meta)
        elif self._metadata is not None:
            self._metadata.extend([None] * len(new_docs))
        self._index = _rustfuzz.BM25Index(
            self._corpus, self._k1, self._b, self._normalize
        )
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    def remove_documents(self, indices: list[int]) -> None:
        """
        Remove documents by index (rebuilds internally).

        Parameters
        ----------
        indices : list[int]
            Indices of documents to remove.
        """
        to_remove = set(indices)
        self._corpus = [d for i, d in enumerate(self._corpus) if i not in to_remove]
        if self._metadata is not None:
            self._metadata = [
                m for i, m in enumerate(self._metadata) if i not in to_remove
            ]
        self._index = _rustfuzz.BM25Index(
            self._corpus, self._k1, self._b, self._normalize
        )
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    def get_top_n_reranked(
        self,
        query: str,
        n: int = 5,
        reranker: Any = None,
        rerank_candidates: int = 50,
    ) -> list[_Result] | list[_MetaResult]:
        """
        BM25 retrieval + external reranker callback.

        Parameters
        ----------
        query : str
        n : int
            Final number of results.
        reranker : Callable[[str, list[str]], list[float]] | None
            Function ``(query, docs) -> scores``. If None, falls back to BM25.
        rerank_candidates : int
            How many BM25 candidates to pass to the reranker.
        """
        if reranker is None:
            return self.get_top_n(query, n)

        candidates = self._index.get_top_n(query, rerank_candidates)
        if not candidates:
            return []

        docs = [d for d, _ in candidates]
        rerank_scores = reranker(query, docs)
        paired = sorted(
            zip(docs, rerank_scores, strict=True), key=lambda x: x[1], reverse=True
        )[:n]
        results: list[_Result] = [(d, s) for d, s in paired]
        return _enrich(results, self._corpus, self._metadata, self._corpus_index)

    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[_Result] | list[_MetaResult]:
        """
        BM25 scoring with phrase proximity boost.

        Documents where query terms appear adjacent or within `proximity_window`
        positions get a multiplicative boost.

        Parameters
        ----------
        query : str
        n : int, default 5
        proximity_window : int, default 3
            Maximum token distance for terms to be considered "adjacent".
        phrase_boost : float, default 2.0
            Multiplicative boost factor (1.0 = no boost).
        """
        return _enrich(
            self._index.get_top_n_phrase(query, n, proximity_window, phrase_boost),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    async def get_top_n_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]:
        """Async wrapper around :meth:`get_top_n` (runs in a thread)."""
        import asyncio

        return await asyncio.to_thread(self.get_top_n, query, n, **kwargs)

    async def search_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]:
        """Async wrapper around :meth:`get_top_n_rrf`."""
        import asyncio

        return await asyncio.to_thread(self.get_top_n_rrf, query, n, **kwargs)

    # ── Fluent query builder ──────────────────────────────

    def filter(self, expression: str) -> Any:
        """Start a filtered query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).filter(expression)

    def sort(self, expression: list[str] | str) -> Any:
        """Start a sorted query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).sort(expression)

    def match(self, query: str, **kwargs: Any) -> Any:
        """Execute a text search with the query builder."""
        return _search_query(self).match(query, **kwargs)

    def rerank(self, model_or_callable: Any, top_k: int = 10) -> Any:
        """Add a Reranker and start a query builder chain."""
        return _search_query(self).rerank(model_or_callable, top_k=top_k)

    def to_hybrid(self, embeddings: Any) -> HybridSearch:
        """
        Convert this BM25 index into a full HybridSearch index by attaching
        dense embeddings. Retains the corpus, k1, b, and metadata configurations.
        """
        return HybridSearch(
            self._corpus,
            embeddings=embeddings,
            k1=self._k1,
            b=self._b,
            metadata=self._metadata,
            algorithm="bm25",
        )

    def __reduce__(
        self,
    ) -> tuple[type, tuple[list[str], float, float, list[Any] | None, bool]]:
        return (
            BM25,
            (self._corpus, self._k1, self._b, self._metadata, self._normalize),
        )


class BM25L:
    """BM25L full-text search index."""

    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
    ):
        corpus_list = _coerce_to_strings(corpus)
        self._corpus = corpus_list
        self._k1 = k1
        self._b = b
        self._delta = delta
        self._normalize = normalize
        self._metadata = _validate_metadata(metadata, len(corpus_list))
        self._corpus_index: dict[str, int] | None = (
            _build_corpus_index(corpus_list) if self._metadata is not None else None
        )
        self._index = _rustfuzz.BM25L(corpus_list, k1, b, delta, normalize)

    @classmethod
    def from_column(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> BM25L:
        """Build a BM25L index from a DataFrame column (Spark, Polars, Pandas)."""
        meta = _extract_metadata(df, metadata_columns) if metadata_columns else None
        return cls(_extract_column(df, column), metadata=meta, **kwargs)

    @property
    def num_docs(self) -> int:
        return self._index.num_docs

    def get_scores(self, query: str) -> list[float]:
        return self._index.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n(query, n),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def get_batch_scores(self, queries: Iterable[str]) -> list[list[float]]:
        return self._index.get_batch_scores(list(queries))

    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n_fuzzy(query, n, bm25_candidates, fuzzy_weight),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def get_top_n_rrf(
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n_rrf(query, n, bm25_candidates, rrf_k),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def fuzzy_only(self, query: str, n: int = 5) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.fuzzy_only(query, n),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def explain(self, query: str, doc: str | int) -> dict[str, Any]:
        """Per-term BM25 score breakdown for a query against a specific document."""
        idx = self._corpus.index(doc) if isinstance(doc, str) else doc
        breakdown = self._index.explain(query, idx)
        terms = [
            {"term": t, "idf": idf, "tf_norm": tf, "score": s}
            for t, idf, tf, s in breakdown
        ]
        return {
            "terms": terms,
            "total_score": sum(float(d["score"]) for d in terms),
            "doc_idx": idx,
            "doc_text": self._corpus[idx],
        }

    def get_idf(self) -> dict[str, float]:
        """Return the IDF map: term → IDF value."""
        return dict(self._index.get_idf_map())

    def get_document_vector(self, doc_idx: int) -> dict[str, float]:
        """Return the normalised TF vector for a specific document."""
        return dict(self._index.get_document_vector(doc_idx))

    def add_documents(
        self, docs: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None:
        """Add documents to the index (rebuilds internally)."""
        new_docs = list(docs)
        self._corpus.extend(new_docs)
        if metadata is not None:
            new_meta = list(metadata)
            if len(new_meta) != len(new_docs):
                raise ValueError("metadata length must match docs length")
            if self._metadata is None:
                self._metadata = [None] * (len(self._corpus) - len(new_docs))
            self._metadata.extend(new_meta)
        elif self._metadata is not None:
            self._metadata.extend([None] * len(new_docs))
        self._index = _rustfuzz.BM25L(
            self._corpus, self._k1, self._b, self._delta, self._normalize
        )
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    def remove_documents(self, indices: list[int]) -> None:
        """Remove documents by index (rebuilds internally)."""
        to_remove = set(indices)
        self._corpus = [d for i, d in enumerate(self._corpus) if i not in to_remove]
        if self._metadata is not None:
            self._metadata = [
                m for i, m in enumerate(self._metadata) if i not in to_remove
            ]
        self._index = _rustfuzz.BM25L(
            self._corpus, self._k1, self._b, self._delta, self._normalize
        )
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    def get_top_n_reranked(
        self, query: str, n: int = 5, reranker: Any = None, rerank_candidates: int = 50
    ) -> list[_Result] | list[_MetaResult]:
        """BM25 retrieval + external reranker callback."""
        if reranker is None:
            return self.get_top_n(query, n)
        candidates = self._index.get_top_n(query, rerank_candidates)
        if not candidates:
            return []
        docs = [d for d, _ in candidates]
        rerank_scores = reranker(query, docs)
        paired = sorted(
            zip(docs, rerank_scores, strict=True), key=lambda x: x[1], reverse=True
        )[:n]
        results: list[_Result] = [(d, s) for d, s in paired]
        return _enrich(results, self._corpus, self._metadata, self._corpus_index)

    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[_Result] | list[_MetaResult]:
        """BM25 scoring with phrase proximity boost."""
        return _enrich(
            self._index.get_top_n_phrase(query, n, proximity_window, phrase_boost),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    async def get_top_n_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]:
        """Async wrapper around get_top_n."""
        import asyncio

        return await asyncio.to_thread(self.get_top_n, query, n, **kwargs)

    async def search_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]:
        """Async wrapper around get_top_n_rrf."""
        import asyncio

        return await asyncio.to_thread(self.get_top_n_rrf, query, n, **kwargs)

    # ── Fluent query builder ──────────────────────────────

    def filter(self, expression: str) -> Any:
        """Start a filtered query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).filter(expression)

    def sort(self, expression: list[str] | str) -> Any:
        """Start a sorted query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).sort(expression)

    def to_hybrid(self, embeddings: Any) -> HybridSearch:
        """Convert this BM25L index into a HybridSearch index with dense embeddings."""
        return HybridSearch(
            self._corpus,
            embeddings=embeddings,
            k1=self._k1,
            b=self._b,
            metadata=self._metadata,
            algorithm="bm25l",
            delta=self._delta,
        )

    def __reduce__(
        self,
    ) -> tuple[type, tuple[list[str], float, float, float, list[Any] | None, bool]]:
        return (
            BM25L,
            (
                self._corpus,
                self._k1,
                self._b,
                self._delta,
                self._metadata,
                self._normalize,
            ),
        )


class BM25Plus:
    """BM25+ (BM25Plus) full-text search index."""

    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
    ):
        corpus_list = _coerce_to_strings(corpus)
        self._corpus = corpus_list
        self._k1 = k1
        self._b = b
        self._delta = delta
        self._normalize = normalize
        self._metadata = _validate_metadata(metadata, len(corpus_list))
        self._corpus_index: dict[str, int] | None = (
            _build_corpus_index(corpus_list) if self._metadata is not None else None
        )
        self._index = _rustfuzz.BM25Plus(corpus_list, k1, b, delta, normalize)

    @classmethod
    def from_column(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> BM25Plus:
        """Build a BM25Plus index from a DataFrame column (Spark, Polars, Pandas)."""
        meta = _extract_metadata(df, metadata_columns) if metadata_columns else None
        return cls(_extract_column(df, column), metadata=meta, **kwargs)

    @property
    def num_docs(self) -> int:
        return self._index.num_docs

    def get_scores(self, query: str) -> list[float]:
        return self._index.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n(query, n),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def get_batch_scores(self, queries: Iterable[str]) -> list[list[float]]:
        return self._index.get_batch_scores(list(queries))

    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n_fuzzy(query, n, bm25_candidates, fuzzy_weight),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def get_top_n_rrf(
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n_rrf(query, n, bm25_candidates, rrf_k),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def fuzzy_only(self, query: str, n: int = 5) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.fuzzy_only(query, n),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def explain(self, query: str, doc: str | int) -> dict[str, Any]:
        """Per-term BM25 score breakdown."""
        idx = self._corpus.index(doc) if isinstance(doc, str) else doc
        breakdown = self._index.explain(query, idx)
        terms = [
            {"term": t, "idf": idf, "tf_norm": tf, "score": s}
            for t, idf, tf, s in breakdown
        ]
        return {
            "terms": terms,
            "total_score": sum(float(d["score"]) for d in terms),
            "doc_idx": idx,
            "doc_text": self._corpus[idx],
        }

    def get_idf(self) -> dict[str, float]:
        return dict(self._index.get_idf_map())

    def get_document_vector(self, doc_idx: int) -> dict[str, float]:
        return dict(self._index.get_document_vector(doc_idx))

    def add_documents(
        self, docs: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None:
        """Add documents (rebuilds internally)."""
        new_docs = list(docs)
        self._corpus.extend(new_docs)
        if metadata is not None:
            new_meta = list(metadata)
            if len(new_meta) != len(new_docs):
                raise ValueError("metadata length must match docs length")
            if self._metadata is None:
                self._metadata = [None] * (len(self._corpus) - len(new_docs))
            self._metadata.extend(new_meta)
        elif self._metadata is not None:
            self._metadata.extend([None] * len(new_docs))
        self._index = _rustfuzz.BM25Plus(
            self._corpus, self._k1, self._b, self._delta, self._normalize
        )
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    def remove_documents(self, indices: list[int]) -> None:
        """Remove documents by index (rebuilds internally)."""
        to_remove = set(indices)
        self._corpus = [d for i, d in enumerate(self._corpus) if i not in to_remove]
        if self._metadata is not None:
            self._metadata = [
                m for i, m in enumerate(self._metadata) if i not in to_remove
            ]
        self._index = _rustfuzz.BM25Plus(
            self._corpus, self._k1, self._b, self._delta, self._normalize
        )
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    def get_top_n_reranked(
        self, query: str, n: int = 5, reranker: Any = None, rerank_candidates: int = 50
    ) -> list[_Result] | list[_MetaResult]:
        if reranker is None:
            return self.get_top_n(query, n)
        candidates = self._index.get_top_n(query, rerank_candidates)
        if not candidates:
            return []
        docs = [d for d, _ in candidates]
        rerank_scores = reranker(query, docs)
        paired = sorted(
            zip(docs, rerank_scores, strict=True), key=lambda x: x[1], reverse=True
        )[:n]
        results: list[_Result] = [(d, s) for d, s in paired]
        return _enrich(results, self._corpus, self._metadata, self._corpus_index)

    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n_phrase(query, n, proximity_window, phrase_boost),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    async def get_top_n_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]:
        import asyncio

        return await asyncio.to_thread(self.get_top_n, query, n, **kwargs)

    async def search_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]:
        import asyncio

        return await asyncio.to_thread(self.get_top_n_rrf, query, n, **kwargs)

    # ── Fluent query builder ──────────────────────────────

    def filter(self, expression: str) -> Any:
        """Start a filtered query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).filter(expression)

    def sort(self, expression: list[str] | str) -> Any:
        """Start a sorted query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).sort(expression)

    def match(self, query: str, **kwargs: Any) -> Any:
        """Execute a text search with the query builder."""
        return _search_query(self).match(query, **kwargs)

    def rerank(self, model_or_callable: Any, top_k: int = 10) -> Any:
        """Add a Reranker and start a query builder chain."""
        return _search_query(self).rerank(model_or_callable, top_k=top_k)

    def to_hybrid(self, embeddings: Any) -> HybridSearch:
        """Convert this BM25Plus index into a HybridSearch index with dense embeddings."""
        return HybridSearch(
            self._corpus,
            embeddings=embeddings,
            k1=self._k1,
            b=self._b,
            metadata=self._metadata,
            algorithm="bm25+",
            delta=self._delta,
        )

    def __reduce__(
        self,
    ) -> tuple[type, tuple[list[str], float, float, float, list[Any] | None, bool]]:
        return (
            BM25Plus,
            (
                self._corpus,
                self._k1,
                self._b,
                self._delta,
                self._metadata,
                self._normalize,
            ),
        )


class BM25T:
    """BM25T full-text search index."""

    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
    ):
        corpus_list = _coerce_to_strings(corpus)
        self._corpus = corpus_list
        self._k1 = k1
        self._b = b
        self._normalize = normalize
        self._metadata = _validate_metadata(metadata, len(corpus_list))
        self._corpus_index: dict[str, int] | None = (
            _build_corpus_index(corpus_list) if self._metadata is not None else None
        )
        self._index = _rustfuzz.BM25T(corpus_list, k1, b, normalize)

    @classmethod
    def from_column(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> BM25T:
        """Build a BM25T index from a DataFrame column (Spark, Polars, Pandas)."""
        meta = _extract_metadata(df, metadata_columns) if metadata_columns else None
        return cls(_extract_column(df, column), metadata=meta, **kwargs)

    @property
    def num_docs(self) -> int:
        return self._index.num_docs

    def get_scores(self, query: str) -> list[float]:
        return self._index.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n(query, n),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def get_batch_scores(self, queries: Iterable[str]) -> list[list[float]]:
        return self._index.get_batch_scores(list(queries))

    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n_fuzzy(query, n, bm25_candidates, fuzzy_weight),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def get_top_n_rrf(
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n_rrf(query, n, bm25_candidates, rrf_k),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def fuzzy_only(self, query: str, n: int = 5) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.fuzzy_only(query, n),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def explain(self, query: str, doc: str | int) -> dict[str, Any]:
        """Per-term BM25 score breakdown."""
        idx = self._corpus.index(doc) if isinstance(doc, str) else doc
        breakdown = self._index.explain(query, idx)
        terms = [
            {"term": t, "idf": idf, "tf_norm": tf, "score": s}
            for t, idf, tf, s in breakdown
        ]
        return {
            "terms": terms,
            "total_score": sum(float(d["score"]) for d in terms),
            "doc_idx": idx,
            "doc_text": self._corpus[idx],
        }

    def get_idf(self) -> dict[str, float]:
        return dict(self._index.get_idf_map())

    def get_document_vector(self, doc_idx: int) -> dict[str, float]:
        return dict(self._index.get_document_vector(doc_idx))

    def add_documents(
        self, docs: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None:
        """Add documents (rebuilds internally)."""
        new_docs = list(docs)
        self._corpus.extend(new_docs)
        if metadata is not None:
            new_meta = list(metadata)
            if len(new_meta) != len(new_docs):
                raise ValueError("metadata length must match docs length")
            if self._metadata is None:
                self._metadata = [None] * (len(self._corpus) - len(new_docs))
            self._metadata.extend(new_meta)
        elif self._metadata is not None:
            self._metadata.extend([None] * len(new_docs))
        self._index = _rustfuzz.BM25T(self._corpus, self._k1, self._b, self._normalize)
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    def remove_documents(self, indices: list[int]) -> None:
        """Remove documents by index (rebuilds internally)."""
        to_remove = set(indices)
        self._corpus = [d for i, d in enumerate(self._corpus) if i not in to_remove]
        if self._metadata is not None:
            self._metadata = [
                m for i, m in enumerate(self._metadata) if i not in to_remove
            ]
        self._index = _rustfuzz.BM25T(self._corpus, self._k1, self._b, self._normalize)
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    def get_top_n_reranked(
        self, query: str, n: int = 5, reranker: Any = None, rerank_candidates: int = 50
    ) -> list[_Result] | list[_MetaResult]:
        if reranker is None:
            return self.get_top_n(query, n)
        candidates = self._index.get_top_n(query, rerank_candidates)
        if not candidates:
            return []
        docs = [d for d, _ in candidates]
        rerank_scores = reranker(query, docs)
        paired = sorted(
            zip(docs, rerank_scores, strict=True), key=lambda x: x[1], reverse=True
        )[:n]
        results: list[_Result] = [(d, s) for d, s in paired]
        return _enrich(results, self._corpus, self._metadata, self._corpus_index)

    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[_Result] | list[_MetaResult]:
        return _enrich(
            self._index.get_top_n_phrase(query, n, proximity_window, phrase_boost),
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    async def get_top_n_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]:
        import asyncio

        return await asyncio.to_thread(self.get_top_n, query, n, **kwargs)

    async def search_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]:
        import asyncio

        return await asyncio.to_thread(self.get_top_n_rrf, query, n, **kwargs)

    # ── Fluent query builder ──────────────────────────────

    def filter(self, expression: str) -> Any:
        """Start a filtered query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).filter(expression)

    def sort(self, expression: list[str] | str) -> Any:
        """Start a sorted query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).sort(expression)

    def match(self, query: str, **kwargs: Any) -> Any:
        """Execute a text search with the query builder."""
        return _search_query(self).match(query, **kwargs)

    def rerank(self, model_or_callable: Any, top_k: int = 10) -> Any:
        """Add a Reranker and start a query builder chain."""
        return _search_query(self).rerank(model_or_callable, top_k=top_k)

    def to_hybrid(self, embeddings: Any) -> HybridSearch:
        """Convert this BM25T index into a HybridSearch index with dense embeddings."""
        return HybridSearch(
            self._corpus,
            embeddings=embeddings,
            k1=self._k1,
            b=self._b,
            metadata=self._metadata,
            algorithm="bm25t",
        )

    def __reduce__(
        self,
    ) -> tuple[type, tuple[list[str], float, float, list[Any] | None, bool]]:
        return (
            BM25T,
            (self._corpus, self._k1, self._b, self._metadata, self._normalize),
        )


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
        Text documents. Accepts:
        - list[str]
        - list[Document] (rustfuzz Document with content + metadata)
        - list[LangChain Document] (duck-typed via page_content)
        - Polars/Pandas Series, PyArrow arrays
    embeddings : matrix-like | Callable[[list[str]], list[list[float]]] | None
        Dense vectors for the corpus. Accepts:
        - A pre-computed matrix (list of lists, numpy array) with shape
          ``(num_docs, dim)``.
        - A **callback** ``(texts: list[str]) -> list[list[float]]`` that
          generates embeddings on-the-fly. The callback is called once at
          init with the corpus texts and again at each ``.search()`` call
          when ``query_embedding`` is omitted.
    k1 : float, default 1.5
        BM25 parameter
    b : float, default 0.75
        BM25 parameter
    metadata : Iterable[Any] | None, default None
        Optional per-document metadata. Overrides metadata from Document objects.

    Examples
    --------
    >>> docs = [Document("Apple iPhone", {"brand": "Apple"}), Document("Samsung Galaxy")]
    >>> hs = HybridSearch(docs, embeddings=[[1, 0], [0, 1]])
    >>> results = hs.search("iphone", query_embedding=[1, 0], n=1)

    Using a callback::

        def my_embed(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]  # your model here

        hs = HybridSearch(corpus, embeddings=my_embed)
        results = hs.search("query")  # query_embedding auto-generated
    """

    def __init__(
        self,
        corpus: Iterable[str] | Iterable[Any] | Any,
        embeddings: Any = None,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
        algorithm: str = "bm25",
        delta: float | None = None,
    ):
        # ── Coerce corpus (handles str, Document, LangChain) ──
        texts, auto_metadata = _coerce_corpus(corpus)
        self._corpus = texts
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
            # Callback path — invoke with corpus, store for later queries
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
            # Static matrix path (unchanged)
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
                    f"Length mismatch: {len(self._corpus)} documents, {len(emb_list)} embeddings"
                )

        # Keep a Python-side reference for pickle support
        self._embeddings = emb_list

        # ── Build Rust index ──
        self._index = _rustfuzz.HybridSearchIndex(
            self._corpus,
            emb_list,
            k1,
            b,
            algorithm,
            delta,
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
    ) -> list[_Result] | list[_MetaResult]:
        """
        Run 3-way hybrid search: BM25 + Fuzzy + Dense via RRF.

        When ``query_embedding`` is omitted and an embedding callback was
        provided at construction, the callback is invoked automatically
        with ``[query]``.  Otherwise, falls back to 2-way RRF (BM25 + fuzzy).

        Parameters
        ----------
        query : str
            The text query.
        query_embedding : Optional list or 1D array
            The semantic embedding for the query. If omitted and an
            embedding callback was provided, it is called automatically.
        n : int, default 5
            Top N results to return.
        rrf_k : int, default 60
            RRF smoothing parameter.
        bm25_candidates : int, default 100
            Number of BM25 candidates to pre-filter. Higher = better recall
            but slower. For million-scale, 100-500 is recommended.
        """
        q_emb: list[float] | None = None
        if query_embedding is not None:
            if hasattr(query_embedding, "tolist"):
                q_emb = query_embedding.tolist()
            else:
                q_emb = list(query_embedding)
        elif self._embed_fn is not None:
            # Auto-generate query embedding via the callback
            q_emb = list(self._embed_fn([query])[0])

        results = self._index.search(query, q_emb, n, rrf_k, bm25_candidates)
        return _enrich(
            results,
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    # ── Fluent query builder ──────────────────────────────

    def filter(self, expression: str) -> Any:
        """Start a filtered query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).filter(expression)

    def sort(self, expression: list[str] | str) -> Any:
        """Start a sorted query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).sort(expression)

    def match(self, query: str, **kwargs: Any) -> Any:
        """Execute a text search with the query builder."""
        return _search_query(self).match(query, **kwargs)

    def rerank(self, model_or_callable: Any, top_k: int = 10) -> Any:
        """Add a Reranker and start a query builder chain."""
        return _search_query(self).rerank(model_or_callable, top_k=top_k)

    def __reduce__(
        self,
    ) -> tuple[
        type, tuple[list[str], Any, float, float, list[Any] | None, str, float | None]
    ]:
        return (
            HybridSearch,
            (
                self._corpus,
                self._embeddings,
                self._k1,
                self._b,
                self._metadata,
                self._algorithm,
                self._delta,
            ),
        )


class Reranker:
    """
    A 2nd-stage Reranker that uses a Cross-Encoder or custom scoring function
    to re-evaluate and re-sort documents retrieved by a 1st-stage search engine.

    Ideal for achieving State-of-the-Art (SOTA) retrieval pipelines. This class
    wraps models from `EmbedAnything`, HuggingFace `sentence-transformers`, or
    any callable that accepts a (query, list_of_texts) and returns float scores.

    Parameters
    ----------
    model_or_callable : Any
        A cross-encoder model object (must have `.predict()` or similar) or
        a callable function `fn(query, texts) -> list[float]`.
    """

    def __init__(self, model_or_callable: Any) -> None:
        self._model = model_or_callable
        # Auto-detect common model interfaces
        if hasattr(model_or_callable, "predict"):
            # Typical sentence-transformers interface: .predict([(q, ctx), (q, ctx)])
            self._score_fn = self._st_predict_wrapper
        elif hasattr(model_or_callable, "compute_scores"):
            # EmbedAnything Reranker ONNX: compute_scores returns nested [[score]] per pair
            def _compute_scores(q: str, texts: list[str]) -> list[float]:
                scores: list[float] = []
                for t in texts:
                    result = model_or_callable.compute_scores([q], [t], 1)
                    # Flatten nested lists: [[0.96]] -> 0.96
                    val = result[0] if isinstance(result[0], (int, float)) else result[0][0]
                    scores.append(float(val))
                return scores
            self._score_fn = _compute_scores
        elif hasattr(model_or_callable, "score"):
            def _score(q: str, texts: list[str]) -> list[float]:
                return model_or_callable.score(q, texts)
            self._score_fn = _score
        elif callable(model_or_callable):
            self._score_fn = model_or_callable
        else:
            raise ValueError(
                "Reranker model must be callable or provide a `.predict()`/`.score()` method."
            )

    def _st_predict_wrapper(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        scores = self._model.predict(pairs)
        if hasattr(scores, "tolist"):
            return scores.tolist()
        return list(scores)

    def rerank(
        self,
        query: str,
        results: list[_Result] | list[_MetaResult],
        top_k: int = 10,
    ) -> list[_Result] | list[_MetaResult]:
        """
        Re-score and re-order the given results from a search engine.

        Parameters
        ----------
        query : str
            The original search query.
        results : list of tuples
            The (text, score) or (text, score, meta) results to rerank.
        top_k : int, default 10
            The number of top re-ranked documents to return.

        Returns
        -------
        list of tuples
            The re-ranked results in descending order of the new score.
        """
        if not results:
            return []

        # Extract texts
        texts = [res[0] for res in results]

        # Get actual scores from the cross-encoder
        try:
            new_scores = self._score_fn(query, texts)
        except Exception as e:
            raise RuntimeError(f"Reranker model failed to score texts: {e}") from e

        if len(new_scores) != len(results):
            raise ValueError(
                f"Reranker returned {len(new_scores)} scores for {len(results)} documents."
            )

        # Re-pack and sort
        reranked = []
        is_meta = len(results[0]) == 3
        for i, new_score in enumerate(new_scores):
            raw = results[i]
            if is_meta:
                # raw: (text, old_score, meta)
                reranked.append((raw[0], float(new_score), raw[2]))
            else:
                reranked.append((raw[0], float(new_score)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


__all__ = ["BM25", "BM25L", "BM25Plus", "BM25T", "Document", "HybridSearch", "Reranker"]
