"""
rustfuzz.search — Full-text IR, BM25, and Hybrid Search capabilities.

Provides the BM25 class backed by Rust for fast indexing and scoring,
and the HybridSearch class which fuses text search with vector embeddings
using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import _rustfuzz
from .compat import _coerce_to_strings, _extract_column, _extract_metadata

# Type aliases for result tuples
_Result = tuple[str, float]
_MetaResult = tuple[str, float, Any]


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

    def __reduce__(
        self,
    ) -> tuple[type, tuple[list[str], float, float, list[Any] | None, bool]]:
        return (
            BM25T,
            (self._corpus, self._k1, self._b, self._metadata, self._normalize),
        )


class HybridSearch:
    """
    Tier-3 Semantic Hybrid Search framework.
    Combines text retrieval (BM25) with vector search via Reciprocal Rank Fusion (RRF).

    This class is agnostic to the embedding model. You provide the text corpus and
    a matrix of embeddings. The search() method fuses scores.

    Parameters
    ----------
    corpus : Iterable[str]
        Text documents.
    embeddings : Optional matrix-like (list of lists, numpy array)
        Vectors associated with the corpus. Shape (num_docs, dim).
        If evaluating without numpy, pass a python list of lists.
    k1 : float, default 1.5
        BM25 parameter
    b : float, default 0.75
        BM25 parameter
    metadata : Iterable[Any] | None, default None
        Optional per-document metadata returned alongside results.
    """

    def __init__(
        self,
        corpus: Iterable[str] | Any,
        embeddings: Any = None,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
    ):
        self._corpus = _coerce_to_strings(corpus)
        self._bm25 = BM25(self._corpus, k1=k1, b=b)
        self._k1 = k1
        self._b = b
        self._metadata = _validate_metadata(metadata, len(self._corpus))
        self._corpus_index: dict[str, int] | None = (
            _build_corpus_index(self._corpus) if self._metadata is not None else None
        )
        self._embeddings: list[list[float]] | None = None

        if embeddings is not None:
            # We try to convert embeddings to a standard list of lists of floats
            # so the Rust boundary can parse it directly as Vec<Vec<f32>>.
            try:
                # If it's a numpy array, tolist() is the fastest conversion to native types
                if hasattr(embeddings, "shape") and hasattr(embeddings, "tolist"):
                    self._embeddings = embeddings.tolist()
                else:
                    self._embeddings = list(embeddings)
            except Exception as e:
                raise ValueError(
                    "embeddings must be convertible to a list of lists of floats"
                ) from e

            emb = self._embeddings
            if emb is not None and len(emb) > 0 and len(self._corpus) != len(emb):
                raise ValueError(
                    f"Length mismatch: {len(self._corpus)} documents, {len(emb)} embeddings"
                )

    @property
    def has_vectors(self) -> bool:
        return self._embeddings is not None

    def search(
        self, query: str, query_embedding: Any = None, n: int = 5, rrf_k: int = 60
    ) -> list[_Result] | list[_MetaResult]:
        """
        Run hybrid search fusing BM25 text relevance and cosine semantic similarity.

        If `query_embedding` is omitted, falls back to BM25 fuzzy RRF.
        Uses Reciprocal Rank Fusion (RRF) where the final score is the sum of `1 / (k + rank)`
        for each retrieval pipeline.

        Parameters
        ----------
        query : str
            The text query.
        query_embedding : Optional list or 1D array
            The semantic embedding for the query.
        n : int, default 5
            Top N results to return.
        rrf_k : int, default 60
            RRF smoothing parameter.
        """
        if query_embedding is None or not self.has_vectors:
            # Fallback to BM25 + fuzzy string RRF
            results = self._bm25.get_top_n_rrf(query, n=n, rrf_k=rrf_k)
            return _enrich(
                results,  # type: ignore[arg-type]
                self._corpus,
                self._metadata,
                self._corpus_index,
            )

        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        # Step 1: BM25 ranks
        # Get ranks for all documents that match at all.
        bm25_scores = self._bm25.get_scores(query)
        bm25_indexed = [(i, score) for i, score in enumerate(bm25_scores) if score > 0]
        bm25_indexed.sort(key=lambda x: x[1], reverse=True)

        # Step 2: Semantic ranks (cosine sim) via Rust fast dot product
        # Vectorise the single query by wrapping in a list
        assert self._embeddings is not None  # guarded by has_vectors above
        q_wrapped = [query_embedding]
        flat_matrix, _, _ = _rustfuzz.cosine_similarity_matrix(
            q_wrapped, self._embeddings
        )
        semantic_indexed = [(i, score) for i, score in enumerate(flat_matrix)]
        semantic_indexed.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Reciprocal Rank Fusion
        rrf_scores = [0.0] * len(self._corpus)

        for rank, (doc_idx, _) in enumerate(bm25_indexed):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)

        for rank, (doc_idx, _) in enumerate(semantic_indexed):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)

        # Assemble and sort final
        final_results: list[_Result] = [
            (self._corpus[i], score) for i, score in enumerate(rrf_scores)
        ]
        final_results.sort(key=lambda x: x[1], reverse=True)
        return _enrich(
            final_results[:n],
            self._corpus,
            self._metadata,
            self._corpus_index,
        )

    def __reduce__(
        self,
    ) -> tuple[type, tuple[list[str], Any, float, float, list[Any] | None]]:
        return (
            HybridSearch,
            (
                self._corpus,
                self._embeddings,
                self._k1,
                self._b,
                self._metadata,
            ),
        )


__all__ = ["BM25", "BM25L", "BM25Plus", "BM25T", "HybridSearch"]
