"""BM25 variant implementations: BM25 (Okapi), BM25L, BM25Plus, BM25T."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:
    from ._hybrid import HybridSearch

from .. import _rustfuzz
from .._types import MetaResult, Result, _search_query
from ..compat import _coerce_to_strings, _extract_column, _extract_metadata
from ..document import Document  # noqa: F401 — re-export for backward compat
from ._helpers import (
    _blend_reranked_scores,
    _build_corpus_index,
    _enrich,
    _validate_metadata,
)

BM25Algorithm = Literal["bm25", "bm25okapi", "bm25l", "bm25+", "bm25plus", "bm25t"]

# Keep backward-compatible aliases
_Result = Result
_MetaResult = MetaResult


# ---------------------------------------------------------------------------
# _BaseBM25 — shared implementation for all BM25 variants
# ---------------------------------------------------------------------------


class _BaseBM25:
    """
    Base class for all BM25 variants.

    Provides all search, ranking, mutation, and query-builder methods.
    Subclasses only need to override ``__init__``, ``to_hybrid``,
    and ``__reduce__``.

    Attributes set by subclass ``__init__``:
        _corpus, _k1, _b, _normalize, _normalize_scores,
        _metadata, _corpus_index, _index
    """

    _corpus: list[str]
    _k1: float
    _b: float
    _normalize: bool
    _normalize_scores: bool
    _metadata: list[Any] | None
    _corpus_index: dict[str, int] | None
    _index: Any  # Rust BM25 index object

    # ── Class methods ─────────────────────────────────────────

    @classmethod
    def from_column(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> Self:
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
        return cls(_extract_column(df, column), metadata=meta, **kwargs)  # type: ignore[call-arg]

    # ── Properties ────────────────────────────────────────────

    @property
    def num_docs(self) -> int:
        """Number of documents in the index."""
        return self._index.num_docs

    # ── Scoring / retrieval ───────────────────────────────────

    def get_scores(self, query: str) -> list[float]:
        """Return BM25 scores for the query against every document."""
        return self._index.get_scores(query)

    def get_top_n(self, query: str, n: int = 5) -> list[Result] | list[MetaResult]:
        """Return the top N matching documents for the query."""
        res = self._index.get_top_n(query, n=n)
        if self._normalize_scores:
            res = _rustfuzz.normalize_bayes(res)
        return _enrich(res, self._corpus, self._metadata, self._corpus_index)

    def get_top_n_filtered(
        self, query: str, allowed: list[bool], n: int = 5
    ) -> list[Result] | list[MetaResult]:
        """Return the top N matching documents, considering only the allowed subset."""
        res = self._index.get_top_n_filtered(query, allowed=allowed, n=n)
        if self._normalize_scores:
            res = _rustfuzz.normalize_bayes(res)
        return _enrich(res, self._corpus, self._metadata, self._corpus_index)

    def get_batch_scores(self, queries: Iterable[str]) -> list[list[float]]:
        """Return BM25 scores for a batch of queries (parallelised via Rayon)."""
        return self._index.get_batch_scores(list(queries))

    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[Result] | list[MetaResult]:
        """
        Hybrid search combining BM25 with Levenshtein-based fuzzy matching.

        Parameters
        ----------
        query : str
        n : int, default 5
            Number of documents to return.
        bm25_candidates : int, default 50
            BM25 candidates to send to the fuzzy stage.
        fuzzy_weight : float, default 0.3
            Weight for fuzzy similarity [0.0 - 1.0].
        """
        res = self._index.get_top_n_fuzzy(query, n, bm25_candidates, fuzzy_weight)
        if self._normalize_scores:
            res = _rustfuzz.normalize_bayes(res)
        return _enrich(res, self._corpus, self._metadata, self._corpus_index)

    def get_top_n_rrf(
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[Result] | list[MetaResult]:
        """Hybrid search via Reciprocal Rank Fusion (BM25 + fuzzy)."""
        res = self._index.get_top_n_rrf(query, n, bm25_candidates, rrf_k)
        if self._normalize_scores:
            res = _rustfuzz.normalize_bayes(res)
        return _enrich(res, self._corpus, self._metadata, self._corpus_index)

    def fuzzy_only(self, query: str, n: int = 5) -> list[Result] | list[MetaResult]:
        """Pure fuzzy string ranking over the corpus (no BM25 scores)."""
        res = self._index.fuzzy_only(query, n)
        if self._normalize_scores:
            res = _rustfuzz.normalize_bayes(res)
        return _enrich(res, self._corpus, self._metadata, self._corpus_index)

    # ── Inspection ────────────────────────────────────────────

    def explain(self, query: str, doc: str | int) -> dict[str, Any]:
        """Per-term BM25 score breakdown for a query against a document."""
        if isinstance(doc, str):
            try:
                idx = self._corpus.index(doc)
            except ValueError as e:
                raise ValueError(f"Document not found in corpus: {doc!r}") from e
        else:
            idx = doc
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
        """Return the entire IDF map: term → IDF value."""
        return dict(self._index.get_idf_map())

    def get_document_vector(self, doc_idx: int) -> dict[str, float]:
        """Return the normalised TF vector for a specific document."""
        return dict(self._index.get_document_vector(doc_idx))

    # ── Mutation ──────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        """Rebuild the Rust index after corpus mutation. Override in subclass."""
        raise NotImplementedError

    def add_documents(
        self, docs: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None:
        """Add documents to the index.

        .. note::

           This method triggers a full index rebuild (O(n) where n is total
           corpus size). For bulk additions, prefer constructing a new index.
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
        self._rebuild_index()
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    def remove_documents(self, indices: list[int]) -> None:
        """Remove documents by index.

        .. note::

           This method triggers a full index rebuild (O(n) where n is
           remaining corpus size). For bulk removals, consider constructing
           a new index from the surviving documents.
        """
        to_remove = set(indices)
        self._corpus = [d for i, d in enumerate(self._corpus) if i not in to_remove]
        if self._metadata is not None:
            self._metadata = [
                m for i, m in enumerate(self._metadata) if i not in to_remove
            ]
        self._rebuild_index()
        if self._metadata is not None:
            self._corpus_index = _build_corpus_index(self._corpus)

    # ── Reranking ─────────────────────────────────────────────

    def get_top_n_reranked(
        self,
        query: str,
        n: int = 5,
        reranker: Any = None,
        rerank_candidates: int = 50,
        blend_alpha: float = 0.0,
    ) -> list[Result] | list[MetaResult]:
        """
        BM25 retrieval + external reranker callback.

        If ``blend_alpha > 0.0``, blends BM25 normalised ranks with reranker
        min-max normalised scores.
        """
        if reranker is None:
            return self.get_top_n(query, n)

        candidates = self._index.get_top_n(query, rerank_candidates)
        if not candidates:
            return []

        docs = [d for d, _ in candidates]
        rerank_scores = reranker(query, docs)

        if blend_alpha > 0.0 and len(rerank_scores) > 0:
            rerank_scores = _blend_reranked_scores(
                docs, candidates, rerank_scores, blend_alpha
            )

        paired = sorted(
            zip(docs, rerank_scores, strict=True), key=lambda x: x[1], reverse=True
        )[:n]
        results: list[Result] = [(d, s) for d, s in paired]
        if self._normalize_scores:
            results = _rustfuzz.normalize_bayes(results)
        return _enrich(results, self._corpus, self._metadata, self._corpus_index)

    # ── Phrase search ─────────────────────────────────────────

    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[Result] | list[MetaResult]:
        """BM25 scoring with phrase proximity boost."""
        res = self._index.get_top_n_phrase(query, n, proximity_window, phrase_boost)
        if self._normalize_scores:
            res = _rustfuzz.normalize_bayes(res)
        return _enrich(res, self._corpus, self._metadata, self._corpus_index)

    # ── Async wrappers ────────────────────────────────────────

    async def get_top_n_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[Result] | list[MetaResult]:
        """Async wrapper around :meth:`get_top_n` (runs in a thread)."""
        import asyncio

        return await asyncio.to_thread(self.get_top_n, query, n, **kwargs)

    async def search_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[Result] | list[MetaResult]:
        """Async wrapper around :meth:`get_top_n_rrf`."""
        import asyncio

        return await asyncio.to_thread(self.get_top_n_rrf, query, n, **kwargs)

    # ── Fluent query builder ──────────────────────────────────

    def filter(self, expression: str) -> Any:
        """Start a filtered query chain. Returns a :class:`SearchQuery` builder."""
        return _search_query(self).filter(expression)

    def sort(self, expression: list[str] | str) -> Any:
        """Start a sorted query chain. Returns a :class:`SearchQuery` builder."""
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

    def search_filtered_sorted(
        self,
        query: str,
        *,
        n: int = 5,
        query_embedding: Any | None = None,
        rrf_k: int = 60,
        bm25_candidates: int = 100,
        filter_json: str | None = None,
        sort_keys: list[tuple[str, bool]] | None = None,
    ) -> list[Result] | list[MetaResult]:
        """Search with pushed-down filters and sorting using Rust metadata evaluation."""
        res = self._index.search_filtered_sorted(
            query,
            query_embedding=query_embedding,
            n=n,
            rrf_k=rrf_k,
            bm25_candidates=bm25_candidates,
            filter_json=filter_json,
            sort_keys=sort_keys,
        )
        if self._normalize_scores:
            res = _rustfuzz.normalize_bayes(res)
        return _enrich(res, self._corpus, self._metadata, self._corpus_index)


# ---------------------------------------------------------------------------
# Concrete BM25 variants (thin subclasses)
# ---------------------------------------------------------------------------


def _init_common(
    self: _BaseBM25,
    corpus: Iterable[str] | Any,
    k1: float,
    b: float,
    metadata: Iterable[Any] | None,
    normalize: bool,
    normalize_scores: bool,
) -> list[str]:
    """Shared initialisation logic for all BM25 variants. Returns corpus_list."""
    corpus_list = _coerce_to_strings(corpus)
    self._corpus = corpus_list
    self._k1 = k1
    self._b = b
    self._normalize = normalize
    self._normalize_scores = normalize_scores
    self._metadata = _validate_metadata(metadata, len(corpus_list))
    self._corpus_index = (
        _build_corpus_index(corpus_list) if self._metadata is not None else None
    )
    return corpus_list


class BM25(_BaseBM25):
    """
    BM25Okapi full-text search index.

    Parameters
    ----------
    corpus : Iterable[str]
        Documents to index.
    k1 : float, default 1.5
        Term frequency saturation parameter.
    b : float, default 0.75
        Length normalisation factor.
    metadata : Iterable[Any] | None, default None
        Optional per-document metadata.
    normalize : bool, default False
        Whether to normalize BM25 scores.
    """

    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
        normalize_scores: bool = False,
    ) -> None:
        corpus_list = _init_common(
            self, corpus, k1, b, metadata, normalize, normalize_scores
        )
        self._index = _rustfuzz.BM25Index(corpus_list, k1, b, normalize)

    def _rebuild_index(self) -> None:
        self._index = _rustfuzz.BM25Index(
            self._corpus, self._k1, self._b, self._normalize
        )

    def to_hybrid(self, embeddings: Any) -> HybridSearch:
        """Convert this BM25 index into a HybridSearch index."""
        from ._hybrid import HybridSearch

        return HybridSearch(
            self._corpus,
            embeddings=embeddings,
            k1=self._k1,
            b=self._b,
            metadata=self._metadata,
            algorithm="bm25",
            normalize_scores=self._normalize_scores,
        )

    def __reduce__(
        self,
    ) -> tuple[type, tuple[list[str], float, float, list[Any] | None, bool, bool]]:
        return (
            BM25,
            (
                self._corpus,
                self._k1,
                self._b,
                self._metadata,
                self._normalize,
                self._normalize_scores,
            ),
        )


class BM25L(_BaseBM25):
    """BM25L full-text search index."""

    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
        normalize_scores: bool = False,
    ) -> None:
        corpus_list = _init_common(
            self, corpus, k1, b, metadata, normalize, normalize_scores
        )
        self._delta = delta
        self._index = _rustfuzz.BM25L(corpus_list, k1, b, delta, normalize)

    def _rebuild_index(self) -> None:
        self._index = _rustfuzz.BM25L(
            self._corpus, self._k1, self._b, self._delta, self._normalize
        )

    def to_hybrid(self, embeddings: Any) -> HybridSearch:
        """Convert this BM25L index into a HybridSearch index."""
        from ._hybrid import HybridSearch

        return HybridSearch(
            self._corpus,
            embeddings=embeddings,
            k1=self._k1,
            b=self._b,
            metadata=self._metadata,
            algorithm="bm25l",
            delta=self._delta,
            normalize_scores=self._normalize_scores,
        )

    def __reduce__(
        self,
    ) -> tuple[
        type, tuple[list[str], float, float, float, list[Any] | None, bool, bool]
    ]:
        return (
            BM25L,
            (
                self._corpus,
                self._k1,
                self._b,
                self._delta,
                self._metadata,
                self._normalize,
                self._normalize_scores,
            ),
        )


class BM25Plus(_BaseBM25):
    """BM25+ (BM25Plus) full-text search index."""

    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
        normalize_scores: bool = False,
    ) -> None:
        corpus_list = _init_common(
            self, corpus, k1, b, metadata, normalize, normalize_scores
        )
        self._delta = delta
        self._index = _rustfuzz.BM25Plus(corpus_list, k1, b, delta, normalize)

    def _rebuild_index(self) -> None:
        self._index = _rustfuzz.BM25Plus(
            self._corpus, self._k1, self._b, self._delta, self._normalize
        )

    def to_hybrid(self, embeddings: Any) -> HybridSearch:
        """Convert this BM25Plus index into a HybridSearch index."""
        from ._hybrid import HybridSearch

        return HybridSearch(
            self._corpus,
            embeddings=embeddings,
            k1=self._k1,
            b=self._b,
            metadata=self._metadata,
            algorithm="bm25+",
            delta=self._delta,
            normalize_scores=self._normalize_scores,
        )

    def __reduce__(
        self,
    ) -> tuple[
        type, tuple[list[str], float, float, float, list[Any] | None, bool, bool]
    ]:
        return (
            BM25Plus,
            (
                self._corpus,
                self._k1,
                self._b,
                self._delta,
                self._metadata,
                self._normalize,
                self._normalize_scores,
            ),
        )


class BM25T(_BaseBM25):
    """BM25T full-text search index."""

    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
        normalize_scores: bool = False,
    ) -> None:
        corpus_list = _init_common(
            self, corpus, k1, b, metadata, normalize, normalize_scores
        )
        self._index = _rustfuzz.BM25T(corpus_list, k1, b, normalize)

    def _rebuild_index(self) -> None:
        self._index = _rustfuzz.BM25T(self._corpus, self._k1, self._b, self._normalize)

    def to_hybrid(self, embeddings: Any) -> HybridSearch:
        """Convert this BM25T index into a HybridSearch index."""
        from ._hybrid import HybridSearch

        return HybridSearch(
            self._corpus,
            embeddings=embeddings,
            k1=self._k1,
            b=self._b,
            metadata=self._metadata,
            algorithm="bm25t",
            normalize_scores=self._normalize_scores,
        )

    def __reduce__(
        self,
    ) -> tuple[type, tuple[list[str], float, float, list[Any] | None, bool, bool]]:
        return (
            BM25T,
            (
                self._corpus,
                self._k1,
                self._b,
                self._metadata,
                self._normalize,
                self._normalize_scores,
            ),
        )
