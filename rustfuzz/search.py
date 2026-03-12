"""
rustfuzz.search — Full-text IR, BM25, and Hybrid Search capabilities.

Provides BM25 variants backed by Rust for fast indexing and scoring,
and the HybridSearch class which fuses text search with vector embeddings
using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal, Self, cast

from . import _rustfuzz
from ._types import MetaResult, Result, _search_query
from .compat import _coerce_to_strings, _extract_column, _extract_metadata
from .document import Document  # noqa: F401 — re-export for backward compat

BM25Algorithm = Literal["bm25", "bm25okapi", "bm25l", "bm25+", "bm25plus", "bm25t"]

# Keep backward-compatible aliases
_Result = Result
_MetaResult = MetaResult


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
        return cls(_extract_column(df, column), metadata=meta, **kwargs)

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


# ---------------------------------------------------------------------------
# Corpus coercion helper (supports Document, LangChain, DataFrames)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HybridSearch
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """
    A 2nd-stage Reranker that uses a Cross-Encoder or custom scoring function
    to re-evaluate and re-sort documents retrieved by a 1st-stage search engine.

    Parameters
    ----------
    model_or_callable : Any
        A cross-encoder model object or callable ``fn(query, texts) -> list[float]``.
    blend_alpha : float, default 0.0
        Weight of original retrieval rank (0.0 to 1.0).
    adaptive_blend : bool, default False
        Dynamically adjusts blend_alpha based on reranker score variance.
    """

    def __init__(
        self,
        model_or_callable: Any,
        blend_alpha: float = 0.0,
        adaptive_blend: bool = False,
    ) -> None:
        self._model = model_or_callable
        self.blend_alpha = blend_alpha
        self.adaptive_blend = adaptive_blend
        self._score_fn = self._resolve_score_fn(model_or_callable)

    @staticmethod
    def _resolve_score_fn(
        model: Any,
    ) -> Callable[[str, list[str]], list[float]]:
        """Auto-detect the scoring interface of a reranker model.

        Dispatch order:
        1. ``.predict(pairs)`` — SentenceTransformers CrossEncoder
        2. ``.compute_scores(queries, docs, n)`` — FlagEmbedding / BGE
        3. ``.score(query, texts)`` — direct score method
        4. Bare callable ``fn(query, texts) -> list[float]``
        """
        if hasattr(model, "predict"):

            def _predict_wrapper(query: str, texts: list[str]) -> list[float]:
                pairs = [(query, t) for t in texts]
                scores = model.predict(pairs)
                return scores.tolist() if hasattr(scores, "tolist") else list(scores)

            return _predict_wrapper

        if hasattr(model, "compute_scores"):

            def _compute_wrapper(query: str, texts: list[str]) -> list[float]:
                scores: list[float] = []
                for t in texts:
                    result = model.compute_scores([query], [t], 1)
                    val = (
                        result[0]
                        if isinstance(result[0], (int, float))
                        else result[0][0]
                    )
                    scores.append(float(val))
                return scores

            return _compute_wrapper

        if hasattr(model, "score"):
            return model.score

        if callable(model):
            return model

        raise ValueError(
            "Reranker model must be callable or provide a "
            "`.predict()`/`.compute_scores()`/`.score()` method."
        )

    def rerank(
        self,
        query: str,
        results: list[Result] | list[MetaResult],
        top_k: int = 10,
    ) -> list[Result] | list[MetaResult]:
        """Re-score and re-order results from a search engine."""
        if not results:
            return []

        texts = [res[0] for res in results]

        try:
            new_scores = cast("list[float]", self._score_fn(query, texts))
        except Exception as e:
            raise RuntimeError(f"Reranker model failed to score texts: {e}") from e

        if len(new_scores) != len(results):
            raise ValueError(
                f"Reranker returned {len(new_scores)} scores "
                f"for {len(results)} documents."
            )

        # Apply score blending if configured
        alpha = self.blend_alpha
        if self.adaptive_blend and len(new_scores) > 1:
            mean_s = sum(float(x) for x in new_scores) / len(new_scores)
            variance = sum((float(s) - mean_s) ** 2 for s in new_scores) / len(
                new_scores
            )
            std_dev = variance**0.5
            alpha = max(0.0, min(1.0, 0.5 - std_dev * 0.1))

        if alpha > 0.0 and new_scores:
            new_scores = _blend_reranked_scores(texts, results, new_scores, alpha)

        # Re-pack and sort
        reranked = []
        is_meta = len(results[0]) == 3
        for i, new_score in enumerate(new_scores):
            raw = results[i]
            if is_meta:
                raw_meta = cast(MetaResult, raw)
                reranked.append(
                    (raw_meta[0], float(new_score), raw_meta[2])  # type: ignore[misc]
                )
            else:
                reranked.append((raw[0], float(new_score)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


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
]
