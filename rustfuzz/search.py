"""
rustfuzz.search — Full-text IR, BM25, and Hybrid Search capabilities.

Provides the BM25 class backed by Rust for fast indexing and scoring,
and the HybridSearch class which fuses text search with vector embeddings
using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from . import _rustfuzz


class AbstractSearchIndex(ABC):
    """
    Abstract base class for all rustfuzz search indices.

    Custom search backends should inherit from this class and implement
    the ``search`` method so they are interchangeable with the built-in
    :class:`BM25` and :class:`HybridSearch` implementations.
    """

    @abstractmethod
    def search(self, query: str, *, n: int = 5) -> list[tuple[str, float]]:
        """Return the *n* best matching documents with their relevance scores.

        Parameters
        ----------
        query:
            The search query string.
        n:
            Maximum number of results to return.
        """
        ...


class BM25(AbstractSearchIndex):
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
    """

    def __init__(self, corpus: Iterable[str], k1: float = 1.5, b: float = 0.75):
        corpus_list = list(corpus)
        # Type assertion for safety
        if corpus_list and not isinstance(corpus_list[0], str):
            raise TypeError("corpus must be an iterable of strings")
        self._index = _rustfuzz.BM25Index(corpus_list, k1, b)

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

    def get_top_n(self, query: str, n: int = 5) -> list[tuple[str, float]]:
        """
        Return the top N matching documents and their BM25 scores.
        Only documents with score > 0.0 are returned (up to n).
        """
        return self._index.get_top_n(query, n)

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
    ) -> list[tuple[str, float]]:
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
        return self._index.get_top_n_fuzzy(query, n, bm25_candidates, fuzzy_weight)

    def get_top_n_rrf(
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[tuple[str, float]]:
        """
        Hybrid search combining BM25 and Levenshtein distance through Reciprocal Rank Fusion (RRF).

        This is generally more robust than `get_top_n_fuzzy` because it uses RRF,
        shielding the combined metric from score-scale variances.
        """
        return self._index.get_top_n_rrf(query, n, bm25_candidates, rrf_k)

    @override
    def search(self, query: str, *, n: int = 5) -> list[tuple[str, float]]:
        """
        Return the top N results using BM25 + Levenshtein Reciprocal Rank Fusion.

        This is the default search method, providing robust results even for
        misspelled queries.  Delegates to :meth:`get_top_n_rrf`.
        """
        return self.get_top_n_rrf(query, n=n)


class HybridSearch(AbstractSearchIndex):
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
    """

    def __init__(
        self,
        corpus: Iterable[str],
        embeddings: Any = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._corpus = list(corpus)
        self._bm25 = BM25(self._corpus, k1=k1, b=b)
        self._embeddings = None

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

            if len(self._embeddings) > 0 and len(self._corpus) != len(self._embeddings):
                raise ValueError(
                    f"Length mismatch: {len(self._corpus)} documents, {len(self._embeddings)} embeddings"
                )

    @property
    def has_vectors(self) -> bool:
        return self._embeddings is not None

    @override
    def search(
        self, query: str, *, query_embedding: Any = None, n: int = 5, rrf_k: int = 60
    ) -> list[tuple[str, float]]:
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
            return self._bm25.get_top_n_rrf(query, n=n, rrf_k=rrf_k)

        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        # Step 1: BM25 ranks
        # Get ranks for all documents that match at all.
        bm25_scores = self._bm25.get_scores(query)
        bm25_indexed = [(i, score) for i, score in enumerate(bm25_scores) if score > 0]
        bm25_indexed.sort(key=lambda x: x[1], reverse=True)

        # Step 2: Semantic ranks (cosine sim) via Rust fast dot product
        # Vectorise the single query by wrapping in a list
        q_wrapped = [query_embedding]
        assert self._embeddings is not None  # has_vectors guard above ensures this
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
        final_results = [(self._corpus[i], score) for i, score in enumerate(rrf_scores)]
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:n]


__all__ = ["AbstractSearchIndex", "BM25", "HybridSearch"]
