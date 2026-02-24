"""
rustfuzz.join — Multi-array fuzzy full join.

Performs a cross-array "full join" using fuzzy text (BM25 + indel), sparse
dot-product, and/or dense cosine similarity, fused via Reciprocal Rank Fusion.
The heavy computation runs in Rust/Rayon outside the GIL.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import _rustfuzz


class MultiJoiner:
    """
    Multi-array fuzzy full joiner backed by Rust.

    Each array element may carry a text string, a sparse vector
    ({token_id: weight}), and/or a dense embedding vector. All provided
    channels are fused with weighted Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    text_weight : float, default 1.0
        Relative weight of the text channel.
    sparse_weight : float, default 1.0
        Relative weight of the sparse vector channel.
    dense_weight : float, default 1.0
        Relative weight of the dense vector channel.
    bm25_k1 : float, default 1.5
        BM25 term-frequency saturation parameter.
    bm25_b : float, default 0.75
        BM25 length-normalisation factor.
    rrf_k : int, default 60
        RRF smoothing constant (Cormack et al., 2009).
    """

    def __init__(
        self,
        *,
        text_weight: float = 1.0,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
    ) -> None:
        self._inner = _rustfuzz.MultiJoiner(
            text_weight,
            sparse_weight,
            dense_weight,
            bm25_k1,
            bm25_b,
            rrf_k,
        )

    @property
    def num_arrays(self) -> int:
        """Number of arrays registered so far."""
        return self._inner.num_arrays

    def add_array(
        self,
        name: str,
        texts: Iterable[str | None] | None = None,
        sparse: Iterable[dict[int, float] | None] | None = None,
        dense: Iterable[list[float] | None] | None = None,
    ) -> MultiJoiner:
        """
        Register a named array.

        At least one of *texts*, *sparse*, *dense* must be provided.
        All provided iterables must have the same length.

        Parameters
        ----------
        name : str
            Unique name for this array.
        texts : iterable of str | None, optional
            Text per element — drives BM25 + fuzzy channel.
        sparse : iterable of dict[int, float] | None, optional
            Sparse vectors per element as ``{token_id: weight}``.
        dense : iterable of list[float] | None, optional
            Dense embedding vectors (pre-normalised for cosine similarity).

        Returns
        -------
        MultiJoiner
            Self, for method chaining.
        """
        self._inner.add_array(
            name,
            list(texts) if texts is not None else None,
            list(sparse) if sparse is not None else None,  # type: ignore[arg-type]
            list(dense) if dense is not None else None,
        )
        return self

    def join(self, n: int = 1) -> list[dict[str, Any]]:
        """
        Full join across every ordered pair of arrays (src ≠ tgt).

        Parameters
        ----------
        n : int, default 1
            Number of top target matches to return per source element.

        Returns
        -------
        list[dict]
            Each row contains:
            ``src_array``, ``src_idx``, ``src_text``,
            ``tgt_array``, ``tgt_idx``, ``tgt_text``,
            ``score``, ``text_score``, ``sparse_score``, ``dense_score``.
        """
        return self._inner.join(n)

    def join_pair(
        self, src_name: str, tgt_name: str, n: int = 1
    ) -> list[dict[str, Any]]:
        """
        Join a single ordered pair of arrays.

        Parameters
        ----------
        src_name : str
        tgt_name : str
        n : int, default 1
        """
        return self._inner.join_pair(src_name, tgt_name, n)


def fuzzy_join(
    arrays: dict[str, list[str]],
    *,
    sparse: dict[str, list[dict[int, float]]] | None = None,
    dense: dict[str, list[list[float]]] | None = None,
    text_weight: float = 1.0,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
    n: int = 1,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Convenience wrapper: fuzzy full join across N text arrays.

    Parameters
    ----------
    arrays : dict[str, list[str]]
        Named text arrays, e.g. ``{"A": [...], "B": [...]}``.
    sparse : dict[str, list[dict[int, float]]], optional
        Named sparse vector arrays aligned to ``arrays``.
    dense : dict[str, list[list[float]]], optional
        Named dense embedding arrays aligned to ``arrays``.
    text_weight : float, default 1.0
    sparse_weight : float, default 1.0
    dense_weight : float, default 1.0
    n : int, default 1
        Top-N matches per element.
    bm25_k1, bm25_b, rrf_k
        BM25 / RRF tuning parameters.

    Returns
    -------
    list[dict]
        One row per (src_array, src_element, tgt_array, top-N match).

    Examples
    --------
    >>> from rustfuzz.join import fuzzy_join
    >>> rows = fuzzy_join(
    ...     {"products": ["Apple iPhone 14", "Galaxy S23"],
    ...      "listings": ["iphone 14 pro",   "galaxy s23 ultra"]},
    ...     n=1,
    ... )
    >>> rows[0]["tgt_text"]
    'iphone 14 pro'
    """
    joiner = MultiJoiner(
        text_weight=text_weight,
        sparse_weight=sparse_weight,
        dense_weight=dense_weight,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        rrf_k=rrf_k,
    )
    for name, texts in arrays.items():
        joiner.add_array(
            name,
            texts=texts,
            sparse=sparse.get(name) if sparse else None,
            dense=dense.get(name) if dense else None,
        )
    return joiner.join(n=n)


__all__ = ["MultiJoiner", "fuzzy_join"]
