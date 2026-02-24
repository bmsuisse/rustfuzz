"""
rustfuzz.join — Multi-array fuzzy full join.

Performs a cross-array join using fuzzy text (BM25 + indel), sparse
dot-product, and/or dense cosine similarity, fused via Reciprocal Rank Fusion.
The heavy computation runs in Rust/Rayon outside the GIL.

Join types
----------
``how="full"``  (default)
    Return the top-N target matches for every source element, regardless of
    score.  Equivalent to a SQL cross join reduced to best matches.
``how="inner"``
    Only return rows where ``score >= score_cutoff``.  Rows with no good
    match are silently dropped — equivalent to a filtered inner join.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from . import _rustfuzz

_HOW = Literal["full", "inner"]


def _apply_filter(
    rows: list[dict[str, Any]],
    how: _HOW,
    score_cutoff: float | None,
) -> list[dict[str, Any]]:
    if how == "inner":
        cutoff = score_cutoff if score_cutoff is not None else 0.0
        return [r for r in rows if r["score"] >= cutoff]
    return rows


class MultiJoiner:
    """
    Multi-array fuzzy joiner backed by Rust.

    Supports any number of named arrays. Each element may carry a text
    string, a sparse vector (``{token_id: weight}``), and/or a dense
    embedding vector. All active channels are fused with weighted
    Reciprocal Rank Fusion (RRF).

    Sparse auto-fallback
    --------------------
    When ``sparse=None`` is passed to :meth:`add_array` but text is
    available, the sparse channel automatically falls back to **pure BM25
    scores** — no extra work required.

    Parameters
    ----------
    text_weight : float, default 1.0
    sparse_weight : float, default 1.0
    dense_weight : float, default 1.0
    bm25_k1 : float, default 1.5
    bm25_b : float, default 0.75
    rrf_k : int, default 60
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
            text_weight, sparse_weight, dense_weight, bm25_k1, bm25_b, rrf_k
        )
        self._names: list[str] = []

    @property
    def num_arrays(self) -> int:
        """Number of arrays registered so far."""
        return self._inner.num_arrays

    @property
    def array_names(self) -> list[str]:
        """Ordered list of registered array names."""
        return list(self._names)

    def add_array(
        self,
        name: str,
        texts: Iterable[str | None] | None = None,
        sparse: Iterable[dict[int, float] | None] | None = None,
        dense: Iterable[list[float] | None] | None = None,
    ) -> MultiJoiner:
        """
        Register a named array (chainable).

        At least one of *texts*, *sparse*, *dense* must be non-``None``.
        All provided iterables must have the same length.

        Parameters
        ----------
        name : str
        texts : iterable of str | None, optional
            Text per element — drives text + sparse-fallback channels.
        sparse : iterable of dict[int, float] | None, optional
            Sparse vectors as ``{token_id: weight}``.
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
        self._names.append(name)
        return self

    # ------------------------------------------------------------------
    # Long-form (pairwise) joins
    # ------------------------------------------------------------------

    def join(
        self,
        n: int = 1,
        *,
        how: _HOW = "full",
        score_cutoff: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Join every ordered pair of arrays (src ≠ tgt).

        Works with any number of arrays — 2, 5, 10, … — producing one set
        of rows for every ordered (src, tgt) pair.

        Parameters
        ----------
        n : int, default 1
            Top-N target matches per source element.
        how : {"full", "inner"}, default "full"
            *"full"*  — return all top-N rows regardless of score.
            *"inner"* — only return rows where ``score >= score_cutoff``.
        score_cutoff : float | None, default None
            Minimum score for *"inner"* mode.  Ignored when ``how="full"``.
            Defaults to ``0.0`` when ``how="inner"`` and left as ``None``.

        Returns
        -------
        list[dict]
            Keys: ``src_array``, ``src_idx``, ``src_text``, ``tgt_array``,
            ``tgt_idx``, ``tgt_text``, ``score``, ``text_score``,
            ``sparse_score``, ``dense_score``.
        """
        rows = self._inner.join(n)
        return _apply_filter(rows, how, score_cutoff)

    def join_pair(
        self,
        src_name: str,
        tgt_name: str,
        n: int = 1,
        *,
        how: _HOW = "full",
        score_cutoff: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Join a single ordered pair of arrays.

        Parameters
        ----------
        src_name, tgt_name : str
        n : int, default 1
        how : {"full", "inner"}, default "full"
        score_cutoff : float | None
        """
        rows = self._inner.join_pair(src_name, tgt_name, n)
        return _apply_filter(rows, how, score_cutoff)

    # ------------------------------------------------------------------
    # Wide (pivoted) join — one row per source element
    # ------------------------------------------------------------------

    def join_wide(
        self,
        src_name: str | None = None,
        n: int = 1,
        *,
        how: _HOW = "full",
        score_cutoff: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Pivoted join — one row per source element with match columns for
        every other registered array.

        This is the natural format when you have many arrays (e.g. 10
        catalogues) and want a single output table keyed by source element.

        Parameters
        ----------
        src_name : str | None, default None
            Source array to pivot from.  If ``None``, the first registered
            array is used.
        n : int, default 1
            Top-N matches per target array.  When ``n > 1`` the match
            columns are lists instead of scalars.
        how : {"full", "inner"}, default "full"
            *"inner"* drops source rows that have **no** match above
            ``score_cutoff`` in **any** target array.
        score_cutoff : float | None
            Per-target score threshold for *"inner"* mode.

        Returns
        -------
        list[dict]
            One dict per source element.  Fixed keys:
            ``src_array``, ``src_idx``, ``src_text``.
            Per-target keys (for each other array ``X``):
            ``match_X`` (str | list[str] | None),
            ``score_X`` (float | list[float] | None).

        Examples
        --------
        >>> joiner = (
        ...     MultiJoiner()
        ...     .add_array("products", texts=["iPhone", "Galaxy"])
        ...     .add_array("listings", texts=["iphone 14", "galaxy s23"])
        ...     .add_array("inventory", texts=["Apple iPhone", "Samsung Gal"])
        ... )
        >>> rows = joiner.join_wide("products", n=1)
        >>> rows[0]
        {'src_array': 'products', 'src_idx': 0, 'src_text': 'iPhone',
         'match_listings': 'iphone 14', 'score_listings': 0.025,
         'match_inventory': 'Apple iPhone', 'score_inventory': 0.025}
        """
        pivot_src = src_name if src_name is not None else (self._names[0] if self._names else "")
        tgt_names = [name for name in self._names if name != pivot_src]

        # Gather all pairwise rows for this source
        all_rows = self._inner.join(max(n, 1))
        src_rows = [r for r in all_rows if r["src_array"] == pivot_src]

        # Build wide rows indexed by src_idx
        src_elements: dict[int, dict[str, Any]] = {}
        for r in src_rows:
            idx = r["src_idx"]
            if idx not in src_elements:
                src_elements[idx] = {
                    "src_array": pivot_src,
                    "src_idx": idx,
                    "src_text": r["src_text"],
                }

        # Fill in per-target match columns
        cutoff = score_cutoff if score_cutoff is not None else 0.0
        for tgt in tgt_names:
            tgt_rows_map: dict[int, list[dict[str, Any]]] = {}
            for r in src_rows:
                if r["tgt_array"] != tgt:
                    continue
                tgt_rows_map.setdefault(r["src_idx"], []).append(r)

            for idx, wide_row in src_elements.items():
                matches = tgt_rows_map.get(idx, [])
                # Sort by score descending, apply cutoff in inner mode
                matches = sorted(matches, key=lambda r: r["score"], reverse=True)[:n]
                if how == "inner":
                    matches = [m for m in matches if m["score"] >= cutoff]

                if not matches:
                    if n == 1:
                        wide_row[f"match_{tgt}"] = None
                        wide_row[f"score_{tgt}"] = None
                    else:
                        wide_row[f"match_{tgt}"] = []
                        wide_row[f"score_{tgt}"] = []
                elif n == 1:
                    wide_row[f"match_{tgt}"] = matches[0]["tgt_text"]
                    wide_row[f"score_{tgt}"] = matches[0]["score"]
                else:
                    wide_row[f"match_{tgt}"] = [m["tgt_text"] for m in matches]
                    wide_row[f"score_{tgt}"] = [m["score"] for m in matches]

        result = list(src_elements.values())

        # inner mode: drop rows that have no match in ANY target
        if how == "inner":
            def _has_any_match(row: dict[str, Any]) -> bool:
                return any(
                    row.get(f"score_{t}") not in (None, [])
                    for t in tgt_names
                )
            result = [r for r in result if _has_any_match(r)]

        return sorted(result, key=lambda r: r["src_idx"])


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def fuzzy_join(
    arrays: dict[str, list[str]],
    *,
    sparse: dict[str, list[dict[int, float]]] | None = None,
    dense: dict[str, list[list[float]]] | None = None,
    text_weight: float = 1.0,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
    n: int = 1,
    how: _HOW = "full",
    score_cutoff: float | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Fuzzy full join (or inner join) across N named text arrays.

    Parameters
    ----------
    arrays : dict[str, list[str]]
        Named text arrays, e.g. ``{"A": [...], "B": [...], "C": [...]}``.
        Any number of arrays is supported.
    sparse : dict[str, list[dict[int, float]]], optional
        Named sparse vectors aligned to ``arrays``.
    dense : dict[str, list[list[float]]], optional
        Named dense embeddings aligned to ``arrays``.
    text_weight, sparse_weight, dense_weight : float, default 1.0
    n : int, default 1
        Top-N matches per element.
    how : {"full", "inner"}, default "full"
        *"full"*  — return all rows (classic full join).
        *"inner"* — only rows with ``score >= score_cutoff`` (inner join).
    score_cutoff : float | None
        Minimum score for inner join mode.
    bm25_k1, bm25_b, rrf_k
        BM25 / RRF tuning parameters.

    Returns
    -------
    list[dict]
        One row per (src_array, src_element, tgt_array, top-N match).

    Examples
    --------
    >>> from rustfuzz.join import fuzzy_join
    >>> rows = fuzzy_join({"A": ["iPhone"], "B": ["iphone 14"]}, n=1)
    >>> rows[0]["tgt_text"]
    'iphone 14'
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
    return joiner.join(n=n, how=how, score_cutoff=score_cutoff)


__all__ = ["MultiJoiner", "fuzzy_join"]
