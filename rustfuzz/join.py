"""
rustfuzz.join — Multi-array fuzzy full join.

All heavy computation (BM25, indel fuzzy, cosine, RRF) runs in Rust/Rayon
outside the GIL. Python is glue only.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from . import _rustfuzz
from .compat import _coerce_to_strings

_HOW = Literal["full", "inner"]


class MultiJoiner:
    """
    Multi-array fuzzy joiner backed by Rust.

    Seamlessly combines lexical vectors (BM25 + Indel) with semantic vectors
    (Dense Embeddings) to perform true Hybrid Search natively outside the Python GIL.

    Parameters
    ----------
    text_weight, sparse_weight, dense_weight : float, default 1.0
        Weights applied to each channel during Reciprocal Rank Fusion (RRF).
    bm25_k1 : float, default 1.5
    bm25_b : float, default 0.75
    bm25_variant : str, default "BM25Okapi"
        The mathematical BM25 formulation used for the text channel. Works
        perfectly in tandem with Dense vectors (provided via `add_array(dense=...)`).
    rrf_k : int, default 60
    bm25_candidates : int, default 100
        Max number of BM25 top-k docs passed to the fuzzy re-ranker.
        Limits O(n²) fuzzy scoring to O(n×K). Lower = faster (less recall).
    """

    def __init__(
        self,
        *,
        text_weight: float = 1.0,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        bm25_variant: str = "BM25Okapi",
        rrf_k: int = 60,
        bm25_candidates: int = 100,
    ) -> None:
        self._inner = _rustfuzz.MultiJoiner(
            text_weight,
            sparse_weight,
            dense_weight,
            bm25_k1,
            bm25_b,
            bm25_variant,
            rrf_k,
            bm25_candidates,
        )
        self._names: list[str] = []

    @property
    def num_arrays(self) -> int:
        return self._inner.num_arrays

    @property
    def array_names(self) -> list[str]:
        return list(self._names)

    def add_array(
        self,
        name: str,
        texts: Iterable[str | None] | Any | None = None,
        sparse: Iterable[dict[int, float] | None] | None = None,
        dense: Iterable[list[float] | None] | None = None,
    ) -> MultiJoiner:
        """Register a named array (chainable). At least one channel required."""
        self._inner.add_array(
            name,
            _coerce_to_strings(texts) if texts is not None else None,
            list(sparse) if sparse is not None else None,  # type: ignore[arg-type]
            list(dense) if dense is not None else None,
        )
        self._names.append(name)
        return self

    # ------------------------------------------------------------------
    # Pairwise joins
    # ------------------------------------------------------------------

    def join(
        self,
        n: int = 1,
        *,
        how: _HOW = "full",
        score_cutoff: float | None = None,
    ) -> list[dict[str, Any]]:
        """Full or inner join across every ordered array pair."""
        cutoff = score_cutoff if how == "inner" else None
        return self._inner.join(n, cutoff)  # type: ignore[return-value]

    def join_pair(
        self,
        src_name: str,
        tgt_name: str,
        n: int = 1,
        *,
        how: _HOW = "full",
        score_cutoff: float | None = None,
    ) -> list[dict[str, Any]]:
        """Join a single ordered pair."""
        cutoff = score_cutoff if how == "inner" else None
        return self._inner.join_pair(src_name, tgt_name, n, cutoff)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Wide (pivoted) join
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
        Pivoted join — one row per source element with match_X/score_X columns
        for every other registered array.
        """
        pivot_src = (
            src_name
            if src_name is not None
            else (self._names[0] if self._names else "")
        )
        tgt_names = [nm for nm in self._names if nm != pivot_src]
        cutoff = score_cutoff if how == "inner" else None

        all_rows: list[dict[str, Any]] = self._inner.join(max(n, 1), None)  # type: ignore[assignment]
        src_rows = [r for r in all_rows if r["src_array"] == pivot_src]

        src_elements: dict[int, dict[str, Any]] = {}
        for r in src_rows:
            idx = r["src_idx"]
            if idx not in src_elements:
                src_elements[idx] = {
                    "src_array": pivot_src,
                    "src_idx": idx,
                    "src_text": r["src_text"],
                }

        for tgt in tgt_names:
            tgt_map: dict[int, list[dict[str, Any]]] = {}
            for r in src_rows:
                if r["tgt_array"] == tgt:
                    tgt_map.setdefault(r["src_idx"], []).append(r)

            for idx, wide_row in src_elements.items():
                matches = sorted(
                    tgt_map.get(idx, []), key=lambda r: r["score"], reverse=True
                )[:n]
                if cutoff is not None:
                    matches = [m for m in matches if m["score"] >= cutoff]

                if not matches:
                    wide_row[f"match_{tgt}"] = None if n == 1 else []
                    wide_row[f"score_{tgt}"] = None if n == 1 else []
                elif n == 1:
                    wide_row[f"match_{tgt}"] = matches[0]["tgt_text"]
                    wide_row[f"score_{tgt}"] = matches[0]["score"]
                else:
                    wide_row[f"match_{tgt}"] = [m["tgt_text"] for m in matches]
                    wide_row[f"score_{tgt}"] = [m["score"] for m in matches]

        result = sorted(src_elements.values(), key=lambda r: r["src_idx"])

        if how == "inner":
            result = [
                r
                for r in result
                if any(r.get(f"score_{t}") not in (None, []) for t in tgt_names)
            ]

        return result


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
    bm25_variant: str = "BM25Okapi",
    rrf_k: int = 60,
    bm25_candidates: int = 100,
) -> list[dict[str, Any]]:
    """Fuzzy full/inner join across N named arrays."""
    joiner = MultiJoiner(
        text_weight=text_weight,
        sparse_weight=sparse_weight,
        dense_weight=dense_weight,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        bm25_variant=bm25_variant,
        rrf_k=rrf_k,
        bm25_candidates=bm25_candidates,
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
