from typing import Any

class MultiJoiner:
    """Rust-backed multi-array fuzzy full joiner."""

    def __init__(
        self,
        *,
        text_weight: float = 1.0,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
    ) -> None: ...
    @property
    def num_arrays(self) -> int: ...
    def add_array(
        self,
        name: str,
        texts: list[str | None] | None = None,
        sparse: list[dict[int, float] | None] | None = None,
        dense: list[list[float] | None] | None = None,
    ) -> MultiJoiner: ...
    def join(self, n: int = 1) -> list[dict[str, Any]]: ...
    def join_pair(
        self, src_name: str, tgt_name: str, n: int = 1
    ) -> list[dict[str, Any]]: ...

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
) -> list[dict[str, Any]]: ...

__all__ = ["MultiJoiner", "fuzzy_join"]
