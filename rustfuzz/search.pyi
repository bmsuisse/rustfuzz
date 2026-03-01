from collections.abc import Callable, Iterable
from typing import Any

_Result = tuple[str, float]
_MetaResult = tuple[str, float, Any]

class BM25:
    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
    ) -> None: ...
    @classmethod
    def from_column(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> BM25: ...
    @property
    def num_docs(self) -> int: ...
    def get_scores(self, query: str) -> list[float]: ...
    def get_top_n(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_batch_scores(self, queries: Iterable[str]) -> list[list[float]]: ...
    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_top_n_rrf(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 100,
        rrf_k: int = 60,
    ) -> list[_Result] | list[_MetaResult]: ...
    def fuzzy_only(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def to_hybrid(self, embeddings: Any) -> HybridSearch: ...

class BM25L:
    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        metadata: Iterable[Any] | None = None,
    ) -> None: ...
    @classmethod
    def from_column(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> BM25L: ...
    @property
    def num_docs(self) -> int: ...
    def get_scores(self, query: str) -> list[float]: ...
    def get_top_n(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_batch_scores(self, queries: Iterable[str]) -> list[list[float]]: ...
    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_top_n_rrf(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 100,
        rrf_k: int = 60,
    ) -> list[_Result] | list[_MetaResult]: ...
    def fuzzy_only(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def to_hybrid(self, embeddings: Any) -> HybridSearch: ...

class BM25Plus:
    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        metadata: Iterable[Any] | None = None,
    ) -> None: ...
    @classmethod
    def from_column(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> BM25Plus: ...
    @property
    def num_docs(self) -> int: ...
    def get_scores(self, query: str) -> list[float]: ...
    def get_top_n(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_batch_scores(self, queries: Iterable[str]) -> list[list[float]]: ...
    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_top_n_rrf(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 100,
        rrf_k: int = 60,
    ) -> list[_Result] | list[_MetaResult]: ...
    def fuzzy_only(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def to_hybrid(self, embeddings: Any) -> HybridSearch: ...

class BM25T:
    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
    ) -> None: ...
    @classmethod
    def from_column(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> BM25T: ...
    @property
    def num_docs(self) -> int: ...
    def get_scores(self, query: str) -> list[float]: ...
    def get_top_n(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_batch_scores(self, queries: Iterable[str]) -> list[list[float]]: ...
    def get_top_n_fuzzy(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 50,
        fuzzy_weight: float = 0.3,
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_top_n_rrf(
        self,
        query: str,
        n: int = 5,
        bm25_candidates: int = 100,
        rrf_k: int = 60,
    ) -> list[_Result] | list[_MetaResult]: ...
    def fuzzy_only(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def to_hybrid(self, embeddings: Any) -> HybridSearch: ...

class Document:
    """A lightweight document with content and metadata."""

    content: str
    metadata: dict[str, Any]
    def __init__(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class HybridSearch:
    """3-way hybrid search: BM25 + Fuzzy + Dense via RRF, all in Rust."""

    def __init__(
        self,
        corpus: Iterable[str] | Iterable[Any] | Any,
        embeddings: Any | Callable[[list[str]], list[list[float]]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
        algorithm: str = "bm25",
        delta: float | None = None,
    ) -> None: ...
    @property
    def has_vectors(self) -> bool: ...
    @property
    def num_docs(self) -> int: ...
    def search(
        self,
        query: str,
        query_embedding: Any = None,
        n: int = 5,
        rrf_k: int = 60,
        bm25_candidates: int = 100,
    ) -> list[_Result] | list[_MetaResult]: ...

__all__ = ["BM25", "BM25L", "BM25Plus", "BM25T", "Document", "HybridSearch"]
