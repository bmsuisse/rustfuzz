from collections.abc import Callable, Iterable
from typing import Any, Literal

BM25Algorithm = Literal["bm25", "bm25okapi", "bm25l", "bm25+", "bm25plus", "bm25t"]

_Result = tuple[str, float]
_MetaResult = tuple[str, float, Any]

class BM25:
    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
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
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[_Result] | list[_MetaResult]: ...
    def fuzzy_only(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def explain(self, query: str, doc: str | int) -> dict[str, Any]: ...
    def get_idf(self) -> dict[str, float]: ...
    def get_document_vector(self, doc_idx: int) -> dict[str, float]: ...
    def add_documents(
        self, docs: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None: ...
    def remove_documents(self, indices: list[int]) -> None: ...
    def get_top_n_reranked(
        self,
        query: str,
        n: int = 5,
        reranker: Any = None,
        rerank_candidates: int = 50,
        blend_alpha: float = 0.0,
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[_Result] | list[_MetaResult]: ...
    async def get_top_n_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]: ...
    async def search_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]: ...
    def filter(self, expression: str) -> Any: ...
    def sort(self, expression: list[str] | str) -> Any: ...
    def match(self, query: str, **kwargs: Any) -> Any: ...
    def rerank(
        self,
        model_or_callable: Any,
        top_k: int = 10,
        blend_alpha: float = 0.0,
        adaptive_blend: bool = False,
    ) -> Any: ...
    def to_hybrid(self, embeddings: Any) -> HybridSearch: ...

class BM25L:
    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
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
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[_Result] | list[_MetaResult]: ...
    def fuzzy_only(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def explain(self, query: str, doc: str | int) -> dict[str, Any]: ...
    def get_idf(self) -> dict[str, float]: ...
    def get_document_vector(self, doc_idx: int) -> dict[str, float]: ...
    def add_documents(
        self, docs: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None: ...
    def remove_documents(self, indices: list[int]) -> None: ...
    def get_top_n_reranked(
        self,
        query: str,
        n: int = 5,
        reranker: Any = None,
        rerank_candidates: int = 50,
        blend_alpha: float = 0.0,
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[_Result] | list[_MetaResult]: ...
    async def get_top_n_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]: ...
    async def search_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]: ...
    def filter(self, expression: str) -> Any: ...
    def sort(self, expression: list[str] | str) -> Any: ...
    def match(self, query: str, **kwargs: Any) -> Any: ...
    def rerank(
        self,
        model_or_callable: Any,
        top_k: int = 10,
        blend_alpha: float = 0.0,
        adaptive_blend: bool = False,
    ) -> Any: ...
    def to_hybrid(self, embeddings: Any) -> HybridSearch: ...

class BM25Plus:
    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
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
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[_Result] | list[_MetaResult]: ...
    def fuzzy_only(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def explain(self, query: str, doc: str | int) -> dict[str, Any]: ...
    def get_idf(self) -> dict[str, float]: ...
    def get_document_vector(self, doc_idx: int) -> dict[str, float]: ...
    def add_documents(
        self, docs: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None: ...
    def remove_documents(self, indices: list[int]) -> None: ...
    def get_top_n_reranked(
        self,
        query: str,
        n: int = 5,
        reranker: Any = None,
        rerank_candidates: int = 50,
        blend_alpha: float = 0.0,
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[_Result] | list[_MetaResult]: ...
    async def get_top_n_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]: ...
    async def search_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]: ...
    def filter(self, expression: str) -> Any: ...
    def sort(self, expression: list[str] | str) -> Any: ...
    def match(self, query: str, **kwargs: Any) -> Any: ...
    def rerank(
        self,
        model_or_callable: Any,
        top_k: int = 10,
        blend_alpha: float = 0.0,
        adaptive_blend: bool = False,
    ) -> Any: ...
    def to_hybrid(self, embeddings: Any) -> HybridSearch: ...

class BM25T:
    def __init__(
        self,
        corpus: Iterable[str] | Any,
        k1: float = 1.5,
        b: float = 0.75,
        metadata: Iterable[Any] | None = None,
        normalize: bool = False,
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
        self, query: str, n: int = 5, bm25_candidates: int = 100, rrf_k: int = 60
    ) -> list[_Result] | list[_MetaResult]: ...
    def fuzzy_only(
        self, query: str, n: int = 5
    ) -> list[_Result] | list[_MetaResult]: ...
    def explain(self, query: str, doc: str | int) -> dict[str, Any]: ...
    def get_idf(self) -> dict[str, float]: ...
    def get_document_vector(self, doc_idx: int) -> dict[str, float]: ...
    def add_documents(
        self, docs: Iterable[str], metadata: Iterable[Any] | None = None
    ) -> None: ...
    def remove_documents(self, indices: list[int]) -> None: ...
    def get_top_n_reranked(
        self,
        query: str,
        n: int = 5,
        reranker: Any = None,
        rerank_candidates: int = 50,
        blend_alpha: float = 0.0,
    ) -> list[_Result] | list[_MetaResult]: ...
    def get_top_n_phrase(
        self,
        query: str,
        n: int = 5,
        proximity_window: int = 3,
        phrase_boost: float = 2.0,
    ) -> list[_Result] | list[_MetaResult]: ...
    async def get_top_n_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]: ...
    async def search_async(
        self, query: str, n: int = 5, **kwargs: Any
    ) -> list[_Result] | list[_MetaResult]: ...
    def filter(self, expression: str) -> Any: ...
    def sort(self, expression: list[str] | str) -> Any: ...
    def match(self, query: str, **kwargs: Any) -> Any: ...
    def rerank(
        self,
        model_or_callable: Any,
        top_k: int = 10,
        blend_alpha: float = 0.0,
        adaptive_blend: bool = False,
    ) -> Any: ...
    def to_hybrid(self, embeddings: Any) -> HybridSearch: ...

class Document:
    """A lightweight document with content and metadata."""

    content: str
    metadata: dict[str, Any]
    _vector: list[float] | None
    def __init__(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        _vector: list[float] | None = None,
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
        algorithm: BM25Algorithm = "bm25",
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
    def filter(self, expression: str) -> Any: ...
    def sort(self, expression: list[str] | str) -> Any: ...
    def match(self, query: str, **kwargs: Any) -> Any: ...
    def rerank(
        self,
        model_or_callable: Any,
        top_k: int = 10,
        blend_alpha: float = 0.0,
        adaptive_blend: bool = False,
    ) -> Any: ...

class Reranker:
    def __init__(
        self,
        model_or_callable: Any,
        blend_alpha: float = 0.0,
        adaptive_blend: bool = False,
    ) -> None: ...
    def rerank(
        self,
        query: str,
        results: list[_Result] | list[_MetaResult],
        top_k: int = 10,
    ) -> list[_Result] | list[_MetaResult]: ...

__all__ = [
    "BM25",
    "BM25Algorithm",
    "BM25L",
    "BM25Plus",
    "BM25T",
    "Document",
    "HybridSearch",
    "Reranker",
]
