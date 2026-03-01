import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

_Result = tuple[str, float]
_MetaResult = tuple[str, float, Any]

@dataclasses.dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for :class:`Retriever`."""

    algorithm: str = "bm25plus"
    k1: float = 1.5
    b: float = 0.75
    delta: float | None = None
    normalize: bool = True
    rerank_top_k: int = 10

class Retriever:
    """Batteries-included retriever — SOTA search in 3 lines."""

    def __init__(
        self,
        corpus: Iterable[str] | Iterable[Any] | Any,
        *,
        embeddings: Any | str | bool | Callable[[list[str]], list[list[float]]] | None = None,
        reranker: Any | None = None,
        config: RetrieverConfig | None = None,
        metadata: Iterable[Any] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        algorithm: str = "bm25plus",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float | None = None,
        normalize: bool = True,
        rerank_top_k: int = 10,
    ) -> None: ...
    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        column: str,
        metadata_columns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> Retriever: ...
    @property
    def config(self) -> RetrieverConfig: ...
    @property
    def num_docs(self) -> int: ...
    @property
    def has_embeddings(self) -> bool: ...
    @property
    def has_reranker(self) -> bool: ...
    def search(
        self,
        query: str,
        *,
        n: int = 10,
        query_embedding: Any = None,
    ) -> list[_Result] | list[_MetaResult]: ...
    def filter(self, expression: str) -> Any: ...
    def sort(self, expression: list[str] | str) -> Any: ...
    def match(self, query: str, **kwargs: Any) -> Any: ...
    def rerank(self, model_or_callable: Any, top_k: int = 10) -> Any: ...
    def get_top_n(
        self,
        query: str,
        n: int = 5,
        *,
        query_embedding: Any = None,
    ) -> list[_Result] | list[_MetaResult]: ...
    def to_hybrid(
        self, embeddings: Any | str | bool | None = True
    ) -> Retriever: ...

__all__ = ["Retriever", "RetrieverConfig"]
