"""LangChain retriever integration for rustfuzz.search.BM25."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun  # type: ignore
    from langchain_core.documents import Document  # type: ignore

from rustfuzz.search import BM25


def _import_langchain() -> tuple[Any, Any, Any, Any]:
    """Lazy-import langchain_core and pydantic, raising a helpful error."""
    try:
        from langchain_core.callbacks import (
            CallbackManagerForRetrieverRun,  # type: ignore
        )
        from langchain_core.documents import Document  # type: ignore
        from langchain_core.retrievers import BaseRetriever  # type: ignore
        from pydantic import Field  # type: ignore
    except ImportError as err:
        raise ImportError(
            "Could not import langchain_core or pydantic. "
            "Please install it with `pip install langchain-core pydantic`."
        ) from err
    return BaseRetriever, Document, CallbackManagerForRetrieverRun, Field


def _build_retriever_class() -> type:
    """Build the retriever class lazily so imports only happen on first use."""
    BaseRetriever, Document, CallbackManagerForRetrieverRun, Field = _import_langchain()

    class RustfuzzBM25Retriever(BaseRetriever):
        """
        `rustfuzz` BM25 retriever for LangChain.

        Setup:
            Install ``rustfuzz`` and ``langchain-core``:

            .. code-block:: bash

                pip install rustfuzz langchain-core

        Instantiate:
            .. code-block:: python

                from rustfuzz.langchain import RustfuzzBM25Retriever
                from langchain_core.documents import Document

                docs = [
                    Document(page_content="The dog chased the cat."),
                    Document(page_content="The cat chased the mouse."),
                ]
                retriever = RustfuzzBM25Retriever.from_documents(docs)

        Usage:
            .. code-block:: python

                retriever.invoke("cat")
        """

        vectorizer: Any
        """The Rustfuzz BM25 vectorizer instance."""
        docs: list[Document] = Field(repr=False)  # type: ignore[assignment]
        """List of documents."""
        k: int = 4
        """Number of documents to return."""

        class Config:
            """Configuration for this pydantic object."""

            arbitrary_types_allowed = True

        @classmethod
        def from_texts(
            cls,
            texts: Iterable[str],
            metadatas: Iterable[dict[Any, Any]] | None = None,
            k1: float = 1.5,
            b: float = 0.75,
            **kwargs: Any,
        ) -> RustfuzzBM25Retriever:
            """
            Create a RustfuzzBM25Retriever from a list of texts.
            """
            texts_list = list(texts)
            if metadatas is not None:
                docs = [
                    Document(page_content=t, metadata=m)
                    for t, m in zip(texts_list, metadatas, strict=False)
                ]
            else:
                docs = [Document(page_content=t) for t in texts_list]

            vectorizer = BM25(texts_list, k1=k1, b=b)
            return cls(vectorizer=vectorizer, docs=docs, **kwargs)

        @classmethod
        def from_documents(
            cls,
            documents: Iterable[Document],
            k1: float = 1.5,
            b: float = 0.75,
            **kwargs: Any,
        ) -> RustfuzzBM25Retriever:
            """
            Create a RustfuzzBM25Retriever from a list of Documents.
            """
            docs = list(documents)
            texts = [doc.page_content for doc in docs]
            vectorizer = BM25(texts, k1=k1, b=b)
            return cls(vectorizer=vectorizer, docs=docs, **kwargs)

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            """Get documents relevant to a query."""
            scores = self.vectorizer.get_scores(query)

            # Collect positive scores and sort descending
            scored_docs = [
                (score, idx) for idx, score in enumerate(scores) if score > 0.0
            ]
            scored_docs.sort(reverse=True, key=lambda x: x[0])

            top_k = scored_docs[: self.k]

            results = []
            for score, idx in top_k:
                doc = self.docs[idx]
                # Create a copy with the score added to the metadata
                meta = doc.metadata.copy() if doc.metadata else {}
                meta["score"] = score
                results.append(Document(page_content=doc.page_content, metadata=meta))

            return results

    return RustfuzzBM25Retriever


def __getattr__(name: str) -> Any:
    if name == "RustfuzzBM25Retriever":
        cls = _build_retriever_class()
        globals()["RustfuzzBM25Retriever"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RustfuzzBM25Retriever"]  # noqa: F822
