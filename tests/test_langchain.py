"""Tests for rustfuzz.langchain — LangChain BM25 retriever integration."""

from __future__ import annotations

import pytest

# langchain_core may not be installed in all environments
langchain = pytest.importorskip("langchain_core")
pydantic = pytest.importorskip("pydantic")

from langchain_core.documents import Document  # type: ignore  # noqa: E402


@pytest.fixture
def sample_docs() -> list[Document]:
    """Fixture providing sample LangChain documents."""
    return [
        Document(page_content="The dog chased the cat."),
        Document(page_content="The cat chased the mouse."),
        Document(page_content="A bird flew over the house."),
        Document(page_content="The fish swam in the pond."),
    ]


@pytest.fixture
def sample_texts() -> list[str]:
    """Fixture providing sample text strings."""
    return [
        "The dog chased the cat.",
        "The cat chased the mouse.",
        "A bird flew over the house.",
        "The fish swam in the pond.",
    ]


class TestRustfuzzBM25Retriever:
    """Tests for the lazy-loaded RustfuzzBM25Retriever class."""

    def test_lazy_import(self) -> None:
        """Test that RustfuzzBM25Retriever is importable via lazy __getattr__."""
        from rustfuzz.langchain import RustfuzzBM25Retriever

        assert RustfuzzBM25Retriever is not None

    def test_from_texts(self, sample_texts: list[str]) -> None:
        """Test creating retriever from plain text strings."""
        from rustfuzz.langchain import RustfuzzBM25Retriever

        retriever = RustfuzzBM25Retriever.from_texts(sample_texts)
        assert retriever is not None
        assert len(retriever.docs) == 4

    def test_from_texts_with_metadata(self, sample_texts: list[str]) -> None:
        """Test creating retriever with metadata."""
        from rustfuzz.langchain import RustfuzzBM25Retriever

        metadatas = [{"idx": i} for i in range(len(sample_texts))]
        retriever = RustfuzzBM25Retriever.from_texts(sample_texts, metadatas=metadatas)
        assert len(retriever.docs) == 4
        assert retriever.docs[0].metadata == {"idx": 0}

    def test_from_documents(self, sample_docs: list[Document]) -> None:
        """Test creating retriever from LangChain Document objects."""
        from rustfuzz.langchain import RustfuzzBM25Retriever

        retriever = RustfuzzBM25Retriever.from_documents(sample_docs)
        assert len(retriever.docs) == 4

    def test_invoke_returns_relevant_documents(
        self, sample_docs: list[Document]
    ) -> None:
        """Test that invoke returns relevant documents sorted by score."""
        from rustfuzz.langchain import RustfuzzBM25Retriever

        retriever = RustfuzzBM25Retriever.from_documents(sample_docs, k1=1.5, b=0.75)
        results = retriever.invoke("cat")

        assert len(results) > 0
        assert all(isinstance(d, Document) for d in results)
        # "cat" appears in first two documents
        result_texts = [d.page_content for d in results]
        assert any("cat" in t for t in result_texts)

    def test_invoke_returns_score_in_metadata(
        self, sample_docs: list[Document]
    ) -> None:
        """Test that results include BM25 score in metadata."""
        from rustfuzz.langchain import RustfuzzBM25Retriever

        retriever = RustfuzzBM25Retriever.from_documents(sample_docs)
        results = retriever.invoke("dog")

        assert len(results) > 0
        for doc in results:
            assert "score" in doc.metadata
            assert doc.metadata["score"] > 0.0

    def test_invoke_respects_k_parameter(self, sample_docs: list[Document]) -> None:
        """Test that k parameter limits number of results."""
        from rustfuzz.langchain import RustfuzzBM25Retriever

        retriever = RustfuzzBM25Retriever.from_documents(sample_docs)
        retriever.k = 2
        results = retriever.invoke("the")

        assert len(results) <= 2

    def test_invoke_no_results_for_unknown_query(
        self, sample_docs: list[Document]
    ) -> None:
        """Test that query with no matches returns empty list."""
        from rustfuzz.langchain import RustfuzzBM25Retriever

        retriever = RustfuzzBM25Retriever.from_documents(sample_docs)
        results = retriever.invoke("xyzzyplugh")

        assert len(results) == 0

    def test_getattr_raises_for_unknown(self) -> None:
        """Test that accessing unknown attributes raises AttributeError."""
        import rustfuzz.langchain as langchain_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = langchain_mod.NonExistentClass  # type: ignore

    def test_custom_k1_b_parameters(self, sample_texts: list[str]) -> None:
        """Test that custom BM25 parameters are passed through."""
        from rustfuzz.langchain import RustfuzzBM25Retriever

        retriever = RustfuzzBM25Retriever.from_texts(sample_texts, k1=2.0, b=0.5)
        assert retriever is not None
        results = retriever.invoke("cat")
        assert len(results) > 0
