"""Tests for 3-way hybrid search: BM25 + Fuzzy + Dense via RRF."""

from __future__ import annotations

from typing import Any

from rustfuzz.search import Document, HybridSearch

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumped over a lazy dog",
    "a lazy dog",
    "the fast brown fox",
    "jumping over dogs",
]

# Dummy embeddings: 5 docs, 3 dimensions
# Doc 0 and Doc 3 are "fox-like" (close to [1,0,0])
# Doc 2 and Doc 4 are "dog-like" (close to [0,1,0])
EMBEDDINGS = [
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [0.0, 1.0, 0.0],
    [0.95, 0.05, 0.0],
    [0.0, 0.8, 0.2],
]


class TestHybridSearch3Way:
    """Test 3-way RRF (BM25 + fuzzy + dense)."""

    def test_basic_3way(self) -> None:
        hs = HybridSearch(CORPUS, embeddings=EMBEDDINGS)
        assert hs.has_vectors
        assert hs.num_docs == 5

        # Query with embedding matching "fox" docs
        q_emb = [1.0, 0.0, 0.0]
        results = hs.search("fox", query_embedding=q_emb, n=3)
        assert len(results) == 3
        # Top result should be a fox document
        assert "fox" in results[0][0]

    def test_3way_boosts_semantic_match(self) -> None:
        """When BM25 and dense agree, 3-way should rank higher."""
        hs = HybridSearch(CORPUS, embeddings=EMBEDDINGS)
        q_emb = [1.0, 0.0, 0.0]
        results = hs.search("brown fox", query_embedding=q_emb, n=5)

        # Doc 0 and 3 match both BM25 ("fox") and dense (close to [1,0,0])
        top_docs = [r[0] for r in results[:2]]
        assert any("fox" in d for d in top_docs)

    def test_fallback_2way_no_embedding(self) -> None:
        """Without query_embedding, falls back to 2-way BM25+fuzzy."""
        hs = HybridSearch(CORPUS, embeddings=EMBEDDINGS)
        results = hs.search("fox", n=3)
        assert len(results) == 3
        assert "fox" in results[0][0]

    def test_fallback_2way_no_vectors(self) -> None:
        """Without corpus embeddings, falls back to 2-way BM25+fuzzy."""
        hs = HybridSearch(CORPUS)
        assert not hs.has_vectors
        results = hs.search("fox", n=3)
        assert len(results) == 3
        assert "fox" in results[0][0]

    def test_typo_resilience(self) -> None:
        """Typos should be handled by fuzzy signal."""
        hs = HybridSearch(CORPUS, embeddings=EMBEDDINGS)
        q_emb = [1.0, 0.0, 0.0]
        results = hs.search("quik brwn fxo", query_embedding=q_emb, n=3)
        assert len(results) > 0
        # Fuzzy + dense should still surface "fox" documents
        assert any("fox" in r[0] for r in results)

    def test_empty_corpus(self) -> None:
        hs = HybridSearch([])
        assert hs.num_docs == 0
        results = hs.search("anything", n=5)
        assert results == []

    def test_bm25_candidates_param(self) -> None:
        """bm25_candidates controls the candidate pool size."""
        hs = HybridSearch(CORPUS, embeddings=EMBEDDINGS)
        q_emb = [1.0, 0.0, 0.0]

        # Small candidate pool
        r1 = hs.search("fox", query_embedding=q_emb, n=3, bm25_candidates=3)
        # Large candidate pool
        r2 = hs.search("fox", query_embedding=q_emb, n=3, bm25_candidates=100)

        # Both should return results
        assert len(r1) > 0
        assert len(r2) > 0


class TestDocument:
    """Test Document class."""

    def test_basic(self) -> None:
        doc = Document("hello world")
        assert doc.content == "hello world"
        assert doc.metadata == {}

    def test_with_metadata(self) -> None:
        doc = Document("hello", metadata={"id": 1, "category": "greeting"})
        assert doc.content == "hello"
        assert doc.metadata == {"id": 1, "category": "greeting"}

    def test_repr(self) -> None:
        doc = Document("short text")
        assert "short text" in repr(doc)

    def test_repr_long(self) -> None:
        doc = Document("a" * 100)
        r = repr(doc)
        assert "..." in r

    def test_equality(self) -> None:
        d1 = Document("hello", {"a": 1})
        d2 = Document("hello", {"a": 1})
        d3 = Document("hello", {"a": 2})
        assert d1 == d2
        assert d1 != d3

    def test_not_equal_to_other_types(self) -> None:
        doc = Document("hello")
        assert doc != "hello"


class TestDocumentCorpus:
    """Test HybridSearch with Document objects."""

    def test_document_list(self) -> None:
        docs = [
            Document("Apple iPhone 15 Pro", {"brand": "Apple", "id": 1}),
            Document("Samsung Galaxy S24", {"brand": "Samsung", "id": 2}),
            Document("Google Pixel 8 Pro", {"brand": "Google", "id": 3}),
        ]
        hs = HybridSearch(docs)
        results = hs.search("apple iphone", n=1)
        assert len(results) == 1
        text, score, meta = results[0]
        assert "Apple" in text
        assert meta["brand"] == "Apple"
        assert meta["id"] == 1

    def test_document_with_embeddings(self) -> None:
        docs = [
            Document("Apple iPhone", {"brand": "Apple"}),
            Document("Samsung Galaxy", {"brand": "Samsung"}),
        ]
        emb = [[1.0, 0.0], [0.0, 1.0]]
        hs = HybridSearch(docs, embeddings=emb)
        assert hs.has_vectors
        results = hs.search("iphone", query_embedding=[1.0, 0.0], n=1)
        assert len(results) == 1
        assert results[0][2]["brand"] == "Apple"  # type: ignore[index]

    def test_explicit_metadata_overrides_document_metadata(self) -> None:
        docs = [
            Document("Apple iPhone", {"brand": "Apple"}),
            Document("Samsung Galaxy", {"brand": "Samsung"}),
        ]
        override_meta = [{"custom": "a"}, {"custom": "b"}]
        hs = HybridSearch(docs, metadata=override_meta)
        results = hs.search("apple iphone", n=1)
        assert results[0][2]["custom"] == "a"  # type: ignore[index]


class TestLangChainCompat:
    """Test duck-typed LangChain Document compatibility."""

    def test_langchain_documents(self) -> None:
        """Simulate LangChain Document objects without importing langchain."""

        class FakeLCDoc:
            def __init__(
                self, page_content: str, metadata: dict[str, Any] | None = None
            ):
                self.page_content = page_content
                self.metadata = metadata or {}

        lc_docs = [
            FakeLCDoc("Apple iPhone 15 Pro Max", {"source": "catalog", "price": 999}),
            FakeLCDoc("Samsung Galaxy S24 Ultra", {"source": "catalog", "price": 899}),
            FakeLCDoc("Google Pixel 8 Pro", {"source": "catalog", "price": 699}),
        ]

        hs = HybridSearch(lc_docs)
        results = hs.search("apple iphone", n=1)
        assert len(results) == 1
        text, score, meta = results[0]
        assert "Apple" in text
        assert meta["source"] == "catalog"
        assert meta["price"] == 999

    def test_langchain_with_embeddings(self) -> None:
        class FakeLCDoc:
            def __init__(
                self, page_content: str, metadata: dict[str, Any] | None = None
            ):
                self.page_content = page_content
                self.metadata = metadata or {}

        lc_docs = [
            FakeLCDoc("Apple iPhone", {"brand": "Apple"}),
            FakeLCDoc("Samsung Galaxy", {"brand": "Samsung"}),
        ]
        emb = [[1.0, 0.0], [0.0, 1.0]]
        hs = HybridSearch(lc_docs, embeddings=emb)

        results = hs.search("iphone", query_embedding=[1.0, 0.0], n=1)
        assert len(results) == 1
        assert results[0][2]["brand"] == "Apple"  # type: ignore[index]


class TestMetadata:
    """Test metadata handling in HybridSearch."""

    def test_string_corpus_with_metadata(self) -> None:
        hs = HybridSearch(
            CORPUS,
            metadata=[{"i": i} for i in range(len(CORPUS))],
        )
        results = hs.search("fox", n=2)
        assert len(results) == 2
        for _text, _score, meta in results:
            assert isinstance(meta, dict)
            assert "i" in meta

    def test_no_metadata(self) -> None:
        hs = HybridSearch(CORPUS)
        results = hs.search("fox", n=2)
        assert len(results) == 2
        for r in results:
            assert len(r) == 2  # (text, score) — no metadata


class TestEmbeddingCallback:
    """Test embedding callback support in HybridSearch."""

    @staticmethod
    def _dummy_embed(texts: list[str]) -> list[list[float]]:
        """Simple callback: hash-based deterministic embeddings."""
        return [[float(hash(t) % 100) / 100, 1.0 - float(hash(t) % 100) / 100] for t in texts]

    @staticmethod
    def _fox_embed(texts: list[str]) -> list[list[float]]:
        """Callback that biases 'fox' texts toward [1, 0]."""
        result: list[list[float]] = []
        for t in texts:
            if "fox" in t.lower():
                result.append([1.0, 0.0])
            else:
                result.append([0.0, 1.0])
        return result

    def test_callback_basic(self) -> None:
        """Callable embeddings should produce has_vectors=True."""
        hs = HybridSearch(CORPUS, embeddings=self._dummy_embed)
        assert hs.has_vectors
        assert hs.num_docs == 5

    def test_callback_auto_query_embedding(self) -> None:
        """When query_embedding is omitted, callback generates it automatically."""
        hs = HybridSearch(CORPUS, embeddings=self._fox_embed)
        # Search without explicit query_embedding — callback should fire
        results = hs.search("fox", n=3)
        assert len(results) == 3
        # "fox" query → [1,0] embedding → should boost fox documents
        assert "fox" in results[0][0]

    def test_callback_explicit_override(self) -> None:
        """Explicit query_embedding takes precedence over callback."""
        hs = HybridSearch(CORPUS, embeddings=self._fox_embed)
        # Pass a "dog-like" embedding explicitly → should boost dog docs
        results = hs.search("fox", query_embedding=[0.0, 1.0], n=3)
        assert len(results) == 3
        # We still get results (BM25 + fuzzy still match fox), just different ranking

    def test_callback_wrong_length_raises(self) -> None:
        """Callback returning wrong number of vectors should raise ValueError."""
        def bad_embed(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]  # always 1 vector regardless of corpus size

        import pytest

        try:
            HybridSearch(CORPUS, embeddings=bad_embed)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            assert "5 documents" in str(e) or "1 vectors" in str(e)

    def test_callback_with_documents(self) -> None:
        """Callback should work with Document objects."""
        docs = [
            Document("Apple iPhone", {"brand": "Apple"}),
            Document("Samsung Galaxy", {"brand": "Samsung"}),
        ]

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "Apple" in t else [0.0, 1.0] for t in texts]

        hs = HybridSearch(docs, embeddings=embed_fn)
        assert hs.has_vectors
        # Auto-generated query embedding for "Apple" → [1, 0]
        results = hs.search("apple iphone", n=1)
        assert len(results) == 1
        assert results[0][2]["brand"] == "Apple"  # type: ignore[index]

    def test_callback_empty_corpus(self) -> None:
        """Callback with empty corpus should work (callback not called)."""
        call_count = 0

        def counting_embed(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            return [[1.0] for _ in texts]

        hs = HybridSearch([], embeddings=counting_embed)
        assert hs.num_docs == 0
        assert call_count == 0  # Not called for empty corpus

    def test_callback_not_pickled(self) -> None:
        """Pickle round-trip uses the computed matrix, not the callback."""
        import pickle

        hs = HybridSearch(CORPUS, embeddings=self._fox_embed)
        assert hs.has_vectors

        data = pickle.dumps(hs)
        hs2 = pickle.loads(data)
        assert hs2.has_vectors
        assert hs2.num_docs == 5
        # Callback is NOT preserved — search still works via stored embeddings
        # but auto-query-embedding won't work (no callback)
        results = hs2.search("fox", query_embedding=[1.0, 0.0], n=3)
        assert len(results) == 3

