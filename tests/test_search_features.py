"""Tests for new search features: explain, IDF/TF export, incremental updates,
reranker pipeline, phrase search, and async wrappers."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rustfuzz.search import BM25, BM25L, BM25T, BM25Plus

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumped over a lazy dog",
    "a lazy dog",
    "the fast brown fox",
    "jumping over dogs",
    "New York is a big city",
    "York new style fashion",
]

METADATA = [
    {"id": 1, "category": "animals"},
    {"id": 2, "category": "animals"},
    {"id": 3, "category": "pets"},
    {"id": 4, "category": "animals"},
    {"id": 5, "category": "animals"},
    {"id": 6, "category": "geography"},
    {"id": 7, "category": "fashion"},
]


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


class TestExplain:
    def test_explain_by_index(self) -> None:
        bm25 = BM25(CORPUS)
        result = bm25.explain("quick fox", 0)
        assert "terms" in result
        assert "total_score" in result
        assert result["doc_idx"] == 0
        assert result["doc_text"] == CORPUS[0]
        assert len(result["terms"]) == 2
        for t in result["terms"]:
            assert "term" in t
            assert "idf" in t
            assert "tf_norm" in t
            assert "score" in t

    def test_explain_by_string(self) -> None:
        bm25 = BM25(CORPUS)
        result = bm25.explain("fox", "the fast brown fox")
        assert result["doc_idx"] == 3
        assert result["total_score"] > 0

    def test_explain_no_match(self) -> None:
        bm25 = BM25(CORPUS)
        result = bm25.explain("xylophone", 0)
        assert result["total_score"] == 0.0
        assert all(t["score"] == 0.0 for t in result["terms"])

    def test_explain_out_of_range(self) -> None:
        bm25 = BM25(CORPUS)
        with pytest.raises(IndexError):
            bm25.explain("fox", 100)

    @pytest.mark.parametrize("cls", [BM25L, BM25Plus, BM25T])
    def test_explain_variants(self, cls: type) -> None:
        bm25 = cls(CORPUS[:5])
        result = bm25.explain("fox", 0)
        assert result["total_score"] > 0


# ---------------------------------------------------------------------------
# get_idf / get_document_vector
# ---------------------------------------------------------------------------


class TestIdfAndDocVector:
    def test_get_idf(self) -> None:
        bm25 = BM25(CORPUS)
        idf = bm25.get_idf()
        assert isinstance(idf, dict)
        assert len(idf) > 0
        assert "fox" in idf
        assert idf["fox"] > 0

    def test_get_document_vector(self) -> None:
        bm25 = BM25(CORPUS)
        vec = bm25.get_document_vector(0)
        assert isinstance(vec, dict)
        assert "fox" in vec
        assert vec["fox"] > 0

    def test_get_document_vector_out_of_range(self) -> None:
        bm25 = BM25(CORPUS)
        with pytest.raises(IndexError):
            bm25.get_document_vector(100)

    @pytest.mark.parametrize("cls", [BM25L, BM25Plus, BM25T])
    def test_idf_and_vector_variants(self, cls: type) -> None:
        bm25 = cls(CORPUS[:5])
        idf = bm25.get_idf()
        assert len(idf) > 0
        vec = bm25.get_document_vector(0)
        assert len(vec) > 0


# ---------------------------------------------------------------------------
# add / remove documents
# ---------------------------------------------------------------------------


class TestIncrementalUpdates:
    def test_add_documents(self) -> None:
        bm25 = BM25(CORPUS[:3])
        assert bm25.num_docs == 3
        bm25.add_documents(["a new document about cats"])
        assert bm25.num_docs == 4
        scores = bm25.get_scores("cats")
        assert scores[3] > 0

    def test_add_documents_with_metadata(self) -> None:
        bm25 = BM25(CORPUS[:3], metadata=METADATA[:3])
        bm25.add_documents(
            ["cats are great"],
            metadata=[{"id": 99, "category": "pets"}],
        )
        assert bm25.num_docs == 4
        results = bm25.get_top_n("cats", 1)
        assert len(results) == 1
        assert results[0][2]["id"] == 99  # type: ignore[index]

    def test_remove_documents(self) -> None:
        bm25 = BM25(CORPUS[:5])
        assert bm25.num_docs == 5
        bm25.remove_documents([0, 1])
        assert bm25.num_docs == 3

    def test_remove_documents_with_metadata(self) -> None:
        bm25 = BM25(CORPUS[:5], metadata=METADATA[:5])
        bm25.remove_documents([0])
        assert bm25.num_docs == 4
        results = bm25.get_top_n("fox", 2)
        # All results should have metadata
        for r in results:
            assert len(r) == 3  # (text, score, metadata)

    @pytest.mark.parametrize("cls", [BM25L, BM25Plus, BM25T])
    def test_add_remove_variants(self, cls: type) -> None:
        kw: dict[str, Any] = {}
        if cls in (BM25L, BM25Plus):
            kw["delta"] = 0.5
        bm25 = cls(CORPUS[:3], **kw)
        bm25.add_documents(["cats are great"])
        assert bm25.num_docs == 4
        bm25.remove_documents([0])
        assert bm25.num_docs == 3


# ---------------------------------------------------------------------------
# reranker
# ---------------------------------------------------------------------------


class TestReranker:
    def test_reranker_callback(self) -> None:
        bm25 = BM25(CORPUS)

        # Simple reranker: reverse the order by returning descending scores
        def reverse_reranker(q: str, docs: list[str]) -> list[float]:
            return list(range(len(docs), 0, -1))

        results = bm25.get_top_n_reranked("fox", n=3, reranker=reverse_reranker)
        assert len(results) <= 3
        # The "worst" BM25 result should now be first
        assert results[0][1] > results[-1][1]

    def test_reranker_none_falls_back(self) -> None:
        bm25 = BM25(CORPUS)
        r1 = bm25.get_top_n("fox", 3)
        r2 = bm25.get_top_n_reranked("fox", n=3, reranker=None)
        assert [d for d, _ in r1] == [d for d, _ in r2]

    @pytest.mark.parametrize("cls", [BM25L, BM25Plus, BM25T])
    def test_reranker_variants(self, cls: type) -> None:
        kw: dict[str, Any] = {}
        if cls in (BM25L, BM25Plus):
            kw["delta"] = 0.5
        bm25 = cls(CORPUS[:5], **kw)
        results = bm25.get_top_n_reranked("fox", n=2, reranker=None)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# phrase / proximity search
# ---------------------------------------------------------------------------


class TestPhraseSearch:
    def test_phrase_boost_adjacent(self) -> None:
        bm25 = BM25(CORPUS)
        # "New York" — both docs contain "new" and "york"
        # Phrase search should boost both since terms are within proximity window
        phrase_results = bm25.get_top_n_phrase("New York", n=3)
        normal_results = bm25.get_top_n("New York", n=3)
        assert len(phrase_results) > 0
        assert len(normal_results) > 0
        # With phrase_boost > 1.0, all matching docs should have higher scores
        phrase_scores = {d: s for d, s in phrase_results}
        normal_scores = {d: s for d, s in normal_results}
        for doc in phrase_scores:
            if doc in normal_scores:
                assert phrase_scores[doc] >= normal_scores[doc]

    def test_phrase_no_boost_with_1(self) -> None:
        bm25 = BM25(CORPUS)
        r1 = bm25.get_top_n("fox", n=3)
        r2 = bm25.get_top_n_phrase("fox", n=3, phrase_boost=1.0)
        # With boost=1.0, scores should be identical to normal BM25
        for (d1, s1), (d2, s2) in zip(r1, r2, strict=True):
            assert d1 == d2
            assert abs(s1 - s2) < 1e-10

    @pytest.mark.parametrize("cls", [BM25L, BM25Plus, BM25T])
    def test_phrase_search_variants(self, cls: type) -> None:
        kw: dict[str, Any] = {}
        if cls in (BM25L, BM25Plus):
            kw["delta"] = 0.5
        bm25 = cls(CORPUS, **kw)
        results = bm25.get_top_n_phrase("quick fox", n=3)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# async wrappers
# ---------------------------------------------------------------------------


class TestAsync:
    def test_get_top_n_async(self) -> None:
        bm25 = BM25(CORPUS)
        results = asyncio.run(bm25.get_top_n_async("fox", 3))
        assert len(results) > 0

    def test_search_async(self) -> None:
        bm25 = BM25(CORPUS)
        results = asyncio.run(bm25.search_async("fox", 3))
        assert len(results) > 0

    @pytest.mark.parametrize("cls", [BM25L, BM25Plus, BM25T])
    def test_async_variants(self, cls: type) -> None:
        kw: dict[str, Any] = {}
        if cls in (BM25L, BM25Plus):
            kw["delta"] = 0.5
        bm25 = cls(CORPUS[:5], **kw)
        results = asyncio.run(bm25.get_top_n_async("fox", 3))
        assert len(results) > 0
