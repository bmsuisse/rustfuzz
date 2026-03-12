"""Tests for BM25 variant API parity — all variants share the same interface via _BaseBM25."""

from __future__ import annotations

import pytest

from rustfuzz.search import BM25, BM25L, BM25T, BM25Plus, _BaseBM25

# All BM25 variant classes to test
BM25_VARIANTS = [BM25, BM25L, BM25Plus, BM25T]

SAMPLE_CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "A fast red car drives over the slow bridge",
    "The cat sat on the mat in the kitchen",
    "Dogs and cats are popular household pets",
    "A brown bear walked through the forest trail",
]


@pytest.fixture(params=BM25_VARIANTS, ids=lambda cls: cls.__name__)
def bm25_instance(request: pytest.FixtureRequest) -> _BaseBM25:
    """Fixture creating an instance of each BM25 variant."""
    cls = request.param
    return cls(SAMPLE_CORPUS)


@pytest.fixture(params=BM25_VARIANTS, ids=lambda cls: cls.__name__)
def bm25_with_metadata(request: pytest.FixtureRequest) -> _BaseBM25:
    """Fixture creating each BM25 variant with metadata."""
    cls = request.param
    meta = [{"idx": i, "source": f"doc_{i}"} for i in range(len(SAMPLE_CORPUS))]
    return cls(SAMPLE_CORPUS, metadata=meta)


class TestBM25VariantInheritance:
    """Verify all variants are subclasses of _BaseBM25."""

    @pytest.mark.parametrize("cls", BM25_VARIANTS, ids=lambda c: c.__name__)
    def test_is_subclass(self, cls: type) -> None:
        assert issubclass(cls, _BaseBM25)

    @pytest.mark.parametrize("cls", BM25_VARIANTS, ids=lambda c: c.__name__)
    def test_isinstance(self, cls: type) -> None:
        instance = cls(SAMPLE_CORPUS)
        assert isinstance(instance, _BaseBM25)


class TestBM25VariantScoring:
    """Verify scoring methods work identically across variants."""

    def test_get_scores_length(self, bm25_instance: _BaseBM25) -> None:
        scores = bm25_instance.get_scores("fox")
        assert len(scores) == len(SAMPLE_CORPUS)

    def test_get_scores_returns_floats(self, bm25_instance: _BaseBM25) -> None:
        scores = bm25_instance.get_scores("fox")
        assert all(isinstance(s, float) for s in scores)

    def test_get_top_n_returns_results(self, bm25_instance: _BaseBM25) -> None:
        results = bm25_instance.get_top_n("fox", n=3)
        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_get_top_n_fuzzy(self, bm25_instance: _BaseBM25) -> None:
        results = bm25_instance.get_top_n_fuzzy("foks", n=3)  # misspelling
        assert len(results) <= 3

    def test_get_top_n_rrf(self, bm25_instance: _BaseBM25) -> None:
        results = bm25_instance.get_top_n_rrf("fox", n=3)
        assert len(results) <= 3

    def test_fuzzy_only(self, bm25_instance: _BaseBM25) -> None:
        results = bm25_instance.fuzzy_only("fox", n=3)
        assert len(results) <= 3

    def test_get_batch_scores(self, bm25_instance: _BaseBM25) -> None:
        batch = bm25_instance.get_batch_scores(["fox", "cat"])
        assert len(batch) == 2
        assert all(len(row) == len(SAMPLE_CORPUS) for row in batch)

    def test_get_top_n_filtered(self, bm25_instance: _BaseBM25) -> None:
        allowed = [True, False, True, False, True]
        results = bm25_instance.get_top_n_filtered("fox", allowed=allowed, n=3)
        assert len(results) <= 3


class TestBM25VariantProperties:
    """Verify properties work across all variants."""

    def test_num_docs(self, bm25_instance: _BaseBM25) -> None:
        assert bm25_instance.num_docs == len(SAMPLE_CORPUS)

    def test_get_idf(self, bm25_instance: _BaseBM25) -> None:
        idf = bm25_instance.get_idf()
        assert isinstance(idf, dict)
        assert len(idf) > 0

    def test_get_document_vector(self, bm25_instance: _BaseBM25) -> None:
        vec = bm25_instance.get_document_vector(0)
        assert isinstance(vec, dict)


class TestBM25VariantExplain:
    """Verify explain works across all variants."""

    def test_explain_by_index(self, bm25_instance: _BaseBM25) -> None:
        explanation = bm25_instance.explain("fox", 0)
        assert "terms" in explanation
        assert "total_score" in explanation
        assert "doc_idx" in explanation
        assert "doc_text" in explanation

    def test_explain_by_text(self, bm25_instance: _BaseBM25) -> None:
        explanation = bm25_instance.explain("fox", SAMPLE_CORPUS[0])
        assert explanation["doc_idx"] == 0

    def test_explain_not_found_raises(self, bm25_instance: _BaseBM25) -> None:
        with pytest.raises(ValueError, match="Document not found"):
            bm25_instance.explain("fox", "nonexistent document text")


class TestBM25VariantMetadata:
    """Verify metadata handling works across all variants."""

    def test_top_n_with_metadata(self, bm25_with_metadata: _BaseBM25) -> None:
        results = bm25_with_metadata.get_top_n("fox", n=3)
        assert all(len(r) == 3 for r in results)  # (text, score, metadata)

    def test_top_n_without_metadata(self, bm25_instance: _BaseBM25) -> None:
        results = bm25_instance.get_top_n("fox", n=3)
        assert all(len(r) == 2 for r in results)  # (text, score)


class TestBM25VariantMutation:
    """Verify add/remove document methods work across all variants."""

    def test_add_documents(self, bm25_instance: _BaseBM25) -> None:
        original_count = bm25_instance.num_docs
        bm25_instance.add_documents(["New document about foxes"])
        assert bm25_instance.num_docs == original_count + 1

    def test_remove_documents(self, bm25_instance: _BaseBM25) -> None:
        original_count = bm25_instance.num_docs
        bm25_instance.remove_documents([0])
        assert bm25_instance.num_docs == original_count - 1


class TestBM25VariantPickle:
    """Verify all variants are picklable."""

    def test_pickle_roundtrip(self, bm25_instance: _BaseBM25) -> None:
        import pickle

        pickled = pickle.dumps(bm25_instance)
        restored = pickle.loads(pickled)  # noqa: S301
        assert restored.num_docs == bm25_instance.num_docs
        original_scores = bm25_instance.get_scores("fox")
        restored_scores = restored.get_scores("fox")
        assert original_scores == pytest.approx(restored_scores, rel=1e-6)
