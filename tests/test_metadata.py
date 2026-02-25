"""Tests for metadata support in BM25 search classes."""

from __future__ import annotations

import pickle

import pytest

from rustfuzz.search import BM25, BM25L, BM25T, BM25Plus, HybridSearch

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumped over a lazy dog",
    "a lazy dog",
    "the fast brown fox",
    "jumping over dogs",
]

METADATA = [
    {"id": 1, "category": "animals"},
    {"id": 2, "category": "animals"},
    {"id": 3, "category": "pets"},
    {"id": 4, "category": "animals"},
    {"id": 5, "category": "sports"},
]


# ─── BM25 basic metadata ─────────────────────────────────────


def test_bm25_metadata_get_top_n() -> None:
    bm25 = BM25(CORPUS, metadata=METADATA)
    results = bm25.get_top_n("fox jumps", n=2)
    assert len(results) == 2
    # Results should be 3-tuples
    for r in results:
        assert len(r) == 3
        text, score, meta = r
        assert isinstance(text, str)
        assert isinstance(score, float)
        assert isinstance(meta, dict)
        assert "id" in meta  # type: ignore[operator]
    # Top result should be doc 0 with metadata id=1
    assert results[0][2]["id"] == 1  # type: ignore[index]


def test_bm25_metadata_get_top_n_fuzzy() -> None:
    bm25 = BM25(CORPUS, metadata=METADATA)
    results = bm25.get_top_n_fuzzy("quik brwn fx", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 3


def test_bm25_metadata_get_top_n_rrf() -> None:
    bm25 = BM25(CORPUS, metadata=METADATA)
    results = bm25.get_top_n_rrf("fox", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 3


def test_bm25_metadata_fuzzy_only() -> None:
    bm25 = BM25(CORPUS, metadata=METADATA)
    results = bm25.fuzzy_only("lazy dog", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 3


# ─── No metadata = unchanged behavior ────────────────────────


def test_bm25_no_metadata_unchanged() -> None:
    bm25 = BM25(CORPUS)
    results = bm25.get_top_n("fox", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 2  # (text, score) — no metadata


# ─── Metadata length validation ──────────────────────────────


def test_bm25_metadata_length_mismatch() -> None:
    with pytest.raises(ValueError, match="metadata length"):
        BM25(CORPUS, metadata=[{"id": 1}, {"id": 2}])


# ─── All BM25 variants support metadata ──────────────────────


@pytest.mark.parametrize("cls", [BM25, BM25L, BM25Plus, BM25T])
def test_bm25_variants_metadata(cls: type) -> None:
    bm25 = cls(CORPUS, metadata=METADATA)
    results = bm25.get_top_n("fox", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 3
        assert isinstance(r[2], dict)


@pytest.mark.parametrize("cls", [BM25, BM25L, BM25Plus, BM25T])
def test_bm25_variants_no_metadata(cls: type) -> None:
    bm25 = cls(CORPUS)
    results = bm25.get_top_n("fox", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 2


# ─── from_column with metadata_columns ────────────────────────


def test_bm25_from_column_metadata_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = pl.DataFrame(
        {
            "name": CORPUS,
            "id": [1, 2, 3, 4, 5],
            "cat": ["a", "a", "b", "a", "c"],
        }
    )
    bm25 = BM25.from_column(df, "name", metadata_columns=["id", "cat"])
    results = bm25.get_top_n("fox", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 3
        meta = r[2]
        assert "id" in meta  # type: ignore[operator]
        assert "cat" in meta  # type: ignore[operator]


def test_bm25_from_column_metadata_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame(
        {
            "name": CORPUS,
            "id": [1, 2, 3, 4, 5],
        }
    )
    bm25 = BM25.from_column(df, "name", metadata_columns="id")
    results = bm25.get_top_n("fox", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 3
        assert "id" in r[2]  # type: ignore[operator]


# ─── Pickle / unpickle preserves metadata ─────────────────────


def test_bm25_pickle_with_metadata() -> None:
    bm25 = BM25(CORPUS, metadata=METADATA)
    data = pickle.dumps(bm25)
    restored: BM25 = pickle.loads(data)  # noqa: S301
    results = restored.get_top_n("fox", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 3


def test_bm25_pickle_without_metadata() -> None:
    bm25 = BM25(CORPUS)
    data = pickle.dumps(bm25)
    restored: BM25 = pickle.loads(data)  # noqa: S301
    results = restored.get_top_n("fox", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 2


# ─── HybridSearch with metadata ───────────────────────────────


def test_hybrid_search_metadata() -> None:
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.8, 0.2],
    ]
    hybrid = HybridSearch(CORPUS, embeddings=embeddings, metadata=METADATA)
    results = hybrid.search("fox", query_embedding=[1.0, 0.0, 0.0], n=3)
    assert len(results) == 3
    for r in results:
        assert len(r) == 3
        assert isinstance(r[2], dict)


def test_hybrid_search_no_metadata() -> None:
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.8, 0.2],
    ]
    hybrid = HybridSearch(CORPUS, embeddings=embeddings)
    results = hybrid.search("fox", query_embedding=[1.0, 0.0, 0.0], n=3)
    assert len(results) == 3
    for r in results:
        assert len(r) == 2


# ─── Metadata with simple values (not just dicts) ────────────


def test_bm25_metadata_simple_values() -> None:
    """Metadata can be any type — not just dicts."""
    ids = [101, 102, 103, 104, 105]
    bm25 = BM25(CORPUS, metadata=ids)
    results = bm25.get_top_n("fox", n=2)
    assert len(results) > 0
    for r in results:
        assert len(r) == 3
        assert isinstance(r[2], int)
