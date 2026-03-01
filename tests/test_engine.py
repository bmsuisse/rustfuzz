"""Tests for the easy-to-use Retriever API."""

import pickle

import pytest

from rustfuzz import Retriever
from rustfuzz.search import BM25Plus, Reranker

# ── Fixtures ──────────────────────────────────────────────────────────

CORPUS = [
    "Apple iPhone 15 Pro Max",
    "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro",
    "Apple MacBook Pro M3",
    "Sony WH-1000XM5 Wireless Headphones",
]

META = [
    {"brand": "Apple", "category": "phone", "price": 1199},
    {"brand": "Samsung", "category": "phone", "price": 1299},
    {"brand": "Google", "category": "phone", "price": 899},
    {"brand": "Apple", "category": "laptop", "price": 2499},
    {"brand": "Sony", "category": "audio", "price": 349},
]


class DummyCrossEncoder:
    """Mock cross-encoder: scores by word overlap."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        res = []
        for q, ctx in pairs:
            q_words = set(q.lower().split())
            c_words = set(ctx.lower().split())
            res.append(float(len(q_words & c_words)))
        return res


# ── Basic tests ───────────────────────────────────────────────────────


def test_retriever_basic_search():
    """Simplest usage: corpus only, no extras."""
    r = Retriever(CORPUS)
    results = r.search("iphone", n=3)
    assert len(results) > 0
    assert "iPhone" in results[0][0]


def test_retriever_default_algorithm():
    """Default algorithm should be BM25Plus."""
    r = Retriever(CORPUS)
    assert r.config.algorithm == "bm25plus"
    assert isinstance(r._bm25, BM25Plus)


def test_retriever_num_docs():
    r = Retriever(CORPUS)
    assert r.num_docs == len(CORPUS)


def test_retriever_repr():
    r = Retriever(CORPUS)
    s = repr(r)
    assert "Retriever" in s
    assert "bm25plus" in s


# ── Metadata + filter ────────────────────────────────────────────────


def test_retriever_with_metadata():
    r = Retriever(CORPUS, metadata=META)
    results = r.search("pro", n=5)
    assert len(results) > 0
    # Should have metadata in the result tuple
    assert len(results[0]) == 3


def test_retriever_filter():
    r = Retriever(CORPUS, metadata=META)
    results = r.filter('brand = "Apple"').match("pro", n=10)
    # All results should be Apple products
    for _text, _score, meta in results:
        assert meta["brand"] == "Apple"


def test_retriever_sort():
    r = Retriever(CORPUS, metadata=META)
    results = r.filter('category = "phone"').sort("price:asc").match("phone", n=10)
    if len(results) >= 2:
        assert results[0][2]["price"] <= results[1][2]["price"]


# ── Embeddings (static matrix) ──────────────────────────────────────


def test_retriever_with_static_embeddings():
    embeddings = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    r = Retriever(CORPUS, embeddings=embeddings)
    assert r.has_embeddings is True
    assert r._hybrid is not None

    results = r.search("iphone", query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0], n=2)
    assert len(results) > 0


def test_retriever_with_callback_embeddings():
    def dummy_embed(texts: list[str]) -> list[list[float]]:
        return [[float(i)] * 4 for i, _ in enumerate(texts)]

    r = Retriever(CORPUS, embeddings=dummy_embed)
    assert r.has_embeddings is True

    results = r.search("iphone", n=2)
    assert len(results) > 0


# ── Reranker ─────────────────────────────────────────────────────────


def test_retriever_with_reranker():
    r = Retriever(CORPUS, reranker=DummyCrossEncoder())
    assert r.has_reranker is True

    results = r.search("apple pro", n=2)
    assert len(results) == 2
    # Top results should have "apple" or "pro" in them
    assert any("Apple" in text for text, _ in results)


def test_retriever_reranker_is_wrapped():
    r = Retriever(CORPUS, reranker=DummyCrossEncoder())
    assert isinstance(r._reranker, Reranker)


# ── Algorithm override ───────────────────────────────────────────────


def test_retriever_algorithm_override_bm25l():
    r = Retriever(CORPUS, algorithm="bm25l")
    assert r.config.algorithm == "bm25l"
    results = r.search("iphone", n=2)
    assert len(results) > 0


def test_retriever_algorithm_override_bm25t():
    r = Retriever(CORPUS, algorithm="bm25t")
    assert r.config.algorithm == "bm25t"
    results = r.search("iphone", n=2)
    assert len(results) > 0


def test_retriever_invalid_algorithm():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        Retriever(CORPUS, algorithm="not_real")


# ── Upgrade to hybrid ───────────────────────────────────────────────


def test_retriever_to_hybrid():
    r = Retriever(CORPUS)
    assert r.has_embeddings is False

    embeddings = [[float(i)] * 4 for i in range(len(CORPUS))]
    r2 = r.to_hybrid(embeddings=embeddings)
    assert r2.has_embeddings is True
    assert r2.num_docs == len(CORPUS)


# ── Pickle round-trip ────────────────────────────────────────────────


def test_retriever_pickle_basic():
    r = Retriever(CORPUS)
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.num_docs == len(CORPUS)
    results = r2.search("iphone", n=2)
    assert len(results) > 0


def test_retriever_pickle_with_embeddings():
    embeddings = [[float(i)] * 4 for i in range(len(CORPUS))]
    r = Retriever(CORPUS, embeddings=embeddings)
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.has_embeddings is True
    assert r2.num_docs == len(CORPUS)


# ── Fuzzy search (typo tolerance) ────────────────────────────────────


def test_retriever_typo_tolerance():
    """Verify the RRF fuzzy path handles typos."""
    r = Retriever(CORPUS)
    results = r.search("samung galxy", n=2)
    assert len(results) > 0
    # Should still find Samsung despite the typo
    assert "Samsung" in results[0][0]
