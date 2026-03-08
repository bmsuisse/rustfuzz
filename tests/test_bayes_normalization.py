from rustfuzz.search import BM25Plus, HybridSearch


def test_bayes_normalization_bm25():
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog",
        "Some completely different text about space and planets",
        "More text that is irrelevant",
        "A slightly related text about a fox",
    ]

    # 1. Without normalization
    idx_raw = BM25Plus(corpus, normalize_scores=False)
    res_raw = idx_raw.get_top_n("brown fox", n=5)

    assert len(res_raw) > 0
    scores_raw = [score for doc, score in res_raw]
    assert any(s > 1.0 for s in scores_raw), "Raw BM25Plus scores usually exceed 1.0"

    # 2. With Bayes normalization
    idx_norm = BM25Plus(corpus, normalize_scores=True)
    res_norm = idx_norm.get_top_n("brown fox", n=5)

    assert len(res_norm) > 0
    scores_norm = [score for doc, score in res_norm]

    # Check bounds
    for s in scores_norm:
        assert 0.0 < s <= 1.0, f"Normalized score {s} is out of bounds (0, 1]"

    # Check ordering is preserved
    docs_raw = [doc for doc, score in res_raw]
    docs_norm = [doc for doc, score in res_norm]
    assert docs_raw == docs_norm, "Normalization should preserve ranking order"


def test_bayes_normalization_hybrid():
    corpus = [
        "Data science and machine learning",
        "Artificial intelligence in modern tech",
        "Web development with React and Node.js",
        "Cloud computing and distributed systems",
    ]

    idx_norm = HybridSearch(corpus, normalize_scores=True)
    res_norm = idx_norm.search("machine learning AI", n=3)

    assert len(res_norm) > 0
    for _, score in res_norm:
        assert 0.0 < score <= 1.0, (
            f"Normalized RRF score {score} is out of bounds (0, 1]"
        )
