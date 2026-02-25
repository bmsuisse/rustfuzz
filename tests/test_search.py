from rustfuzz.search import BM25, HybridSearch

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumped over a lazy dog",
    "a lazy dog",
    "the fast brown fox",
    "jumping over dogs",
]


def test_bm25_basic() -> None:
    bm25 = BM25(CORPUS)
    assert bm25.num_docs == 5

    # "fox" is in doc 0, 1, 3
    scores = bm25.get_scores("fox")
    assert len(scores) == 5
    assert scores[0] > 0
    assert scores[1] > 0
    assert scores[2] == 0  # no "fox"
    assert scores[3] > 0

    top = bm25.get_top_n("fox jumps", n=2)
    assert len(top) == 2
    assert top[0][0] == CORPUS[0]  # exact match for "fox jumps"

    batch = bm25.get_batch_scores(["fox", "dog"])
    assert len(batch) == 2
    assert batch[0] == scores


def test_bm25_rrf() -> None:
    bm25 = BM25(CORPUS)
    # Misspelled query — BM25 alone might fail, but RRF with fuzzy will catch it
    top_rrf = bm25.get_top_n_rrf("quik brwn fx")
    # Top result should be doc 0 or 1 despite misspellings
    assert len(top_rrf) > 0
    assert "fox" in top_rrf[0][0]


def test_hybrid_search() -> None:
    # Dummy embeddings: 5 docs, 3 dimensions
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.8, 0.2],
    ]
    hybrid = HybridSearch(CORPUS, embeddings=embeddings)
    assert hybrid.has_vectors

    # Query embedding matching the first concept
    q_emb = [1.0, 0.0, 0.0]
    results = hybrid.search("fox", query_embedding=q_emb, n=3)

    assert len(results) == 3
    # Top results should be the "fox" documents
    docs = [r[0] for r in results]
    assert CORPUS[0] in docs or CORPUS[3] in docs

    # Without embedding fallback
    results_no_emb = hybrid.search("fox", n=3)
    assert len(results_no_emb) == 3


def test_bm25_normalized() -> None:
    # Test charabia tokenization and normalization
    vocab = ["Café", "naïf", "Thé", "hallo"]
    bm25 = BM25(vocab, normalize=True)

    # "Thé" should be normalized to "the"
    scores = bm25.get_scores("the")
    assert scores[2] > 0

    # "Café" should match "cafe"
    scores = bm25.get_scores("cafe")
    assert scores[0] > 0

    # "naïf" should match "naif"
    scores = bm25.get_scores("naif")
    assert scores[1] > 0


def test_bm25_not_normalized() -> None:
    # Test that without normalization, diacritics don't match base characters
    vocab = ["Café", "naïf", "Thé", "hallo"]
    bm25 = BM25(vocab, normalize=False)

    scores = bm25.get_scores("the")
    assert scores[2] == 0

    scores = bm25.get_scores("cafe")
    assert scores[0] == 0
