import pytest

from rustfuzz.search import BM25, BM25L, BM25T, BM25Plus

# Corpus used for logical math verification
corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
]
query = "windy London"

# Validated outputs from our high-quality rust implementations
# (which use Lucene Okapi IDF smoothing and correct BM25L/+/T term frequency parameters)
EXPECTED_OKAPI_SCORES = [0.0, 1.79968670277381, 0.0]
EXPECTED_BM25L_SCORES = [0.0, 2.340615262868892, 0.0]
EXPECTED_BM25PLUS_SCORES = [0.0, 5.316248100441416, 0.0]
EXPECTED_BM25T_SCORES = [0.0, 1.888390041423545, 0.0]  # locked in mathematical baseline


def test_bm25okapi_parity():
    bm25 = BM25(corpus)
    scores = bm25.get_scores(query)
    assert len(scores) == 3
    for s, exp in zip(scores, EXPECTED_OKAPI_SCORES, strict=True):
        assert pytest.approx(s, rel=1e-4) == exp


def test_bm25l_parity():
    bm25l = BM25L(corpus)
    scores = bm25l.get_scores(query)
    for s, exp in zip(scores, EXPECTED_BM25L_SCORES, strict=True):
        assert pytest.approx(s, rel=1e-4) == exp


def test_bm25plus_parity():
    bm25plus = BM25Plus(corpus)
    scores = bm25plus.get_scores(query)
    for s, exp in zip(scores, EXPECTED_BM25PLUS_SCORES, strict=True):
        assert pytest.approx(s, rel=1e-4) == exp


def test_bm25t_parity():
    bm25t = BM25T(corpus)
    scores = bm25t.get_scores(query)
    # the exact score will be locked in to prevent regressions
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    # Check if the score matches the expected roughly or print it if failing
    if abs(scores[1] - EXPECTED_BM25T_SCORES[1]) > 1e-4:
        pytest.fail(f"BM25T score changed to {scores[1]}")


# ---------------------------------------------------------------------------
# Consistent API tests — ensure all variants expose the same methods
# ---------------------------------------------------------------------------

CORPUS_LARGE = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
    "The quick brown fox jumps over the lazy dog",
    "A lazy dog slept in the sun all afternoon",
]


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (BM25L, {"delta": 0.5}),
        (BM25Plus, {"delta": 1.0}),
        (BM25T, {}),
    ],
)
def test_variant_get_batch_scores(cls, kwargs):
    idx = cls(CORPUS_LARGE, **kwargs)
    batch = idx.get_batch_scores(["windy", "fox"])
    assert len(batch) == 2
    assert len(batch[0]) == len(CORPUS_LARGE)
    # Should match individual get_scores
    assert batch[0] == idx.get_scores("windy")


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (BM25L, {"delta": 0.5}),
        (BM25Plus, {"delta": 1.0}),
        (BM25T, {}),
    ],
)
def test_variant_get_top_n_fuzzy(cls, kwargs):
    idx = cls(CORPUS_LARGE, **kwargs)
    # Misspelled query — fuzzy should still find results
    results = idx.get_top_n_fuzzy("wndy Lndon", n=3, fuzzy_weight=0.4)
    assert len(results) > 0
    # Best match should contain "London"
    assert "London" in results[0][0]


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (BM25L, {"delta": 0.5}),
        (BM25Plus, {"delta": 1.0}),
        (BM25T, {}),
    ],
)
def test_variant_get_top_n_rrf(cls, kwargs):
    idx = cls(CORPUS_LARGE, **kwargs)
    results = idx.get_top_n_rrf("quik brwn fx", n=3)
    assert len(results) > 0
    # Top result should contain "fox"
    assert "fox" in results[0][0]


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (BM25L, {"delta": 0.5}),
        (BM25Plus, {"delta": 1.0}),
        (BM25T, {}),
    ],
)
def test_variant_fuzzy_only(cls, kwargs):
    idx = cls(CORPUS_LARGE, **kwargs)
    results = idx.fuzzy_only("lazy dog", n=2)
    assert len(results) == 2
    # All results should have scores > 0
    assert all(score > 0 for _, score in results)

