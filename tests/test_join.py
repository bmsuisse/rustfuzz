"""Tests for rustfuzz.join — multi-array fuzzy full join."""

from __future__ import annotations

import pytest

from rustfuzz.join import MultiJoiner, fuzzy_join


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRODUCTS = ["Apple iPhone 14", "Samsung Galaxy S23", "Google Pixel 7"]
LISTINGS = ["iphone 14 pro", "galaxy s23 ultra", "pixel 7a"]
TYPOS = ["Apple Iphone", "Samsung Galxy", "Google Pxal"]


def _top_match(rows: list[dict], src_array: str, src_idx: int, tgt_array: str) -> dict:
    """Return the single top match row for a given source element and target array."""
    matches = [
        r
        for r in rows
        if r["src_array"] == src_array
        and r["src_idx"] == src_idx
        and r["tgt_array"] == tgt_array
    ]
    assert matches, f"No match found for ({src_array}[{src_idx}] → {tgt_array})"
    return max(matches, key=lambda r: r["score"])


# ---------------------------------------------------------------------------
# Text-only join
# ---------------------------------------------------------------------------


def test_text_only_basic_structure() -> None:
    """join() returns one row per (src, src_idx, tgt) combination (n=1)."""
    rows = fuzzy_join({"A": PRODUCTS, "B": LISTINGS}, n=1)

    # 3 src elements × 2 direction pairs (A→B and B→A) × n=1 = 6 rows
    assert len(rows) == 6
    for r in rows:
        assert "src_array" in r
        assert "tgt_array" in r
        assert "score" in r
        assert r["src_array"] != r["tgt_array"]


def test_text_only_top_match_correct() -> None:
    """Top text match resolves to the right target document."""
    rows = fuzzy_join({"products": PRODUCTS, "listings": LISTINGS}, n=1)

    # Products[0] = "Apple iPhone 14"  →  should match "iphone 14 pro" (idx 0)
    m = _top_match(rows, "products", 0, "listings")
    assert m["tgt_idx"] == 0
    assert "iphone" in m["tgt_text"].lower()

    # Products[1] = "Samsung Galaxy S23" → should match "galaxy s23 ultra" (idx 1)
    m = _top_match(rows, "products", 1, "listings")
    assert m["tgt_idx"] == 1

    # Products[2] = "Google Pixel 7" → should match "pixel 7a" (idx 2)
    m = _top_match(rows, "products", 2, "listings")
    assert m["tgt_idx"] == 2


def test_text_only_three_arrays() -> None:
    """Works with 3 arrays: 6 ordered pairs, 3 elements each → 18 rows."""
    rows = fuzzy_join(
        {"A": PRODUCTS, "B": LISTINGS, "C": TYPOS}, n=1
    )
    assert len(rows) == 18  # 3 arrays × 2 directions × 3 elements


def test_typo_robustness() -> None:
    """Deliberately misspelled text still resolves to the correct match."""
    rows = fuzzy_join({"clean": PRODUCTS, "typos": TYPOS}, n=1)
    for idx in range(3):
        m = _top_match(rows, "clean", idx, "typos")
        assert m["tgt_idx"] == idx, (
            f"Expected idx {idx}, got {m['tgt_idx']} (text: {m['tgt_text']!r})"
        )


def test_top_n_returns_multiple() -> None:
    """n=2 returns two tgt matches per source element."""
    rows = fuzzy_join({"A": PRODUCTS, "B": LISTINGS + ["bonus item"]}, n=2)
    src_0_rows = [r for r in rows if r["src_array"] == "A" and r["src_idx"] == 0]
    assert len(src_0_rows) == 2


# ---------------------------------------------------------------------------
# Dense-only join
# ---------------------------------------------------------------------------

EMBEDDINGS_A = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]
EMBEDDINGS_B = [
    [0.99, 0.01, 0.0],   # closest to A[0]
    [0.01, 0.99, 0.0],   # closest to A[1]
    [0.0,  0.01, 0.99],  # closest to A[2]
]


def test_dense_only_join() -> None:
    """Dense cosine channel correctly picks nearest neighbour."""
    joiner = MultiJoiner(text_weight=0.0, sparse_weight=0.0, dense_weight=1.0)
    joiner.add_array("A", dense=EMBEDDINGS_A)
    joiner.add_array("B", dense=EMBEDDINGS_B)
    rows = joiner.join(n=1)

    for idx in range(3):
        m = _top_match(rows, "A", idx, "B")
        assert m["tgt_idx"] == idx, (
            f"Dense join: A[{idx}] should match B[{idx}], got B[{m['tgt_idx']}]"
        )
        assert m["dense_score"] is not None
        assert m["text_score"] is None
        assert m["sparse_score"] is None


def test_dense_scores_in_range() -> None:
    """Cosine scores on unit vectors are between -1 and 1."""
    joiner = MultiJoiner(text_weight=0.0, sparse_weight=0.0, dense_weight=1.0)
    joiner.add_array("A", dense=EMBEDDINGS_A)
    joiner.add_array("B", dense=EMBEDDINGS_B)
    for r in joiner.join(n=3):
        ds = r["dense_score"]
        assert ds is not None
        assert -1.01 <= ds <= 1.01


# ---------------------------------------------------------------------------
# Sparse-only join
# ---------------------------------------------------------------------------

SPARSE_A = [
    {0: 1.0, 2: 0.5},
    {1: 1.0, 3: 0.8},
    {4: 1.0},
]
SPARSE_B = [
    {0: 0.9, 2: 0.6},   # best match for A[0]
    {1: 0.7, 3: 0.9},   # best match for A[1]
    {4: 0.95},           # best match for A[2]
]


def test_sparse_only_join() -> None:
    """Sparse dot product channel correctly picks highest-overlap match."""
    joiner = MultiJoiner(text_weight=0.0, sparse_weight=1.0, dense_weight=0.0)
    joiner.add_array("A", sparse=SPARSE_A)
    joiner.add_array("B", sparse=SPARSE_B)
    rows = joiner.join(n=1)

    for idx in range(3):
        m = _top_match(rows, "A", idx, "B")
        assert m["tgt_idx"] == idx
        assert m["sparse_score"] is not None
        assert m["text_score"] is None
        assert m["dense_score"] is None


# ---------------------------------------------------------------------------
# Mixed text + dense join
# ---------------------------------------------------------------------------


def test_mixed_text_dense_join() -> None:
    """Text + dense channels combined should still resolve correctly."""
    joiner = MultiJoiner(text_weight=0.5, dense_weight=0.5, sparse_weight=0.0)
    joiner.add_array("A", texts=PRODUCTS, dense=EMBEDDINGS_A)
    joiner.add_array("B", texts=LISTINGS, dense=EMBEDDINGS_B)
    rows = joiner.join(n=1)

    for idx in range(3):
        m = _top_match(rows, "A", idx, "B")
        assert m["tgt_idx"] == idx
        assert m["text_score"] is not None
        assert m["dense_score"] is not None


# ---------------------------------------------------------------------------
# join_pair
# ---------------------------------------------------------------------------


def test_join_pair_only_returns_one_direction() -> None:
    """join_pair('A','B') only returns A→B rows, not B→A."""
    joiner = MultiJoiner()
    joiner.add_array("A", texts=PRODUCTS)
    joiner.add_array("B", texts=LISTINGS)
    rows = joiner.join_pair("A", "B", n=1)
    assert all(r["src_array"] == "A" and r["tgt_array"] == "B" for r in rows)
    assert len(rows) == 3


def test_join_pair_unknown_array_raises() -> None:
    """join_pair with an unknown name raises KeyError."""
    joiner = MultiJoiner()
    joiner.add_array("A", texts=PRODUCTS)
    joiner.add_array("B", texts=LISTINGS)
    with pytest.raises(KeyError):
        joiner.join_pair("A", "NOPE")


# ---------------------------------------------------------------------------
# MultiJoiner API
# ---------------------------------------------------------------------------


def test_num_arrays_property() -> None:
    joiner = MultiJoiner()
    assert joiner.num_arrays == 0
    joiner.add_array("X", texts=PRODUCTS)
    assert joiner.num_arrays == 1
    joiner.add_array("Y", texts=LISTINGS)
    assert joiner.num_arrays == 2


def test_add_array_chaining() -> None:
    """add_array returns self for method chaining."""
    joiner = (
        MultiJoiner()
        .add_array("A", texts=PRODUCTS)
        .add_array("B", texts=LISTINGS)
    )
    assert joiner.num_arrays == 2


def test_add_array_no_channels_raises() -> None:
    """add_array with no channels raises ValueError."""
    with pytest.raises(ValueError):
        MultiJoiner().add_array("X")


def test_add_array_length_mismatch_raises() -> None:
    """add_array raises if texts and dense have different lengths."""
    with pytest.raises((ValueError, Exception)):
        MultiJoiner().add_array(
            "X",
            texts=["a", "b"],
            dense=[[1.0], [0.5], [0.2]],  # 3 != 2
        )


# ---------------------------------------------------------------------------
# fuzzy_join convenience function
# ---------------------------------------------------------------------------


def test_fuzzy_join_convenience() -> None:
    """fuzzy_join(dict) is equivalent to manually building a MultiJoiner."""
    rows = fuzzy_join({"A": PRODUCTS, "B": LISTINGS}, n=1)
    assert len(rows) == 6
    assert all("score" in r for r in rows)


def test_fuzzy_join_with_dense() -> None:
    """fuzzy_join accepts dense kwarg."""
    rows = fuzzy_join(
        {"A": PRODUCTS, "B": LISTINGS},
        dense={"A": EMBEDDINGS_A, "B": EMBEDDINGS_B},
        text_weight=0.5,
        dense_weight=0.5,
        n=1,
    )
    assert len(rows) == 6


# ---------------------------------------------------------------------------
# BM25 auto-fallback for sparse channel
# ---------------------------------------------------------------------------


def test_sparse_bm25_fallback_activates() -> None:
    """When sparse_weight>0 but no explicit sparse vectors, sparse channel uses BM25."""
    joiner = MultiJoiner(text_weight=0.0, sparse_weight=1.0, dense_weight=0.0)
    joiner.add_array("A", texts=PRODUCTS)
    joiner.add_array("B", texts=LISTINGS)
    rows = joiner.join(n=1)

    # sparse_score should be populated (BM25 auto-fallback)
    for r in rows:
        assert r["sparse_score"] is not None, "Expected BM25 auto-fallback for sparse_score"
        assert r["text_score"] is None      # text channel disabled
        assert r["dense_score"] is None     # dense channel disabled


def test_sparse_bm25_fallback_correct_match() -> None:
    """BM25 sparse auto-fallback resolves to correct target document."""
    joiner = MultiJoiner(text_weight=0.0, sparse_weight=1.0, dense_weight=0.0)
    joiner.add_array("products", texts=PRODUCTS)
    joiner.add_array("listings", texts=LISTINGS)
    rows = joiner.join(n=1)

    for idx in range(len(PRODUCTS)):
        m = _top_match(rows, "products", idx, "listings")
        assert m["tgt_idx"] == idx, (
            f"BM25 fallback: products[{idx}] should match listings[{idx}], "
            f"got listings[{m['tgt_idx']}] ({m['tgt_text']!r})"
        )
