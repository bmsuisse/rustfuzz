"""Tests for process.dedupe() BK-Tree deduplication."""
from __future__ import annotations

import pytest

from rustfuzz import process


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_bktree_dedupe_threshold_kw() -> None:
    """threshold= kwarg groups strings within 1 edit under the first seen."""
    choices = ["apple", "apples", "banana", "appl", "cherry"]
    unique = process.dedupe(choices, threshold=1)

    assert "apple" in unique
    assert "banana" in unique
    assert "cherry" in unique
    assert "apples" not in unique
    assert "appl" not in unique


def test_bktree_dedupe_max_edits_alias() -> None:
    """max_edits= is a backward-compat alias for threshold=."""
    choices = ["apple", "apples", "banana", "appl", "cherry"]
    unique = process.dedupe(choices, max_edits=1)

    assert "apple" in unique
    assert "banana" in unique
    assert "cherry" in unique
    assert "apples" not in unique
    assert "appl" not in unique


def test_bktree_dedupe_defaults_equal() -> None:
    """Calling with threshold=2 and max_edits=2 yield the same result."""
    choices = ["cat", "cats", "cut", "car", "dog"]
    r1 = sorted(process.dedupe(choices, threshold=2))
    r2 = sorted(process.dedupe(choices, max_edits=2))
    assert r1 == r2


def test_bktree_dedupe_exact() -> None:
    """threshold=0 removes only exact duplicates."""
    choices = ["foo", "bar", "foo", "baz"]
    unique = process.dedupe(choices, threshold=0)
    assert sorted(unique) == ["bar", "baz", "foo"]


def test_bktree_dedupe_exact_max_edits_alias() -> None:
    """max_edits=0 (alias) removes only exact duplicates."""
    choices = ["foo", "bar", "foo", "baz"]
    unique = process.dedupe(choices, max_edits=0)
    assert sorted(unique) == ["bar", "baz", "foo"]


def test_bktree_dedupe_default_threshold() -> None:
    """Default threshold is 2 — strings ≤2 edits apart are merged."""
    choices = ["hello", "helo", "he", "world"]
    unique = process.dedupe(choices)  # default threshold=2
    # "helo" is 1 edit from "hello" → merged; "he" is 3 edits from "hello" → kept
    assert "hello" in unique
    assert "world" in unique
    assert "helo" not in unique


def test_bktree_dedupe_no_dupes() -> None:
    """Completely distinct strings → all are retained."""
    choices = ["alpha", "beta", "gamma", "delta"]
    unique = process.dedupe(choices, threshold=1)
    assert sorted(unique) == sorted(choices)


def test_bktree_dedupe_all_same() -> None:
    """All identical strings → only one is retained."""
    choices = ["rust", "rust", "rust"]
    unique = process.dedupe(choices, threshold=0)
    assert unique == ["rust"]


def test_bktree_dedupe_empty() -> None:
    """Empty input → empty output."""
    assert process.dedupe([], threshold=1) == []


def test_bktree_dedupe_single() -> None:
    """Single-element list → returned unchanged."""
    assert process.dedupe(["only"], threshold=1) == ["only"]


def test_bktree_dedupe_both_params_raises() -> None:
    """Passing both threshold= and max_edits= raises TypeError."""
    with pytest.raises(TypeError, match="threshold.*max_edits|max_edits.*threshold"):
        process.dedupe(["a", "b"], threshold=1, max_edits=1)


def test_bktree_dedupe_order_preserved() -> None:
    """The first-seen occurrence is always the canonical representative."""
    choices = ["apple", "apples", "banana"]
    unique = process.dedupe(choices, threshold=1)
    # "apple" appears before "apples", so "apple" must win
    assert "apple" in unique
    assert "apples" not in unique
    # Verify order: apple comes before banana
    assert unique.index("apple") < unique.index("banana")
