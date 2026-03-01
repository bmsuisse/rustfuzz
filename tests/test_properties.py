"""Property-based tests for rustfuzz using Hypothesis."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

import rustfuzz.fuzz as fuzz
import rustfuzz.utils as utils
from rustfuzz.distance import (
    OSA,
    DamerauLevenshtein,
    Hamming,
    Indel,
    Jaro,
    JaroWinkler,
    LCSseq,
    Levenshtein,
    Postfix,
    Prefix,
)

# ---------------------------------------------------------------------------
# Fuzz / String Matching Properties
# ---------------------------------------------------------------------------

@given(st.text())
def test_fuzz_ratio_identity(s: str) -> None:
    """The ratio of a string with itself should be 100.0. Token metrics may be 0 for empty strings."""
    assert fuzz.ratio(s, s) == 100.0
    assert fuzz.partial_ratio(s, s) == 100.0

    if s and utils.default_process(s):
        assert fuzz.token_sort_ratio(s, s) == 100.0
        assert fuzz.token_set_ratio(s, s) == 100.0
        assert fuzz.token_ratio(s, s) == 100.0
        assert fuzz.partial_token_sort_ratio(s, s) == 100.0
        assert fuzz.partial_token_set_ratio(s, s) == 100.0
        assert fuzz.partial_token_ratio(s, s) == 100.0
        assert fuzz.WRatio(s, s) == 100.0
        assert fuzz.QRatio(s, s) == 100.0


@given(st.text(), st.text())
def test_fuzz_ratio_bounds(s1: str, s2: str) -> None:
    """The ratio should always be between 0.0 and 100.0."""
    methods = [
        fuzz.ratio,
        fuzz.partial_ratio,
        fuzz.token_sort_ratio,
        fuzz.token_set_ratio,
        fuzz.token_ratio,
        fuzz.partial_token_sort_ratio,
        fuzz.partial_token_set_ratio,
        fuzz.partial_token_ratio,
        fuzz.WRatio,
        fuzz.QRatio,
    ]
    for method in methods:
        score = method(s1, s2)
        assert 0.0 <= score <= 100.0, f"{method.__name__} returned {score} which is out of bounds."


@given(st.text(), st.text())
def test_fuzz_ratio_symmetry(s1: str, s2: str) -> None:
    """Distance measures like simple ratio should be symmetric."""
    assert fuzz.ratio(s1, s2) == fuzz.ratio(s2, s1)


# ---------------------------------------------------------------------------
# Utils Properties
# ---------------------------------------------------------------------------

@given(st.text())
def test_utils_default_process(s: str) -> None:
    """Default process should not crash and always return a lowercased string."""
    processed = utils.default_process(s)
    assert isinstance(processed, str)
    assert processed == processed.lower()


# ---------------------------------------------------------------------------
# Distance Properties
# ---------------------------------------------------------------------------

DISTANCE_METRICS = [
    OSA,
    DamerauLevenshtein,
    Indel,
    LCSseq,
    Levenshtein,
    Postfix,
    Prefix,
]

@given(st.text())
def test_distance_identity(s: str) -> None:
    """The distance of a string to itself should be 0, and normalized metrics bounded perfectly."""
    for metric in DISTANCE_METRICS:
        assert metric.distance(s, s) == 0
        if hasattr(metric, "normalized_distance"):
            assert metric.normalized_distance(s, s) == 0.0
        if hasattr(metric, "normalized_similarity"):
            assert metric.normalized_similarity(s, s) == 1.0


@given(st.text(), st.text())
def test_distance_bounds(s1: str, s2: str) -> None:
    """Distances should be non-negative and normalized bounds should be [0.0, 1.0]."""
    for metric in DISTANCE_METRICS:
        dist = metric.distance(s1, s2)
        assert dist >= 0

        if hasattr(metric, "normalized_distance"):
            nd = metric.normalized_distance(s1, s2)
            assert 0.0 <= nd <= 1.0

        if hasattr(metric, "normalized_similarity"):
            ns = metric.normalized_similarity(s1, s2)
            assert 0.0 <= ns <= 1.0


@given(st.text(), st.text())
def test_distance_symmetry(s1: str, s2: str) -> None:
    """Levenshtein distance should be symmetric."""
    assert Levenshtein.distance(s1, s2) == Levenshtein.distance(s2, s1)


@given(st.text(), st.text(), st.text())
def test_levenshtein_triangle_inequality(s1: str, s2: str, s3: str) -> None:
    """Levenshtein distance should satisfy the triangle inequality."""
    d12 = Levenshtein.distance(s1, s2)
    d23 = Levenshtein.distance(s2, s3)
    d13 = Levenshtein.distance(s1, s3)
    assert d13 <= d12 + d23


# ---------------------------------------------------------------------------
# Hamming Properties
# ---------------------------------------------------------------------------

@given(st.text())
def test_hamming_identity(s: str) -> None:
    """Hamming distance of string to itself should be 0."""
    assert Hamming.distance(s, s) == 0
    assert Hamming.normalized_similarity(s, s) == 1.0


@given(st.text(), st.text())
def test_hamming_bounds(s1: str, s2: str) -> None:
    """Hamming distance limits."""
    dist = Hamming.distance(s1, s2)
    assert dist >= abs(len(s1) - len(s2))
    assert dist <= max(len(s1), len(s2))


# ---------------------------------------------------------------------------
# Jaro / JaroWinkler Properties
# ---------------------------------------------------------------------------

@given(st.text(min_size=1))
def test_jaro_identity_non_empty(s: str) -> None:
    """Jaro metrics for identical strings (non-empty) should be 1.0."""
    assert Jaro.similarity(s, s) == 1.0
    assert JaroWinkler.similarity(s, s) == 1.0

    if hasattr(Jaro, "normalized_similarity"):
        assert Jaro.normalized_similarity(s, s) == 1.0
        assert JaroWinkler.normalized_similarity(s, s) == 1.0


@given(st.text(), st.text())
def test_jaro_bounds(s1: str, s2: str) -> None:
    """Jaro similarities should be bounded between 0.0 and 1.0."""
    assert 0.0 <= Jaro.similarity(s1, s2) <= 1.0
    assert 0.0 <= JaroWinkler.similarity(s1, s2) <= 1.0
