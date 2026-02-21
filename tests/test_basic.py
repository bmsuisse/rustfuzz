"""Smoke tests for the rustfuzz Python API surface."""

from __future__ import annotations

import pytest

import rustfuzz
import rustfuzz.fuzz as fuzz
import rustfuzz.process as process
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
from rustfuzz.distance._initialize import (
    Editop,
    Editops,
    MatchingBlock,
    Opcode,
    Opcodes,
    ScoreAlignment,
)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
def test_version() -> None:
    assert isinstance(rustfuzz.__version__, str)
    assert rustfuzz.__version__ != ""


# ---------------------------------------------------------------------------
# fuzz
# ---------------------------------------------------------------------------
class TestFuzz:
    def test_ratio_identical(self) -> None:
        assert fuzz.ratio("hello", "hello") == 100.0

    def test_ratio_empty(self) -> None:
        assert fuzz.ratio("", "") == 100.0

    def test_ratio_different(self) -> None:
        score = fuzz.ratio("hello", "world")
        assert 0.0 <= score <= 100.0

    def test_partial_ratio(self) -> None:
        score = fuzz.partial_ratio("hello", "say hello world")
        assert score == 100.0

    def test_partial_ratio_alignment_returns_alignment(self) -> None:
        result = fuzz.partial_ratio_alignment("hello", "say hello world")
        assert result is not None
        assert isinstance(result.score, float)

    def test_token_sort_ratio(self) -> None:
        assert fuzz.token_sort_ratio("fuzzy wuzzy", "wuzzy fuzzy") == 100.0

    def test_token_set_ratio(self) -> None:
        score = fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
        assert score == 100.0

    def test_token_ratio(self) -> None:
        score = fuzz.token_ratio("fuzzy wuzzy", "wuzzy fuzzy")
        assert score == 100.0

    def test_token_ratio_none(self) -> None:
        assert fuzz.token_ratio(None, "test") == 0.0  # type: ignore[arg-type]
        assert fuzz.token_ratio("test", None) == 0.0  # type: ignore[arg-type]

    def test_partial_token_sort_ratio(self) -> None:
        score = fuzz.partial_token_sort_ratio("hello world", "world hello foo")
        assert score > 80.0

    def test_partial_token_set_ratio(self) -> None:
        score = fuzz.partial_token_set_ratio("hello world", "world hello foo")
        assert score >= 0.0

    def test_partial_token_ratio(self) -> None:
        score = fuzz.partial_token_ratio("hello", "hello world")
        assert score > 80.0

    def test_wratio(self) -> None:
        score = fuzz.WRatio("Hello World", "Hello World!")
        assert score > 90.0

    def test_qratio(self) -> None:
        score = fuzz.QRatio("hello", "hello")
        assert score == 100.0

    def test_score_cutoff(self) -> None:
        assert fuzz.ratio("hello", "world", score_cutoff=99.0) == 0.0

    def test_none_returns_zero(self) -> None:
        assert fuzz.ratio(None, "foo") == 0.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------
class TestUtils:
    def test_default_process_basic(self) -> None:
        result = utils.default_process("Hello, World!")
        assert result == "hello  world"

    def test_default_process_lowercase(self) -> None:
        assert utils.default_process("ABC") == "abc"

    def test_default_process_none(self) -> None:
        result = utils.default_process(None)  # type: ignore[arg-type]
        assert result == ""


# ---------------------------------------------------------------------------
# distance — Levenshtein
# ---------------------------------------------------------------------------
class TestLevenshtein:
    def test_distance_identical(self) -> None:
        assert Levenshtein.distance("hello", "hello") == 0

    def test_distance_single_insert(self) -> None:
        assert Levenshtein.distance("hello", "helllo") == 1

    def test_similarity(self) -> None:
        s = Levenshtein.similarity("hello", "hello")
        assert s == 5

    def test_normalized_distance(self) -> None:
        nd = Levenshtein.normalized_distance("hello", "hello")
        assert nd == 0.0

    def test_normalized_similarity(self) -> None:
        ns = Levenshtein.normalized_similarity("hello", "hello")
        assert ns == 1.0

    def test_editops(self) -> None:
        ops = Levenshtein.editops("hello", "world")
        assert isinstance(ops, Editops)

    def test_opcodes(self) -> None:
        ops = Levenshtein.opcodes("hello", "world")
        assert isinstance(ops, Opcodes)


# ---------------------------------------------------------------------------
# distance — Hamming
# ---------------------------------------------------------------------------
class TestHamming:
    def test_distance_identical(self) -> None:
        assert Hamming.distance("hello", "hello") == 0

    def test_distance(self) -> None:
        assert Hamming.distance("hello", "jello") == 1

    def test_normalized_similarity(self) -> None:
        assert Hamming.normalized_similarity("hello", "hello") == 1.0


# ---------------------------------------------------------------------------
# distance — Indel
# ---------------------------------------------------------------------------
class TestIndel:
    def test_distance_identical(self) -> None:
        assert Indel.distance("hello", "hello") == 0

    def test_normalized_similarity(self) -> None:
        assert Indel.normalized_similarity("hello", "hello") == 1.0


# ---------------------------------------------------------------------------
# distance — Jaro / JaroWinkler
# ---------------------------------------------------------------------------
class TestJaro:
    def test_identical(self) -> None:
        assert Jaro.similarity("hello", "hello") == 1.0

    def test_normalized_similarity(self) -> None:
        assert Jaro.normalized_similarity("hello", "hello") == 1.0


class TestJaroWinkler:
    def test_identical(self) -> None:
        assert JaroWinkler.similarity("hello", "hello") == 1.0


# ---------------------------------------------------------------------------
# distance — LCSseq, OSA, DamerauLevenshtein, Prefix, Postfix
# ---------------------------------------------------------------------------
class TestLCSseq:
    def test_identical(self) -> None:
        assert LCSseq.distance("hello", "hello") == 0


class TestOSA:
    def test_identical(self) -> None:
        assert OSA.distance("hello", "hello") == 0

    def test_transposition(self) -> None:
        # OSA treats adjacent transposition as cost 1
        assert OSA.distance("ab", "ba") == 1


class TestDamerauLevenshtein:
    def test_identical(self) -> None:
        assert DamerauLevenshtein.distance("hello", "hello") == 0


class TestPrefix:
    def test_identical(self) -> None:
        assert Prefix.distance("hello", "hello") == 0

    def test_normalized_similarity(self) -> None:
        assert Prefix.normalized_similarity("he", "hello") == pytest.approx(2 / 5)


class TestPostfix:
    def test_identical(self) -> None:
        assert Postfix.distance("hello", "hello") == 0


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------
class TestProcess:
    def test_extract_returns_list(self) -> None:
        results = process.extract("hello", ["hello world", "goodbye", "hello there"])
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_cdist(self) -> None:
        queries = ["apple", "banana"]
        choices = ["apple", "mango", "banana"]
        matrix = process.cdist(queries, choices, scorer=fuzz.ratio)
        assert matrix.shape == (2, 3)
        assert matrix[0, 0] == 100.0
        assert matrix[1, 2] == 100.0
        assert matrix[0, 1] < 100.0

    def test_extract_bests(self) -> None:
        choices = ["apple", "mango", "banana", "pineapple"]
        bests = process.extractBests(
            "apple", choices, scorer=fuzz.ratio, score_cutoff=50.0
        )
        assert len(bests) == 2
        docs = [x[0] for x in bests]
        assert "apple" in docs
        assert "pineapple" in docs
        assert "mango" not in docs

    def test_extract_iter(self) -> None:
        results = list(process.extract_iter("hello", ["hello", "world", "helo"]))
        assert any(r[0] == "hello" for r in results)

    def test_extract_score_cutoff(self) -> None:
        results = process.extract("hello", ["hello", "world"], score_cutoff=99.0)
        assert all(r[1] >= 99.0 for r in results)


# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------
class TestDataTypes:
    def test_editop_is_importable(self) -> None:
        assert Editop is not None

    def test_editops_is_importable(self) -> None:
        assert Editops is not None

    def test_opcode_is_importable(self) -> None:
        assert Opcode is not None

    def test_opcodes_is_importable(self) -> None:
        assert Opcodes is not None

    def test_matching_block_is_importable(self) -> None:
        assert MatchingBlock is not None

    def test_score_alignment_is_importable(self) -> None:
        assert ScoreAlignment is not None
