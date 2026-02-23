"""Tests for pickle serialization — required for PySpark / Databricks compatibility."""

from __future__ import annotations

import pickle

import rustfuzz.fuzz as fuzz
from rustfuzz.distance import (
    Jaro,
    Levenshtein,
)
from rustfuzz.distance._initialize import (
    Editop,
    Editops,
    MatchingBlock,
    Opcode,
    Opcodes,
    ScoreAlignment,
)
from rustfuzz.search import BM25, HybridSearch


# ---------------------------------------------------------------------------
# Helper: pickle round-trip
# ---------------------------------------------------------------------------
def _round_trip(obj: object) -> object:
    """Pickle + unpickle an object and return it."""
    data = pickle.dumps(obj)
    return pickle.loads(data)  # noqa: S301


# ---------------------------------------------------------------------------
# Rust-backed data types
# ---------------------------------------------------------------------------
class TestPickleEditop:
    def test_round_trip(self) -> None:
        op = Editop("replace", 1, 2)
        restored = _round_trip(op)
        assert isinstance(restored, Editop)
        assert restored.tag == "replace"
        assert restored.src_pos == 1
        assert restored.dest_pos == 2

    def test_equality_after_round_trip(self) -> None:
        op = Editop("insert", 0, 5)
        assert _round_trip(op) == op


class TestPickleOpcode:
    def test_round_trip(self) -> None:
        op = Opcode("replace", 0, 3, 0, 4)
        restored = _round_trip(op)
        assert isinstance(restored, Opcode)
        assert restored.tag == "replace"
        assert restored.src_start == 0
        assert restored.src_end == 3
        assert restored.dest_start == 0
        assert restored.dest_end == 4

    def test_equality_after_round_trip(self) -> None:
        op = Opcode("equal", 0, 5, 0, 5)
        assert _round_trip(op) == op


class TestPickleMatchingBlock:
    def test_round_trip(self) -> None:
        mb = MatchingBlock(1, 2, 3)
        restored = _round_trip(mb)
        assert isinstance(restored, MatchingBlock)
        assert restored.a == 1
        assert restored.b == 2
        assert restored.size == 3


class TestPickleScoreAlignment:
    def test_round_trip(self) -> None:
        sa = ScoreAlignment(95.5, 0, 5, 3, 8)
        restored = _round_trip(sa)
        assert isinstance(restored, ScoreAlignment)
        assert restored.score == 95.5
        assert restored.src_start == 0
        assert restored.src_end == 5
        assert restored.dest_start == 3
        assert restored.dest_end == 8


class TestPickleEditops:
    def test_round_trip(self) -> None:
        ops = Levenshtein.editops("hello", "world")
        restored = _round_trip(ops)
        assert isinstance(restored, Editops)
        assert len(restored) == len(ops)
        assert restored.src_len == ops.src_len
        assert restored.dest_len == ops.dest_len

    def test_empty(self) -> None:
        ops = Levenshtein.editops("hello", "hello")
        restored = _round_trip(ops)
        assert isinstance(restored, Editops)
        assert len(restored) == 0


class TestPickleOpcodes:
    def test_round_trip(self) -> None:
        ops = Levenshtein.opcodes("hello", "world")
        restored = _round_trip(ops)
        assert isinstance(restored, Opcodes)
        assert len(restored) == len(ops)
        assert restored.src_len == ops.src_len
        assert restored.dest_len == ops.dest_len

    def test_empty(self) -> None:
        ops = Levenshtein.opcodes("hello", "hello")
        restored = _round_trip(ops)
        assert isinstance(restored, Opcodes)


# ---------------------------------------------------------------------------
# Python wrapper classes
# ---------------------------------------------------------------------------
class TestPickleBM25:
    def test_round_trip(self) -> None:
        corpus = ["the cat sat on the mat", "the dog chased the cat", "the mouse ran fast"]
        bm25 = BM25(corpus, k1=1.2, b=0.8)
        restored = _round_trip(bm25)
        assert isinstance(restored, BM25)
        assert restored.num_docs == 3

    def test_scores_match_after_round_trip(self) -> None:
        corpus = ["hello world", "foo bar", "hello foo"]
        bm25 = BM25(corpus)
        restored = _round_trip(bm25)
        assert isinstance(restored, BM25)
        assert bm25.get_scores("hello") == restored.get_scores("hello")

    def test_top_n_after_round_trip(self) -> None:
        corpus = ["apple pie", "banana split", "apple sauce"]
        bm25 = BM25(corpus)
        original = bm25.get_top_n("apple", n=2)
        restored = _round_trip(bm25)
        assert isinstance(restored, BM25)
        assert restored.get_top_n("apple", n=2) == original


class TestPickleHybridSearch:
    def test_round_trip_without_embeddings(self) -> None:
        corpus = ["hello world", "foo bar", "hello foo"]
        hs = HybridSearch(corpus)
        restored = _round_trip(hs)
        assert isinstance(restored, HybridSearch)
        assert not restored.has_vectors

    def test_round_trip_with_embeddings(self) -> None:
        corpus = ["hello", "world"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        hs = HybridSearch(corpus, embeddings=embeddings)
        restored = _round_trip(hs)
        assert isinstance(restored, HybridSearch)
        assert restored.has_vectors


# ---------------------------------------------------------------------------
# Stateless functions (pickle by reference)
# ---------------------------------------------------------------------------
class TestPickleStatelessFunctions:
    """Stateless module-level functions already pickle by reference.
    These tests document that guarantee."""

    def test_fuzz_ratio(self) -> None:
        restored = _round_trip(fuzz.ratio)
        assert restored("hello", "hello") == 100.0  # type: ignore[operator]

    def test_fuzz_wratio(self) -> None:
        restored = _round_trip(fuzz.WRatio)
        assert restored("hello", "hello") == 100.0  # type: ignore[operator]

    def test_levenshtein_distance(self) -> None:
        restored = _round_trip(Levenshtein.distance)
        assert restored("hello", "hello") == 0  # type: ignore[operator]

    def test_jaro_similarity(self) -> None:
        restored = _round_trip(Jaro.similarity)
        assert restored("hello", "hello") == 1.0  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Spark module (import only, no live Spark session required)
# ---------------------------------------------------------------------------
class TestSparkModule:
    def test_import(self) -> None:
        from rustfuzz import spark
        assert hasattr(spark, "ratio_udf")
        assert hasattr(spark, "levenshtein_distance_udf")
        assert hasattr(spark, "jaro_winkler_similarity_udf")

    def test_all_exports(self) -> None:
        from rustfuzz import spark
        expected = [
            "ratio_udf",
            "partial_ratio_udf",
            "token_sort_ratio_udf",
            "token_set_ratio_udf",
            "wratio_udf",
            "qratio_udf",
            "levenshtein_distance_udf",
            "levenshtein_similarity_udf",
            "levenshtein_normalized_similarity_udf",
            "jaro_similarity_udf",
            "jaro_winkler_similarity_udf",
        ]
        for name in expected:
            assert name in spark.__all__
