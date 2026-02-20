"""
rustfuzz — performance regression benchmarks.

Run once to establish a baseline:
    uv run pytest tests/test_benchmarks.py --benchmark-save=baseline

On subsequent runs, compare against the baseline:
    uv run pytest tests/test_benchmarks.py --benchmark-compare=baseline --benchmark-compare-fail=mean:10%

A 10% regression on any benchmark will fail the run.
"""

from __future__ import annotations

import pytest

import rustfuzz.fuzz as fuzz
from rustfuzz import process
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
# Fixtures — representative string pairs
# ---------------------------------------------------------------------------
SHORT_A = "hello"
SHORT_B = "hallo"

MEDIUM_A = "the quick brown fox jumps over the lazy dog"
MEDIUM_B = "the quick brown fox jumped over a lazy dog"

LONG_A = "a" * 200 + "b" * 200
LONG_B = "a" * 195 + "c" * 210


CHOICES = [
    "New York",
    "New Orleans",
    "Newark",
    "Los Angeles",
    "San Francisco",
    "Nashville",
    "Boston",
    "Denver",
    "Miami",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
]


# ---------------------------------------------------------------------------
# fuzz — short strings
# ---------------------------------------------------------------------------
def test_fuzz_ratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.ratio, SHORT_A, SHORT_B)


def test_fuzz_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.ratio, MEDIUM_A, MEDIUM_B)


def test_fuzz_ratio_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.ratio, LONG_A, LONG_B)


def test_fuzz_partial_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.partial_ratio, MEDIUM_A, MEDIUM_B)


def test_fuzz_token_sort_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.token_sort_ratio, MEDIUM_A, MEDIUM_B)


def test_fuzz_token_set_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.token_set_ratio, MEDIUM_A, MEDIUM_B)


def test_fuzz_token_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.token_ratio, MEDIUM_A, MEDIUM_B)


def test_fuzz_wratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.WRatio, MEDIUM_A, MEDIUM_B)


def test_fuzz_qratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.QRatio, MEDIUM_A, MEDIUM_B)


# ---------------------------------------------------------------------------
# distance — Levenshtein
# ---------------------------------------------------------------------------
def test_levenshtein_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.distance, SHORT_A, SHORT_B)


def test_levenshtein_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.distance, MEDIUM_A, MEDIUM_B)


def test_levenshtein_distance_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.distance, LONG_A, LONG_B)


def test_levenshtein_normalized_similarity_medium(
    benchmark: pytest.FixtureRequest,
) -> None:
    benchmark(Levenshtein.normalized_similarity, MEDIUM_A, MEDIUM_B)


def test_levenshtein_editops_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.editops, MEDIUM_A, MEDIUM_B)


def test_levenshtein_opcodes_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.opcodes, MEDIUM_A, MEDIUM_B)


# ---------------------------------------------------------------------------
# distance — other metrics
# ---------------------------------------------------------------------------
def test_hamming_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Hamming.distance, SHORT_A, SHORT_B)


def test_indel_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Indel.distance, MEDIUM_A, MEDIUM_B)


def test_jaro_similarity_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Jaro.similarity, MEDIUM_A, MEDIUM_B)


def test_jaro_winkler_similarity_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(JaroWinkler.similarity, MEDIUM_A, MEDIUM_B)


def test_lcs_seq_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(LCSseq.distance, MEDIUM_A, MEDIUM_B)


def test_osa_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(OSA.distance, MEDIUM_A, MEDIUM_B)


def test_damerau_levenshtein_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(DamerauLevenshtein.distance, MEDIUM_A, MEDIUM_B)


def test_prefix_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Prefix.distance, SHORT_A, SHORT_B)


def test_postfix_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Postfix.distance, SHORT_A, SHORT_B)


# ---------------------------------------------------------------------------
# process — batch operations
# ---------------------------------------------------------------------------
def test_process_extract(benchmark: pytest.FixtureRequest) -> None:
    benchmark(process.extract, "new york", CHOICES, limit=5)


def test_process_extract_one(benchmark: pytest.FixtureRequest) -> None:
    benchmark(process.extractOne, "new york", CHOICES)


def test_process_extract_iter(benchmark: pytest.FixtureRequest) -> None:
    def _run() -> None:
        list(process.extract_iter("new york", CHOICES))

    benchmark(_run)
