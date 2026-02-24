"""
rustfuzz — performance regression benchmarks vs rapidfuzz.

Run once to establish a baseline:
    uv run pytest tests/test_benchmarks.py --benchmark-save=baseline

Compare against baseline (10% regression threshold):
    uv run pytest tests/test_benchmarks.py --benchmark-compare=baseline --benchmark-compare-fail=mean:10%

Run only a single group:
    uv run pytest tests/test_benchmarks.py -k "wratio"
"""

from __future__ import annotations

import random
import string

import pytest
import rapidfuzz.fuzz as rf_fuzz
from rapidfuzz import process as rf_process
from rapidfuzz.distance import (
    OSA as rf_OSA,
)
from rapidfuzz.distance import (
    DamerauLevenshtein as rf_DamerauLevenshtein,
)
from rapidfuzz.distance import (
    Hamming as rf_Hamming,
)
from rapidfuzz.distance import (
    Indel as rf_Indel,
)
from rapidfuzz.distance import (
    Jaro as rf_Jaro,
)
from rapidfuzz.distance import (
    JaroWinkler as rf_JaroWinkler,
)
from rapidfuzz.distance import (
    LCSseq as rf_LCSseq,
)
from rapidfuzz.distance import (
    Levenshtein as rf_Levenshtein,
)
from rapidfuzz.distance import (
    Postfix as rf_Postfix,
)
from rapidfuzz.distance import (
    Prefix as rf_Prefix,
)

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
# Representative string pairs — keyed by scenario
# ---------------------------------------------------------------------------

# Short: 5-char "typo" pair
SHORT_A = "hello"
SHORT_B = "hallo"

# Medium: 43-char natural language, high similarity (~95%)
MEDIUM_A = "the quick brown fox jumps over the lazy dog"
MEDIUM_B = "the quick brown fox jumped over a lazy dog"

# Medium low-sim: same length, low similarity
MEDIUM_LOW_A = "abcdefghijklmnopqrs"
MEDIUM_LOW_B = "zyxwvutsrqponmlkjih"

# Long: 400-char synthetic, ~50% similarity
LONG_A = "a" * 200 + "b" * 200
LONG_B = "a" * 195 + "c" * 210

# Long high-sim: very similar long strings
LONG_HI_A = "x" * 300 + "abcdefghijklmnopqrstuvwxyz" * 3
LONG_HI_B = "x" * 298 + "y" * 2 + "abcdefghijklmnopqrstuvwxyz" * 3

# Unicode (non-ASCII) pair
UNICODE_A = "Héllo wörld — café naïf"
UNICODE_B = "Hello world — cafe naif"

# process.extract fixtures — different batch sizes
CHOICES_SMALL = [
    "New York",
    "New Orleans",
    "Newark",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
]

rng = random.Random(42)
_words = [
    "apple",
    "banana",
    "cherry",
    "date",
    "elderberry",
    "fig",
    "grape",
    "honeydew",
    "kiwi",
    "lemon",
    "mango",
    "nectarine",
    "orange",
    "papaya",
    "quince",
    "raspberry",
    "strawberry",
    "tangerine",
    "ugli",
    "vanilla",
    "watermelon",
    "xigua",
    "yam",
    "zucchini",
    "almond",
    "brazil",
    "cashew",
    "dill",
    "eggplant",
    "fennel",
    "garlic",
    "horseradish",
    "iceapple",
    "jalapeño",
    "kale",
    "leek",
    "mushroom",
    "nutmeg",
    "oregano",
]

CHOICES_MEDIUM = [f"{rng.choice(_words)} {rng.choice(_words)}" for _ in range(100)]
CHOICES_LARGE = [
    "".join(rng.choices(string.ascii_lowercase + " ", k=rng.randint(5, 30)))
    for _ in range(1000)
]


# ---------------------------------------------------------------------------
# fuzz.ratio — short
# ---------------------------------------------------------------------------
def test_fuzz_ratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.ratio, SHORT_A, SHORT_B)


def test_rf_fuzz_ratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.ratio, SHORT_A, SHORT_B)


# ---------------------------------------------------------------------------
# fuzz.ratio — medium high-sim
# ---------------------------------------------------------------------------
def test_fuzz_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.ratio, MEDIUM_A, MEDIUM_B)


def test_rf_fuzz_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.ratio, MEDIUM_A, MEDIUM_B)


# ---------------------------------------------------------------------------
# fuzz.ratio — medium low-sim (early exit wins)
# ---------------------------------------------------------------------------
def test_fuzz_ratio_medium_low_sim(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.ratio, MEDIUM_LOW_A, MEDIUM_LOW_B)


def test_rf_fuzz_ratio_medium_low_sim(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.ratio, MEDIUM_LOW_A, MEDIUM_LOW_B)


# ---------------------------------------------------------------------------
# fuzz.ratio — long
# ---------------------------------------------------------------------------
def test_fuzz_ratio_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.ratio, LONG_A, LONG_B)


def test_rf_fuzz_ratio_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.ratio, LONG_A, LONG_B)


# ---------------------------------------------------------------------------
# fuzz.ratio — long high-sim
# ---------------------------------------------------------------------------
def test_fuzz_ratio_long_hi_sim(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.ratio, LONG_HI_A, LONG_HI_B)


def test_rf_fuzz_ratio_long_hi_sim(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.ratio, LONG_HI_A, LONG_HI_B)


# ---------------------------------------------------------------------------
# fuzz.partial_ratio — short / medium / long
# ---------------------------------------------------------------------------
def test_fuzz_partial_ratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.partial_ratio, SHORT_A, SHORT_B)


def test_rf_fuzz_partial_ratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.partial_ratio, SHORT_A, SHORT_B)


def test_fuzz_partial_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.partial_ratio, MEDIUM_A, MEDIUM_B)


def test_rf_fuzz_partial_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.partial_ratio, MEDIUM_A, MEDIUM_B)


def test_fuzz_partial_ratio_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.partial_ratio, LONG_HI_A, LONG_HI_B)


def test_rf_fuzz_partial_ratio_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.partial_ratio, LONG_HI_A, LONG_HI_B)


# ---------------------------------------------------------------------------
# fuzz — token scorers (medium)
# ---------------------------------------------------------------------------
def test_fuzz_token_sort_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.token_sort_ratio, MEDIUM_A, MEDIUM_B)


def test_rf_fuzz_token_sort_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.token_sort_ratio, MEDIUM_A, MEDIUM_B)


def test_fuzz_token_set_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.token_set_ratio, MEDIUM_A, MEDIUM_B)


def test_rf_fuzz_token_set_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.token_set_ratio, MEDIUM_A, MEDIUM_B)


def test_fuzz_token_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.token_ratio, MEDIUM_A, MEDIUM_B)


def test_rf_fuzz_token_ratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.token_ratio, MEDIUM_A, MEDIUM_B)


# ---------------------------------------------------------------------------
# fuzz — token scorers (short)
# ---------------------------------------------------------------------------
def test_fuzz_token_sort_ratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.token_sort_ratio, SHORT_A, SHORT_B)


def test_rf_fuzz_token_sort_ratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.token_sort_ratio, SHORT_A, SHORT_B)


def test_fuzz_token_set_ratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.token_set_ratio, SHORT_A, SHORT_B)


def test_rf_fuzz_token_set_ratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.token_set_ratio, SHORT_A, SHORT_B)


# ---------------------------------------------------------------------------
# fuzz — WRatio / QRatio (short / medium / long)
# ---------------------------------------------------------------------------
def test_fuzz_wratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.WRatio, SHORT_A, SHORT_B)


def test_rf_fuzz_wratio_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.WRatio, SHORT_A, SHORT_B)


def test_fuzz_wratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.WRatio, MEDIUM_A, MEDIUM_B)


def test_rf_fuzz_wratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.WRatio, MEDIUM_A, MEDIUM_B)


def test_fuzz_wratio_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.WRatio, LONG_HI_A, LONG_HI_B)


def test_rf_fuzz_wratio_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.WRatio, LONG_HI_A, LONG_HI_B)


def test_fuzz_qratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(fuzz.QRatio, MEDIUM_A, MEDIUM_B)


def test_rf_fuzz_qratio_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_fuzz.QRatio, MEDIUM_A, MEDIUM_B)


# ---------------------------------------------------------------------------
# distance — Levenshtein
# ---------------------------------------------------------------------------
def test_levenshtein_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.distance, SHORT_A, SHORT_B)


def test_rf_levenshtein_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Levenshtein.distance, SHORT_A, SHORT_B)


def test_levenshtein_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.distance, MEDIUM_A, MEDIUM_B)


def test_rf_levenshtein_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Levenshtein.distance, MEDIUM_A, MEDIUM_B)


def test_levenshtein_distance_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.distance, LONG_A, LONG_B)


def test_rf_levenshtein_distance_long(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Levenshtein.distance, LONG_A, LONG_B)


def test_levenshtein_normalized_similarity_medium(
    benchmark: pytest.FixtureRequest,
) -> None:
    benchmark(Levenshtein.normalized_similarity, MEDIUM_A, MEDIUM_B)


def test_rf_levenshtein_normalized_similarity_medium(
    benchmark: pytest.FixtureRequest,
) -> None:
    benchmark(rf_Levenshtein.normalized_similarity, MEDIUM_A, MEDIUM_B)


def test_levenshtein_editops_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.editops, MEDIUM_A, MEDIUM_B)


def test_rf_levenshtein_editops_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Levenshtein.editops, MEDIUM_A, MEDIUM_B)


def test_levenshtein_opcodes_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Levenshtein.opcodes, MEDIUM_A, MEDIUM_B)


def test_rf_levenshtein_opcodes_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Levenshtein.opcodes, MEDIUM_A, MEDIUM_B)


# ---------------------------------------------------------------------------
# distance — other metrics
# ---------------------------------------------------------------------------
def test_hamming_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Hamming.distance, SHORT_A, SHORT_B)


def test_rf_hamming_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Hamming.distance, SHORT_A, SHORT_B)


def test_indel_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Indel.distance, MEDIUM_A, MEDIUM_B)


def test_rf_indel_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Indel.distance, MEDIUM_A, MEDIUM_B)


def test_jaro_similarity_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Jaro.similarity, MEDIUM_A, MEDIUM_B)


def test_rf_jaro_similarity_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Jaro.similarity, MEDIUM_A, MEDIUM_B)


def test_jaro_winkler_similarity_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(JaroWinkler.similarity, MEDIUM_A, MEDIUM_B)


def test_rf_jaro_winkler_similarity_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_JaroWinkler.similarity, MEDIUM_A, MEDIUM_B)


def test_lcs_seq_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(LCSseq.distance, MEDIUM_A, MEDIUM_B)


def test_rf_lcs_seq_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_LCSseq.distance, MEDIUM_A, MEDIUM_B)


def test_osa_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(OSA.distance, MEDIUM_A, MEDIUM_B)


def test_rf_osa_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_OSA.distance, MEDIUM_A, MEDIUM_B)


def test_damerau_levenshtein_distance_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(DamerauLevenshtein.distance, MEDIUM_A, MEDIUM_B)


def test_rf_damerau_levenshtein_distance_medium(
    benchmark: pytest.FixtureRequest,
) -> None:
    benchmark(rf_DamerauLevenshtein.distance, MEDIUM_A, MEDIUM_B)


def test_prefix_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Prefix.distance, SHORT_A, SHORT_B)


def test_rf_prefix_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Prefix.distance, SHORT_A, SHORT_B)


def test_postfix_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(Postfix.distance, SHORT_A, SHORT_B)


def test_rf_postfix_distance_short(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_Postfix.distance, SHORT_A, SHORT_B)


# ---------------------------------------------------------------------------
# process — small batch (14 choices)
# ---------------------------------------------------------------------------
def test_process_extract(benchmark: pytest.FixtureRequest) -> None:
    benchmark(process.extract, "new york", CHOICES_SMALL, limit=5)


def test_rf_process_extract(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_process.extract, "new york", CHOICES_SMALL, limit=5)


def test_process_extract_one_low_sim(benchmark: pytest.FixtureRequest) -> None:
    benchmark(process.extractOne, "zimbabwe", CHOICES_SMALL)


# ---------------------------------------------------------------------------
# process.cdist Benchmarks
# ---------------------------------------------------------------------------


def test_cdist_rf(benchmark: pytest.FixtureRequest) -> None:
    # We test with a small subset to keep benchmark times reasonable
    q = CHOICES_SMALL[:10]
    c = CHOICES_SMALL
    benchmark(rf_process.cdist, q, c, scorer=rf_fuzz.ratio)


def test_cdist_rustfuzz(benchmark: pytest.FixtureRequest) -> None:
    q = CHOICES_SMALL[:10]
    c = CHOICES_SMALL
    benchmark(process.cdist, q, c, scorer=fuzz.ratio)


# ---------------------------------------------------------------------------
# BM25 / Hybrid Search Benchmarks
# ---------------------------------------------------------------------------


def test_bm25_rank_bm25(benchmark: pytest.FixtureRequest) -> None:
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [doc.split(" ") for doc in BM25_CORPUS]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = BM25_QUERY.split(" ")
        benchmark(bm25.get_scores, tokenized_query)
    except ImportError:
        pytest.skip("rank_bm25 not installed")


def test_bm25_rustfuzz(benchmark: pytest.FixtureRequest) -> None:
    from rustfuzz.search import BM25

    bm25 = BM25(BM25_CORPUS)
    benchmark(bm25.get_scores, BM25_QUERY)


def test_bm25l_rustfuzz(benchmark: pytest.FixtureRequest) -> None:
    from rustfuzz.search import BM25L

    bm25 = BM25L(BM25_CORPUS)
    benchmark(bm25.get_scores, BM25_QUERY)


def test_bm25plus_rustfuzz(benchmark: pytest.FixtureRequest) -> None:
    from rustfuzz.search import BM25Plus

    bm25 = BM25Plus(BM25_CORPUS)
    benchmark(bm25.get_scores, BM25_QUERY)


def test_bm25t_rustfuzz(benchmark: pytest.FixtureRequest) -> None:
    from rustfuzz.search import BM25T

    bm25 = BM25T(BM25_CORPUS)
    benchmark(bm25.get_scores, BM25_QUERY)


def test_bm25_rustfuzz_top_n(benchmark: pytest.FixtureRequest) -> None:
    from rustfuzz.search import BM25

    bm25 = BM25(BM25_CORPUS)
    benchmark(bm25.get_top_n, BM25_QUERY, n=5)


def test_bm25_rustfuzz_fuzzy_rrf(benchmark: pytest.FixtureRequest) -> None:
    from rustfuzz.search import BM25

    bm25 = BM25(BM25_CORPUS)
    benchmark(bm25.get_top_n_rrf, BM25_QUERY, n=5)


def test_process_extract_one(benchmark: pytest.FixtureRequest) -> None:
    benchmark(process.extractOne, "new york", CHOICES_SMALL)


def test_rf_process_extract_one(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_process.extractOne, "new york", CHOICES_SMALL)


def test_process_extract_iter(benchmark: pytest.FixtureRequest) -> None:
    def _run() -> None:
        list(process.extract_iter("new york", CHOICES_SMALL))

    benchmark(_run)


def test_rf_process_extract_iter(benchmark: pytest.FixtureRequest) -> None:
    def _run() -> None:
        list(rf_process.extract_iter("new york", CHOICES_SMALL))

    benchmark(_run)


# ---------------------------------------------------------------------------
# process — medium batch (100 choices)
# ---------------------------------------------------------------------------
def test_process_extract_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(process.extract, "apple banana", CHOICES_MEDIUM, limit=5)


def test_rf_process_extract_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_process.extract, "apple banana", CHOICES_MEDIUM, limit=5)


def test_process_extract_one_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(process.extractOne, "apple banana", CHOICES_MEDIUM)


def test_rf_process_extract_one_medium(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_process.extractOne, "apple banana", CHOICES_MEDIUM)


# ---------------------------------------------------------------------------
# process — large batch (1000 choices)
# ---------------------------------------------------------------------------
def test_process_extract_large(benchmark: pytest.FixtureRequest) -> None:
    benchmark(process.extract, "hello world foo", CHOICES_LARGE, limit=10)


def test_rf_process_extract_large(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_process.extract, "hello world foo", CHOICES_LARGE, limit=10)


def test_process_extract_one_large(benchmark: pytest.FixtureRequest) -> None:
    benchmark(process.extractOne, "hello world foo", CHOICES_LARGE)


def test_rf_process_extract_one_large(benchmark: pytest.FixtureRequest) -> None:
    benchmark(rf_process.extractOne, "hello world foo", CHOICES_LARGE)


# BM25 Corpus (medium size for benchmarking)
BM25_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the fast brown fox jumped",
    "a lazy dog sleeps all day",
    "sphinx of black quartz judge my vow",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
] * 100
BM25_QUERY = "quick brown fox"


# ---------------------------------------------------------------------------
# process — with score_cutoff (filters most items early)
# ---------------------------------------------------------------------------
def test_process_extract_cutoff_large(benchmark: pytest.FixtureRequest) -> None:
    benchmark(
        process.extract, "hello world foo", CHOICES_LARGE, limit=10, score_cutoff=80
    )


def test_rf_process_extract_cutoff_large(benchmark: pytest.FixtureRequest) -> None:
    benchmark(
        rf_process.extract, "hello world foo", CHOICES_LARGE, limit=10, score_cutoff=80
    )
