"""
Pytest-benchmark suite for MultiJoiner.

Run with:
    uv run pytest tests/bench_join.py -v --benchmark-sort=mean --benchmark-columns=mean,min,max,rounds

Compare before/after:
    uv run pytest tests/bench_join.py --benchmark-save=baseline
    # rebuild, then:
    uv run pytest tests/bench_join.py --benchmark-compare=baseline
"""

from __future__ import annotations

import math
import random

import pytest

from rustfuzz.join import MultiJoiner, fuzzy_join

# ---------------------------------------------------------------------------
# Fixed shared data  (single RNG, generated ONCE at import time)
# All benchmark functions that parametrise over the same size share the same
# arrays so the comparisons aren't poisoned by RNG drift.
# ---------------------------------------------------------------------------

_RNG = random.Random(42)

_WORDS = [
    "apple",
    "iphone",
    "samsung",
    "galaxy",
    "google",
    "pixel",
    "pro",
    "ultra",
    "max",
    "mini",
    "air",
    "plus",
    "note",
    "tab",
    "watch",
    "book",
    "pad",
    "studio",
    "chip",
    "nano",
    "micro",
    "lite",
    "edge",
    "fold",
    "flip",
    "zoom",
    "arc",
    "prime",
    "neo",
    "opus",
    "nova",
    "apex",
    "flex",
    "core",
    "smart",
    "turbo",
    "speed",
    "rapid",
    "swift",
    "light",
    "wave",
    "peak",
]


def _rand_text(n_words: int = 4) -> str:
    return " ".join(_RNG.choices(_WORDS, k=n_words))


def _norm(v: list[float]) -> list[float]:
    m = math.sqrt(sum(x * x for x in v))
    return [x / m for x in v] if m else v


def _texts(n: int) -> list[str]:
    return [_rand_text() for _ in range(n)]


def _embeddings(n: int, dim: int = 128) -> list[list[float]]:
    return [_norm([_RNG.gauss(0, 1) for _ in range(dim)]) for _ in range(n)]


# Pre-generate all data at module import so all benchmark functions see the
# same distributions (no RNG state contamination between parametrize values).
_TEXTS: dict[int, tuple[list[str], list[str]]] = {
    sz: (_texts(sz), _texts(sz)) for sz in [100, 500, 1000, 2000]
}
_EMBS: dict[tuple[int, int], tuple] = {
    (sz, dim): (_embeddings(sz, dim), _embeddings(sz, dim))
    for sz, dim in [(100, 128), (500, 128), (1000, 128), (500, 512)]
}
# Shared 1000-elem arrays for bm25_candidates sweep
_SWEEP_A, _SWEEP_B = _texts(1000), _texts(1000)
# N-array data (200 elements each)
_NARRAYS: dict[int, dict[str, list[str]]] = {
    n: {f"arr{i}": _texts(200) for i in range(n)} for n in [2, 3, 5, 10]
}


# ---------------------------------------------------------------------------
# 1. Text-only: varying array size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [100, 500, 1000, 2000])
def test_bench_text_join(benchmark, size):
    a, b = _TEXTS[size]
    benchmark(fuzzy_join, {"A": a, "B": b}, n=1)


# ---------------------------------------------------------------------------
# 2. Dense-only: varying size and embedding dimension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size,dim", [(100, 128), (500, 128), (1000, 128), (500, 512)])
def test_bench_dense_join(benchmark, size, dim):
    a, b = _EMBS[(size, dim)]

    def _run():
        return (
            MultiJoiner(text_weight=0, sparse_weight=0, dense_weight=1)
            .add_array("A", dense=a)
            .add_array("B", dense=b)
            .join(n=1)
        )

    benchmark(_run)


# ---------------------------------------------------------------------------
# 3. bm25_candidates sweep — quality/speed tradeoff of the pre-filter
#    Uses the SAME array pair for all values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bm25_candidates", [10, 25, 50, 100, 200, 500, 1000])
def test_bench_bm25_candidates(benchmark, bm25_candidates):
    """Shows the quality/speed tradeoff of the BM25 candidate pre-filter."""
    a, b = _SWEEP_A, _SWEEP_B

    def _run():
        return fuzzy_join({"A": a, "B": b}, n=1, bm25_candidates=bm25_candidates)

    benchmark(_run)


# ---------------------------------------------------------------------------
# 4. N-array scaling (200 elements/array, text only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_arrays", [2, 3, 5, 10])
def test_bench_n_arrays(benchmark, n_arrays):
    arrays = _NARRAYS[n_arrays]
    benchmark(fuzzy_join, arrays, n=1)


# ---------------------------------------------------------------------------
# 5. Top-N scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_n", [1, 5, 10, 50])
def test_bench_top_n(benchmark, top_n):
    a, b = _TEXTS[500]
    benchmark(fuzzy_join, {"A": a, "B": b}, n=top_n)


# ---------------------------------------------------------------------------
# 6. Mixed channels (text + dense, same data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [100, 500, 1000])
def test_bench_mixed_join(benchmark, size):
    ta, tb = _TEXTS[size]
    ea, eb = _EMBS[(size, 128)]

    def _run():
        return (
            MultiJoiner(text_weight=0.5, sparse_weight=0.5, dense_weight=0.5)
            .add_array("A", texts=ta, dense=ea)
            .add_array("B", texts=tb, dense=eb)
            .join(n=1)
        )

    benchmark(_run)


# ---------------------------------------------------------------------------
# 7. join_wide overhead vs join (3 arrays x 200 elements)
# ---------------------------------------------------------------------------


def test_bench_join_long(benchmark):
    arrays = _NARRAYS[3]
    j = MultiJoiner()
    for k, v in arrays.items():
        j.add_array(k, texts=v)
    benchmark(j.join, 1)


def test_bench_join_wide(benchmark):
    arrays = _NARRAYS[3]
    j = MultiJoiner()
    for k, v in arrays.items():
        j.add_array(k, texts=v)
    benchmark(j.join_wide, "arr0", 1)


# ---------------------------------------------------------------------------
# 8. Inner join (score_cutoff now in Rust, not Python)
# ---------------------------------------------------------------------------


def test_bench_inner_join(benchmark):
    a, b = _TEXTS[1000]
    benchmark(fuzzy_join, {"A": a, "B": b}, n=1, how="inner", score_cutoff=0.01)
