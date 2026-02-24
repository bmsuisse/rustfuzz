"""
Scalability benchmark for MultiJoiner.
Measures search latency of N_src queries against N_tgt documents.

Usage:
    uv run pytest tests/bench_scaling.py -v --benchmark-sort=mean
"""

from __future__ import annotations

import random
import time

import pytest

_RNG = random.Random(42)


def _rand_text(n_words: int = 4) -> str:
    # Use a large vocabulary (100k) so that documents are sparse.
    # A 25-word vocab means every document matches every query!
    return " ".join(f"word_{_RNG.randint(1, 100_000)}" for _ in range(n_words))


def _texts(n: int) -> list[str]:
    return [_rand_text(4) for _ in range(n)]


# Pre-generate target corpora of different sizes
# We'll use a small fixed number of queries (N_src=100) to measure the O(N_tgt) lookup cost
N_SRC = 100
QUERIES = _texts(N_SRC)

# Pre-generate sizes up to 2 million.
# Note: generating 2M strings takes a few seconds in Python and uses ~150MB RAM.
SIZES = [200, 2_000, 20_000, 200_000, 2_000_000]

print(f"Pre-generating data for sizes {SIZES}...")
t0 = time.time()
_CORPORA = {sz: _texts(sz) for sz in SIZES}
print(f"Done in {time.time() - t0:.2f}s")


@pytest.mark.parametrize("n_tgt", SIZES)
def test_bench_text_scaling(benchmark, n_tgt):
    """
    Measures the time to execute N_SRC (100) queries against n_tgt documents.
    This exposes the O(N_tgt) scaling factor per query.
    """
    queries = QUERIES
    corpus = _CORPORA[n_tgt]

    # Pre-build the joiner outside the timing loop!
    # Otherwise we measure PyO3 deep-copying 2 million Python strings into Rust on every iteration.
    from rustfuzz.join import MultiJoiner

    joiner = MultiJoiner()
    joiner.add_array("queries", texts=queries)
    joiner.add_array("corpus", texts=corpus)

    def _run():
        return joiner.join_pair(
            "queries",
            "corpus",
            n=1,
            score_cutoff=0.01,
        )

    benchmark(_run)
