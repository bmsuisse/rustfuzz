"""
Scaling benchmark for HybridSearch 3-way RRF.

Measures index build time and single-query latency at various scales.

Usage:
    uv run python tests/bench_hybrid.py
"""

from __future__ import annotations

import random
import time

_RNG = random.Random(42)

DIM = 128  # Embedding dimensionality (typical for small models like MiniLM)


def _rand_text(n_words: int = 5) -> str:
    words = [
        "apple",
        "samsung",
        "google",
        "microsoft",
        "amazon",
        "iphone",
        "galaxy",
        "pixel",
        "surface",
        "echo",
        "pro",
        "max",
        "ultra",
        "plus",
        "mini",
        "laptop",
        "phone",
        "tablet",
        "headphones",
        "watch",
        "black",
        "white",
        "silver",
        "gold",
        "blue",
        "128gb",
        "256gb",
        "512gb",
        "1tb",
        "wireless",
    ]
    return " ".join(_RNG.choice(words) for _ in range(n_words))


def _rand_embedding() -> list[float]:
    return [_RNG.gauss(0, 1) for _ in range(DIM)]


def _generate(n: int) -> tuple[list[str], list[list[float]]]:
    """Generate n documents with random text and embeddings."""
    texts = [f"{_rand_text()} ID-{i}" for i in range(n)]
    embeddings = [_rand_embedding() for _ in range(n)]
    return texts, embeddings


def bench_build(texts: list[str], embeddings: list[list[float]], label: str) -> object:
    """Benchmark index build time."""
    from rustfuzz.search import HybridSearch

    t0 = time.perf_counter()
    hs = HybridSearch(texts, embeddings=embeddings)
    t1 = time.perf_counter()
    print(f"  [{label}] Build {len(texts):>10,} docs: {(t1 - t0) * 1000:>10.1f} ms")
    return hs


def bench_search_3way(
    hs: object,
    query: str,
    q_emb: list[float],
    label: str,
    n: int = 10,
    bm25_cands: int = 200,
) -> None:
    """Benchmark 3-way search (BM25 + fuzzy + dense)."""
    # Warmup
    hs.search(query, query_embedding=q_emb, n=n, bm25_candidates=bm25_cands)  # type: ignore[union-attr]

    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        hs.search(query, query_embedding=q_emb, n=n, bm25_candidates=bm25_cands)  # type: ignore[union-attr]
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / runs * 1000
    print(
        f"  [{label}] 3-way search (n={n}, cands={bm25_cands}): {avg_ms:>8.3f} ms/query"
    )


def bench_search_2way(
    hs: object, query: str, label: str, n: int = 10, bm25_cands: int = 200
) -> None:
    """Benchmark 2-way search (BM25 + fuzzy, no embedding)."""
    # Warmup
    hs.search(query, n=n, bm25_candidates=bm25_cands)  # type: ignore[union-attr]

    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        hs.search(query, n=n, bm25_candidates=bm25_cands)  # type: ignore[union-attr]
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / runs * 1000
    print(
        f"  [{label}] 2-way search (n={n}, cands={bm25_cands}): {avg_ms:>8.3f} ms/query"
    )


def main() -> None:
    print("=" * 65)
    print("  HybridSearch 3-Way RRF Benchmark")
    print("=" * 65)
    print(f"  Embedding dim: {DIM}")
    print()

    sizes = [1_000, 10_000, 100_000, 1_000_000]

    query = "apple iphone pro max 256gb"
    q_emb = _rand_embedding()

    for size in sizes:
        label = f"{size:>10,}"
        print(f"\n--- {label} documents ---")
        t0 = time.perf_counter()
        texts, embeddings = _generate(size)
        gen_time = (time.perf_counter() - t0) * 1000
        print(f"  [{label}] Gen data: {gen_time:>10.1f} ms")

        hs = bench_build(texts, embeddings, label)
        bench_search_3way(hs, query, q_emb, label, n=10, bm25_cands=200)
        bench_search_2way(hs, query, label, n=10, bm25_cands=200)

        # Also test with larger candidate pools for quality comparison
        if size >= 100_000:
            bench_search_3way(hs, query, q_emb, label, n=10, bm25_cands=1000)

        # Free memory before next iteration
        del hs, texts, embeddings

    print(f"\n{'=' * 65}")
    print("  ✅ Benchmark complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
