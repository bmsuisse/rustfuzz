"""
Benchmark: HybridSearch with filter + sort — Rust vs Python path.

Measures latency at 1K / 10K / 100K scale using Faker-generated data.

Usage:
    uv run python tests/bench_hybrid_filter_sort.py
"""

from __future__ import annotations

import random
import time

_RNG = random.Random(42)
DIM = 64  # Small embedding dim for fast bench

BRANDS = ["Apple", "Samsung", "Google", "Microsoft", "Amazon", "Sony", "LG", "OnePlus"]
CATEGORIES = ["phone", "tablet", "laptop", "watch", "headphones", "tv", "camera"]


def _rand_embedding() -> list[float]:
    return [_RNG.gauss(0, 1) for _ in range(DIM)]


def _generate(n: int) -> tuple[list[str], list[list[float]], list[dict[str, object]]]:
    """Generate n documents with random text, embeddings, and metadata."""
    try:
        from faker import Faker

        fake = Faker()
        Faker.seed(42)
    except ImportError:
        fake = None

    texts: list[str] = []
    embeddings: list[list[float]] = []
    metadata: list[dict[str, object]] = []

    for i in range(n):
        brand = _RNG.choice(BRANDS)
        category = _RNG.choice(CATEGORIES)
        price = round(_RNG.uniform(50, 3000), 2)
        rating = round(_RNG.uniform(1.0, 5.0), 1)
        year = _RNG.choice([2021, 2022, 2023, 2024])
        in_stock = _RNG.choice([True, False])

        if fake:
            text = f"{brand} {fake.catch_phrase()} {category} {i}"
        else:
            text = f"{brand} product-{i} {category}"

        texts.append(text)
        embeddings.append(_rand_embedding())
        metadata.append(
            {
                "brand": brand,
                "category": category,
                "price": price,
                "rating": rating,
                "year": year,
                "in_stock": in_stock,
            }
        )

    return texts, embeddings, metadata


def bench_build(
    texts: list[str],
    embeddings: list[list[float]],
    metadata: list[dict[str, object]],
    label: str,
) -> object:
    from rustfuzz.search import HybridSearch

    t0 = time.perf_counter()
    hs = HybridSearch(texts, embeddings=embeddings, metadata=metadata)
    t1 = time.perf_counter()
    print(f"  [{label}] Build {len(texts):>10,} docs: {(t1 - t0) * 1000:>10.1f} ms")
    return hs


def bench_search_no_filter(hs: object, label: str, n: int = 10) -> None:
    from rustfuzz.search import HybridSearch

    assert isinstance(hs, HybridSearch)
    q_emb = _rand_embedding()

    # Warmup
    hs.search("apple phone pro", query_embedding=q_emb, n=n)

    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        hs.search("apple phone pro", query_embedding=q_emb, n=n)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / runs * 1000
    print(f"  [{label}] Search (no filter):          {avg_ms:>8.3f} ms/query")


def bench_filter_search(hs: object, label: str, n: int = 10) -> None:
    from rustfuzz.search import HybridSearch

    assert isinstance(hs, HybridSearch)
    q_emb = _rand_embedding()

    # Warmup
    hs.filter('brand = "Apple"').get_top_n("apple phone", n=n, query_embedding=q_emb)

    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        hs.filter('brand = "Apple"').get_top_n(
            "apple phone", n=n, query_embedding=q_emb
        )
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / runs * 1000
    print(f"  [{label}] Filter search:               {avg_ms:>8.3f} ms/query")


def bench_filter_sort_search(hs: object, label: str, n: int = 10) -> None:
    from rustfuzz.search import HybridSearch

    assert isinstance(hs, HybridSearch)
    q_emb = _rand_embedding()

    # Warmup
    hs.filter("price > 500").sort("price:asc").get_top_n(
        "phone pro", n=n, query_embedding=q_emb
    )

    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        hs.filter("price > 500").sort("price:asc").get_top_n(
            "phone pro", n=n, query_embedding=q_emb
        )
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / runs * 1000
    print(f"  [{label}] Filter+Sort search:          {avg_ms:>8.3f} ms/query")


def bench_complex_filter_search(hs: object, label: str, n: int = 10) -> None:
    from rustfuzz.search import HybridSearch

    assert isinstance(hs, HybridSearch)
    q_emb = _rand_embedding()
    expr = 'price > 200 AND (brand = "Apple" OR brand = "Samsung") AND in_stock = true'

    # Warmup
    hs.filter(expr).sort("rating:desc").get_top_n(
        "pro phone", n=n, query_embedding=q_emb
    )

    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        hs.filter(expr).sort("rating:desc").get_top_n(
            "pro phone", n=n, query_embedding=q_emb
        )
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / runs * 1000
    print(f"  [{label}] Complex filter+sort search:  {avg_ms:>8.3f} ms/query")


def bench_filter_mask_only(metadata: list[dict[str, object]], label: str) -> None:
    """Benchmark just the filter mask evaluation (Python path)."""
    from rustfuzz.filter import evaluate_filter, parse_filter

    node = parse_filter('brand = "Apple" AND price > 500')

    # Warmup
    [evaluate_filter(node, m) for m in metadata]

    runs = 20
    t0 = time.perf_counter()
    for _ in range(runs):
        [evaluate_filter(node, m) for m in metadata]
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / runs * 1000
    print(f"  [{label}] Python filter mask:          {avg_ms:>8.3f} ms")


def bench_rust_filter_mask(
    hs: object, metadata: list[dict[str, object]], label: str
) -> None:
    """Benchmark Rust-side filter mask evaluation."""
    from rustfuzz.filter import filter_to_json, parse_filter

    node = parse_filter('brand = "Apple" AND price > 500')
    filter_json = filter_to_json(node)

    # Warmup
    hs._index.evaluate_filter_mask(filter_json)  # type: ignore[union-attr]

    runs = 20
    t0 = time.perf_counter()
    for _ in range(runs):
        hs._index.evaluate_filter_mask(filter_json)  # type: ignore[union-attr]
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / runs * 1000
    print(f"  [{label}] Rust filter mask:            {avg_ms:>8.3f} ms")


def main() -> None:
    print("=" * 70)
    print("  HybridSearch Filter+Sort Benchmark (Rust-accelerated)")
    print("=" * 70)
    print(f"  Embedding dim: {DIM}")
    print()

    sizes = [1_000, 10_000, 100_000]

    for size in sizes:
        label = f"{size:>10,}"
        print(f"\n--- {label} documents ---")

        t0 = time.perf_counter()
        texts, embeddings, metadata = _generate(size)
        gen_time = (time.perf_counter() - t0) * 1000
        print(f"  [{label}] Gen data: {gen_time:>10.1f} ms")

        hs = bench_build(texts, embeddings, metadata, label)

        bench_search_no_filter(hs, label)
        bench_filter_search(hs, label)
        bench_filter_sort_search(hs, label)
        bench_complex_filter_search(hs, label)

        # Compare Python vs Rust filter mask evaluation
        print()
        bench_filter_mask_only(metadata, label)
        bench_rust_filter_mask(hs, metadata, label)

        # Compute speedup
        from rustfuzz.filter import evaluate_filter, filter_to_json, parse_filter

        node = parse_filter('brand = "Apple" AND price > 500')
        fj = filter_to_json(node)

        runs = 10
        t0 = time.perf_counter()
        for _ in range(runs):
            [evaluate_filter(node, m) for m in metadata]
        py_time = (time.perf_counter() - t0) / runs

        t0 = time.perf_counter()
        for _ in range(runs):
            hs._index.evaluate_filter_mask(fj)  # type: ignore[union-attr]
        rs_time = (time.perf_counter() - t0) / runs

        if rs_time > 0:
            speedup = py_time / rs_time
            print(f"  [{label}] ⚡ Rust filter mask speedup: {speedup:>6.1f}x")

        del hs, texts, embeddings, metadata

    print(f"\n{'=' * 70}")
    print("  ✅ Benchmark complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
