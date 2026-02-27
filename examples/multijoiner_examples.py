"""
MultiJoiner — Comprehensive examples showing every join mode and channel.

Run:  uv run python examples/multijoiner_examples.py
"""

from __future__ import annotations

import math
import random


def divider(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────────────
# Dummy data
# ──────────────────────────────────────────────────────────────────────

PRODUCTS = [
    "Apple iPhone 15 Pro Max",
    "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro",
    "OnePlus 12",
    "Sony WH-1000XM5 Headphones",
]

LISTINGS = [
    "Apple iPhone 15 Pro Max 256GB Black",
    "Samsung Galaxy S24 Ultra 512GB",
    "Pixel 8 Pro by Google",
    "OnePlus Twelve 5G",
    "Sony WH1000XM5 Wireless Headphones",
]

ERP = [
    "APPLE IPHONE 15 PROMAX",
    "SAMSUNG GALAXY S24",
    "GOOGLE PIXEL 8",
    "ONEPLUS 12 PHONE",
    "SONY HEADPHONES WH1000XM5",
]


def _make_embeddings(texts: list[str], dim: int = 32, seed: int = 42) -> list[list[float]]:
    """Generate deterministic dummy embeddings and L2-normalise them."""
    rng = random.Random(seed)
    vecs: list[list[float]] = []
    for _ in texts:
        raw = [rng.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw))
        vecs.append([x / norm for x in raw])
    return vecs


# ──────────────────────────────────────────────────────────────────────
# 1. Text-only join (BM25 + Indel fuzzy re-ranking)
# ──────────────────────────────────────────────────────────────────────

def example_text_only() -> None:
    divider("1 · Text-Only Join (BM25 + Indel)")

    from rustfuzz.join import MultiJoiner

    joiner = (
        MultiJoiner(text_weight=1.0, sparse_weight=0.0, dense_weight=0.0)
        .add_array("products", texts=PRODUCTS)
        .add_array("listings", texts=LISTINGS)
    )

    results = joiner.join(n=1)
    print("  Each product matched to its best listing (text channel only):\n")
    for r in results:
        if r["src_array"] == "products":
            print(
                f"    {r['src_text']:35s} → {r['tgt_text']:40s}  "
                f"score={r['score']:.4f}"
            )


# ──────────────────────────────────────────────────────────────────────
# 2. Dense-only join (cosine similarity on embeddings)
# ──────────────────────────────────────────────────────────────────────

def example_dense_only() -> None:
    divider("2 · Dense-Only Join (Cosine Similarity)")

    from rustfuzz.join import MultiJoiner

    emb_products = _make_embeddings(PRODUCTS, seed=1)
    emb_listings = _make_embeddings(LISTINGS, seed=2)

    joiner = (
        MultiJoiner(text_weight=0.0, sparse_weight=0.0, dense_weight=1.0)
        .add_array("products", dense=emb_products)
        .add_array("listings", dense=emb_listings)
    )

    results = joiner.join(n=1)
    print("  Matching via embeddings only (no text):\n")
    for r in results:
        if r["src_array"] == "products":
            print(
                f"    product[{r['src_idx']}] → listing[{r['tgt_idx']}]  "
                f"score={r['score']:.4f}  dense={r['dense_score']:.4f}"
            )

    print("\n  (With dummy embeddings the matches are random — use real embeddings")
    print("   from sentence-transformers or similar for meaningful results.)")


# ──────────────────────────────────────────────────────────────────────
# 3. Sparse-only join (explicit sparse vectors)
# ──────────────────────────────────────────────────────────────────────

def example_sparse_only() -> None:
    divider("3 · Sparse-Only Join (Explicit Sparse Vectors)")

    from rustfuzz.join import MultiJoiner

    # Simulate sparse vectors (e.g., from a learned sparse model like SPLADE)
    # Keys are token IDs, values are weights.
    sparse_a = [
        {10: 1.5, 20: 0.8, 30: 0.3},  # "Apple iPhone"
        {40: 1.2, 50: 0.9},            # "Samsung Galaxy"
        {60: 1.0, 70: 0.7, 80: 0.2},   # "Google Pixel"
    ]
    sparse_b = [
        {10: 1.3, 20: 0.6, 35: 0.1},   # overlaps with entry 0
        {40: 1.0, 50: 0.8, 55: 0.4},   # overlaps with entry 1
        {60: 0.9, 70: 0.5},            # overlaps with entry 2
        {99: 1.0},                      # no overlap with any
    ]

    joiner = (
        MultiJoiner(text_weight=0.0, sparse_weight=1.0, dense_weight=0.0)
        .add_array("src", sparse=sparse_a)
        .add_array("tgt", sparse=sparse_b)
    )

    results = joiner.join(n=1)
    print("  Sparse-vector join (merge-join dot-product):\n")
    for r in results:
        if r["src_array"] == "src":
            print(
                f"    src[{r['src_idx']}] → tgt[{r['tgt_idx']}]  "
                f"score={r['score']:.4f}  sparse={r['sparse_score']:.4f}"
            )


# ──────────────────────────────────────────────────────────────────────
# 4. Hybrid join (text + dense + sparse via RRF)
# ──────────────────────────────────────────────────────────────────────

def example_hybrid() -> None:
    divider("4 · Hybrid Join (Text + Dense + Sparse → RRF)")

    from rustfuzz.join import MultiJoiner

    emb_products = _make_embeddings(PRODUCTS, seed=10)
    emb_listings = _make_embeddings(LISTINGS, seed=20)

    # Simple sparse vectors derived from text length (just a demo)
    sparse_products = [{i: 1.0} for i in range(len(PRODUCTS))]
    sparse_listings = [{i: 1.0} for i in range(len(LISTINGS))]

    joiner = (
        MultiJoiner(text_weight=0.5, sparse_weight=0.3, dense_weight=0.5)
        .add_array("products", texts=PRODUCTS, dense=emb_products, sparse=sparse_products)
        .add_array("listings", texts=LISTINGS, dense=emb_listings, sparse=sparse_listings)
    )

    results = joiner.join(n=1)
    print("  Hybrid join — all three channels fused via RRF:\n")
    for r in results:
        if r["src_array"] == "products":
            print(
                f"    {r['src_text']:35s} → {r['tgt_text']:40s}\n"
                f"      combined={r['score']:.4f}  "
                f"text={r['text_score']:.4f}  "
                f"sparse={r['sparse_score']:.4f}  "
                f"dense={r['dense_score']:.4f}"
            )


# ──────────────────────────────────────────────────────────────────────
# 5. join() vs join_pair() vs join_wide()
# ──────────────────────────────────────────────────────────────────────

def example_join_modes() -> None:
    divider("5 · Join Modes: join() vs join_pair() vs join_wide()")

    from rustfuzz.join import MultiJoiner

    joiner = (
        MultiJoiner()
        .add_array("products", texts=PRODUCTS)
        .add_array("listings", texts=LISTINGS)
        .add_array("erp", texts=ERP)
    )

    # --- join(): all directed pairs ---
    all_rows = joiner.join(n=1)
    pairs_seen = {(r["src_array"], r["tgt_array"]) for r in all_rows}
    print(f"  join(n=1): {len(all_rows)} rows across {len(pairs_seen)} directed pairs:")
    for pair in sorted(pairs_seen):
        count = sum(1 for r in all_rows if (r["src_array"], r["tgt_array"]) == pair)
        print(f"    {pair[0]:10s} → {pair[1]:10s}  ({count} rows)")

    # --- join_pair(): single directed pair ---
    print("\n  join_pair('products', 'erp', n=1):")
    pair_rows = joiner.join_pair("products", "erp", n=1)
    for r in pair_rows:
        print(
            f"    {r['src_text']:35s} → {r['tgt_text']:30s}  "
            f"score={r['score']:.4f}"
        )

    # --- join_wide(): pivoted view ---
    print("\n  join_wide('products', n=1) — one row per product, columns per target:")
    wide = joiner.join_wide("products", n=1)
    for r in wide:
        print(
            f"    {r['src_text']:35s}\n"
            f"      → listings: {str(r.get('match_listings', '—')):40s}  "
            f"score={r.get('score_listings', 0):.4f}\n"
            f"      → erp:      {str(r.get('match_erp', '—')):40s}  "
            f"score={r.get('score_erp', 0):.4f}"
        )


# ──────────────────────────────────────────────────────────────────────
# 6. Inner join with score_cutoff
# ──────────────────────────────────────────────────────────────────────

def example_inner_join() -> None:
    divider("6 · Inner Join with score_cutoff")

    from rustfuzz.join import MultiJoiner

    src = ["Apple iPhone 15", "Samsung Galaxy", "totally unrelated item"]
    tgt = ["Apple iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra"]

    joiner = (
        MultiJoiner()
        .add_array("src", texts=src)
        .add_array("tgt", texts=tgt)
    )

    # Full join — every src gets a match (even bad ones)
    full = joiner.join(n=1, how="full")
    print("  how='full' (default) — every src appears:")
    for r in full:
        if r["src_array"] == "src":
            print(f"    {r['src_text']:30s} → {str(r['tgt_text']):30s}  score={r['score']:.4f}")

    # Inner join — only rows above the cutoff
    inner = joiner.join(n=1, how="inner", score_cutoff=0.005)
    print(f"\n  how='inner', score_cutoff=0.005 — {len([r for r in inner if r['src_array'] == 'src'])} rows survive:")
    for r in inner:
        if r["src_array"] == "src":
            print(f"    {r['src_text']:30s} → {str(r['tgt_text']):30s}  score={r['score']:.4f}")


# ──────────────────────────────────────────────────────────────────────
# 7. Top-N matches (multiple candidates per source)
# ──────────────────────────────────────────────────────────────────────

def example_top_n() -> None:
    divider("7 · Top-N Matches (n=3)")

    from rustfuzz.join import MultiJoiner

    queries = ["Apple phone", "Samsung tablet"]
    catalogue = [
        "Apple iPhone 15 Pro Max",
        "Apple iPhone 14",
        "Apple iPad Air M2",
        "Samsung Galaxy S24 Ultra",
        "Samsung Galaxy Tab S9",
        "Samsung Galaxy Buds",
        "Google Pixel 8 Pro",
    ]

    joiner = (
        MultiJoiner()
        .add_array("queries", texts=queries)
        .add_array("catalogue", texts=catalogue)
    )

    results = joiner.join(n=3)
    for q in queries:
        print(f'  query: "{q}"')
        q_rows = sorted(
            [r for r in results if r["src_text"] == q and r["src_array"] == "queries"],
            key=lambda r: r["score"],
            reverse=True,
        )
        for r in q_rows:
            print(f"    → {r['tgt_text']:35s}  score={r['score']:.4f}")
        print()


# ──────────────────────────────────────────────────────────────────────
# 8. fuzzy_join() convenience function
# ──────────────────────────────────────────────────────────────────────

def example_fuzzy_join() -> None:
    divider("8 · fuzzy_join() Convenience Function")

    from rustfuzz.join import fuzzy_join

    results = fuzzy_join(
        arrays={
            "crm": ["Apple Inc.", "Microsoft Corp.", "Alphabet Inc."],
            "invoices": ["Apple Incorporated", "Microsft Corporation", "Google LLC"],
        },
        n=1,
        how="full",
    )

    print("  One-liner fuzzy join across two arrays:\n")
    for r in results:
        if r["src_array"] == "crm":
            print(
                f"    {r['src_text']:25s} → {r['tgt_text']:25s}  "
                f"score={r['score']:.4f}"
            )


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_text_only()
    example_dense_only()
    example_sparse_only()
    example_hybrid()
    example_join_modes()
    example_inner_join()
    example_top_n()
    example_fuzzy_join()

    print("\n✅  All MultiJoiner examples completed!\n")
