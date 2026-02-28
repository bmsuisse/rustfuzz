"""
FastEmbed + rustfuzz — Real-world embedding examples.

FastEmbed is a lightweight, ONNX-based embedding library by Qdrant.
No PyTorch, no GPU required — just fast local embeddings.

Install:  uv add fastembed
Run:      uv run python examples/fastembed_examples.py
"""

from __future__ import annotations


def divider(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────────────
# 1. HybridSearch with dense embeddings  (TextEmbedding)
# ──────────────────────────────────────────────────────────────────────

def example_hybrid_search_dense() -> None:
    divider("1 · HybridSearch + FastEmbed Dense Embeddings")

    from fastembed import TextEmbedding
    from rustfuzz.search import Document, HybridSearch

    # Lightweight ONNX model — ~33 MB, downloads once, runs on CPU
    model = TextEmbedding("BAAI/bge-small-en-v1.5")

    # A small product catalogue
    docs = [
        Document("Apple iPhone 15 Pro Max 256GB",    {"brand": "Apple",   "category": "phone"}),
        Document("Samsung Galaxy S24 Ultra",          {"brand": "Samsung", "category": "phone"}),
        Document("Google Pixel 8 Pro",                {"brand": "Google",  "category": "phone"}),
        Document("Apple MacBook Pro M3 Max",          {"brand": "Apple",   "category": "laptop"}),
        Document("Dell XPS 15 Laptop",                {"brand": "Dell",    "category": "laptop"}),
        Document("Sony WH-1000XM5 Headphones",        {"brand": "Sony",    "category": "audio"}),
        Document("Bose QuietComfort Ultra Earbuds",    {"brand": "Bose",    "category": "audio"}),
        Document("Apple iPad Air M2",                  {"brand": "Apple",   "category": "tablet"}),
    ]

    # Embed all documents
    texts = [d.content for d in docs]
    embeddings = [e.tolist() for e in model.embed(texts)]

    print(f"  Model: BAAI/bge-small-en-v1.5 (dim={len(embeddings[0])})")
    print(f"  Corpus: {len(docs)} documents\n")

    # Build 3-way hybrid index
    hs = HybridSearch(docs, embeddings=embeddings)

    # --- Query 1: exact keyword + semantic ---
    query = "apple phone"
    query_emb = list(model.embed([query]))[0].tolist()

    print(f'  query: "{query}" (3-way: BM25 + fuzzy + dense)')
    for text, score, meta in hs.search(query, query_embedding=query_emb, n=3):
        print(f"    [{score:.6f}] [{meta['category']:6s}] {text}")

    # --- Query 2: typos — fuzzy channel shines ---
    query = "samung galxy"
    query_emb = list(model.embed([query]))[0].tolist()

    print(f'\n  query: "{query}" (typos — fuzzy helps)')
    for text, score, meta in hs.search(query, query_embedding=query_emb, n=3):
        print(f"    [{score:.6f}] [{meta['category']:6s}] {text}")

    # --- Query 3: semantic — dense channel shines ---
    query = "wireless noise cancelling headset"
    query_emb = list(model.embed([query]))[0].tolist()

    print(f'\n  query: "{query}" (semantic — dense helps)')
    for text, score, meta in hs.search(query, query_embedding=query_emb, n=3):
        print(f"    [{score:.6f}] [{meta['category']:6s}] {text}")

    # --- Query 4: without embedding — falls back to BM25 + fuzzy ---
    query = "macbok pro"
    print(f'\n  query: "{query}" (no embedding → BM25 + fuzzy fallback)')
    for text, score, meta in hs.search(query, n=3):
        print(f"    [{score:.6f}] [{meta['category']:6s}] {text}")


# ──────────────────────────────────────────────────────────────────────
# 2. MultiJoiner with dense + sparse embeddings
# ──────────────────────────────────────────────────────────────────────

def example_multijoiner_dense_sparse() -> None:
    divider("2 · MultiJoiner + Dense + Sparse (SPLADE)")

    from fastembed import SparseTextEmbedding, TextEmbedding
    from rustfuzz.join import MultiJoiner

    products = [
        "Apple iPhone 15 Pro Max",
        "Samsung Galaxy S24 Ultra",
        "Google Pixel 8 Pro",
        "Apple MacBook Pro M3 Max",
        "Sony WH-1000XM5 Headphones",
    ]

    queries = [
        "apple smartphone",               # semantic match
        "wireless noise cancelling headset",  # semantic — no keyword overlap
        "samung galxy",                    # typo-heavy
    ]

    # Dense embeddings (semantic meaning)
    dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    emb_products = [e.tolist() for e in dense_model.embed(products)]
    emb_queries = [e.tolist() for e in dense_model.embed(queries)]

    # Sparse embeddings (keyword importance via SPLADE)
    sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
    sparse_products_raw = list(sparse_model.embed(products))
    sparse_queries_raw = list(sparse_model.embed(queries))

    # Convert SparseEmbedding → dict[int, float] for rustfuzz
    sparse_products = [
        {int(idx): float(val) for idx, val in zip(emb.indices, emb.values)}
        for emb in sparse_products_raw
    ]
    sparse_queries = [
        {int(idx): float(val) for idx, val in zip(emb.indices, emb.values)}
        for emb in sparse_queries_raw
    ]

    print(f"  Dense model:  BAAI/bge-small-en-v1.5 (dim={len(emb_products[0])})")
    print(f"  Sparse model: prithivida/Splade_PP_en_v1")
    print(f"  Products: {len(products)}, Queries: {len(queries)}\n")

    # Hybrid join using all 3 channels: text + dense + sparse
    joiner = (
        MultiJoiner(text_weight=0.4, dense_weight=0.4, sparse_weight=0.3)
        .add_array("queries", texts=queries, dense=emb_queries, sparse=sparse_queries)
        .add_array("products", texts=products, dense=emb_products, sparse=sparse_products)
    )

    results = joiner.join(n=2)

    for q in queries:
        print(f'  query: "{q}"')
        q_rows = sorted(
            [r for r in results if r["src_text"] == q and r["src_array"] == "queries"],
            key=lambda r: r["score"],
            reverse=True,
        )
        for r in q_rows:
            print(
                f"    → {r['tgt_text']:35s}  "
                f"score={r['score']:.4f}  "
                f"text={r['text_score']:.4f}  "
                f"dense={r['dense_score']:.4f}  "
                f"sparse={r['sparse_score']:.4f}"
            )
        print()


# ──────────────────────────────────────────────────────────────────────
# 3. MultiJoiner with FastEmbed dense embeddings
# ──────────────────────────────────────────────────────────────────────

def example_multijoiner_fastembed() -> None:
    divider("3 · MultiJoiner + FastEmbed Dense Embeddings")

    from fastembed import TextEmbedding
    from rustfuzz.join import MultiJoiner

    model = TextEmbedding("BAAI/bge-small-en-v1.5")

    # Two messy data sources to reconcile
    crm_names = [
        "Apple Inc.",
        "Microsoft Corporation",
        "Alphabet Inc.",
        "Amazon.com Inc.",
        "Tesla Motors",
    ]

    invoice_names = [
        "Apple Incorporated",
        "Microsft Corp",         # typo
        "Google LLC",            # different name for Alphabet
        "Amazon Inc",
        "Tesla Inc",
    ]

    # Embed both arrays
    emb_crm = [e.tolist() for e in model.embed(crm_names)]
    emb_inv = [e.tolist() for e in model.embed(invoice_names)]

    # Hybrid join: text (BM25 + fuzzy) + dense (cosine similarity)
    joiner = (
        MultiJoiner(text_weight=0.5, dense_weight=0.5)
        .add_array("crm", texts=crm_names, dense=emb_crm)
        .add_array("invoices", texts=invoice_names, dense=emb_inv)
    )

    results = joiner.join(n=1)

    print("  Fuzzy join with real embeddings (text + dense):\n")
    for r in results:
        if r["src_array"] == "crm":
            print(
                f"    {r['src_text']:25s} → {r['tgt_text']:25s}  "
                f"score={r['score']:.4f}  "
                f"text={r['text_score']:.4f}  "
                f"dense={r['dense_score']:.4f}"
            )

    print("\n  ✨ 'Alphabet Inc.' ↔ 'Google LLC' matched via semantic similarity!")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import fastembed  # noqa: F401
    except ImportError:
        print("❌  fastembed not installed. Install with: uv add fastembed")
        print("   Then re-run: uv run python examples/fastembed_examples.py")
        raise SystemExit(1)

    example_hybrid_search_dense()
    example_multijoiner_dense_sparse()
    example_multijoiner_fastembed()

    print("\n✅  All FastEmbed examples completed!\n")
