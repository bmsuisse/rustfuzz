"""
EmbedAnything + rustfuzz — Real-world embedding examples.

EmbedAnything is a high-performance Rust-native embedding library by
StarlightSearch. No PyTorch, uses Candle/ONNX backends — fast local
embeddings with minimal memory footprint.

Install:  uv add embed-anything
Run:      uv run python examples/embed_anything_examples.py
"""

from __future__ import annotations


def divider(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────────────
# 1. HybridSearch with dense embeddings (EmbeddingModel)
# ──────────────────────────────────────────────────────────────────────

def example_hybrid_search_dense() -> None:
    divider("1 · HybridSearch + EmbedAnything Dense Embeddings")

    import embed_anything
    from embed_anything import EmbeddingModel

    from rustfuzz.search import Document, HybridSearch

    # Lightweight Rust-native model — no PyTorch, Candle backend
    model = EmbeddingModel.from_pretrained_hf(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )

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
    embed_data = embed_anything.embed_query(texts, embedder=model)
    embeddings = [item.embedding for item in embed_data]

    print(f"  Model: all-MiniLM-L6-v2 via Candle (dim={len(embeddings[0])})")
    print(f"  Corpus: {len(docs)} documents\n")

    # Build 3-way hybrid index
    hs = HybridSearch(docs, embeddings=embeddings)

    # --- Query 1: exact keyword + semantic ---
    query = "apple phone"
    query_emb = embed_anything.embed_query([query], embedder=model)[0].embedding

    print(f'  query: "{query}" (3-way: BM25 + fuzzy + dense)')
    for text, score, meta in hs.search(query, query_embedding=query_emb, n=3):
        print(f"    [{score:.6f}] [{meta['category']:6s}] {text}")

    # --- Query 2: typos — fuzzy channel shines ---
    query = "samung galxy"
    query_emb = embed_anything.embed_query([query], embedder=model)[0].embedding

    print(f'\n  query: "{query}" (typos — fuzzy helps)')
    for text, score, meta in hs.search(query, query_embedding=query_emb, n=3):
        print(f"    [{score:.6f}] [{meta['category']:6s}] {text}")

    # --- Query 3: semantic — dense channel shines ---
    query = "wireless noise cancelling headset"
    query_emb = embed_anything.embed_query([query], embedder=model)[0].embedding

    print(f'\n  query: "{query}" (semantic — dense helps)')
    for text, score, meta in hs.search(query, query_embedding=query_emb, n=3):
        print(f"    [{score:.6f}] [{meta['category']:6s}] {text}")

    # --- Query 4: without embedding — falls back to BM25 + fuzzy ---
    query = "macbok pro"
    print(f'\n  query: "{query}" (no embedding → BM25 + fuzzy fallback)')
    for text, score, meta in hs.search(query, n=3):
        print(f"    [{score:.6f}] [{meta['category']:6s}] {text}")


# ──────────────────────────────────────────────────────────────────────
# 2. HybridSearch with embedding callback  (auto query embedding)
# ──────────────────────────────────────────────────────────────────────

def example_hybrid_callback() -> None:
    divider("2 · HybridSearch + EmbedAnything Callback (auto-embed)")

    import embed_anything
    from embed_anything import EmbeddingModel

    from rustfuzz.search import HybridSearch

    model = EmbeddingModel.from_pretrained_hf(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Define a callback that maps texts → dense vectors
    def embed_fn(texts: list[str]) -> list[list[float]]:
        results = embed_anything.embed_query(texts, embedder=model)
        return [r.embedding for r in results]

    corpus = [
        "Python is a popular programming language",
        "Rust provides memory safety without garbage collection",
        "JavaScript runs in the browser and on the server",
        "Machine learning models require large datasets",
        "Database indexing improves query performance",
        "Cloud computing enables scalable infrastructure",
        "TypeScript adds static typing to JavaScript",
        "Docker containers simplify application deployment",
    ]

    # Callback path: embeddings are generated automatically at init AND search()
    hs = HybridSearch(corpus, embeddings=embed_fn)
    print(f"  has_vectors: {hs.has_vectors}")
    print(f"  corpus size: {hs.num_docs}\n")

    # No query_embedding needed — the callback does it for us!
    for query in ["fast programming language", "containerised deployment", "neural network"]:
        print(f'  query: "{query}" (auto-embedded via callback)')
        for text, score in hs.search(query, n=2):
            print(f"    [{score:.6f}] {text}")
        print()


# ──────────────────────────────────────────────────────────────────────
# 3. MultiJoiner with EmbedAnything dense embeddings
# ──────────────────────────────────────────────────────────────────────

def example_multijoiner() -> None:
    divider("3 · MultiJoiner + EmbedAnything Dense Embeddings")

    import embed_anything
    from embed_anything import EmbeddingModel

    from rustfuzz.join import MultiJoiner

    model = EmbeddingModel.from_pretrained_hf(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )

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

    # Embed both arrays via EmbedAnything
    crm_data = embed_anything.embed_query(crm_names, embedder=model)
    inv_data = embed_anything.embed_query(invoice_names, embedder=model)
    emb_crm = [item.embedding for item in crm_data]
    emb_inv = [item.embedding for item in inv_data]

    print(f"  Model: all-MiniLM-L6-v2 (dim={len(emb_crm[0])})")
    print(f"  CRM names: {len(crm_names)}, Invoice names: {len(invoice_names)}\n")

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
# 4. Sparse Embeddings (SPLADE) with MultiJoiner
# ──────────────────────────────────────────────────────────────────────

def example_sparse_embeddings() -> None:
    divider("4 · MultiJoiner + EmbedAnything SPLADE (Sparse)")

    import embed_anything
    from embed_anything import EmbeddingModel

    from rustfuzz.join import MultiJoiner

    # SPLADE model for sparse keyword-weighted embeddings
    sparse_model = EmbeddingModel.from_pretrained_hf(
        model_id="prithivida/Splade_PP_en_v1",
    )

    # Dense model for semantic similarity
    dense_model = EmbeddingModel.from_pretrained_hf(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )

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

    # Dense embeddings
    emb_products = [d.embedding for d in embed_anything.embed_query(products, embedder=dense_model)]
    emb_queries = [d.embedding for d in embed_anything.embed_query(queries, embedder=dense_model)]

    # Sparse embeddings (SPLADE) — convert to dict[int, float]
    sparse_products_raw = embed_anything.embed_query(products, embedder=sparse_model)
    sparse_queries_raw = embed_anything.embed_query(queries, embedder=sparse_model)

    sparse_products = [
        {i: v for i, v in enumerate(item.embedding) if v != 0.0}
        for item in sparse_products_raw
    ]
    sparse_queries = [
        {i: v for i, v in enumerate(item.embedding) if v != 0.0}
        for item in sparse_queries_raw
    ]

    print(f"  Dense model:  all-MiniLM-L6-v2 (dim={len(emb_products[0])})")
    print(f"  Sparse model: Splade_PP_en_v1")
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
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import embed_anything  # noqa: F401
    except ImportError:
        print("❌  embed-anything not installed. Install with: uv add embed-anything")
        print("   Then re-run: uv run python examples/embed_anything_examples.py")
        raise SystemExit(1)

    example_hybrid_search_dense()
    example_hybrid_callback()
    example_multijoiner()

    # SPLADE may not work with all models — run separately
    try:
        example_sparse_embeddings()
    except Exception as e:
        print(f"\n  ⚠️  Sparse example skipped: {e}")

    print("\n✅  All EmbedAnything examples completed!\n")
