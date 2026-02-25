"""
rustfuzz — Quick-start examples with dummy data.

Run:  uv run python examples/quickstart.py
"""

from __future__ import annotations


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────────────────────
# 1. Basic fuzzy scoring  (rustfuzz.fuzz)
# ──────────────────────────────────────────────────────────────

def example_fuzz() -> None:
    divider("1 · Fuzzy Scoring (rustfuzz.fuzz)")

    from rustfuzz import fuzz

    pairs = [
        ("Apple Inc.", "apple inc"),
        ("New York Mets", "New York Meats"),
        ("Los Angeles Lakers", "LA Lakers"),
        ("Microsoft Corporation", "Microsft Corp"),
        ("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"),
    ]

    for s1, s2 in pairs:
        print(f'  ratio("{s1}", "{s2}")                = {fuzz.ratio(s1, s2):.1f}')
        print(f'  partial_ratio                        = {fuzz.partial_ratio(s1, s2):.1f}')
        print(f'  token_sort_ratio                     = {fuzz.token_sort_ratio(s1, s2):.1f}')
        print(f'  token_set_ratio                      = {fuzz.token_set_ratio(s1, s2):.1f}')
        print(f'  WRatio (weighted)                    = {fuzz.WRatio(s1, s2):.1f}')
        print()


# ──────────────────────────────────────────────────────────────
# 2. Batch extraction  (rustfuzz.process)
# ──────────────────────────────────────────────────────────────

def example_process() -> None:
    divider("2 · Batch Extraction (rustfuzz.process)")

    from rustfuzz import fuzz, process

    products = [
        "Apple iPhone 15 Pro Max",
        "Samsung Galaxy S24 Ultra",
        "Google Pixel 8 Pro",
        "OnePlus 12",
        "Apple iPad Air M2",
        "Samsung Galaxy Tab S9",
        "Apple MacBook Pro M3",
        "Dell XPS 15",
        "Lenovo ThinkPad X1 Carbon",
        "Apple AirPods Pro 2",
    ]

    query = "aple iphone"

    # Top 3 matches (default scorer = WRatio)
    print(f'  query: "{query}"')
    print("  --- extract (top 3) ---")
    results = process.extract(query, products, limit=3)
    for match, score, idx in results:
        print(f"    [{idx}] {match:35s}  score={score:.1f}")

    # Best single match
    best = process.extractOne(query, products)
    if best:
        match, score, idx = best
        print(f"\n  extractOne → [{idx}] {match}  score={score:.1f}")

    # With a different scorer
    print(f'\n  --- extract with token_set_ratio, score_cutoff=60 ---')
    results = process.extract(
        query, products, scorer=fuzz.token_set_ratio, limit=5, score_cutoff=60.0
    )
    for match, score, idx in results:
        print(f"    [{idx}] {match:35s}  score={score:.1f}")


# ──────────────────────────────────────────────────────────────
# 3. Pairwise distance matrix  (process.cdist)
# ──────────────────────────────────────────────────────────────

def example_cdist() -> None:
    divider("3 · Pairwise Distance Matrix (process.cdist)")

    from rustfuzz import process

    cities_a = ["New York", "Los Angeles", "Chicago", "Houston"]
    cities_b = ["New Yrok", "LA", "Chigaco", "Housten", "Dallas"]

    matrix = process.cdist(cities_a, cities_b)
    print("  Similarity matrix (rows=queries, cols=choices):\n")
    header = "".ljust(16) + "".join(c.center(10) for c in cities_b)
    print(f"  {header}")
    print(f"  {'─' * len(header)}")
    for i, city in enumerate(cities_a):
        row = city.ljust(16) + "".join(f"{matrix[i][j]:.1f}".center(10) for j in range(len(cities_b)))
        print(f"  {row}")


# ──────────────────────────────────────────────────────────────
# 4. Deduplication  (process.dedupe)
# ──────────────────────────────────────────────────────────────

def example_dedupe() -> None:
    divider("4 · Deduplication (process.dedupe)")

    from rustfuzz import process

    names = [
        "John Smith",
        "Jon Smith",
        "Jane Doe",
        "John Smyth",
        "Janet Doe",
        "Alice Wonderland",
        "Bob Builder",
        "Bobby Builder",
        "Alyce Wonderland",
    ]

    deduped = process.dedupe(names, threshold=2)
    print(f"  Input  ({len(names)} items): {names}")
    print(f"  Output ({len(deduped)} items): {deduped}")


# ──────────────────────────────────────────────────────────────
# 5. String distance metrics  (rustfuzz.distance)
# ──────────────────────────────────────────────────────────────

def example_distance() -> None:
    divider("5 · Distance Metrics (rustfuzz.distance)")

    from rustfuzz.distance import (
        DamerauLevenshtein,
        Hamming,
        Indel,
        Jaro,
        JaroWinkler,
        LCSseq,
        Levenshtein,
    )

    s1, s2 = "kitten", "sitting"

    print(f'  Comparing: "{s1}" vs "{s2}"\n')
    print(f"  Levenshtein distance        = {Levenshtein.distance(s1, s2)}")
    print(f"  Levenshtein similarity       = {Levenshtein.similarity(s1, s2)}")
    print(f"  Levenshtein norm. similarity = {Levenshtein.normalized_similarity(s1, s2):.4f}")
    print(f"  DamerauLevenshtein distance  = {DamerauLevenshtein.distance(s1, s2)}")
    print(f"  Hamming distance             = {Hamming.distance(s1, s2)}")
    print(f"  Indel distance               = {Indel.distance(s1, s2)}")
    print(f"  Jaro similarity              = {Jaro.similarity(s1, s2):.4f}")
    print(f"  Jaro-Winkler similarity      = {JaroWinkler.similarity(s1, s2):.4f}")
    print(f"  LCSseq similarity            = {LCSseq.similarity(s1, s2)}")

    # Edit operations
    print(f"\n  Levenshtein editops:")
    for op in Levenshtein.editops(s1, s2):
        print(f"    {op}")


# ──────────────────────────────────────────────────────────────
# 6. BM25 full-text search  (rustfuzz.search)
# ──────────────────────────────────────────────────────────────

def example_bm25() -> None:
    divider("6 · BM25 Full-Text Search (rustfuzz.search)")

    from rustfuzz.search import BM25

    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast red fox leaps across the sleepy hound",
        "The lazy dog slept in the sun all afternoon",
        "Quick brown foxes are surprisingly fast animals",
        "The dog chased the cat around the garden",
        "Machine learning and artificial intelligence news",
        "Python programming for data science beginners",
        "Rust programming language for fast systems",
        "Natural language processing with transformers",
        "Deep learning frameworks comparison guide",
    ]

    bm25 = BM25(documents)
    print(f"  Indexed {bm25.num_docs} documents\n")

    for query in ["quick fox", "programming language", "deep lerning"]:
        print(f'  query: "{query}"')
        for doc, score in bm25.get_top_n(query, n=3):
            print(f"    score={score:.4f}  {doc[:60]}")
        print()

    # Fuzzy search (handles typos!)
    query = "qiuck brwon fxo"
    print(f'  --- Fuzzy search (typos): "{query}" ---')
    for doc, score in bm25.get_top_n_fuzzy(query, n=3, fuzzy_weight=0.4):
        print(f"    score={score:.4f}  {doc[:60]}")


# ──────────────────────────────────────────────────────────────
# 6b. BM25 variants comparison  (BM25L, BM25Plus, BM25T)
# ──────────────────────────────────────────────────────────────

def example_bm25_variants() -> None:
    divider("6b · BM25 Variants — Consistent API")

    from rustfuzz.search import BM25, BM25L, BM25Plus, BM25T

    corpus = [
        "the cat sat on the mat",
        "the cat sat on the mat the cat sat on the mat",
        "a dog barked loudly in the park",
        "the quick brown fox jumps over the lazy dog",
        "machine learning for natural language understanding",
        "deep neural networks and transformers",
    ]

    variants: list[tuple[str, BM25 | BM25L | BM25Plus | BM25T]] = [
        ("BM25 Okapi", BM25(corpus)),
        ("BM25L",      BM25L(corpus, delta=0.5)),
        ("BM25Plus",   BM25Plus(corpus, delta=1.0)),
        ("BM25T",      BM25T(corpus)),
    ]

    # All variants now share the same API
    query = "cat mat"
    print(f'  query: "{query}" — get_top_n (pure BM25)')
    print(f"  {'Variant':<14s}  {'#1 Doc':42s}  Score")
    print(f"  {'─' * 66}")
    for name, index in variants:
        top = index.get_top_n(query, n=1)
        if top:
            doc, score = top[0]
            print(f"  {name:<14s}  {doc[:42]:<42s}  {score:.4f}")

    # get_batch_scores — available on ALL variants now
    print(f"\n  get_batch_scores(['cat mat', 'neural']):")
    for name, index in variants:
        batch = index.get_batch_scores(["cat mat", "neural"])
        non_zero = [i for i, s in enumerate(batch[1]) if s > 0]
        print(f"  {name:<14s}  neural hits docs: {non_zero}")

    # Length normalisation comparison
    print(f"\n  --- Length normalisation (query='cat mat') ---")
    for name, index in variants:
        scores = index.get_scores(query)
        short, long = scores[0], scores[1]
        ratio = f"{long / short:.2f}x" if short > 0 else "n/a"
        print(f"  {name:<14s}  short={short:.4f}  long(2x)={long:.4f}  ratio={ratio}")


# ──────────────────────────────────────────────────────────────
# 6c. BM25 + Fuzzy  (typo-tolerant search)
# ──────────────────────────────────────────────────────────────

def example_bm25_fuzzy() -> None:
    divider("6c · BM25 + Fuzzy (typo-tolerant search)")

    from rustfuzz.search import BM25, BM25L, BM25Plus, BM25T

    corpus = [
        "Apple iPhone 15 Pro Max 256GB",
        "Samsung Galaxy S24 Ultra",
        "Google Pixel 8 Pro",
        "Apple MacBook Pro M3 Max",
        "Dell XPS 15 Laptop",
        "Lenovo ThinkPad X1 Carbon Gen 11",
        "Sony WH-1000XM5 Headphones",
        "Bose QuietComfort Ultra Earbuds",
    ]

    queries = [
        ("appel iphon", "misspelled brand + product"),
        ("makbok pro",  "phonetic misspelling"),
        ("thinkpad crbon", "missing letters"),
    ]

    # Show all three search modes side by side
    bm25 = BM25(corpus)

    for query, desc in queries:
        print(f'  query: "{query}" ({desc})')

        # Pure BM25 (keyword only)
        top_bm25 = bm25.get_top_n(query, n=1)
        bm25_result = top_bm25[0] if top_bm25 else ("(no match)", 0.0)

        # BM25 + Fuzzy (weighted hybrid)
        top_fuzzy = bm25.get_top_n_fuzzy(query, n=1, fuzzy_weight=0.4)
        fuzzy_result = top_fuzzy[0] if top_fuzzy else ("(no match)", 0.0)

        # BM25 + Fuzzy via RRF (rank fusion)
        top_rrf = bm25.get_top_n_rrf(query, n=1)
        rrf_result = top_rrf[0] if top_rrf else ("(no match)", 0.0)

        print(f"    BM25 only:    {bm25_result[0]:40s}  score={bm25_result[1]:.4f}")
        print(f"    BM25 + fuzzy: {fuzzy_result[0]:40s}  score={fuzzy_result[1]:.4f}")
        print(f"    BM25 + RRF:   {rrf_result[0]:40s}  score={rrf_result[1]:.6f}")
        print()

    # Demonstrate that ALL variants now support fuzzy search
    print("  --- All variants support get_top_n_fuzzy('appel iphon') ---")
    for name, cls, kw in [
        ("BM25 Okapi", BM25, {}),
        ("BM25L",      BM25L, {"delta": 0.5}),
        ("BM25Plus",   BM25Plus, {"delta": 1.0}),
        ("BM25T",      BM25T, {}),
    ]:
        idx = cls(corpus, **kw)
        top = idx.get_top_n_fuzzy("appel iphon", n=1, fuzzy_weight=0.4)
        doc, score = top[0] if top else ("(no match)", 0.0)
        print(f"    {name:<14s} → {doc:40s}  score={score:.4f}")


# ──────────────────────────────────────────────────────────────
# 6d. Hybrid Search  (BM25 + dense vectors via RRF)
# ──────────────────────────────────────────────────────────────

def example_hybrid_search() -> None:
    divider("6d · Hybrid Search (BM25 + Embeddings via RRF)")

    import random

    from rustfuzz.search import HybridSearch

    documents = [
        "Python is a popular programming language",
        "Rust provides memory safety without garbage collection",
        "JavaScript runs in the browser and on the server",
        "Machine learning models require large datasets",
        "Database indexing improves query performance",
        "Cloud computing enables scalable infrastructure",
        "TypeScript adds static typing to JavaScript",
        "Docker containers simplify application deployment",
    ]

    # Create dummy embeddings (in practice, use a real model like sentence-transformers)
    random.seed(42)
    dim = 32
    embeddings = [[random.gauss(0, 1) for _ in range(dim)] for _ in documents]

    hs = HybridSearch(documents, embeddings=embeddings)
    print(f"  has_vectors: {hs.has_vectors}")
    print(f"  corpus size: {len(documents)}\n")

    # Search WITH a query embedding (full hybrid: BM25 + cosine → RRF)
    query = "programming language"
    query_emb = [random.gauss(0, 1) for _ in range(dim)]

    print(f'  query: "{query}" (with embedding → hybrid RRF)')
    for doc, score in hs.search(query, query_embedding=query_emb, n=3):
        print(f"    rrf={score:.6f}  {doc}")

    # Search WITHOUT a query embedding (falls back to BM25 + fuzzy RRF)
    print(f'\n  query: "{query}" (no embedding → BM25 + fuzzy fallback)')
    for doc, score in hs.search(query, n=3):
        print(f"    rrf={score:.6f}  {doc}")

    # Typo-heavy query — hybrid search should still surface relevant docs
    query_typo = "progamming languege"
    print(f'\n  query: "{query_typo}" (typos, no embedding)')
    for doc, score in hs.search(query_typo, n=3):
        print(f"    rrf={score:.6f}  {doc}")


# ──────────────────────────────────────────────────────────────
# 7. Fuzzy join across arrays  (rustfuzz.join)
# ──────────────────────────────────────────────────────────────

def example_join() -> None:
    divider("7 · Fuzzy Join (rustfuzz.join)")

    from rustfuzz.join import MultiJoiner, fuzzy_join

    # Scenario: match company names from two different data sources
    crm_names = [
        "Apple Inc.",
        "Microsoft Corporation",
        "Alphabet Inc.",
        "Amazon.com Inc.",
        "Tesla Motors",
    ]

    invoice_names = [
        "Apple Incorporated",
        "Microsft Corp",
        "Google LLC",
        "Amazon Inc",
        "Tesla Inc",
    ]

    erp_names = [
        "APPLE INC",
        "MICROSOFT CORP",
        "ALPHABET / GOOGLE",
        "AMAZON.COM",
        "TESLA",
    ]

    # Simple convenience function
    print("  --- fuzzy_join (convenience) ---")
    results = fuzzy_join(
        {"crm": crm_names, "invoices": invoice_names},
        n=1,
        how="full",
    )
    for row in results[:5]:
        print(f"    {row['src_text']:30s} → {row['tgt_text']:30s}  score={row['score']:.2f}")

    # MultiJoiner with three arrays
    print("\n  --- MultiJoiner (3-way, wide pivot) ---")
    joiner = (
        MultiJoiner()
        .add_array("crm", texts=crm_names)
        .add_array("invoices", texts=invoice_names)
        .add_array("erp", texts=erp_names)
    )
    for row in joiner.join_wide("crm", n=1):
        crm = row["src_text"]
        inv = row.get("match_invoices", "—")
        erp = row.get("match_erp", "—")
        s_inv = row.get("score_invoices")
        s_erp = row.get("score_erp")
        print(
            f"    {crm:25s} │ inv: {str(inv):25s} ({s_inv:.2f})"
            f"  │ erp: {str(erp):25s} ({s_erp:.2f})"
        )


# ──────────────────────────────────────────────────────────────
# 8. score_cutoff and processor
# ──────────────────────────────────────────────────────────────

def example_advanced() -> None:
    divider("8 · Advanced: score_cutoff & custom processor")

    from rustfuzz import fuzz

    # score_cutoff: returns 0.0 if score is below the cutoff
    s1, s2 = "hello world", "helo wrld"
    score = fuzz.ratio(s1, s2, score_cutoff=90.0)
    print(f'  ratio("{s1}", "{s2}", score_cutoff=90) = {score:.1f}  (below cutoff → 0)')

    score = fuzz.ratio(s1, s2, score_cutoff=70.0)
    print(f'  ratio("{s1}", "{s2}", score_cutoff=70) = {score:.1f}  (above cutoff → actual)')

    # Using a custom processor
    def strip_ltd(s: str) -> str:
        return s.replace(" Ltd", "").replace(" GmbH", "").strip().lower()

    pairs = [
        ("Acme Ltd", "acme"),
        ("Widgets GmbH", "widgets"),
        ("FooBar Ltd", "foobar"),
    ]
    print("\n  Custom processor (strip Ltd/GmbH, lowercase):")
    for a, b in pairs:
        score = fuzz.ratio(a, b, processor=strip_ltd)
        print(f'    ratio("{a}", "{b}") → {score:.1f}')


# ──────────────────────────────────────────────────────────────
# 9. Data framework integration  (Polars, Pandas, PyArrow)
# ──────────────────────────────────────────────────────────────

def example_data_frameworks() -> None:
    divider("9 · Data Framework Integration (Polars, Pandas, PyArrow)")

    from rustfuzz.search import BM25
    from rustfuzz.join import MultiJoiner

    products = [
        "Apple iPhone 15 Pro Max",
        "Samsung Galaxy S24 Ultra",
        "Google Pixel 8 Pro",
        "OnePlus 12",
        "Dell XPS 15 Laptop",
        "Lenovo ThinkPad X1 Carbon",
        "Apple MacBook Pro M3",
        "Sony WH-1000XM5 Headphones",
    ]

    # ── Polars ──
    try:
        import polars as pl

        df = pl.DataFrame({"name": products, "id": list(range(len(products)))})

        # Direct from Polars Series — no .to_list() needed!
        bm25 = BM25(df["name"])
        print(f"  Polars Series → BM25 ({bm25.num_docs} docs)")
        for doc, score in bm25.get_top_n("apple iphone", n=2):
            print(f"    score={score:.4f}  {doc}")

        # Or use from_column classmethod
        bm25_col = BM25.from_column(df, "name")
        print(f"\n  BM25.from_column(df, 'name') → {bm25_col.num_docs} docs")

        # MultiJoiner also accepts Polars Series
        joiner = MultiJoiner()
        joiner.add_array("products", texts=df["name"])
        joiner.add_array("queries", texts=pl.Series(["appel iphon", "samung galxy"]))
        results = joiner.join(n=1)
        print(f"\n  MultiJoiner with Polars: {len(results)} matches")
        for r in results[:2]:
            print(f"    {r['src_text']:30s} → {r['tgt_text']:30s}  score={r['score']:.2f}")
    except ImportError:
        print("  [skipped — polars not installed]")

    # ── Pandas ──
    try:
        import pandas as pd

        pdf = pd.DataFrame({"name": products})
        bm25 = BM25(pdf["name"])
        print(f"\n  Pandas Series → BM25 ({bm25.num_docs} docs)")
        for doc, score in bm25.get_top_n("dell laptop", n=2):
            print(f"    score={score:.4f}  {doc}")
    except ImportError:
        print("  [skipped — pandas not installed]")

    # ── PyArrow ──
    try:
        import pyarrow as pa

        arr = pa.array(products)
        bm25 = BM25(arr)
        print(f"\n  PyArrow Array → BM25 ({bm25.num_docs} docs)")
        for doc, score in bm25.get_top_n("sony headphones", n=2):
            print(f"    score={score:.4f}  {doc}")
    except ImportError:
        print("  [skipped — pyarrow not installed]")

    print("\n  ✨  All data frameworks feed directly into BM25 — no manual conversion!")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_fuzz()
    example_process()
    example_cdist()
    example_dedupe()
    example_distance()
    example_bm25()
    example_bm25_variants()
    example_bm25_fuzzy()
    example_hybrid_search()
    example_join()
    example_advanced()
    example_data_frameworks()

    print("\n✅  All examples completed!\n")
