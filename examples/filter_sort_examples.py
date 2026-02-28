"""
rustfuzz — Filter & Sort examples.

Demonstrates the Meilisearch-style filter expression language, the fluent
SearchQuery builder, and sort capabilities across BM25 and HybridSearch.

Run:  uv run python examples/filter_sort_examples.py
"""

from __future__ import annotations


def divider(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────────────
# Shared product catalogue
# ──────────────────────────────────────────────────────────────────────

PRODUCTS = [
    "Apple iPhone 15 Pro Max 256GB",
    "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro",
    "Apple MacBook Pro M3 Max",
    "Dell XPS 15 Laptop",
    "Sony WH-1000XM5 Headphones",
    "Bose QuietComfort Ultra Earbuds",
    "Apple iPad Air M2",
    "Samsung Galaxy Tab S9",
    "Lenovo ThinkPad X1 Carbon Gen 11",
    "Apple AirPods Pro 2",
    "Google Pixel Watch 2",
    "Microsoft Surface Pro 9",
    "OnePlus 12 5G",
    "Samsung Galaxy Buds3 Pro",
]

METADATA = [
    {"brand": "Apple",     "category": "phone",     "price": 1199, "rating": 4.8, "in_stock": True,  "year": 2024, "specs": {"weight_g": 221}},
    {"brand": "Samsung",   "category": "phone",     "price": 1299, "rating": 4.7, "in_stock": True,  "year": 2024, "specs": {"weight_g": 234}},
    {"brand": "Google",    "category": "phone",     "price":  899, "rating": 4.5, "in_stock": True,  "year": 2023, "specs": {"weight_g": 213}},
    {"brand": "Apple",     "category": "laptop",    "price": 3499, "rating": 4.9, "in_stock": False, "year": 2024, "specs": {"weight_g": 1610}},
    {"brand": "Dell",      "category": "laptop",    "price": 1499, "rating": 4.3, "in_stock": True,  "year": 2023, "specs": {"weight_g": 1860}},
    {"brand": "Sony",      "category": "audio",     "price":  349, "rating": 4.7, "in_stock": True,  "year": 2023, "specs": {"weight_g": 250}},
    {"brand": "Bose",      "category": "audio",     "price":  299, "rating": 4.6, "in_stock": False, "year": 2024, "specs": {"weight_g": 24}},
    {"brand": "Apple",     "category": "tablet",    "price":  799, "rating": 4.6, "in_stock": True,  "year": 2024, "specs": {"weight_g": 462}},
    {"brand": "Samsung",   "category": "tablet",    "price":  849, "rating": 4.4, "in_stock": True,  "year": 2023, "specs": {"weight_g": 498}},
    {"brand": "Lenovo",    "category": "laptop",    "price": 1649, "rating": 4.5, "in_stock": True,  "year": 2024, "specs": {"weight_g": 1120}},
    {"brand": "Apple",     "category": "audio",     "price":  249, "rating": 4.5, "in_stock": True,  "year": 2023, "specs": {"weight_g": 51}},
    {"brand": "Google",    "category": "wearable",  "price":  349, "rating": 4.2, "in_stock": True,  "year": 2023, "specs": {"weight_g": 36}},
    {"brand": "Microsoft", "category": "laptop",    "price": 1599, "rating": 4.4, "in_stock": False, "year": 2023, "specs": {"weight_g": 879}},
    {"brand": "OnePlus",   "category": "phone",     "price":  799, "rating": 4.3, "in_stock": True,  "year": 2024, "specs": {"weight_g": 220}},
    {"brand": "Samsung",   "category": "audio",     "price":  229, "rating": 4.3, "in_stock": True,  "year": 2024, "specs": {"weight_g": 27}},
]


# ──────────────────────────────────────────────────────────────────────
# 1. Low-level filter API  (parse → evaluate → apply)
# ──────────────────────────────────────────────────────────────────────

def example_filter_api() -> None:
    divider("1 · Low-Level Filter API (parse_filter / evaluate_filter / apply_filter)")

    from rustfuzz.filter import apply_filter, evaluate_filter, parse_filter

    # Parse a Meilisearch-style expression into an AST
    ast = parse_filter('brand = "Apple" AND price > 500')
    print(f"  Parsed AST type: {type(ast).__name__}")
    print()

    # Evaluate against individual metadata dicts
    test_items = [
        {"brand": "Apple",   "price": 1199},
        {"brand": "Apple",   "price": 249},
        {"brand": "Samsung", "price": 1299},
    ]
    for item in test_items:
        result = evaluate_filter(ast, item)
        print(f"  evaluate({item}) → {result}")

    # apply_filter on a results list  (text, score, metadata)
    print()
    results = [(name, 1.0, meta) for name, meta in zip(PRODUCTS, METADATA, strict=True)]
    filtered = apply_filter(results, 'brand = "Apple" AND price > 500')
    print(f"  apply_filter(brand='Apple' AND price>500): {len(filtered)} / {len(results)} docs")
    for name, _, meta in filtered:
        print(f"    ${meta['price']:,}  {name}")


# ──────────────────────────────────────────────────────────────────────
# 2. Filter expression showcase  (operators tour)
# ──────────────────────────────────────────────────────────────────────

def example_filter_expressions() -> None:
    divider("2 · Filter Expression Showcase")

    from rustfuzz.filter import apply_filter

    results = [(name, 1.0, meta) for name, meta in zip(PRODUCTS, METADATA, strict=True)]

    expressions = [
        # Comparisons
        ('price > 1000',                       "Expensive items (>$1000)"),
        ('rating >= 4.7',                      "Top-rated (≥4.7)"),
        ('year = 2024',                        "Released in 2024"),

        # Range
        ('price 500 TO 1000',                  "Mid-range ($500–$1000)"),

        # Boolean
        ('in_stock = true',                    "In stock only"),

        # Logical operators
        ('brand = "Apple" OR brand = "Google"', "Apple or Google"),
        ('NOT in_stock = true',                "Out of stock"),
        ('brand = "Samsung" AND price < 500',  "Samsung under $500"),

        # IN operator
        ('category IN ["phone", "tablet"]',    "Phones & tablets"),

        # Dot notation (nested attributes)
        ('specs.weight_g < 100',               "Ultra-light (<100g)"),

        # Complex compound
        ('(brand = "Apple" OR brand = "Samsung") AND price > 800 AND in_stock = true',
         "Apple/Samsung, >$800, in stock"),
    ]

    for expr, desc in expressions:
        filtered = apply_filter(results, expr)
        names = [n for n, _, _ in filtered]
        print(f"  {desc}")
        print(f"    filter: {expr}")
        print(f"    → {len(filtered)} hit(s): {', '.join(names)}")
        print()


# ──────────────────────────────────────────────────────────────────────
# 3. Sort API  (apply_sort)
# ──────────────────────────────────────────────────────────────────────

def example_sort_api() -> None:
    divider("3 · Sort API (apply_sort)")

    from rustfuzz.sort import apply_sort

    results = [(name, 1.0, meta) for name, meta in zip(PRODUCTS, METADATA, strict=True)]

    # Single key ascending
    print("  --- price:asc ---")
    sorted_r = apply_sort(results, "price:asc")
    for name, _, meta in sorted_r[:5]:
        print(f"    ${meta['price']:>5,}  {name}")

    # Single key descending
    print("\n  --- rating:desc ---")
    sorted_r = apply_sort(results, "rating:desc")
    for name, _, meta in sorted_r[:5]:
        print(f"    ★{meta['rating']}  {name}")

    # Multi-key sort: brand alphabetical, then price descending
    print("\n  --- brand:asc, price:desc (multi-key) ---")
    sorted_r = apply_sort(results, ["brand:asc", "price:desc"])
    for name, _, meta in sorted_r:
        print(f"    {meta['brand']:<10s}  ${meta['price']:>5,}  {name}")

    # Dot-notation sort
    print("\n  --- specs.weight_g:asc (nested attribute) ---")
    sorted_r = apply_sort(results, "specs.weight_g:asc")
    for name, _, meta in sorted_r[:5]:
        print(f"    {meta['specs']['weight_g']:>5}g  {name}")


# ──────────────────────────────────────────────────────────────────────
# 4. SearchQuery builder  (fluent chaining)
# ──────────────────────────────────────────────────────────────────────

def example_search_query_builder() -> None:
    divider("4 · SearchQuery Builder (fluent chain)")

    from rustfuzz.search import BM25

    bm25 = BM25(PRODUCTS, metadata=METADATA)
    print(f"  Indexed {bm25.num_docs} products with metadata\n")

    # Chain: filter → sort → search
    print("  --- filter(in_stock=true) → sort(price:asc) → search('pro') ---")
    results = (
        bm25
        .filter("in_stock = true")
        .sort("price:asc")
        .search("pro", n=5)
        .collect()
    )
    for text, score, meta in results:
        print(f"    ${meta['price']:>5,}  score={score:.4f}  {text}")

    # Multiple chained filters (AND'd together)
    print("\n  --- filter(brand='Apple') → filter(price>500) → get_top_n('pro') ---")
    results = (
        bm25
        .filter('brand = "Apple"')
        .filter("price > 500")
        .get_top_n("pro", n=5)
    )
    for text, _score, meta in results:
        print(f"    ${meta['price']:>5,}  {meta['brand']:8s}  {text}")

    # .match() shortcut (equivalent to .search().collect())
    print("\n  --- filter(category IN ['phone']) → match('samsung', n=3) ---")
    results = bm25.filter('category IN ["phone"]').match("samsung", n=3)
    for text, score, meta in results:
        print(f"    {meta['category']:8s}  score={score:.4f}  {text}")

    # RRF variant through the builder
    print("\n  --- filter(price>300) → get_top_n_rrf('phone', n=3) ---")
    results = bm25.filter("price > 300").get_top_n_rrf("phone", n=3)
    for text, score, meta in results:
        print(f"    ${meta['price']:>5,}  rrf={score:.6f}  {text}")


# ──────────────────────────────────────────────────────────────────────
# 5. Real-world combo: filter + sort + fuzzy search
# ──────────────────────────────────────────────────────────────────────

def example_combo() -> None:
    divider("5 · Real-World Combo: Filter + Sort + Fuzzy Search")

    from rustfuzz.search import BM25

    bm25 = BM25(PRODUCTS, metadata=METADATA)

    # Scenario: customer types "appel pro" (typo) and wants in-stock Apple products sorted by price
    query = "appel pro"
    print(f'  Customer search: "{query}"')
    print("  Filters: in_stock = true, brand = Apple")
    print("  Sort: price ascending\n")

    # BM25 keyword search (may miss due to typo)
    print("  --- BM25 keyword only ---")
    results = bm25.filter('brand = "Apple" AND in_stock = true').sort("price:asc").get_top_n(query, n=5)
    for text, score, meta in results:
        print(f"    ${meta['price']:>5,}  score={score:.4f}  {text}")

    # BM25 + fuzzy (handles typo)
    print("\n  --- BM25 + fuzzy (typo-tolerant) ---")
    results = bm25.filter('brand = "Apple" AND in_stock = true').sort("price:asc").get_top_n_fuzzy(query, n=5, fuzzy_weight=0.4)
    for text, score, meta in results:
        print(f"    ${meta['price']:>5,}  score={score:.4f}  {text}")

    # BM25 + RRF
    print("\n  --- BM25 + RRF (rank fusion) ---")
    results = bm25.filter('brand = "Apple" AND in_stock = true').sort("price:asc").get_top_n_rrf(query, n=5)
    for text, score, meta in results:
        print(f"    ${meta['price']:>5,}  rrf={score:.6f}  {text}")

    # Scenario: find phones under $1000, sorted by rating
    print("\n\n  Customer browse: phones under $1000, sorted by best rating")
    results = (
        bm25
        .filter('category IN ["phone"] AND price < 1000')
        .sort("rating:desc")
        .get_top_n("phone", n=10)
    )
    for text, _score, meta in results:
        print(f"    ★{meta['rating']}  ${meta['price']:>5,}  {meta['brand']:8s}  {text}")


# ──────────────────────────────────────────────────────────────────────
# 6. All BM25 variants with filter & sort
# ──────────────────────────────────────────────────────────────────────

def example_all_variants() -> None:
    divider("6 · All BM25 Variants + Filter & Sort")

    from rustfuzz.search import BM25, BM25L, BM25T, BM25Plus

    variants: list[tuple[str, BM25 | BM25L | BM25Plus | BM25T]] = [
        ("BM25 Okapi", BM25(PRODUCTS, metadata=METADATA)),
        ("BM25L",      BM25L(PRODUCTS, metadata=METADATA, delta=0.5)),
        ("BM25Plus",   BM25Plus(PRODUCTS, metadata=METADATA, delta=1.0)),
        ("BM25T",      BM25T(PRODUCTS, metadata=METADATA)),
    ]

    query = "samsung phone"
    filter_expr = "in_stock = true AND price > 200"
    sort_expr = "price:desc"

    print(f'  query:  "{query}"')
    print(f"  filter: {filter_expr}")
    print(f"  sort:   {sort_expr}\n")

    print(f"  {'Variant':<14s}  {'#1 Result':40s}  {'Price':>7s}  Score")
    print(f"  {'─' * 75}")

    for name, index in variants:
        results = (
            index
            .filter(filter_expr)
            .sort(sort_expr)
            .get_top_n(query, n=1)
        )
        if results:
            text, score, meta = results[0]
            print(f"  {name:<14s}  {text:40s}  ${meta['price']:>5,}  {score:.4f}")
        else:
            print(f"  {name:<14s}  (no results)")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_filter_api()
    example_filter_expressions()
    example_sort_api()
    example_search_query_builder()
    example_combo()
    example_all_variants()

    print("\n✅  All filter & sort examples completed!\n")
