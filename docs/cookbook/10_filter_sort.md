# Filtering & Sorting Search Results

`rustfuzz` provides a **Meilisearch-compatible** filtering and sorting API that lets you narrow down search results by metadata attributes — with filtering executed at the **Rust level** for maximum performance.

The API uses a **fluent builder pattern**, so you can chain operations naturally:

```python
results = (
    bm25
    .filter('brand = "Apple" AND price > 500')
    .sort("price:asc")
    .get_top_n("iphone pro max", n=10)
)
```

---

## Setup

Filtering and sorting require **metadata** to be attached to your documents. The easiest way is using `rustfuzz.Document`:

```python
from rustfuzz import Document
from rustfuzz.search import BM25

docs = [
    Document("Apple iPhone 15 Pro Max 256GB", {"brand": "Apple",   "category": "phone",  "price": 1199, "in_stock": True,  "year": 2024}),
    Document("Samsung Galaxy S24 Ultra",      {"brand": "Samsung", "category": "phone",  "price": 1299, "in_stock": True,  "year": 2024}),
    Document("Google Pixel 8 Pro",            {"brand": "Google",  "category": "phone",  "price": 699,  "in_stock": False, "year": 2023}),
    Document("Apple MacBook Pro M3 Max",      {"brand": "Apple",   "category": "laptop", "price": 2499, "in_stock": True,  "year": 2024}),
    Document("Samsung Galaxy Tab S9",         {"brand": "Samsung", "category": "tablet", "price": 799,  "in_stock": True,  "year": 2023}),
]

bm25 = BM25(docs)
```

Works with all search engines: **`BM25`**, **`BM25L`**, **`BM25Plus`**, **`BM25T`**, and **`HybridSearch`**.

---

## Filtering

### Basic Filter

```python
# Only Apple products
results = bm25.filter('brand = "Apple"').get_top_n("pro", n=5)

for text, score, meta in results:
    print(f"  {text} — ${meta['price']}")
```

### Filter Syntax Reference

The filter language mirrors the [Meilisearch filter syntax](https://www.meilisearch.com/docs/learn/filtering_and_sorting/filter_search_results):

| Operator | Example | Description |
|---|---|---|
| `=` | `brand = "Apple"` | Equality |
| `!=` | `brand != "Apple"` | Inequality |
| `>` | `price > 1000` | Greater than |
| `>=` | `price >= 1000` | Greater than or equal |
| `<` | `price < 500` | Less than |
| `<=` | `price <= 500` | Less than or equal |
| `TO` | `price 100 TO 500` | Inclusive range |
| `IN` | `brand IN ["Apple", "Google"]` | Value in list |
| `EXISTS` | `brand EXISTS` | Attribute exists |
| `IS NULL` | `discount IS NULL` | Value is null |
| `IS EMPTY` | `tags IS EMPTY` | Empty string or list |
| `AND` | `brand = "Apple" AND price > 500` | Logical AND |
| `OR` | `brand = "Apple" OR brand = "Google"` | Logical OR |
| `NOT` | `NOT brand = "Apple"` | Logical negation |
| `()` | `(brand = "Apple" OR brand = "Google") AND price > 500` | Grouping |

### Nested Attributes (Dot Notation)

Access nested metadata with dot notation:

```python
metadata = [
    {"specs": {"weight_g": 200, "battery_mah": 4000}},
    {"specs": {"weight_g": 350, "battery_mah": 5000}},
]

results = bm25.filter("specs.weight_g < 300").get_top_n("phone", n=5)
```

### Combining Filters

Chain multiple `.filter()` calls — they're combined with AND:

```python
results = (
    bm25
    .filter('brand = "Samsung"')
    .filter("price < 1000")
    .filter("in_stock = true")
    .get_top_n("galaxy", n=10)
)
```

Or use a single expression:

```python
results = bm25.filter(
    '(brand = "Apple" OR brand = "Samsung") AND price 500 TO 2000 AND in_stock = true'
).get_top_n("pro", n=10)
```

---

## Sorting

### Basic Sort

```python
# Sort by price ascending
results = bm25.sort("price:asc").get_top_n("phone", n=10)

# Sort by price descending
results = bm25.sort("price:desc").get_top_n("phone", n=10)
```

### Multi-Key Sort

```python
# Sort by brand ascending, then by price descending within each brand
results = bm25.sort(["brand:asc", "price:desc"]).get_top_n("phone", n=10)
```

You can also use a comma-separated string:

```python
results = bm25.sort("brand:asc, price:desc").get_top_n("phone", n=10)
```

### Missing Values

Documents with missing sort attributes are always placed at the **end** of results, regardless of sort direction.

---

## Fluent Builder API

The `.filter()` and `.sort()` methods return a **`SearchQuery`** builder that accumulates operations lazily. You can terminate the chain with any search method:

### Terminal Methods

| Method | Description |
|---|---|
| `.match(query, n)` | Search with filter/sort (BM25 + HybridSearch) |
| `.get_top_n(query, n)` | BM25 / Hybrid top-N search |
| `.get_top_n_rrf(query, n)` | BM25 + fuzzy RRF search |
| `.get_top_n_fuzzy(query, n)` | BM25 + fuzzy hybrid |
| `.collect()` | Execute a deferred `.search()` call |

### Using `.match()` (recommended)

`.match()` is the simplest terminal method — it executes immediately:

```python
# Filter → sort → match (executes immediately)
results = (
    bm25
    .filter('brand = "Apple" AND price > 500')
    .sort("price:asc")
    .match("pro max", n=10)
)
```

### Lazy Execution with `.search()` + `.collect()`

For advanced use cases, `.search()` is lazy and requires `.collect()`:

```python
# Build the query lazily, execute with .collect()
results = (
    bm25
    .filter('category IN ["phone", "tablet"]')
    .filter("in_stock = true")
    .sort(["price:asc", "rating:desc"])
    .search("pro max", n=20)
    .collect()
)
```

---

## HybridSearch Integration

Filter and sort work seamlessly with `HybridSearch` (BM25 + Fuzzy + Dense embeddings):

```python
from rustfuzz import Document
from rustfuzz.search import HybridSearch

docs = [
    Document("Apple iPhone 15 Pro Max", {"brand": "Apple", "price": 1199}),
    Document("Samsung Galaxy S24 Ultra", {"brand": "Samsung", "price": 1299}),
    Document("Google Pixel 8 Pro",       {"brand": "Google", "price": 699}),
]

# Provide dense embeddings for your corpus (e.g. from EmbedAnything or OpenAI)
embeddings = [
    [0.9, 0.1, 0.0], # Apple
    [0.1, 0.8, 0.1], # Samsung
    [0.0, 0.2, 0.8], # Google
]

hs = HybridSearch(docs, embeddings=embeddings)

# Provide the embedding for your search query 
query_emb = [0.85, 0.15, 0.0]  # Embedding for "iphone pro"

# Filter + sort + semantic search (executes immediately)
results = (
    hs
    .filter('brand = "Apple"')
    .sort("price:asc")
    .match("iphone pro", n=5, query_embedding=query_emb)
)
```

---

## Performance

Filtering is implemented at the **Rust level**:

1. Python evaluates the filter expression against metadata to create a **boolean mask**
2. The mask is passed to Rust's `get_top_n_filtered()` which **zeroes out scores** for excluded documents
3. Rust's optimised heap selection skips masked documents entirely

This means filtering adds near-zero overhead — the dominant cost is still the BM25 scoring, which runs in Rust with Rayon parallelism.

---

## DataFrame Integration

Build a filtered index directly from a DataFrame:

```python
import polars as pl

df = pl.DataFrame({
    "product": ["iPhone 15", "Galaxy S24", "Pixel 8"],
    "brand": ["Apple", "Samsung", "Google"],
    "price": [1199, 1299, 699],
})

bm25 = BM25.from_column(df, "product", metadata_columns=["brand", "price"])

# Filter by metadata extracted from the DataFrame
results = bm25.filter("price > 1000").get_top_n("phone", n=5)
```
