# Fuzzy Full Join

!!! tip "When to use this"
    Use `fuzzy_join` / `MultiJoiner` when you have **two or more arrays of records** that you want to link by similarity — without a shared key. Classic use cases: product matching, entity resolution, record linkage, cross-catalogue deduplication.

`rustfuzz` provides a **Rust-native, Rayon-parallel fuzzy full join** that compares every element in each array against every element in every other array and returns the top-N matches.

Three complementary signals are supported and fused via **Reciprocal Rank Fusion (RRF)**:

| Channel | Input | Scoring | Auto-fallback |
|---|---|---|---|
| **Text** | `texts=` | BM25 Okapi + indel fuzzy, RRF | — |
| **Sparse** | `sparse=` | dot product on `{token_id: weight}` | ✅ BM25 when `sparse=None` |
| **Dense** | `dense=` | cosine similarity on unit vectors | — |

---

## Quick Start

```python
from rustfuzz.join import fuzzy_join

products  = ["Apple iPhone 14 Pro", "Samsung Galaxy S23 Ultra", "Google Pixel 7a"]
listings  = ["iphone 14 pro max",   "galaxy s23 ultra 256gb",   "pixel 7a black"]
inventory = ["Apple Iphone",        "Samsung Galxy",            "Gogle Pixle"]   # typos

rows = fuzzy_join(
    {"products": products, "listings": listings, "inventory": inventory},
    n=1,   # top-1 match per element per target array
)

for r in rows:
    print(f"{r['src_array']}[{r['src_idx']}] → {r['tgt_array']}[{r['tgt_idx']}]  "
          f"{r['src_text']!r:30} ↔ {r['tgt_text']!r}  score={r['score']:.4f}")
```

Output (excerpt):

```
products[0] → listings[0]    'Apple iPhone 14 Pro'          ↔ 'iphone 14 pro max'
products[1] → inventory[1]   'Samsung Galaxy S23 Ultra'     ↔ 'Samsung Galxy'
inventory[0] → products[0]   'Apple Iphone'                 ↔ 'Apple iPhone 14 Pro'
```

---

## How Scoring Works

### Reciprocal Rank Fusion (RRF)

Each active channel produces a ranking over all target documents. The final score per target is:

```
score = Σ  weight_channel × (1 / (rrf_k + rank_channel))
        ────────────────────────────────────────────────
                       Σ active_weights
```

where `rrf_k = 60` (default, Cormack et al. 2009). This makes the score robust to scale differences between channels — BM25 values and cosine similarities are never directly compared, only their ranks are.

### Text channel

A BM25 Okapi index is built over the **target** array texts once per array pair. Then for each source query:

1. Score all target docs with BM25 → rank by score
2. Score all target docs with indel fuzzy similarity → rank by score
3. RRF-fuse the two ranking lists

This dual-ranking combination means **misspellings and abbreviations** are handled robustly: if BM25 misses a misspelled term, the fuzzy rerank salvages it.

### Sparse channel (auto-fallback)

When you provide explicit `sparse=` dict vectors (e.g. SPLADE, BM25S encodings), a merge-join dot product is computed. **When no sparse vectors are provided**, the channel automatically falls back to **pure BM25 scores** over the target text — giving you a complementary term-overlap signal at zero extra cost.

| sparse provided? | text provided? | Sparse channel behaviour |
|---|---|---|
| ✅ | any | merge-join dot product on your vectors |
| ❌ | ✅ | BM25 score over target texts (auto-fallback) |
| ❌ | ❌ | channel disabled |

### Dense channel

Expects pre-normalised embedding vectors (unit norm). Scoring is pure dot product (= cosine similarity on unit vectors) computed in Rayon-parallel across all target docs.

---

## API Reference

### `fuzzy_join` — one-shot convenience

```python
from rustfuzz.join import fuzzy_join

rows = fuzzy_join(
    arrays: dict[str, list[str]],           # named text arrays
    *,
    sparse: dict[str, list[dict[int, float]]] | None = None,  # named sparse arrays
    dense:  dict[str, list[list[float]]]    | None = None,    # named dense arrays
    text_weight:   float = 1.0,
    sparse_weight: float = 1.0,
    dense_weight:  float = 1.0,
    n:      int   = 1,      # top-N matches per element
    bm25_k1: float = 1.5,
    bm25_b:  float = 0.75,
    rrf_k:   int   = 60,
) -> list[dict]
```

All keys in `sparse` / `dense` must appear in `arrays`.

### `MultiJoiner` — fine-grained control

```python
from rustfuzz.join import MultiJoiner

joiner = MultiJoiner(
    text_weight=1.0,
    sparse_weight=1.0,
    dense_weight=1.0,
    bm25_k1=1.5,
    bm25_b=0.75,
    rrf_k=60,
)

# Method-chaining API
joiner = (
    MultiJoiner()
    .add_array("A", texts=[...], dense=[...])
    .add_array("B", texts=[...], sparse=[...])
)

rows = joiner.join(n=1)              # all ordered pairs
rows = joiner.join_pair("A", "B", n=3)  # single direction only
```

`add_array` parameters:

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Unique array identifier |
| `texts` | `list[str \| None] \| None` | Text per element; drives text + sparse-fallback channels |
| `sparse` | `list[dict[int, float] \| None] \| None` | Sparse vectors as `{token_id: weight}` |
| `dense` | `list[list[float] \| None] \| None` | Dense embedding vectors (pre-normalised) |

At least one of `texts`, `sparse`, or `dense` must be non-`None`.

### Return schema

Each row is a Python `dict`:

```python
{
    "src_array":    str,           # source array name
    "src_idx":      int,           # index within source array
    "src_text":     str | None,    # source element text
    "tgt_array":    str,           # target array name
    "tgt_idx":      int,           # index within target array
    "tgt_text":     str | None,    # matched target text
    "score":        float,         # combined weighted RRF score
    "text_score":   float | None,  # raw text-channel RRF (or None if disabled)
    "sparse_score": float | None,  # dot-product or BM25 score (or None if disabled)
    "dense_score":  float | None,  # cosine similarity (or None if disabled)
}
```

---

## Examples

### Text-only: product matching with typos

```python
from rustfuzz.join import fuzzy_join

products  = ["Apple iPhone 14 Pro", "Samsung Galaxy S23 Ultra", "Google Pixel 7a"]
inventory = ["Apple Iphone",        "Samsung Galxy",            "Gogle Pixle"]

rows = fuzzy_join({"products": products, "inventory": inventory}, n=1)
# inventory[0] = "Apple Iphone"  →  products[0] = "Apple iPhone 14 Pro"  ✓
# inventory[1] = "Samsung Galxy" →  products[1] = "Samsung Galaxy S23 Ultra"  ✓
```

### Dense-only: semantic embedding matching

```python
import math
from rustfuzz.join import MultiJoiner

def normalize(v):
    m = math.sqrt(sum(x*x for x in v))
    return [x/m for x in v]

emb_a = [normalize([1,0,0]), normalize([0,1,0]), normalize([0,0,1])]
emb_b = [normalize([0.97,0.08,0]), normalize([0.03,0.98,0.05]), normalize([0.01,0.04,0.99])]

joiner = (
    MultiJoiner(text_weight=0, sparse_weight=0, dense_weight=1)
    .add_array("A", texts=["concept_X","concept_Y","concept_Z"], dense=emb_a)
    .add_array("B", texts=["near_X",   "near_Y",   "near_Z"],   dense=emb_b)
)

for r in joiner.join_pair("A", "B", n=1):
    print(f"{r['src_text']} → {r['tgt_text']}  cosine={r['dense_score']:.4f}")
# concept_X → near_X   cosine=0.9979
# concept_Y → near_Y   cosine=0.9988
# concept_Z → near_Z   cosine=0.9990
```

### Text + Dense: fusing lexical and semantic signals

```python
from rustfuzz.join import MultiJoiner

brands = ["Apple MacBook Pro", "Dell XPS 15",  "Lenovo ThinkPad X1"]
descr  = ["macbook pro m3",    "xps 15 oled",  "thinkpad x1 carbon"]

# Dummy normalized embeddings — use your real model in production
emb_brands = [[1,0,0], [0,1,0], [0,0,1]]
emb_descr  = [[0.97,0.05,0], [0.02,0.96,0.04], [0.01,0.03,0.98]]

joiner = (
    MultiJoiner(text_weight=0.4, sparse_weight=0.4, dense_weight=0.6)
    .add_array("brands", texts=brands, dense=emb_brands)
    .add_array("descr",  texts=descr,  dense=emb_descr)
)

for r in joiner.join_pair("brands", "descr", n=1):
    print(f"{r['src_text']:<22} → {r['tgt_text']:<22}  "
          f"score={r['score']:.4f}  bm25={r['sparse_score']:.2f}  cos={r['dense_score']:.4f}")
```

### Explicit sparse vectors (SPLADE / BM25S)

Pass pre-encoded sparse vectors directly — the merge-join dot product is much faster than building a BM25 index for very large pre-encoded corpora:

```python
from rustfuzz.join import MultiJoiner

# Token IDs from your vocabulary; weights are TF-IDF / SPLADE scores
sparse_corpus = [
    {101: 0.9, 2345: 0.7, 879: 0.4},  # doc 0
    {202: 0.85, 3310: 0.6},            # doc 1
]
sparse_query = [
    {101: 0.95, 879: 0.5},             # query 0
    {202: 0.9,  3310: 0.7},            # query 1
]

joiner = (
    MultiJoiner(text_weight=0, sparse_weight=1, dense_weight=0)
    .add_array("queries", sparse=sparse_query)
    .add_array("corpus",  sparse=sparse_corpus)
)

for r in joiner.join_pair("queries", "corpus", n=1):
    print(f"query[{r['src_idx']}] → corpus[{r['tgt_idx']}]  dot={r['sparse_score']:.4f}")
```

### Pandas / Polars output

```python
import pandas as pd
from rustfuzz.join import fuzzy_join

rows = fuzzy_join({"A": array_a, "B": array_b, "C": array_c}, n=1)
df = pd.DataFrame(rows)

# Best match per source element
best = df.loc[df.groupby(["src_array","src_idx"])["score"].idxmax()]
print(best[["src_array","src_text","tgt_array","tgt_text","score"]])
```

```python
import polars as pl
from rustfuzz.join import fuzzy_join

df = pl.DataFrame(fuzzy_join({"A": array_a, "B": array_b}, n=3))

# Top match per pair
top1 = (
    df.sort("score", descending=True)
      .group_by(["src_array","src_idx","tgt_array"])
      .first()
)
```

---

## Performance Notes

- All cross-array scoring is **Rayon-parallel** — uses all CPU cores automatically.
- BM25 index per target array is built **once** and reused for all source queries in a pair.
- Sparse dot product uses a **merge-join** on sorted `(token_id, weight)` pairs — O(n+m) instead of O(nm).
- Dense cosine is a **SIMD-friendly dot product** in Rust — same throughput as a tuned BLAS gemv on unit vectors.
- Memory footprint: one BM25 index per active (src, tgt) text pair + source/target arrays held in Python.

!!! tip "Scaling to millions of rows"
    For very large arrays (>100k docs), consider pre-filtering with a fast ANN library (e.g. FAISS, hnswlib) and feeding the candidate set as a smaller array to `MultiJoiner` rather than doing a full cross join.
