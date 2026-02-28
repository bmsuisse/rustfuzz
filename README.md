<p align="center">
  <img src="docs/logo.svg" alt="rustfuzz logo" width="320"/>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/rustfuzz"><img src="https://badge.fury.io/py/rustfuzz.svg" alt="PyPI version"/></a>
  <a href="https://bmsuisse.github.io/rustfuzz/"><img src="https://img.shields.io/badge/docs-online-a855f7" alt="Docs"/></a>
  <a href="https://github.com/bmsuisse/rustfuzz/actions/workflows/ci.yml"><img src="https://github.com/bmsuisse/rustfuzz/actions/workflows/ci.yml/badge.svg" alt="Tests"/></a>
  <img src="https://img.shields.io/badge/License-MIT-22c55e.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Rust-powered-a855f7?logo=rust" alt="Rust powered"/>
  <img src="https://img.shields.io/badge/Built%20by-AI-6366f1?logo=google" alt="Built by AI"/>
</p>

---

> [!WARNING]
> **🚧 Under Heavy Construction**
>
> This library is actively being developed and APIs may change between releases.
> We're shipping fast — expect frequent updates, new features, and occasional breaking changes.
> Pin your version if stability matters to you: `pip install rustfuzz==0.1.12`

---

> **🤖 This project was built entirely by AI.**
>
> The idea was simple: could an AI agent beat [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) — one of the fastest fuzzy matching libraries in the world — by writing a Rust-backed Python library from scratch, guided only by benchmarks?
>
> The development loop was: **Research → Build → Benchmark → Repeat.**

---

**rustfuzz** is a blazing-fast fuzzy string matching library for Python — implemented entirely in **Rust**. 🚀

Zero Python overhead. Memory safe. Pre-compiled wheels for every major platform.


## Features

| | |
|---|---|
| ⚡ **Blazing Fast** | Core algorithms written in Rust — no Python overhead, no GIL bottlenecks |
| 🧠 **Smart Matching** | Ratio, partial ratio, token sort/set, Levenshtein, Jaro-Winkler, and more |
| 🔒 **Memory Safe** | Rust's borrow checker guarantees — no segfaults, no buffer overflows |
| 🐍 **Pythonic API** | Clean, typed Python interface. Import and go |
| 📦 **Zero Build Step** | Pre-compiled wheels on PyPI for Python 3.10–3.14 on all major platforms |
| 🏔️ **Big Data Ready** | Excels in 1 Billion Row Challenge benchmarks, crushing high-throughput tasks |
| 🔍 **3-Way Hybrid Search** | BM25 + Fuzzy + Dense embeddings via RRF — 25ms at 1M docs, all in Rust |
| 🔎 **Filter & Sort** | Meilisearch-style filtering and sorting with Rust-level performance |
| 📄 **Document Objects** | First-class `Document(content, metadata)` + LangChain compatibility |
| 🧩 **Ecosystem Integrations** | BM25, Hybrid Search, and LangChain Retrievers for Vector DBs (Qdrant, LanceDB, FAISS, etc.) |

## Installation

```sh
pip install rustfuzz
# or, with uv (recommended — much faster):
uv pip install rustfuzz
```

## Quick Start

```python
import rustfuzz.fuzz as fuzz
from rustfuzz.distance import Levenshtein

# Fuzzy ratio
print(fuzz.ratio("hello world", "hello wrold"))          # ~96.0

# Partial ratio (substring match)
print(fuzz.partial_ratio("hello", "say hello world"))    # 100.0

# Token-order-insensitive match
print(fuzz.token_sort_ratio("fuzzy wuzzy", "wuzzy fuzzy")) # 100.0

# Levenshtein distance
print(Levenshtein.distance("kitten", "sitting"))         # 3

# Normalised similarity [0.0 – 1.0]
print(Levenshtein.normalized_similarity("kitten", "kitten")) # 1.0
```

### Batch extraction

```python
from rustfuzz import process

choices = ["New York", "New Orleans", "Newark", "Los Angeles"]
print(process.extractOne("new york", choices))
# ('New York', 100.0, 0)

print(process.extract("new", choices, limit=3))
# [('Newark', ...), ('New York', ...), ('New Orleans', ...)]
```

### 3-Way Hybrid Search (BM25 + Fuzzy + Dense)

```python
from rustfuzz.search import Document, HybridSearch

# Create documents with metadata
docs = [
    Document("Apple iPhone 15 Pro Max 256GB", {"brand": "Apple", "price": 1199}),
    Document("Samsung Galaxy S24 Ultra", {"brand": "Samsung", "price": 1299}),
    Document("Google Pixel 8 Pro", {"brand": "Google", "price": 699}),
]

# Optional: add dense embeddings for semantic search
embeddings = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]

hs = HybridSearch(docs, embeddings=embeddings)

# Handles typos via fuzzy, keywords via BM25, meaning via dense — all in Rust
results = hs.search("appel iphon", query_embedding=[1.0, 0.0, 0.0], n=1)
text, score, meta = results[0]
print(f"{text} — ${meta['price']}")
# Apple iPhone 15 Pro Max 256GB — $1199
```

> Also works with **LangChain** `Document` objects — no dependency required, auto-detected via duck-typing!

### With Real Embeddings (FastEmbed)

Use [FastEmbed](https://github.com/qdrant/fastembed) for lightweight, local, ONNX-based embeddings — no GPU needed:

```python
from fastembed import TextEmbedding
from rustfuzz.search import Document, HybridSearch

model = TextEmbedding("BAAI/bge-small-en-v1.5")  # ~33 MB, CPU-only

docs = [
    Document("Apple iPhone 15 Pro Max 256GB", {"brand": "Apple"}),
    Document("Samsung Galaxy S24 Ultra",      {"brand": "Samsung"}),
    Document("Sony WH-1000XM5 Headphones",    {"brand": "Sony"}),
]

embeddings = [e.tolist() for e in model.embed([d.content for d in docs])]
hs = HybridSearch(docs, embeddings=embeddings)

query = "wireless noise cancelling headset"
query_emb = list(model.embed([query]))[0].tolist()

results = hs.search(query, query_embedding=query_emb, n=1)
text, score, meta = results[0]
print(f"{text} — {meta['brand']}")
# Sony WH-1000XM5 Headphones — Sony
```

### With Rust-Native Embeddings (EmbedAnything)

Use [EmbedAnything](https://github.com/StarlightSearch/EmbedAnything) for Rust-native embeddings via Candle — no PyTorch, no ONNX:

```python
import embed_anything
from embed_anything import EmbeddingModel
from rustfuzz.search import Document, HybridSearch

model = EmbeddingModel.from_pretrained_hf(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
)

docs = [
    Document("Apple iPhone 15 Pro Max 256GB", {"brand": "Apple"}),
    Document("Samsung Galaxy S24 Ultra",      {"brand": "Samsung"}),
    Document("Sony WH-1000XM5 Headphones",    {"brand": "Sony"}),
]

# Embed corpus with EmbedAnything
embed_data = embed_anything.embed_query([d.content for d in docs], embedder=model)
embeddings = [item.embedding for item in embed_data]

hs = HybridSearch(docs, embeddings=embeddings)

query = "wireless noise cancelling headset"
query_emb = embed_anything.embed_query([query], embedder=model)[0].embedding

text, score, meta = hs.search(query, query_embedding=query_emb, n=1)[0]
print(f"{text} — {meta['brand']}")
# Sony WH-1000XM5 Headphones — Sony
```

Or use the **callback pattern** for fully automatic query embedding:

```python
def embed_fn(texts: list[str]) -> list[list[float]]:
    return [r.embedding for r in embed_anything.embed_query(texts, embedder=model)]

hs = HybridSearch(docs, embeddings=embed_fn)
results = hs.search("wireless headset", n=1)  # query auto-embedded!
```

### Filtering & Sorting (Meilisearch-style)

```python
from rustfuzz import Document
from rustfuzz.search import BM25

docs = [
    Document("Apple iPhone 15 Pro Max",  {"brand": "Apple",   "category": "phone",  "price": 1199, "in_stock": True}),
    Document("Samsung Galaxy S24 Ultra", {"brand": "Samsung", "category": "phone",  "price": 1299, "in_stock": True}),
    Document("Google Pixel 8 Pro",       {"brand": "Google",  "category": "phone",  "price": 699,  "in_stock": False}),
    Document("Apple MacBook Pro M3",     {"brand": "Apple",   "category": "laptop", "price": 2499, "in_stock": True}),
]

bm25 = BM25(docs)

# Fluent builder: filter → sort → match (executes immediately)
results = (
    bm25
    .filter('brand = "Apple" AND price > 500')
    .sort("price:asc")
    .match("pro", n=10)
)

for text, score, meta in results:
    print(f"  {text} — ${meta['price']}")

# Supports: =, !=, >, <, >=, <=, TO (range), IN, EXISTS, IS NULL, AND, OR, NOT
# Works with BM25, BM25L, BM25Plus, BM25T, and HybridSearch
```

Filter and sort also work with **HybridSearch** (BM25 + Fuzzy + Dense):

```python
from rustfuzz import Document
from rustfuzz.search import HybridSearch

docs = [
    Document("Apple iPhone 15 Pro Max", {"brand": "Apple", "price": 1199}),
    Document("Samsung Galaxy S24 Ultra", {"brand": "Samsung", "price": 1299}),
    Document("Google Pixel 8 Pro",       {"brand": "Google", "price": 699}),
]

hs = HybridSearch(docs, embeddings=embeddings)

# Filter + sort + semantic search
results = (
    hs
    .filter('brand = "Apple"')
    .sort("price:asc")
    .match("iphone pro", n=5, query_embedding=query_emb)
)
```

## Supported Algorithms

| Module | Algorithms |
|--------|------------|
| `rustfuzz.fuzz` | `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, `token_ratio`, `WRatio`, `QRatio`, `partial_token_*` |
| `rustfuzz.distance` | `Levenshtein`, `Hamming`, `Indel`, `Jaro`, `JaroWinkler`, `LCSseq`, `OSA`, `DamerauLevenshtein`, `Prefix`, `Postfix` |
| `rustfuzz.process` | `extract`, `extractOne`, `extract_iter`, `cdist` |
| `rustfuzz.search` | **`BM25`**, **`BM25L`**, **`BM25Plus`**, **`BM25T`**, **`HybridSearch`**, **`Document`** |
| `rustfuzz.filter` | Meilisearch-style filter parser & evaluator |
| `rustfuzz.sort` | Multi-key sort with dot notation |
| `rustfuzz.query` | Fluent `SearchQuery` builder (`.filter().sort().search().collect()`) |
| `rustfuzz.utils` | `default_process` |

### The BM25 Search Engines

`rustfuzz.search` implements lightning-fast Text Retrieval mathematical variants. The core differences:
- **`BM25` (Okapi)**: The industry standard. Employs term frequency saturation (logarithmic decay) and document length normalization.
- **`BM25L`**: Focuses on **length** penalization corrections. Introduces a static term shift `delta`, guaranteeing that matching terms yield a minimum baseline score even in massive documents where normalisation would normally suppress them.
- **`BM25Plus`**: Also creates a lower-bound for any given matching term, but applies the shift *after* term saturation. Widely considered the best default for highly mixed-length corpuses.
- **`BM25T`**: Introduces *Information Gain* adjustments to dynamically calculate the saturation limit `$k_1$` per term, restricting dominant variance. **`rustfuzz` hyper-optimises this by pre-computing term limits natively within the inverted index.**

> You can see an end-to-end benchmark comparison of these algorithms resolving the BEIR SciFact dataset in `examples/bench_retrieval.py`.

## Documentation

Full cookbook with interactive examples and benchmark results:
👉 **[bmsuisse.github.io/rustfuzz](https://bmsuisse.github.io/rustfuzz/)**

## License

MIT © [BM Suisse](https://github.com/bmsuisse)
