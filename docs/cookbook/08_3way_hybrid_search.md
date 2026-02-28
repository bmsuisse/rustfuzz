# 3-Way Hybrid Search

`rustfuzz` provides a state-of-the-art **3-way hybrid search** that fuses BM25 keyword matching, fuzzy string similarity, and dense vector embeddings via **Reciprocal Rank Fusion (RRF)** — all computed in Rust for million-scale performance.

## Why 3-Way?

| Signal | Catches | Example |
|--------|---------|---------|
| **BM25** | Exact keyword matches | "apple iphone" → "Apple iPhone 15 Pro" |
| **Fuzzy** | Typos and misspellings | "appel iphon" → "Apple iPhone 15 Pro" |
| **Dense** | Semantic similarity | "smartphone" → "Apple iPhone 15 Pro" |

By combining all three signals, you get maximum recall without sacrificing precision.

---

## Quick Start

```python
from rustfuzz.search import HybridSearch

corpus = [
    "Apple iPhone 15 Pro Max 256GB",
    "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro",
    "Apple MacBook Pro M3 Max",
    "Sony WH-1000XM5 Headphones",
]

# Without embeddings — 2-way RRF (BM25 + fuzzy)
hs = HybridSearch(corpus)

# Handles typos beautifully
for doc, score in hs.search("appel iphon", n=3):
    print(f"  [{score:.6f}] {doc}")
```

---

## With Dense Embeddings (3-Way)

```python
from rustfuzz.search import HybridSearch

corpus = [
    "Apple iPhone 15 Pro Max 256GB",
    "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro",
    "Apple MacBook Pro M3 Max",
    "Sony WH-1000XM5 Headphones",
]

# Use your favourite embedding model (sentence-transformers, OpenAI, etc.)
# Here we use dummy embeddings for illustration
embeddings = [
    [1.0, 0.0, 0.0],  # iPhone — "phone" concept
    [0.9, 0.1, 0.0],  # Galaxy — similar to iPhone
    [0.8, 0.2, 0.0],  # Pixel — also phone
    [0.0, 0.0, 1.0],  # MacBook — "laptop" concept
    [0.0, 1.0, 0.0],  # Headphones — "audio" concept
]

hs = HybridSearch(corpus, embeddings=embeddings)

# 3-way search: BM25 + fuzzy + dense
query = "apple phone"
query_emb = [0.95, 0.05, 0.0]  # Close to "phone" concept

results = hs.search(query, query_embedding=query_emb, n=3)
for doc, score in results:
    print(f"  [{score:.6f}] {doc}")
```

---

## Using Document Objects

`rustfuzz` provides a `Document` class for first-class document objects with metadata:

```python
from rustfuzz.search import Document, HybridSearch

docs = [
    Document("Apple iPhone 15 Pro Max", {"brand": "Apple", "price": 1199}),
    Document("Samsung Galaxy S24 Ultra", {"brand": "Samsung", "price": 1299}),
    Document("Google Pixel 8 Pro", {"brand": "Google", "price": 699}),
]

hs = HybridSearch(docs)
results = hs.search("apple iphone", n=1)

text, score, metadata = results[0]
print(f"Found: {text}")
print(f"Brand: {metadata['brand']}, Price: ${metadata['price']}")
# Found: Apple iPhone 15 Pro Max
# Brand: Apple, Price: $1199
```

---

## LangChain Integration

`HybridSearch` accepts LangChain `Document` objects directly — no `langchain` dependency required:

```python
from langchain_core.documents import Document  # Optional dependency

lc_docs = [
    Document(page_content="Apple iPhone 15 Pro Max", metadata={"source": "catalog"}),
    Document(page_content="Samsung Galaxy S24 Ultra", metadata={"source": "catalog"}),
    Document(page_content="Google Pixel 8 Pro", metadata={"source": "catalog"}),
]

# HybridSearch auto-detects LangChain Documents via duck-typing
from rustfuzz.search import HybridSearch

hs = HybridSearch(lc_docs)
results = hs.search("apple iphone", n=1)

text, score, metadata = results[0]
print(f"{text} (source: {metadata['source']})")
```

---

## Real-World: Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
from rustfuzz.search import Document, HybridSearch

# Load your corpus
docs = [
    Document("Machine learning for natural language processing", {"topic": "ML"}),
    Document("Deep learning frameworks comparison guide", {"topic": "DL"}),
    Document("Python programming for data science", {"topic": "Python"}),
    Document("Rust programming language for systems", {"topic": "Rust"}),
    Document("Cloud computing and infrastructure", {"topic": "Cloud"}),
]

# Embed with sentence-transformers
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d.content for d in docs]
embeddings = model.encode(texts)

# Build hybrid index
hs = HybridSearch(docs, embeddings=embeddings)

# Search with 3-way RRF
query = "deep lerning frameworks"  # Note the typo!
query_emb = model.encode(query)

for text, score, meta in hs.search(query, query_embedding=query_emb, n=3):
    print(f"  [{score:.6f}] [{meta['topic']}] {text}")
```

---

## Performance at Scale

The entire search pipeline runs in Rust via Rayon parallelism — designed for million-scale corpora.

**Key trick**: BM25 narrows to `bm25_candidates` first (default 100), then fuzzy + dense only score that small subset. This gives O(N) BM25 + O(candidates) fuzzy + O(candidates × D) dense.

| Corpus Size | Build Time | 3-Way Search | 2-Way Search |
|-------------|-----------|-------------|-------------|
| 1,000 | 3.5 ms | 0.15 ms | 0.11 ms |
| 10,000 | 34 ms | 0.40 ms | 0.31 ms |
| 100,000 | 335 ms | 2.4 ms | 3.9 ms |
| 1,000,000 | 13.0 s | 25 ms | 26 ms |

*Benchmarks on Apple M-series, dim=128, bm25_candidates=200.*

Tune `bm25_candidates` for your quality/speed trade-off:

```python
# Fast — fewer candidates, lower recall
results = hs.search(query, query_embedding=q_emb, n=10, bm25_candidates=50)

# Thorough — more candidates, higher recall
results = hs.search(query, query_embedding=q_emb, n=10, bm25_candidates=500)
```
