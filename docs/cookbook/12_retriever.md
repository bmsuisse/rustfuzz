# Retriever — Batteries-Included Search

The `Retriever` class gives you **SOTA retrieval in 3 lines** — no need to choose BM25 variants, wire up embeddings, or configure rerankers manually. It auto-selects the best pipeline based on what you provide.

## Quick Start

```python
from rustfuzz import Retriever

docs = [
    "Apple iPhone 15 Pro Max 256GB",
    "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro",
    "Apple MacBook Pro M3 Max",
    "Sony WH-1000XM5 Wireless Headphones",
]

# Simplest usage — BM25Plus + Fuzzy RRF (fast, typo-tolerant)
r = Retriever(docs)
results = r.search("wireless headphones", n=3)
for text, score in results:
    print(f"  [{score:.6f}] {text}")
```

---

## Auto-Embeddings

Pass a provider name and the `Retriever` handles model loading and query embedding automatically:

```python
# Local (no API key, no PyTorch — uses embed_anything)
r = Retriever(docs, embeddings=True)                          # default: all-MiniLM-L6-v2
r = Retriever(docs, embeddings="BAAI/bge-small-en-v1.5")     # any HuggingFace model

# OpenAI API
r = Retriever(docs, embeddings="openai")                      # text-embedding-3-small
r = Retriever(docs, embeddings="openai-large")                 # text-embedding-3-large

# Cohere API
r = Retriever(docs, embeddings="cohere")                       # embed-english-v3.0
r = Retriever(docs, embeddings="cohere-multilingual")          # embed-multilingual-v3.0

# Azure
r = Retriever(docs, embeddings="azure-openai")                 # Azure OpenAI
r = Retriever(docs, embeddings="azure-cohere")                 # Azure Cohere
```

### Passing API keys directly

```python
r = Retriever(docs, embeddings="openai", api_key="sk-...")
r = Retriever(docs, embeddings="azure-openai",
              api_key="...", api_base="https://myinstance.openai.azure.com")
```

If `api_key`/`api_base` are not provided, the Retriever reads from environment variables (`OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, etc.).

---

## With Reranker (SOTA Pipeline)

Add a cross-encoder reranker for maximum accuracy:

```python
import embed_anything

reranker_model = embed_anything.Reranker.from_pretrained("llmware/jina-reranker-tiny-onnx")

r = Retriever(docs, embeddings=True, reranker=reranker_model)
results = r.search("wireless headphones", n=5)
```

The Retriever automatically:

1. Retrieves candidates via BM25+ with fuzzy matching (RRF)
2. If embeddings → upgrades to 3-way HybridSearch (BM25 + Fuzzy + Dense)
3. If reranker → re-scores the top candidates for SOTA accuracy

---

## `RetrieverConfig` Dataclass

Group tuning knobs into a reusable config:

```python
from rustfuzz import Retriever, RetrieverConfig

cfg = RetrieverConfig(
    algorithm="bm25l",  # or "bm25", "bm25plus", "bm25t"
    k1=1.2,
    b=0.8,
    delta=0.5,
    rerank_top_k=20,
)

r = Retriever(docs, config=cfg, embeddings="openai")
```

| Field | Default | Description |
|-------|---------|-------------|
| `algorithm` | `"bm25plus"` | BM25 variant |
| `k1` | `1.5` | Term-frequency saturation |
| `b` | `0.75` | Document-length normalisation |
| `delta` | `None` | BM25L/BM25Plus delta (auto-selected) |
| `normalize` | `True` | Unicode NFKD + lowercase |
| `rerank_top_k` | `10` | Max results after reranking |

---

## With Document Objects and Metadata

```python
from rustfuzz import Retriever, Document

docs = [
    Document("Apple iPhone 15 Pro Max", {"brand": "Apple", "price": 1199}),
    Document("Samsung Galaxy S24 Ultra", {"brand": "Samsung", "price": 1299}),
    Document("Google Pixel 8 Pro", {"brand": "Google", "price": 699}),
]

r = Retriever(docs, embeddings=True)

# Filter + sort + search
results = (
    r.filter('brand = "Apple"')
    .sort("price:asc")
    .match("pro", n=10)
)

for text, score, meta in results:
    print(f"  {text} — ${meta['price']}")
```

---

## From DataFrame

```python
import pandas as pd
from rustfuzz import Retriever

df = pd.DataFrame({
    "product": ["iPhone 15", "Galaxy S24", "Pixel 8"],
    "brand": ["Apple", "Samsung", "Google"],
    "price": [1199, 1299, 699],
})

r = Retriever.from_dataframe(df, column="product", metadata_columns=["brand", "price"])
results = r.search("iphone", n=2)
```

---

## Embedding Provider Reference

| Shortcut | Provider | Model | Install | Auth |
|----------|----------|-------|---------|------|
| `True` | embed_anything (local) | `all-MiniLM-L6-v2` | `uv add embed-anything` | — |
| `"openai"` | OpenAI API | `text-embedding-3-small` | `uv add openai` | `OPENAI_API_KEY` |
| `"openai-large"` | OpenAI API | `text-embedding-3-large` | `uv add openai` | `OPENAI_API_KEY` |
| `"azure-openai"` | Azure OpenAI | `text-embedding-3-small` | `uv add openai` | `AZURE_OPENAI_ENDPOINT` + `_API_KEY` |
| `"cohere"` | Cohere API | `embed-english-v3.0` | `uv add cohere` | `COHERE_API_KEY` |
| `"cohere-multilingual"` | Cohere API | `embed-multilingual-v3.0` | `uv add cohere` | `COHERE_API_KEY` |
| `"azure-cohere"` | Azure AI | `embed-english-v3.0` | `uv add azure-ai-inference` | `AZURE_COHERE_ENDPOINT` + `_API_KEY` |
| Any `"org/model"` | embed_anything (local) | specified HF model | `uv add embed-anything` | — |
