# FastEmbed — Lightweight Local Embeddings

[FastEmbed](https://github.com/qdrant/fastembed) is a lightweight, ONNX-based embedding library by Qdrant. No PyTorch, no GPU required — just fast, local embeddings that pair perfectly with `rustfuzz`'s hybrid search.

| | |
|---|---|
| ⚡ **Fast** | ONNX Runtime — no PyTorch overhead |
| 📦 **Tiny** | ~33 MB for `bge-small-en-v1.5` |
| 🖥️ **CPU-only** | No GPU drivers or CUDA needed |
| 🔀 **Dense + Sparse** | Both `TextEmbedding` and `SparseTextEmbedding` (SPLADE) |

```sh
pip install fastembed
# or: uv add fastembed
```

---

## 1. HybridSearch with Dense Embeddings

The most common pattern — embed your corpus with `TextEmbedding`, then use `HybridSearch` for 3-way retrieval (BM25 + fuzzy + dense):

```python
from fastembed import TextEmbedding
from rustfuzz.search import Document, HybridSearch

# Lightweight ONNX model — downloads once, runs on CPU
model = TextEmbedding("BAAI/bge-small-en-v1.5")

docs = [
    Document("Apple iPhone 15 Pro Max 256GB", {"brand": "Apple"}),
    Document("Samsung Galaxy S24 Ultra",      {"brand": "Samsung"}),
    Document("Google Pixel 8 Pro",            {"brand": "Google"}),
    Document("Apple MacBook Pro M3 Max",      {"brand": "Apple"}),
    Document("Sony WH-1000XM5 Headphones",    {"brand": "Sony"}),
]

# Embed the corpus
texts = [d.content for d in docs]
embeddings = [e.tolist() for e in model.embed(texts)]

# Build 3-way hybrid index
hs = HybridSearch(docs, embeddings=embeddings)

# Search — handles typos (fuzzy), keywords (BM25), and meaning (dense)
query = "wireless noise cancelling headset"
query_emb = list(model.embed([query]))[0].tolist()

for text, score, meta in hs.search(query, query_embedding=query_emb, n=3):
    print(f"  [{score:.6f}] [{meta['brand']}] {text}")
```

> **Why this works**: the query *"wireless noise cancelling headset"* has zero keyword overlap with *"Sony WH-1000XM5 Headphones"* — BM25 alone would miss it. The dense embedding captures the semantic similarity.

---

## 2. MultiJoiner with Dense + Sparse (SPLADE)

For maximum matching quality, combine dense embeddings with sparse SPLADE vectors using `MultiJoiner`:

```python
from fastembed import SparseTextEmbedding, TextEmbedding
from rustfuzz.join import MultiJoiner

products = [
    "Apple iPhone 15 Pro Max",
    "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro",
    "Sony WH-1000XM5 Headphones",
]

queries = [
    "apple smartphone",
    "wireless noise cancelling headset",
    "samung galxy",  # typos!
]

# Dense embeddings (semantic meaning)
dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
emb_products = [e.tolist() for e in dense_model.embed(products)]
emb_queries = [e.tolist() for e in dense_model.embed(queries)]

# Sparse embeddings (keyword importance via SPLADE)
sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")

# Convert SparseEmbedding → dict[int, float] for rustfuzz
sparse_products = [
    {int(i): float(v) for i, v in zip(e.indices, e.values)}
    for e in sparse_model.embed(products)
]
sparse_queries = [
    {int(i): float(v) for i, v in zip(e.indices, e.values)}
    for e in sparse_model.embed(queries)
]

# Hybrid join: text + dense + sparse
joiner = (
    MultiJoiner(text_weight=0.4, dense_weight=0.4, sparse_weight=0.3)
    .add_array("queries", texts=queries, dense=emb_queries, sparse=sparse_queries)
    .add_array("products", texts=products, dense=emb_products, sparse=sparse_products)
)

for r in joiner.join(n=1):
    if r["src_array"] == "queries":
        print(
            f"  {r['src_text']:40s} → {r['tgt_text']:30s}  "
            f"score={r['score']:.4f}"
        )
```

> ✨ All three scoring channels (text fuzzy, dense semantic, sparse keyword) are fused via RRF for maximum recall.

---

## 3. MultiJoiner with Dense Embeddings

Use real embeddings in fuzzy joins to catch semantic matches that text-only matching would miss:

```python
from fastembed import TextEmbedding
from rustfuzz.join import MultiJoiner

model = TextEmbedding("BAAI/bge-small-en-v1.5")

crm_names = ["Apple Inc.", "Microsoft Corporation", "Alphabet Inc."]
invoice_names = ["Apple Incorporated", "Microsft Corp", "Google LLC"]

# Embed both arrays
emb_crm = [e.tolist() for e in model.embed(crm_names)]
emb_inv = [e.tolist() for e in model.embed(invoice_names)]

# Hybrid join: text (BM25 + fuzzy) + dense (cosine similarity)
joiner = (
    MultiJoiner(text_weight=0.5, dense_weight=0.5)
    .add_array("crm", texts=crm_names, dense=emb_crm)
    .add_array("invoices", texts=invoice_names, dense=emb_inv)
)

for r in joiner.join(n=1):
    if r["src_array"] == "crm":
        print(f"  {r['src_text']:25s} → {r['tgt_text']:25s}  score={r['score']:.4f}")
```

> ✨ `"Alphabet Inc."` ↔ `"Google LLC"` — matched via semantic similarity despite zero text overlap!

---

## Model Selection Guide

| Model | Dimensions | Size | Best For |
|-------|-----------|------|----------|
| `BAAI/bge-small-en-v1.5` | 384 | 33 MB | Fast, general-purpose (English) |
| `BAAI/bge-base-en-v1.5` | 768 | 110 MB | Higher quality, still fast |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 23 MB | Lightweight alternative |
| `prithivida/Splade_PP_en_v1` | sparse | 110 MB | Sparse (SPLADE++) |

```python
from fastembed import TextEmbedding

# List all available models
for m in TextEmbedding.list_supported_models():
    print(f"  {m['model']}: dim={m.get('dim', '?')}, size={m.get('size_in_GB', '?')} GB")
```

---

## Tips

- **Batch size**: Pass `batch_size=` to `model.embed()` for large corpora — balances memory and throughput.
- **First run**: Models download on first use (~33 MB for `bge-small-en-v1.5`). Subsequent runs use the cache.
- **`.tolist()`**: FastEmbed returns numpy arrays — call `.tolist()` to convert to plain Python lists for rustfuzz.
- **No embedding at query time?** Just omit `query_embedding=` — `HybridSearch` gracefully falls back to BM25 + fuzzy (2-way RRF).

> **See also**: [3-Way Hybrid Search](08_3way_hybrid_search.md) for the full API reference and performance benchmarks.
