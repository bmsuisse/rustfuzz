# EmbedAnything — Rust-Native Embeddings

[EmbedAnything](https://github.com/StarlightSearch/EmbedAnything) is a high-performance embedding library built in Rust by StarlightSearch. Like `rustfuzz`, it uses Rust under the hood — making it a natural pairing for maximum performance.

| | |
|---|---|
| 🦀 **Rust-Native** | Built with Candle — no PyTorch dependency |
| 🧠 **Multi-Backend** | Candle (HuggingFace) + ONNX Runtime |
| 📄 **File Ingestion** | Embed PDFs, text, markdown, images, audio directly |
| 🔀 **Dense + Sparse** | BERT, Jina, SPLADE, ColPali and more |
| 🌊 **Vector Streaming** | Concurrent file processing + inference pipeline |

```sh
pip install embed-anything
# or: uv add embed-anything
```

---

## 1. HybridSearch with Dense Embeddings

Embed your corpus with `EmbeddingModel` from HuggingFace, then use `HybridSearch` for 3-way retrieval (BM25 + fuzzy + dense):

```python
import embed_anything
from embed_anything import EmbeddingModel
from rustfuzz.search import Document, HybridSearch

# Rust-native model via Candle — no PyTorch needed
model = EmbeddingModel.from_pretrained_hf(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
)

docs = [
    Document("Apple iPhone 15 Pro Max 256GB", {"brand": "Apple"}),
    Document("Samsung Galaxy S24 Ultra",      {"brand": "Samsung"}),
    Document("Google Pixel 8 Pro",            {"brand": "Google"}),
    Document("Apple MacBook Pro M3 Max",      {"brand": "Apple"}),
    Document("Sony WH-1000XM5 Headphones",    {"brand": "Sony"}),
]

# Embed the corpus
texts = [d.content for d in docs]
embed_data = embed_anything.embed_query(texts, embedder=model)
embeddings = [item.embedding for item in embed_data]

# Build 3-way hybrid index
hs = HybridSearch(docs, embeddings=embeddings)

# Search — handles typos (fuzzy), keywords (BM25), and meaning (dense)
query = "wireless noise cancelling headset"
query_emb = embed_anything.embed_query([query], embedder=model)[0].embedding

for text, score, meta in hs.search(query, query_embedding=query_emb, n=3):
    print(f"  [{score:.6f}] [{meta['brand']}] {text}")
```

> **Why this works**: the query *"wireless noise cancelling headset"* has zero keyword overlap with *"Sony WH-1000XM5 Headphones"* — BM25 alone would miss it. The dense embedding captures the semantic similarity.

---

## 2. Callback-Based Auto-Embedding

Use the embedding callback pattern so `HybridSearch` automatically embeds queries — no manual `embed_query` calls needed:

```python
import embed_anything
from embed_anything import EmbeddingModel
from rustfuzz.search import HybridSearch

model = EmbeddingModel.from_pretrained_hf(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
)

# Define a callback: texts → dense vectors
def embed_fn(texts: list[str]) -> list[list[float]]:
    results = embed_anything.embed_query(texts, embedder=model)
    return [r.embedding for r in results]

corpus = [
    "Python is a popular programming language",
    "Rust provides memory safety without garbage collection",
    "Machine learning models require large datasets",
    "Docker containers simplify application deployment",
]

# Callback: embeddings generated at init AND at each .search() call
hs = HybridSearch(corpus, embeddings=embed_fn)

# No query_embedding needed — the callback handles it!
for text, score in hs.search("fast programming language", n=2):
    print(f"  [{score:.6f}] {text}")
```

> ✨ The callback is invoked once for the corpus at init, and again for each query at search time — fully automatic.

---

## 3. MultiJoiner with Dense Embeddings

Use real embeddings in fuzzy joins to catch semantic matches that text-only matching would miss:

```python
import embed_anything
from embed_anything import EmbeddingModel
from rustfuzz.join import MultiJoiner

model = EmbeddingModel.from_pretrained_hf(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
)

crm_names = ["Apple Inc.", "Microsoft Corporation", "Alphabet Inc."]
invoice_names = ["Apple Incorporated", "Microsft Corp", "Google LLC"]

# Embed both arrays via EmbedAnything
emb_crm = [d.embedding for d in embed_anything.embed_query(crm_names, embedder=model)]
emb_inv = [d.embedding for d in embed_anything.embed_query(invoice_names, embedder=model)]

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

## 4. Dense + Sparse (SPLADE) with MultiJoiner

For maximum matching quality, combine dense embeddings with sparse SPLADE vectors:

```python
import embed_anything
from embed_anything import EmbeddingModel
from rustfuzz.join import MultiJoiner

dense_model = EmbeddingModel.from_pretrained_hf(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
)
sparse_model = EmbeddingModel.from_pretrained_hf(
    model_id="prithivida/Splade_PP_en_v1",
)

products = ["Apple iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra", "Sony WH-1000XM5 Headphones"]
queries = ["apple smartphone", "wireless noise cancelling headset"]

# Dense embeddings
emb_products = [d.embedding for d in embed_anything.embed_query(products, embedder=dense_model)]
emb_queries = [d.embedding for d in embed_anything.embed_query(queries, embedder=dense_model)]

# Sparse embeddings — filter to non-zero for SPLADE
sparse_products = [
    {i: v for i, v in enumerate(d.embedding) if v != 0.0}
    for d in embed_anything.embed_query(products, embedder=sparse_model)
]
sparse_queries = [
    {i: v for i, v in enumerate(d.embedding) if v != 0.0}
    for d in embed_anything.embed_query(queries, embedder=sparse_model)
]

# 3-channel hybrid join: text + dense + sparse
joiner = (
    MultiJoiner(text_weight=0.4, dense_weight=0.4, sparse_weight=0.3)
    .add_array("queries", texts=queries, dense=emb_queries, sparse=sparse_queries)
    .add_array("products", texts=products, dense=emb_products, sparse=sparse_products)
)

for r in joiner.join(n=1):
    if r["src_array"] == "queries":
        print(f"  {r['src_text']:40s} → {r['tgt_text']}")
```

---

## Tips

- **First run**: Models download from HuggingFace on first use. Subsequent runs use the cache.
- **`embed_query` vs `embed_file`**: Use `embed_query` for text lists, `embed_file` for documents (PDF, TXT, etc.).
- **Callback pattern**: Pass a callable to `HybridSearch(embeddings=fn)` to auto-embed queries at search time.
- **No embedding at query time?** Omit `query_embedding=` — `HybridSearch` gracefully falls back to BM25 + fuzzy (2-way RRF).

> **See also**: [3-Way Hybrid Search](08_3way_hybrid_search.md) for the full API reference.
