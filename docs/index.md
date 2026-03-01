<p align="center">
  <img src="logo.svg" alt="rustfuzz" width="320"/>
</p>

<p align="center">
  <em>Blazing-fast fuzzy string matching — implemented entirely in Rust.</em><br/>
  <strong>Built entirely by AI. Designed to beat RapidFuzz.</strong>
</p>

---

## The Story

**rustfuzz** started as an experiment: *can an AI agent, starting from scratch, build a fuzzy-matching library that outperforms [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) — one of the best-optimised C++ string-matching libraries in the Python ecosystem?*

No human wrote the Rust. No human tuned the algorithm parameters. The AI drove every iteration, read every benchmark result, and decided what to rewrite next.

The answer the AI kept coming back to: **Rust + PyO3 + tight Python-boundary design**.

---

## The Development Loop

Every feature and optimisation went through the same cycle:

```mermaid
flowchart LR
    R["🔍 Research<br>Profiler output<br>& algorithm gaps"]
    B["🦀 Build<br>Rust core<br>via PyO3"]
    T["✅ Test<br>All tests must pass<br>before proceeding"]
    BM["📊 Benchmark<br>vs RapidFuzz<br>& record results"]
    RP["🔁 Repeat<br>Find the next<br>bottleneck"]

    R --> B --> T --> BM --> RP --> R

    style R fill:#6366f1,color:#fff,stroke:none
    style B fill:#a855f7,color:#fff,stroke:none
    style T fill:#ef4444,color:#fff,stroke:none
    style BM fill:#22c55e,color:#fff,stroke:none
    style RP fill:#f59e0b,color:#fff,stroke:none
```

Each iteration asked:

- **Research** — where is the remaining Python overhead? What does the profiler show?
- **Build** — move that hot path into Rust. Eliminate copies, reduce allocations, avoid iterator protocol overhead.
- **Test** — the full test suite must pass before proceeding. No broken correctness, no skipped edge cases.
- **Benchmark** — run head-to-head comparisons vs RapidFuzz. Numbers don't lie.
- **Repeat** — the next bottleneck is always waiting.

---

## Why This Matters

RapidFuzz is exceptional — its C++ core, SIMD intrinsics, and decades of optimisation make it a formidable target. The goal of this project was never to dismiss it, but to prove that:

1. **AI can drive non-trivial systems programming** — not just generate boilerplate.
2. **Rust + PyO3 can match C++ at the Python boundary** — with the added safety guarantees Rust provides.
3. **Iterative AI-driven optimisation works** — each benchmark loop produced measurable gains.

---

## Features

| | |
|---|---|
| ⚡ **Blazing Fast** | Core algorithms in Rust — no Python overhead, no GIL bottlenecks |
| 🧠 **Smart Matching** | ratio, partial_ratio, token sort/set, Levenshtein, Jaro-Winkler, and more |
| 🔒 **Memory Safe** | Rust's borrow checker — no segfaults, no buffer overflows |
| 🐍 **Pythonic API** | Typed Python interface — `import rustfuzz.fuzz as fuzz` and go |
| 📦 **No Build Step** | Pre-compiled wheels for Python 3.10–3.14 on Linux, macOS, and Windows |
| 🏔️ **Big Data Ready** | Excels in 1 Billion Row Challenge benchmarks, crushing high-throughput tasks |
| 🔍 **3-Way Hybrid Search** | BM25 + Fuzzy + Dense embeddings via RRF — 25ms at 1M docs, all in Rust |
| 📄 **Document Objects** | First-class `Document(content, metadata)` + LangChain compatibility |
| 🧩 **Ecosystem Integrations** | BM25, Hybrid Search, and LangChain Retrievers for Vector DBs |
| 🎯 **Retriever** | Batteries-included SOTA search — auto-selects BM25, embeddings (OpenAI/Cohere/HF), and reranker |

---

## Installation

```sh
pip install rustfuzz
# or with uv:
uv pip install rustfuzz
```

---

## Quick Example

```python
import rustfuzz.fuzz as fuzz
from rustfuzz.distance import Levenshtein, JaroWinkler
from rustfuzz import process

# Similarity ratios
fuzz.ratio("hello world", "hello wrold")            # ~96.0
fuzz.partial_ratio("hello", "say hello world")      # 100.0
fuzz.token_sort_ratio("fuzzy wuzzy", "wuzzy fuzzy") # 100.0

# Edit distance
Levenshtein.distance("kitten", "sitting")           # 3
JaroWinkler.similarity("martha", "marhta")          # ~0.96

# Batch matching
process.extractOne("new york", ["New York", "Newark", "Los Angeles"])
# ('New York', 100.0, 0)
```

### 3-Way Hybrid Search

```python
from rustfuzz.search import Document, HybridSearch

docs = [
    Document("Apple iPhone 15 Pro Max", {"brand": "Apple", "price": 1199}),
    Document("Samsung Galaxy S24 Ultra", {"brand": "Samsung", "price": 1299}),
    Document("Google Pixel 8 Pro", {"brand": "Google", "price": 699}),
]

hs = HybridSearch(docs, embeddings=[[1, 0, 0], [0.9, 0.1, 0], [0.1, 0.9, 0]])

# Typo-tolerant + semantic search — all in Rust
results = hs.search("appel iphon", query_embedding=[1, 0, 0], n=1)
text, score, meta = results[0]
print(f"{text} — ${meta['price']}")
# Apple iPhone 15 Pro Max — $1199
```

### Custom BM25 variants via fluent builder

You can seamlessly construct a `HybridSearch` model using *any* of the advanced BM25 variants (`BM25L`, `BM25Plus`, `BM25T`) via the `.to_hybrid()` builder method:

```python
from rustfuzz.search import BM25L

results = (
    BM25L(docs, delta=0.5, b=0.8)
    .to_hybrid(embeddings=embeddings)
    .filter('brand = "Apple"')
    .match("iphone pro", n=10)
)
```

---

## Cookbook Recipes 🧑‍🍳

| Recipe | Description |
|--------|-------------|
| [Introduction](cookbook/01_introduction.md) | Get started — basic matching and terminology |
| [Advanced Matching](cookbook/02_advanced_matching.md) | Partial ratios, token sorts, score cutoffs |
| [Benchmarks](cookbook/03_benchmarks.md) | Head-to-head speed comparisons vs RapidFuzz |
| [Vector DB Hybrid Search](cookbook/04_hybrid_search.md) | BM25 + dense embeddings with Qdrant, LanceDB, FAISS & more |
| [LangChain Integration](cookbook/05_langchain.md) | Use rustfuzz as a LangChain Retriever |
| [Real-World Examples](cookbook/06_real_world.md) | Entity resolution, deduplication & production patterns |
| [Fuzzy Full Join](cookbook/07_fuzzy_join.md) | Multi-array fuzzy joins with MultiJoiner & RRF fusion |
| [**3-Way Hybrid Search**](cookbook/08_3way_hybrid_search.md) | **BM25 + Fuzzy + Dense via RRF — Document & LangChain support** |
| [**EmbedAnything**](cookbook/11_embed_anything.md) | **Rust-native embeddings — dense + sparse, no PyTorch needed** |
| [**Retriever**](cookbook/12_retriever.md) | **Batteries-included SOTA search — auto-selects BM25, embeddings & reranker** |

Start exploring from the navigation menu on the left!
