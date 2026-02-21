<p align="center">
  <img src="docs/logo.svg" alt="rustfuzz logo" width="320"/>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/rustfuzz"><img src="https://badge.fury.io/py/rustfuzz.svg" alt="PyPI version"/></a>
  <a href="https://bmsuisse.github.io/rustfuzz/"><img src="https://img.shields.io/badge/docs-online-a855f7" alt="Docs"/></a>
  <a href="https://github.com/bmsuisse/rustfuzz/actions/workflows/test.yml"><img src="https://github.com/bmsuisse/rustfuzz/actions/workflows/test.yml/badge.svg" alt="Tests"/></a>
  <img src="https://img.shields.io/badge/License-MIT-22c55e.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Rust-powered-a855f7?logo=rust" alt="Rust powered"/>
  <img src="https://img.shields.io/badge/Built%20by-AI-6366f1?logo=google" alt="Built by AI"/>
</p>

---

> **🤖 This project was built entirely by AI.**
>
> The idea was simple: could an AI agent beat [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) — one of the fastest fuzzy matching libraries in the world — by writing a Rust-backed Python library from scratch, guided only by benchmarks?
>
> The development loop was: **Research → Build → Benchmark → Repeat.**

---

**rustfuzz** is a blazing-fast fuzzy string matching library for Python — implemented entirely in **Rust**. 🚀

Zero Python overhead. Memory safe. Pre-compiled wheels for every major platform.

## The Challenge: Beat RapidFuzz

```mermaid
flowchart LR
    R[🔍 Research\nProfiler output\n& algorithm gaps]
    B[🦀 Build\nRust implementation\nvia PyO3]
    BM[📊 Benchmark\nvs RapidFuzz\npython -m pytest bench]
    RP[🔁 Repeat\nfind next\nbottleneck]

    R --> B --> BM --> RP --> R

    style R fill:#6366f1,color:#fff,stroke:none
    style B fill:#a855f7,color:#fff,stroke:none
    style BM fill:#22c55e,color:#fff,stroke:none
    style RP fill:#f59e0b,color:#fff,stroke:none
```

The goal: match or exceed RapidFuzz's throughput on `ratio`, `partial_ratio`, `token_sort_ratio`, and `process.extract` — all from Python. Each iteration starts with profiling, identifies the hottest path, and rewrites it deeper into Rust.

## Features

| | |
|---|---|
| ⚡ **Blazing Fast** | Core algorithms written in Rust — no Python overhead, no GIL bottlenecks |
| 🧠 **Smart Matching** | Ratio, partial ratio, token sort/set, Levenshtein, Jaro-Winkler, and more |
| 🔒 **Memory Safe** | Rust's borrow checker guarantees — no segfaults, no buffer overflows |
| 🐍 **Pythonic API** | Clean, typed Python interface. Import and go |
| 📦 **Zero Build Step** | Pre-compiled wheels on PyPI for Python 3.10–3.13 on all major platforms |

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

## Supported Algorithms

| Module | Algorithms |
|--------|------------|
| `rustfuzz.fuzz` | `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, `token_ratio`, `WRatio`, `QRatio`, `partial_token_*` |
| `rustfuzz.distance` | `Levenshtein`, `Hamming`, `Indel`, `Jaro`, `JaroWinkler`, `LCSseq`, `OSA`, `DamerauLevenshtein`, `Prefix`, `Postfix` |
| `rustfuzz.process` | `extract`, `extractOne`, `extract_iter`, `cdist` |
| `rustfuzz.utils` | `default_process` |

## Documentation

Full cookbook with interactive examples and benchmark results:
👉 **[bmsuisse.github.io/rustfuzz](https://bmsuisse.github.io/rustfuzz/)**

## License

MIT © [BM Suisse](https://github.com/bmsuisse)
