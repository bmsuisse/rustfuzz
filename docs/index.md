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
    R["🔍 Research\nProfiler output\n& algorithm gaps"]
    B["🦀 Build\nRust core\nvia PyO3"]
    BM["📊 Benchmark\nvs RapidFuzz\n& record results"]
    RP["🔁 Repeat\nfind the next\nbottleneck"]

    R --> B --> BM --> RP --> R

    style R fill:#6366f1,color:#fff,stroke:none
    style B fill:#a855f7,color:#fff,stroke:none
    style BM fill:#22c55e,color:#fff,stroke:none
    style RP fill:#f59e0b,color:#fff,stroke:none
```

Each iteration asked:

- **Research** — where is the remaining Python overhead? What does the profiler show?
- **Build** — move that hot path into Rust. Eliminate copies, reduce allocations, avoid iterator protocol overhead.
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
| 📦 **No Build Step** | Pre-compiled wheels for Python 3.10–3.13 on Linux, macOS, and Windows |

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

---

## Cookbook Recipes 🧑‍🍳

| Recipe | Description |
|--------|-------------|
| [Introduction](cookbook/01_introduction.ipynb) | Get started — basic matching and terminology |
| [Advanced Matching](cookbook/02_advanced_matching.ipynb) | Partial ratios, token sorts, score cutoffs |
| [Benchmarks](cookbook/03_benchmarks.ipynb) | Head-to-head speed comparisons vs RapidFuzz |

Start exploring from the navigation menu on the left!
