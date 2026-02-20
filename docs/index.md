<p align="center">
  <img src="logo.svg" alt="rustfuzz" width="320"/>
</p>

<p align="center">
  <em>Blazing-fast fuzzy string matching — implemented entirely in Rust.</em>
</p>

---

## Why rustfuzz?

| | |
|---|---|
| ⚡ **Speed** | Rust core — significantly faster than Python or C++ string matching |
| 🔒 **Safety** | No segfaults, no buffer overflows — Rust's memory model enforces correctness |
| 🐍 **Simple API** | Drop-in typed Python interface — `import rustfuzz.fuzz as fuzz` and go |
| 📦 **No build step** | Pre-compiled wheels for Python 3.10–3.13 on Linux, macOS, and Windows |

## Installation

```sh
pip install rustfuzz
# or with uv:
uv pip install rustfuzz
```

## Quick Example

```python
import rustfuzz.fuzz as fuzz
from rustfuzz.distance import Levenshtein, JaroWinkler
from rustfuzz import process

# Similarity ratios
fuzz.ratio("hello world", "hello wrold")          # ~96.0
fuzz.partial_ratio("hello", "say hello world")    # 100.0
fuzz.token_sort_ratio("fuzzy wuzzy", "wuzzy fuzzy") # 100.0

# Edit distance
Levenshtein.distance("kitten", "sitting")         # 3
JaroWinkler.similarity("martha", "marhta")        # ~0.96

# Batch matching
process.extractOne("new york", ["New York", "Newark", "Los Angeles"])
# ('New York', 100.0, 0)
```

## Cookbook Recipes 🧑‍🍳

| Recipe | Description |
|--------|-------------|
| [Introduction](cookbook/01_introduction.ipynb) | Get started — basic matching and terminology |
| [Advanced Matching](cookbook/02_advanced_matching.ipynb) | Partial ratios, token sorts, score cutoffs |
| [Benchmarks](cookbook/03_benchmarks.ipynb) | Speed comparisons vs other libraries |

Start exploring from the navigation menu on the left!
