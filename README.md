# RustFuzz 🦀✨

[![PyPI version](https://badge.fury.io/py/rustfuzz.svg)](https://badge.fury.io/py/rustfuzz)
[![Documentation Status](https://readthedocs.org/projects/rustfuzz/badge/?version=latest)](https://rustfuzz.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/bmsuisse/rustfuzz/actions/workflows/test.yml/badge.svg)](https://github.com/bmsuisse/rustfuzz/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RustFuzz** is a blazing fast string matching library for Python, implemented entirely in Rust. 🚀

It serves as a high-performance Rust port of the popular `rapidfuzz` library, bringing memory safety and incredible speed to your fuzzy string matching tasks. Whether you are dealing with deduplication, record linkage, or simple spell checking, RustFuzz is engineered to deliver results instantaneously.

## Features

- **Blazing Fast**: Leverages the power of Rust string slices and optimized algorithms to outperform native Python and C++ implementations.
- **Pythonic API**: Designed to be a drop-in replacement for `rapidfuzz`, maintaining familiar interfaces so you don't have to rewrite your existing code.
- **Memory Safe**: Say goodbye to segfaults and buffer overflows. The core logic is built on Rust's strong guarantees.
- **Easy Installation**: Distributed as pre-compiled wheels for all major platforms. Just `pip install` and go! No Rust compiler needed.

## Installation

Install RustFuzz using `pip`:

```sh
pip install rustfuzz
```

Or using `uv` for insanely fast resolution:

```sh
uv pip install rustfuzz
```

## Quick Start
```python
import rustfuzz

# Calculate string similarity ratio
score = rustfuzz.fuzz_ratio("this is a test", "this is a test!")
print(f"Similarity: {score}%")
```

## Documentation

For a comprehensive guide on advanced usage, benchmarks, and interactive examples, please check out our [Cookbook Documentation 📚](https://bmsuisse.github.io/rustfuzz/).
