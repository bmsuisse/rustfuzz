# Architecture & Design

`rustfuzz` is built with a singular goal: provide the fastest fuzzy string matching library for Python by moving all heavy computation into Rust, while maintaining a perfectly Pythonic API.

This document outlines the core architectural decisions that make `rustfuzz` incredibly fast and memory-safe.

## 1. Rust FFI & PyO3 Bindings

The bridge between Python and Rust is built using `PyO3`. We prioritize zero-copy or minimal-copy extraction of Python strings.

### String Extraction
When a Python string is passed to a Rust function, we avoid allocating new Rust `String` objects if possible. 
- For ASCII strings (`PyUnicode_1BYTE_KIND`), we extract direct pointers to the underlying byte array using `PyUnicode_AsUTF8AndSize` and process them as `&[u8]`.
- This ensures that operations like `fuzz.ratio` have zero allocation overhead for the strings themselves.

### Apache Arrow & PyCapsule Management
For high-throughput analytical workloads (e.g., PySpark DataFrames), zero-copy data transfer is facilitated using the Apache Arrow C-Data interface. We strictly manage `PyCapsule` lifetimes during FFI boundary crossings to ensure that memory references remain valid and are not prematurely garbage collected by Python, completely avoiding segmentation faults during batch processing.

## 2. Core Algorithms & Optimizations

The `src/algorithms.rs` and `src/fuzz.rs` files contain highly optimized versions of standard fuzzy matching algorithms.

- **Levenshtein Distance:** We use variations of Myers' bit-parallel algorithm (`PatternMask64`) for strings up to 64 characters long, significantly outperforming naive dynamic programming approaches. 
- **Pattern Masks:** The `PatternMask64` builds a 64-bit integer mask for each character in the query, allowing multiple characters to be processed in a single CPU instruction during the matching matrix calculation.
- **Fast Paths:** Natively supported scorers (e.g., `ratio`, `wratio`, `partial_ratio`) completely bypass Python callback overhead when used in batch processing methods like `process.extract`.

## 3. Batch Processing (`process.rs`)

Batch processing is the area where `rustfuzz` delivers the most significant performance gains over pure Python or Cython implementations.

### The Fast Path
When `process.extract` evaluates a query against a list of choices, it does the following:
1. Validates if the selected scorer is implemented natively (`ScorerType`).
2. If the scorer is native and no custom Python `processor` function is provided, it enters the Native Fast Path (`ratio_fast`).
3. For large lists (defined by `PARALLEL_THRESHOLD = 64`), it switches to a Rayon-powered parallel execution model.

### Concurrency and the GIL
- **Phase 1 (GIL Held):** Safely extract raw byte slice pointers (`*mut PyObject`) from the Python list.
- **Phase 2 (GIL Released):** Releases the Python Global Interpreter Lock (`py.allow_threads()`). Spawns a Rayon parallel iterator (`par_iter`) that evaluates the distance metric across all available CPU cores. No Python objects are interacted with during this phase.
- **Phase 3 (GIL Held):** Re-acquire the GIL and pack the resulting winner pairs `(PyObject, score, index)` into a final Python list.

This architecture scales linearly with CPU cores for functions like `cdist` (distance matrix generation) and `extract_iter`.

## 4. Advanced Data Structures

### BK-Trees for Deduplication
Deduplication is handled natively in Rust using a Burkhard-Keller Tree (`src/distance/bktree.rs`). The tree organizes strings based on a discrete metric distance (Levenshtein). This turns an $O(N^2)$ deduplication problem into a vastly faster tree traversal where entire branches are pruned based on the triangle inequality.

### Hybrid Search & BM25
For integration with LLMs and Vector databases, `src/search.rs` provides a native `BM25Index`.
- It implements the Okapi BM25 scoring algorithm, parallelized using Rayon over document chunks.
- **RRF (Reciprocal Rank Fusion):** Provides hybrid capabilities (`get_top_n_rrf`) that effectively merge traditional lexical token matching (BM25) with character-level fuzzy matching, providing robust retrieval even when queries include typos.
