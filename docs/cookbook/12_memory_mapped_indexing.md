# Memory-Mapped Search Indexing Guide

The `rustfuzz` search library provides native memory-mapped (mmap) capabilities for `BM25` (and its variants `BM25L`, `BM25Plus`, `BM25T`) as well as `HybridSearch`. This allows agents to index and search millions of documents with near-zero Python RAM usage by directly streaming index segments from disk.

## 1. Generator Streaming (Zero-RAM Compilation)
Python agents do NOT need to load the entire corpus into memory (like a `list[str]`) before building. `rustfuzz` supports reading lazily from native Python generator iterables.

```python
from rustfuzz.search import BM25

# 1. Create a lazy generator
def stream_large_corpus():
    for i in range(10_000_000):
        yield f"This is document {i} with extensive text content..."

# 2. Stream directly into a memory-mapped Rust instance
# memory_map accepts either a boolean (uses temp dir) or a string filepath.
bm25 = BM25(
    stream_large_corpus(),
    memory_map="./my_mmap_index" 
)

# 3. Query the index (sub-millisecond latency using page caches)
results = bm25.get_top_n("extensive text", n=10)
```

## 2. Using BM25 Variants & Hybrid Search
The `memory_map` parameter is safely propagated into all BM25 variants, ensuring that the specific `tf_norm` scaling for each algorithm remains mathematically exact even when mapping.

```python
from rustfuzz.search import BM25L, HybridSearch

# Memory-Mapped BM25L
bm25l = BM25L(
    stream_large_corpus(), 
    delta=0.7, 
    memory_map="./my_bm25l_mmap"
)

# Memory-Mapped HybridSearch
# This requires embeddings to be provided, but the textual BM25 index components
# will be flawlessly memory-mapped to disk.
hybrid = HybridSearch(
    corpus=stream_large_corpus(), 
    embeddings=my_embedding_array,
    algorithm="bm25+",  # Pass variant config natively 
    delta=1.5,
    memory_map="./my_hybrid_mmap"
)
```

## 3. Loading Existing Indices (Cold Start)
If agents need to mount pre-existing datasets dynamically without rebuilding them, they should utilize the dedicated disk loaders for instant bootstrapping.

```python
from rustfuzz.search import MmapBM25, MmapHybridSearchIndex

# Load a generic BM25 dataset dynamically
loaded_bm25 = MmapBM25.load("./my_mmap_index")
print(loaded_bm25.get_top_n("search query", n=5))

# Load a Hybrid Search dataset dynamically
loaded_hybrid = MmapHybridSearchIndex.load("./my_hybrid_mmap")
print(loaded_hybrid.search("search query", query_embedding=my_vector, n=5))
```

## 4. Key Considerations for Agents
- **Immutability:** Memory-mapped datasets are immutable (`add_documents` and `remove_documents` will raise a `NotImplementedError`). Once built, they are strictly read-only for fast sequential retrieval.
- **Resource Footprint:** A 1-million document index may create ~3GB of binary disk artifacts (`postings.bin`, `positions.bin`), but it will only consume **~15MB of RAM** natively when mounted due to dictionary serialization.
- **Vector Mappings:** In `HybridSearch`, the embedding list remains in-memory unless the underlying vector array itself is a memory-mapped `numpy` unboxed primitive. Ensure your embedding tensors use `np.memmap` if extreme constraints are required.
