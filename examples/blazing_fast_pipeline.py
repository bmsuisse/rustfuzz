"""
Blazing Fast 3-Way Hybrid Search Pipeline
─────────────────────────────────────────────
This example demonstrates the architectural optimizations from the "Advanced Architectures
for Ultra-Low Latency Information Retrieval" report.

It implements a cascading, hardware-symbiotic pipeline that leverages:
1. Eager BM25 Sparse Matrix Scoring
2. Dense Vector Semantic Search (via FastEmbed)
3. Non-parametric Reciprocal Rank Fusion (RRF)
4. Hardware-accelerated Exact & Fuzzy fallbacks

Usage:
    uv run python examples/blazing_fast_pipeline.py
"""

import sys
import time

try:
    from fastembed import TextEmbedding
except ImportError:
    print("❌ fastembed is required to run the pipeline example.")
    print("Please install via: uv add fastembed")
    sys.exit(1)

from rustfuzz.search import Document, HybridSearch

def _divider(title: str) -> None:
    print(f"\n{title}")
    print("=" * max(len(title), 50))


def build_pipeline_index(docs: list[Document], model: TextEmbedding) -> HybridSearch:
    """Builds the eager-scoring BM25 and Dense vector hybrid index."""
    t0 = time.perf_counter()
    
    print(f"[*] Embedding {len(docs)} documents... (in production, this is pre-computed)")
    texts = [d.content for d in docs]
    
    # 1. Dense Semantic Generation
    # Generates standard float32 vectors mapped to local CPU memory
    embeddings = [e.tolist() for e in model.embed(texts)]
    
    # 2. Build the Hybrid Index
    # This automatically computes eager BM25 scores and builds the internal RRF logic
    pipeline = HybridSearch(docs, embeddings=embeddings)
    
    t1 = time.perf_counter()
    print(f"[+] Pipeline Index built in {((t1 - t0) * 1000):.2f} ms")
    return pipeline


def run_cascading_query(pipeline: HybridSearch, model: TextEmbedding, query: str, limit: int = 3):
    """Executes a 3-way asynchronous query fused by RRF."""
    print(f'\n🔎 QUERY: "{query}"')
    
    t0 = time.perf_counter()
    
    # Generate dense query payload
    query_emb = list(model.embed([query]))[0].tolist()
    t_embed = time.perf_counter()
    
    # Fetch results using the 3-way RRF internal engine
    # (Combines BM25 eager scores + Dense Cosine distance + Fuzzy fallback)
    results = pipeline.search(query, query_embedding=query_emb, n=limit, bm25_candidates=50)
    t_search = time.perf_counter()
    
    ms_embed = (t_embed - t0) * 1000
    ms_search = (t_search - t_embed) * 1000
    ms_total = (t_search - t0) * 1000
    
    print(f"⏱️  Pipeline Latency -> Embed: {ms_embed:.2f} ms | Search + RRF: {ms_search:.2f} ms | Total: {ms_total:.2f} ms")
    print("-" * 50)
    
    for rank, (text, score, meta) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {text}  | Source: {meta.get('domain', 'general')}")


def main():
    _divider("🚀 3-Way Hybrid Search / Agentic RAG Pipeline Initialization")
    
    print("[*] Loading quantization-optimized FastEmbed model [BAAI/bge-small-en-v1.5]")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    
    # Simulate a corpus that requires semantic, keyword, AND fuzzy matching to test the engine
    corpus = [
        Document("The hypervisor layer limits direct CPU vector network extensions (VNNI) access.", {"domain": "sysadmin"}),
        Document("Rustfuzz implements AVX-512 SIMD parallelism to bypass GIL constraints.", {"domain": "engineering"}),
        Document("The CEO announced a stock buyback of $500M starting next quarter.", {"domain": "finance"}),
        Document("Python rank-bm25 incurs massive overhead due to dynamic TF-IDF calculation.", {"domain": "engineering"}),
        Document("Our multi-cloud deployment utilizes Kubernetes clusters distributed globally.", {"domain": "devops"}),
        Document("MRL (Matryoshka Representation Learning) allows truncating dense dimensions safely.", {"domain": "ml_research"}),
        Document("Elasticsearch Lucene indices usually take 10-40ms on billion scale datasets.", {"domain": "ml_research"}),
        Document("Standard python string operations process characters sequentially and lack SIMD.", {"domain": "engineering"}),
        Document("The QAttention kernel replaces MatMul and Scale blocks natively inside ONNX.", {"domain": "ml_research"}),
    ]
    
    # 1. Build the index
    pipeline = build_pipeline_index(corpus, model)
    
    _divider("🔥 High-Speed RRF Query Testing")
    
    # Query A: Keyword bias -> requires BM25 exact match
    run_cascading_query(pipeline, model, "AVX-512 SIMD parallelism")
    
    # Query B: Semantic bias -> specific wording differs from document, requires Dense Vector
    run_cascading_query(pipeline, model, "How to safely scale down embedding dimensionality?")
    
    # Query C: Fuzzy bias -> typing errors that dense/sparse often miss
    run_cascading_query(pipeline, model, "Elastcserch luecene indexes")


if __name__ == "__main__":
    main()
