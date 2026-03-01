import os
import json
import time
import requests
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm import tqdm

from rustfuzz.search import BM25Plus, Reranker
import embed_anything


def download_scifact():
    """Download and extract SciFact dataset if not exists."""
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    filepath = "scifact.zip"
    
    if not os.path.exists("scifact"):
        if not os.path.exists(filepath):
            print("Downloading SciFact dataset...")
            response = requests.get(url, stream=True)
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print("Extracting SciFact...")
        import zipfile
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(".")
    
    return "scifact"

def load_data(data_path: str):
    print("Loading corpus, queries, and qrels...")
    corpus = {}
    doc_text_to_id = {}
    with open(os.path.join(data_path, "corpus.jsonl"), 'r') as f:
        for line in f:
            obj = json.loads(line)
            doc_text = f"{obj.get('title', '')} {obj.get('text', '')}"
            corpus[obj['_id']] = doc_text
            doc_text_to_id[doc_text] = int(obj['_id'])
            
    queries = {}
    with open(os.path.join(data_path, "queries.jsonl"), 'r') as f:
        for line in f:
            obj = json.loads(line)
            queries[obj['_id']] = obj['text']
            
    qrels = {}
    with open(os.path.join(data_path, "qrels", "test.tsv"), 'r') as f:
        next(f)
        for line in f:
            qid, docid, score = line.strip().split('\t')
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][int(docid)] = int(score)
            
    return corpus, doc_text_to_id, queries, qrels

def evaluate_ndcg_at_10(qrels: Dict[str, Dict[str, int]], runs: Dict[str, List[str]]) -> float:
    ndcg_list = []
    import math
    
    for qid in qrels:
        if qid not in runs:
            continue  # Skip queries we didn't evaluate
            
        truth = qrels[qid]
        predicted = runs[qid][:10]
        
        dcg = 0.0
        for i, docid in enumerate(predicted):
            rel = truth.get(docid, 0)
            if rel > 0:
                dcg += (math.pow(2, rel) - 1) / math.log2(i + 2)
                
        ideal_rels = sorted(truth.values(), reverse=True)[:10]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            if rel > 0:
                idcg += (math.pow(2, rel) - 1) / math.log2(i + 2)
                
        if idcg > 0:
            ndcg_list.append(dcg / idcg)
        else:
            ndcg_list.append(0.0)
            
    return sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0

def main():
    data_path = download_scifact()
    corpus, doc_text_to_id, queries, qrels = load_data(data_path)
    corpus_docs = list(corpus.values())
    
    print(f"Loaded {len(corpus_docs)} docs, {len(queries)} queries, {len(qrels)} qrels")
    
    # ── 1. Init SOTA Pipeline Components ──
    print("\\nInitializing Search Engine...")
    t0 = time.perf_counter()
    bm25 = BM25Plus(corpus_docs)
    print(f"BM25 Indexed in {(time.perf_counter() - t0)*1000:.2f}ms")
    
    print("Loading ONNX Jina-Reranker-Tiny via EmbedAnything (126 MB, no PyTorch)...")
    bge_model = embed_anything.Reranker.from_pretrained("llmware/jina-reranker-tiny-onnx")
    
    # Only test a slice for fast local execution (first 50)
    test_qids = [qid for qid in queries if qid in qrels][:50]
    
    # ── Benches ──
    benches = [
        {"name": "BM25+ (Baseline)", "use_fuzzy": False, "use_reranker": False},
        {"name": "Stage 2: BM25+ & Fuzzy", "use_fuzzy": True, "use_reranker": False},
        {"name": "Stage 3: BM25+ & Reranker", "use_fuzzy": False, "use_reranker": True},
        {"name": "SOTA: BM25+ & Fuzzy & Reranker", "use_fuzzy": True, "use_reranker": True},
    ]
    
    results = {}
    
    for bench in benches:
        name = bench["name"]
        print(f"\\nEvaluating {name}...")
        
        runs = {}
        total_retrieval_time = 0.0
        
        for qid in tqdm(test_qids, desc="Retrieving"):
            query_text = queries[qid]
            
            t0 = time.perf_counter()
            
            # Build query chain
            query_builder = bm25
            
            if bench["use_reranker"]:
                # Request 40 fast candidates from BM25+ to feed into the reranker
                query_builder = query_builder.rerank(bge_model, top_k=10)
                
            # Execute
            if bench["use_fuzzy"]:
                res = query_builder.match(query_text, method="get_top_n_fuzzy", n=40 if bench["use_reranker"] else 10)
            else:
                res = query_builder.match(query_text, n=40 if bench["use_reranker"] else 10)
                
            total_retrieval_time += (time.perf_counter() - t0) * 1000
            
            # Map back to doc IDs
            runs[qid] = [doc_text_to_id[doc] for doc, score in res if doc in doc_text_to_id]
            
        avg_latency = total_retrieval_time / len(test_qids)
        ndcg = evaluate_ndcg_at_10(qrels, runs)
        results[name] = {"latency": avg_latency, "ndcg": ndcg}
        print(f"Result: {name} | Latency: {avg_latency:.2f}ms/query | nDCG@10: {ndcg:.4f}")

    # Plot
    names = list(results.keys())
    latencies = [results[n]["latency"] for n in names]
    ndcgs = [results[n]["ndcg"] for n in names]
    
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i in range(len(names)):
        plt.scatter(latencies[i], ndcgs[i], s=150, c=colors[i], label=names[i], alpha=0.8)
        
    for i, name in enumerate(names):
        plt.annotate(name, (latencies[i], ndcgs[i]), xytext=(10, -5), 
                     textcoords='offset points', fontsize=12)

    plt.xlabel('Query Latency (ms)', fontsize=14)
    plt.ylabel('nDCG@10', fontsize=14)
    plt.title('Rustfuzz Pipeline Benchmarks on SciFact', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pipeline_sota_benchmark.png', dpi=300)
    print("\\nSaved benchmark plot to pipeline_sota_benchmark.png")

if __name__ == "__main__":
    main()
