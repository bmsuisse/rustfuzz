import os
import json
import tarfile
import time
import requests
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from rustfuzz.search import BM25, BM25L, BM25Plus, BM25T


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
            # Combine title and text for BM25
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
        next(f) # skip header
        for line in f:
            qid, docid, score = line.strip().split('\t')
            if qid not in qrels:
                qrels[qid] = {}
            # docids are stored as ints because _id in corpus is an int in SciFact
            qrels[qid][int(docid)] = int(score)
            
    return corpus, doc_text_to_id, queries, qrels


def evaluate_ndcg_at_10(qrels: Dict[str, Dict[str, int]], runs: Dict[str, List[str]]) -> float:
    """Calculate average nDCG@10 across all queries."""
    ndcg_list = []
    import math
    
    for qid in qrels:
        if qid not in runs:
            ndcg_list.append(0.0)
            continue
            
        truth = qrels[qid]
        predicted = runs[qid][:10]
        
        # Calculate DCG
        dcg = 0.0
        for i, docid in enumerate(predicted):
            rel = truth.get(docid, 0)
            if rel > 0:
                dcg += (math.pow(2, rel) - 1) / math.log2(i + 2)
                
        # Calculate IDCG
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

def benchmark_variant(variant_class, name, corpus_docs, doc_text_to_id, queries, qrels):
    print(f"\\nEvaluating {name}...")
    
    # Indexing
    t0 = time.perf_counter()
    index = variant_class(corpus_docs)
    indexing_time = (time.perf_counter() - t0) * 1000  # ms
    
    # Retrieval
    runs = {}
    total_retrieval_time = 0.0
    valid_queries = 0
    
    # Only test queries with qrels
    test_qids = [qid for qid in queries if qid in qrels]
    
    for qid in tqdm(test_qids, desc="Retrieving"):
        query_text = queries[qid]
        valid_queries += 1
        
        t0 = time.perf_counter()
        # rustfuzz BM25 implementation already has a fast top-N heap which evaluates par_scores
        results = index.get_top_n(query_text, n=10)
        total_retrieval_time += (time.perf_counter() - t0) * 1000 # ms
        
        # map retrieved doc text back to doc id
        runs[qid] = [doc_text_to_id[doc] for doc, score in results if doc in doc_text_to_id]
        
    avg_latency = total_retrieval_time / valid_queries if valid_queries > 0 else 0
    ndcg = evaluate_ndcg_at_10(qrels, runs)
    
    print(f"{name} - Indexing: {indexing_time:.2f}ms | Latency: {avg_latency:.2f}ms/query | nDCG@10: {ndcg:.4f}")
    return avg_latency, ndcg


def main():
    data_path = download_scifact()
    corpus, doc_text_to_id, queries, qrels = load_data(data_path)
    
    corpus_docs = list(corpus.values())
    
    print(f"Loaded {len(corpus_docs)} docs, {len(queries)} queries, {len(qrels)} qrels")
    
    variants = [
        (BM25, "BM25Okapi"),
        (BM25L, "BM25L"),
        (BM25Plus, "BM25+"),
        (BM25T, "BM25T")
    ]
    
    results = {}
    for variant_class, name in variants:
        latency, ndcg = benchmark_variant(variant_class, name, corpus_docs, doc_text_to_id, queries, qrels)
        results[name] = {"latency": latency, "ndcg": ndcg}
        
    # Plotting
    names = list(results.keys())
    latencies = [results[name]["latency"] for name in names]
    ndcgs = [results[name]["ndcg"] for name in names]
    
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i in range(len(names)):
        plt.scatter(latencies[i], ndcgs[i], s=150, c=colors[i], label=names[i], alpha=0.8)
        
    for i, name in enumerate(names):
        plt.annotate(name, (latencies[i], ndcgs[i]), xytext=(10, -5), 
                     textcoords='offset points', fontsize=12)

    plt.xlabel('Query Latency (ms)', fontsize=14)
    plt.ylabel('nDCG@10', fontsize=14)
    plt.title('BM25 Variants: Retrieval Quality vs Speed on SciFact', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('bm25_variants_benchmark.png', dpi=300)
    print("\\nSaved benchmark plot to bm25_variants_benchmark.png")
    
if __name__ == "__main__":
    main()
