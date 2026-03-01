"""
bench_crush_ndcg.py — Systematically optimize nDCG@10 on SciFact.

Tests every optimization lever in rustfuzz to maximize retrieval quality:
  1. Text preprocessing (Unicode NFKD, lowercasing, punctuation strip)
  2. BM25 parameter tuning (k1, b, delta)
  3. Candidate pool expansion (n=40 → n=100 → n=200)
  4. Multi-method retrieval fusion (BM25 + Phrase + Fuzzy → rerank union)
  5. Score interpolation (blend BM25 + reranker scores)
"""

import os
import json
import time
import math
import re
import unicodedata
import requests
from typing import Dict, List, Any, Callable
from tqdm import tqdm

from rustfuzz.search import BM25Plus, Reranker
import embed_anything


# ── Text Preprocessing ──────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Unicode NFKD normalization + lowercasing + whitespace collapse."""
    # NFKD: fold ligatures, decompose characters
    text = unicodedata.normalize("NFKD", text)
    # Strip combining marks (accents)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Lowercase
    text = text.lower()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_keep_punct(text: str) -> str:
    """NFKD + lowercase + whitespace — keeps punctuation for phrase matching."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_strip_punct(text: str) -> str:
    """NFKD + lowercase + strip punctuation + whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    # Strip punctuation but keep hyphens in words
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Data Loading ────────────────────────────────────────────────

def download_scifact():
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
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(".")
    return "scifact"


def load_data(data_path: str, preprocess: Callable[[str], str] | None = None):
    corpus: dict[str, str] = {}
    doc_text_to_id: dict[str, int] = {}
    with open(os.path.join(data_path, "corpus.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            doc_text = f"{obj.get('title', '')} {obj.get('text', '')}"
            if preprocess:
                doc_text = preprocess(doc_text)
            corpus[obj["_id"]] = doc_text
            doc_text_to_id[doc_text] = int(obj["_id"])

    queries: dict[str, str] = {}
    with open(os.path.join(data_path, "queries.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            q_text = obj["text"]
            if preprocess:
                q_text = preprocess(q_text)
            queries[obj["_id"]] = q_text

    qrels: dict[str, dict[int, int]] = {}
    with open(os.path.join(data_path, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            qid, docid, score = line.strip().split("\t")
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][int(docid)] = int(score)

    return corpus, doc_text_to_id, queries, qrels


def evaluate_ndcg_at_10(
    qrels: Dict[str, Dict[int, int]], runs: Dict[str, List[int]]
) -> float:
    ndcg_list = []
    for qid in qrels:
        if qid not in runs:
            continue
        truth = qrels[qid]
        predicted = runs[qid][:10]
        dcg = 0.0
        for i, docid in enumerate(predicted):
            rel = truth.get(docid, 0)
            if rel > 0:
                dcg += (math.pow(2, rel) - 1) / math.log2(i + 2)
        ideal_rels = sorted(truth.values(), reverse=True)[:10]
        idcg = sum(
            (math.pow(2, rel) - 1) / math.log2(i + 2)
            for i, rel in enumerate(ideal_rels)
            if rel > 0
        )
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0


# ── Retrieval strategies ────────────────────────────────────────

def retrieve_bm25(bm25: BM25Plus, query: str, n: int = 10) -> list[tuple[str, float]]:
    return bm25.get_top_n(query, n)


def retrieve_phrase(bm25: BM25Plus, query: str, n: int = 10) -> list[tuple[str, float]]:
    return bm25.get_top_n_phrase(query, n, proximity_window=4, phrase_boost=2.5)


def retrieve_fuzzy(bm25: BM25Plus, query: str, n: int = 10) -> list[tuple[str, float]]:
    return bm25.get_top_n_fuzzy(query, n, bm25_candidates=100, fuzzy_weight=0.3)


def retrieve_rrf(bm25: BM25Plus, query: str, n: int = 10) -> list[tuple[str, float]]:
    return bm25.get_top_n_rrf(query, n, bm25_candidates=150, rrf_k=60)


def fuse_rrf(
    *result_lists: list[tuple[str, float]], k: int = 60, n: int = 100
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of multiple result lists."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (doc, _score) in enumerate(results):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return sorted_docs


# ── Main Experiment Runner ──────────────────────────────────────

def run_experiment(
    name: str,
    bm25: BM25Plus,
    reranker_model: Any,
    queries: dict[str, str],
    qrels: dict[str, dict[int, int]],
    test_qids: list[str],
    doc_text_to_id: dict[str, int],
    retrieval_fn: Callable,
    candidate_n: int = 40,
    rerank_top_k: int = 10,
    use_reranker: bool = True,
) -> dict[str, Any]:
    """Run a single experiment and return results."""
    runs: dict[str, list[int]] = {}
    total_time = 0.0
    reranker = Reranker(reranker_model) if use_reranker else None

    for qid in tqdm(test_qids, desc=name[:30], leave=False):
        query_text = queries[qid]
        t0 = time.perf_counter()

        # Retrieve candidates
        candidates = retrieval_fn(bm25, query_text, n=candidate_n)

        # Rerank if applicable
        if reranker and candidates:
            candidates = reranker.rerank(query_text, candidates, top_k=rerank_top_k)

        total_time += (time.perf_counter() - t0) * 1000

        # Map to doc IDs
        runs[qid] = [doc_text_to_id[doc] for doc, _ in candidates if doc in doc_text_to_id]

    avg_latency = total_time / len(test_qids)
    ndcg = evaluate_ndcg_at_10(qrels, runs)
    return {"name": name, "latency": avg_latency, "ndcg": ndcg}


def main():
    data_path = download_scifact()

    # Load reranker once
    print("Loading ONNX Jina-Reranker-Tiny (126 MB)...")
    reranker_model = embed_anything.Reranker.from_pretrained("llmware/jina-reranker-tiny-onnx")

    all_results: list[dict] = []

    # ── Experiment Group 1: Text Preprocessing ──────────────────
    preprocess_variants = [
        ("Raw text", None),
        ("NFKD + lowercase", normalize_text_keep_punct),
        ("NFKD + lowercase + strip punct", normalize_text_strip_punct),
    ]

    for pp_name, pp_fn in preprocess_variants:
        corpus, doc_text_to_id, queries, qrels = load_data(data_path, preprocess=pp_fn)
        corpus_docs = list(corpus.values())
        test_qids = [qid for qid in queries if qid in qrels][:50]

        print(f"\n── Preprocessing: {pp_name} ──")

        # Baseline BM25+ (default params)
        bm25 = BM25Plus(corpus_docs)

        # Without reranker
        r = run_experiment(
            f"BM25+ [{pp_name}]", bm25, reranker_model, queries, qrels,
            test_qids, doc_text_to_id, retrieve_bm25,
            candidate_n=10, use_reranker=False,
        )
        all_results.append(r)
        print(f"  {r['name']}: nDCG={r['ndcg']:.4f} lat={r['latency']:.1f}ms")

        # With reranker (n=40)
        r = run_experiment(
            f"BM25+ → Rerank [{pp_name}]", bm25, reranker_model, queries, qrels,
            test_qids, doc_text_to_id, retrieve_bm25,
            candidate_n=40, use_reranker=True,
        )
        all_results.append(r)
        print(f"  {r['name']}: nDCG={r['ndcg']:.4f} lat={r['latency']:.1f}ms")

    # ── Experiment Group 2: BM25 Parameter Tuning ───────────────
    # Use best preprocessing from group 1 (we'll use NFKD+lowercase)
    corpus, doc_text_to_id, queries, qrels = load_data(data_path, preprocess=normalize_text_keep_punct)
    corpus_docs = list(corpus.values())
    test_qids = [qid for qid in queries if qid in qrels][:50]

    print("\n── BM25 Parameter Tuning ──")
    param_variants = [
        ("k1=1.5 b=0.75 (default)", 1.5, 0.75, 1.0),
        ("k1=1.2 b=0.75", 1.2, 0.75, 1.0),
        ("k1=1.0 b=0.75", 1.0, 0.75, 1.0),
        ("k1=1.2 b=0.5", 1.2, 0.5, 1.0),
        ("k1=1.2 b=0.4", 1.2, 0.4, 1.0),
        ("k1=1.5 b=0.5 delta=0.5", 1.5, 0.5, 0.5),
        ("k1=1.5 b=0.5 delta=1.5", 1.5, 0.5, 1.5),
    ]

    for p_name, k1, b, delta in param_variants:
        bm25 = BM25Plus(corpus_docs, k1=k1, b=b, delta=delta)
        r = run_experiment(
            f"Tuned {p_name} → Rerank", bm25, reranker_model, queries, qrels,
            test_qids, doc_text_to_id, retrieve_bm25,
            candidate_n=40, use_reranker=True,
        )
        all_results.append(r)
        print(f"  {r['name']}: nDCG={r['ndcg']:.4f} lat={r['latency']:.1f}ms")

    # ── Experiment Group 3: Candidate Pool Size ─────────────────
    print("\n── Candidate Pool Expansion ──")
    best_bm25 = BM25Plus(corpus_docs, k1=1.2, b=0.75)  # will refine based on group 2

    for pool_n in [40, 80, 120, 200]:
        r = run_experiment(
            f"Pool n={pool_n} → Rerank", best_bm25, reranker_model, queries, qrels,
            test_qids, doc_text_to_id, retrieve_bm25,
            candidate_n=pool_n, use_reranker=True,
        )
        all_results.append(r)
        print(f"  {r['name']}: nDCG={r['ndcg']:.4f} lat={r['latency']:.1f}ms")

    # ── Experiment Group 4: Retrieval Methods ───────────────────
    print("\n── Retrieval Method Variants ──")
    methods = [
        ("BM25 → Rerank", retrieve_bm25),
        ("Phrase → Rerank", retrieve_phrase),
        ("Fuzzy → Rerank", retrieve_fuzzy),
        ("RRF → Rerank", retrieve_rrf),
    ]
    for m_name, m_fn in methods:
        r = run_experiment(
            m_name, best_bm25, reranker_model, queries, qrels,
            test_qids, doc_text_to_id, m_fn,
            candidate_n=100, use_reranker=True,
        )
        all_results.append(r)
        print(f"  {r['name']}: nDCG={r['ndcg']:.4f} lat={r['latency']:.1f}ms")

    # ── Experiment Group 5: Multi-Method Fusion → Rerank ────────
    print("\n── Multi-Method Fusion → Rerank ──")
    reranker = Reranker(reranker_model)
    runs_fusion: dict[str, list[int]] = {}
    total_t = 0.0
    for qid in tqdm(test_qids, desc="Fusion+Rerank", leave=False):
        q = queries[qid]
        t0 = time.perf_counter()
        # Get candidates from multiple methods
        bm25_res = best_bm25.get_top_n(q, 80)
        phrase_res = best_bm25.get_top_n_phrase(q, 80, proximity_window=4, phrase_boost=2.5)
        fuzzy_res = best_bm25.get_top_n_fuzzy(q, 80, bm25_candidates=100, fuzzy_weight=0.3)
        # Fuse with RRF
        fused = fuse_rrf(bm25_res, phrase_res, fuzzy_res, k=60, n=120)
        # Rerank the fused set
        reranked = reranker.rerank(q, fused, top_k=10)
        total_t += (time.perf_counter() - t0) * 1000
        runs_fusion[qid] = [doc_text_to_id[d] for d, _ in reranked if d in doc_text_to_id]

    ndcg_fusion = evaluate_ndcg_at_10(qrels, runs_fusion)
    lat_fusion = total_t / len(test_qids)
    all_results.append({"name": "⭐ Fusion(BM25+Phrase+Fuzzy) → Rerank", "latency": lat_fusion, "ndcg": ndcg_fusion})
    print(f"  ⭐ Fusion → Rerank: nDCG={ndcg_fusion:.4f} lat={lat_fusion:.1f}ms")

    # ── Final Summary ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL RESULTS — Sorted by nDCG@10")
    print("=" * 80)
    all_results.sort(key=lambda x: x["ndcg"], reverse=True)
    for i, r in enumerate(all_results):
        marker = "🏆" if i == 0 else "  "
        print(f"{marker} nDCG={r['ndcg']:.4f} | {r['latency']:8.1f}ms | {r['name']}")


if __name__ == "__main__":
    main()
