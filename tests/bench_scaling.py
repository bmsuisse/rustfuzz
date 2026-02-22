import time

import rapidfuzz.process as rf_process

import rustfuzz.process as rust_process
from rustfuzz.search import BM25


def generate_big_corpus(size):
    print("\n==============================")
    print(f"Generating {size:,} rows corpus...")
    print("==============================\n")
    base_strings = [
        "apple macbook pro m3 max 64gb",
        "apple macbook pro m2 16gb",
        "apple iphone 15 pro max titanium",
        "samsung galaxy s24 ultra",
        "sony playstation 5 slim edition",
        "nintendo switch oled mario kart bundle",
        "dell xps 15 laptop oled 32gb",
        "lenovo thinkpad x1 carbon gen 12",
        "asus rog zephyrus g14 gaming laptop",
        "microsoft surface pro 9 platinum",
    ]
    # Multiply and add some noise
    corpus = []
    for i in range(size):
        base = base_strings[i % len(base_strings)]
        # Add index to make it somewhat unique
        corpus.append(f"{base} variant ID-{i}")
    return corpus

def run_benchmarks(size):
    corpus = generate_big_corpus(size)
    query = "apple macbook m3 max 64gb"

    # 1. Rapidfuzz process.extract (Sequential)
    print("--- 1. RapidFuzz Extract WRatio (Sequential) ---")
    t0 = time.time()
    rf_process.extract(query, corpus, limit=10)
    t1 = time.time()
    print(f"RapidFuzz process.extract: {(t1 - t0)*1000:.2f} ms")

    # 2. Rustfuzz process.extract (Parallel via Rayon)
    print("\n--- 2. RustFuzz Extract WRatio (Parallel) ---")
    t0 = time.time()
    rust_process.extract(query, corpus, limit=10)
    t1 = time.time()
    print(f"RustFuzz process.extract: {(t1 - t0)*1000:.2f} ms")

    # 3. Rustfuzz BM25 Hybrid Pipeline
    print("\n--- 3. RustFuzz BM25 Hybrid Pipeline ---")
    t0 = time.time()
    index = BM25(corpus)
    t1 = time.time()
    print(f"Index build time: {(t1 - t0)*1000:.2f} ms")

    t0 = time.time()
    # Find top 10 using BM25, fallback to fuzzy if needed
    index.get_top_n_fuzzy(query, n=10, bm25_candidates=min(size // 100, 500), fuzzy_weight=0.3)
    t1 = time.time()
    print(f"BM25 + Semantic Hybrid Search: {(t1 - t0)*1000:.2f} ms")

if __name__ == "__main__":
    for size in [10_000, 100_000, 1_000_000]:
        run_benchmarks(size)
