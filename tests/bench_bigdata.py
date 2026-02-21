import time

import rapidfuzz.process as rf_process

import rustfuzz.process as rust_process
from rustfuzz.search import BM25


def generate_big_corpus(size=1_000_000):
    print(f"Generating {size} rows corpus...")
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

def run_benchmarks():
    corpus = generate_big_corpus()
    query = "apple macbook m3 max 64gb"

    # 1. Rapidfuzz process.extract (Sequential)
    print("\n--- 1. RapidFuzz Extract (Sequential) ---")
    t0 = time.time()
    res1 = rf_process.extract(query, corpus, limit=10)
    t1 = time.time()
    print(f"RapidFuzz process.extract: {(t1 - t0)*1000:.2f} ms")

    # 2. Rustfuzz process.extract (Sequential/Parallel)
    print("\n--- 2. RustFuzz Extract ---")
    t0 = time.time()
    res2 = rust_process.extract(query, corpus, limit=10)
    t1 = time.time()
    print(f"RustFuzz process.extract: {(t1 - t0)*1000:.2f} ms")
    print("Top match:", res2[0] if res2 else None)

    # 3. Rustfuzz BM25 Hybrid Pipeline
    print("\n--- 3. RustFuzz BM25 Hybrid Pipeline ---")
    t0 = time.time()
    index = BM25(corpus)
    t1 = time.time()
    print(f"Index build time: {(t1 - t0)*1000:.2f} ms")

    t0 = time.time()
    # Find top 10 using BM25, fallback to fuzzy if needed
    res3 = index.get_top_n_fuzzy(query, n=10, bm25_candidates=500, fuzzy_weight=0.3)
    t1 = time.time()
    print(f"BM25 + Semantic Hybrid Search: {(t1 - t0)*1000:.2f} ms")
    print("Top hybrid match:", res3[0] if res3 else None)

if __name__ == "__main__":
    run_benchmarks()
