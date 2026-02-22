import time
from rapidfuzz import process as rf_process
import rustfuzz.process as rust_process
from examples_realworld_benchmark import load_corpus

def timeit_ms(func, *args, **kwargs):
    t0 = time.perf_counter_ns()
    res = func(*args, **kwargs)
    t1 = time.perf_counter_ns()
    return res, (t1 - t0) / 1_000_000

corpus = load_corpus(target=30_000)
query = "San Fransisco downtown"

_, ms_rf = timeit_ms(rf_process.extract, query, corpus, limit=10)
_, ms_rust = timeit_ms(rust_process.extract, query, corpus, limit=10)

print(f"RapidFuzz: {ms_rf:.2f}ms")
print(f"RustFuzz:  {ms_rust:.2f}ms")
