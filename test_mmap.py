import os

from rustfuzz.search import BM25, MmapBM25

corpus = ["hello world", "test document", "another one"]
bm25 = BM25(corpus)

path = "/tmp/bm25_mmap_test"
if not os.path.exists(path):
    os.makedirs(path)

print("Building MmapBM25...")
MmapBM25.build(path, bm25)
print("Build OK")

print("Loading MmapBM25...")
try:
    m = MmapBM25.load(path)
    print("Load OK")
    print(m.get_top_n("test", n=2))
except Exception as e:
    print(f"Failed: {e}")
