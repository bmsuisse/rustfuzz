"""Bug Reproduction: HybridSearch.search() returns duplicate documents.

When querying a HybridSearch index, the same document (identified by its metadata)
is returned multiple times in the results. Each search hit should be a unique
document — duplicates inflate scores in downstream aggregation and produce
misleading results in user-facing tables.

Reproduction:
    uv run pytest tests/test_hybrid_search_duplicates.py -v

Observed at scale in production with ~370k documents, where a single article_id
appeared up to 19 times in a top-500 result set. This minimal reproduction uses
1000 documents and demonstrates the same behaviour.
"""

from __future__ import annotations

from collections import Counter

from rustfuzz.search import Document, HybridSearch


def _build_index(n: int = 1000) -> HybridSearch:
    """Build an index with *n* unique documents, ~1/7 containing 'Badewanne'."""
    prefixes = [
        "Badewanne",
        "Dusche",
        "Waschtisch",
        "WC",
        "Armatur",
        "Siphon",
        "Ventil",
    ]
    suffixes = ["Classic", "Modern", "Premium", "Standard", "Eco", "Pro", "Plus"]
    materials = ["Stahl", "Acryl", "Keramik", "Mineralguss", "Emaille"]
    sizes = ["150x70", "170x75", "180x80", "160x70", "200x100", "90x90", "120x80"]

    docs: list[Document] = []
    for i in range(n):
        content = (
            f"{prefixes[i % len(prefixes)]} "
            f"{suffixes[i % len(suffixes)]} "
            f"{sizes[i % len(sizes)]} "
            f"{materials[i % len(materials)]} weiss"
        )
        docs.append(Document(content, {"id": str(i), "name": content}))
    return HybridSearch(docs)


class TestDuplicateResults:
    """Every result in a search() call should be a unique document."""

    def test_search_returns_unique_documents(self) -> None:
        """search('badewanne', n=100) should return 100 unique document IDs."""
        idx = _build_index(1000)
        results = idx.search("badewanne", n=100)
        ids = [r[2]["id"] for r in results]
        assert len(ids) == len(set(ids)), (
            f"Expected {len(ids)} unique IDs but got {len(set(ids))}. "
            f"Duplicates: {[(k, v) for k, v in Counter(ids).items() if v > 1]}"
        )

    def test_search_returns_unique_documents_small(self) -> None:
        """Even a small n=20 request should have no duplicates."""
        idx = _build_index(1000)
        results = idx.search("badewanne", n=20)
        ids = [r[2]["id"] for r in results]
        assert len(ids) == len(set(ids)), (
            f"Expected {len(ids)} unique IDs but got {len(set(ids))}. "
            f"Duplicates: {[(k, v) for k, v in Counter(ids).items() if v > 1]}"
        )

    def test_match_returns_unique_documents(self) -> None:
        """match() via filter() should also return unique documents."""
        idx = _build_index(1000)
        # Use match via a trivial filter that matches everything
        # (if filter() is available with a passthrough)
        results = idx.search("waschtisch", n=50)
        ids = [r[2]["id"] for r in results]
        assert len(ids) == len(set(ids)), (
            f"Expected {len(ids)} unique IDs but got {len(set(ids))}. "
            f"Duplicates: {[(k, v) for k, v in Counter(ids).items() if v > 1]}"
        )

    def test_large_scale_duplicates(self) -> None:
        """Reproduce the production scenario: 10k docs, n=500."""
        idx = _build_index(10_000)
        results = idx.search("badewanne", n=500)
        ids = [r[2]["id"] for r in results]
        unique_count = len(set(ids))
        assert len(ids) == unique_count, (
            f"At scale (10k docs, n=500): expected {len(ids)} unique IDs "
            f"but got {unique_count}. Top duplicates: "
            f"{Counter(ids).most_common(5)}"
        )


def _build_corpus(n: int = 1000) -> list[str]:
    """Build a corpus with duplicate texts (due to modular cycling)."""
    prefixes = [
        "Badewanne",
        "Dusche",
        "Waschtisch",
        "WC",
        "Armatur",
        "Siphon",
        "Ventil",
    ]
    suffixes = ["Classic", "Modern", "Premium", "Standard", "Eco", "Pro", "Plus"]
    materials = ["Stahl", "Acryl", "Keramik", "Mineralguss", "Emaille"]
    sizes = ["150x70", "170x75", "180x80", "160x70", "200x100", "90x90", "120x80"]
    return [
        f"{prefixes[i % 7]} {suffixes[i % 7]} {sizes[i % 7]} {materials[i % 5]} weiss"
        for i in range(n)
    ]


class TestBM25VariantDuplicates:
    """All BM25 variant get_top_n_rrf methods should return unique documents."""

    def _assert_no_dupes(self, results: list[tuple[str, float]], label: str) -> None:
        texts = [r[0] for r in results]
        assert len(texts) == len(set(texts)), (
            f"{label}: expected {len(texts)} unique, got {len(set(texts))}. "
            f"Top dupes: {Counter(texts).most_common(3)}"
        )

    def test_bm25_rrf_no_duplicates(self) -> None:
        from rustfuzz.search import BM25

        corpus = _build_corpus(1000)
        idx = BM25(corpus)
        results = idx.get_top_n_rrf("badewanne", n=100)
        self._assert_no_dupes(results, "BM25.get_top_n_rrf")

    def test_bm25l_rrf_no_duplicates(self) -> None:
        from rustfuzz.search import BM25L

        corpus = _build_corpus(1000)
        idx = BM25L(corpus)
        results = idx.get_top_n_rrf("badewanne", n=100)
        self._assert_no_dupes(results, "BM25L.get_top_n_rrf")

    def test_bm25plus_rrf_no_duplicates(self) -> None:
        from rustfuzz.search import BM25Plus

        corpus = _build_corpus(1000)
        idx = BM25Plus(corpus)
        results = idx.get_top_n_rrf("badewanne", n=100)
        self._assert_no_dupes(results, "BM25Plus.get_top_n_rrf")

    def test_bm25t_rrf_no_duplicates(self) -> None:
        from rustfuzz.search import BM25T

        corpus = _build_corpus(1000)
        idx = BM25T(corpus)
        results = idx.get_top_n_rrf("badewanne", n=100)
        self._assert_no_dupes(results, "BM25T.get_top_n_rrf")

    def test_bm25_rrf_large_scale(self) -> None:
        from rustfuzz.search import BM25

        corpus = _build_corpus(10_000)
        idx = BM25(corpus)
        results = idx.get_top_n_rrf("badewanne", n=500)
        self._assert_no_dupes(results, "BM25.get_top_n_rrf@10k")
