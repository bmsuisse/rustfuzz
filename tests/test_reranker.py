from rustfuzz.search import BM25, Reranker


class DummyCrossEncoder:
    def predict(self, pairs):
        # simple mock: length of shared words
        res = []
        for q, ctx in pairs:
            q_words = set(q.lower().split())
            c_words = set(ctx.lower().split())
            score = len(q_words.intersection(c_words))
            res.append(float(score))
        return res


def test_reranker_standalone():
    reranker = Reranker(DummyCrossEncoder())
    results = [
        ("the quick brown fox", 1.0),
        ("a lazy dog", 0.5),
        ("the fast brown fox jumps", 0.8),
    ]

    # "brown fox" shares 2 words with doc 0 and doc 2, but doc 1 shares 0
    reranked = reranker.rerank("brown fox", results, top_k=2)

    assert len(reranked) == 2
    # both doc 0 and 2 should beat doc 1
    assert reranked[0][1] == 2.0
    assert reranked[1][1] == 2.0
    assert "lazy dog" not in [r[0] for r in reranked]


def test_reranker_fluent_chain():
    docs = [
        "apple iphone 15 pro max",
        "samsung galaxy s24 ultra",
        "google pixel 8 pro",
        "apple macbook pro m3",
    ]

    bm25 = BM25(docs)

    # Base search matches 3 things containing "pro"
    base_res = bm25.match("pro", n=10)
    assert len(base_res) == 3

    # Rerank forces it to use the dummy cross encoder
    chain_res = bm25.rerank(DummyCrossEncoder(), top_k=1).match("apple pro", n=10)

    assert len(chain_res) == 1
    # "apple pro" shares 2 words with "apple macbook pro m3" and "apple iphone 15 pro max"
    # The dummy cross encoder gives both a score of 2.0
    assert chain_res[0][1] == 2.0
    assert "apple" in chain_res[0][0]


def test_reranker_with_fuzzy_fallback():
    docs = [
        "apple iphone 15 pro max",
        "samsung galaxy s24 ultra",
    ]
    
    # We'll make our dummy cross encoder slightly smarter for this test
    # so it grants 1 point if "iphn" is in the query and "iphone" in the context
    class SmarterDummy(DummyCrossEncoder):
        def predict(self, pairs):
            res = super().predict(pairs)
            for i, (q, ctx) in enumerate(pairs):
                if "iphn" in q.lower() and "iphone" in ctx.lower():
                    res[i] += 1.0
            return res

    bm25 = BM25(docs)

    # Misspelled query "iphn" -> won't exact match, but fuzzy will catch it.
    # We want to ensure the reranker receives the fuzzy output cleanly.
    # Let's see what the base search engine returns
    base_res = bm25.get_top_n_fuzzy("iphn", n=10)
    print(f"Base fuzzy res: {base_res}")

    chain_res = (
        bm25.rerank(SmarterDummy(), top_k=1)
        .search("iphn", method="get_top_n_fuzzy", n=10)
        .collect()
    )
    
    # Debug what chain produces
    print(f"Chain res: {chain_res}")

    assert len(chain_res) == 1
    # SmarterDummy gives it 1.0 points
    assert chain_res[0][1] == 1.0
    assert "apple" in chain_res[0][0]
