from rustfuzz.distance import Gotoh


def test_gotoh_basic():
    # Identical
    assert Gotoh.distance("apple", "apple") == 0.0

    # 1 replace (cost=1.0)
    assert Gotoh.distance("apple", "appxe") == 1.0

    # Gap extension: "a" vs "abcde"
    # Open=1.0, extend=0.5 -> cost = 1.0 + 4 * 0.5 = 3.0
    assert Gotoh.distance("a", "abcde") == 3.0

def test_gotoh_kwargs():
    # gap open=2.0, extend=1.0
    # "a" vs "ab" -> 1 gap of length 1 -> open(2.0) + 1*1.0 = 3.0
    assert Gotoh.distance("a", "ab", open_penalty=2.0, extend_penalty=1.0) == 3.0


def test_gotoh_similarity():
    # Identical -> max similarity
    sim = Gotoh.similarity("apple", "apple")
    assert sim > 0

    # Different -> lower similarity
    sim_diff = Gotoh.similarity("apple", "banana")
    assert sim_diff < sim

    # Empty + Empty -> 0 similarity (nothing to compare)
    assert Gotoh.similarity("", "") == 0.0


def test_gotoh_normalized():
    # Identical -> normalized_similarity = 1.0, normalized_distance = 0.0
    assert Gotoh.normalized_similarity("apple", "apple") == 1.0
    assert Gotoh.normalized_distance("apple", "apple") == 0.0

    # Different -> between 0 and 1
    nd = Gotoh.normalized_distance("apple", "banana")
    ns = Gotoh.normalized_similarity("apple", "banana")
    assert 0.0 < nd <= 1.0
    assert 0.0 <= ns < 1.0
    assert abs(nd + ns - 1.0) < 1e-9

    # Empty + Empty -> distance=0, similarity=1
    assert Gotoh.normalized_similarity("", "") == 1.0
    assert Gotoh.normalized_distance("", "") == 0.0
