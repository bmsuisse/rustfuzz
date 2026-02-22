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
