from rustfuzz import process


def test_bktree_dedupe():
    # 'apple' and 'apples' are distance 1.
    # 'banana' is very far.
    # 'appl' is distance 1 from 'apple'.
    choices = ["apple", "apples", "banana", "appl", "cherry"]

    # threshold=1 means diff of 1 is duplicate. "apples" and "appl" get grouped into "apple".
    unique = process.dedupe(choices, threshold=1)

    assert "apple" in unique
    assert "banana" in unique
    assert "cherry" in unique
    assert "apples" not in unique
    assert "appl" not in unique


def test_bktree_dedupe_exact():
    choices = ["foo", "bar", "foo", "baz"]
    # Exact deduplication
    unique = process.dedupe(choices, threshold=0)
    assert sorted(unique) == ["bar", "baz", "foo"]
