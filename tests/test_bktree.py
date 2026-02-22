from rustfuzz import process


def test_bktree_dedupe() -> None:
    # 'apple' and 'apples' are distance 1.
    # 'banana' is very far.
    # 'appl' is distance 1 from 'apple'.
    choices = ["apple", "apples", "banana", "appl", "cherry"]

    # max_edits=1 means strings within 1 edit are considered duplicates.
    # "apples" and "appl" get grouped under "apple".
    unique = process.dedupe(choices, max_edits=1)

    assert "apple" in unique
    assert "banana" in unique
    assert "cherry" in unique
    assert "apples" not in unique
    assert "appl" not in unique


def test_bktree_dedupe_exact() -> None:
    choices = ["foo", "bar", "foo", "baz"]
    # Exact deduplication (max_edits=0)
    unique = process.dedupe(choices, max_edits=0)
    assert sorted(unique) == ["bar", "baz", "foo"]
