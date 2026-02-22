from rustfuzz.distance import NGram


def test_sorensen_dice():
    # Bigrams: "night" -> "ni", "ig", "gh", "ht" (4)
    # Bigrams: "nacht" -> "na", "ac", "ch", "ht" (4)
    # Intersection: "ht" (1)
    # SD = 2 * 1 / (4 + 4) = 0.25
    assert NGram.sorensen_dice("night", "nacht", n=2) == 0.25

    # Identical
    assert NGram.sorensen_dice("apple", "apple") == 1.0


def test_jaccard():
    # Intersection: "ht" (1)
    # Union: 4 + 4 - 1 = 7
    # Jaccard = 1 / 7 = 0.142857...
    assert abs(NGram.jaccard("night", "nacht", n=2) - 0.142857) < 1e-4

    # Empty
    assert NGram.jaccard("", "") == 1.0
