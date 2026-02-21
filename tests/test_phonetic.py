from rustfuzz.distance import Soundex


def test_soundex_basic():
    assert Soundex.encode("Smith") == "S530"
    assert Soundex.encode("Smythe") == "S530"
    assert Soundex.encode("Washington") == "W252"
    assert Soundex.encode("Lee") == "L000"
    assert Soundex.encode("Gutierrez") == "G362"


def test_soundex_distance():
    assert Soundex.distance("Smith", "Smythe") == 0
    assert Soundex.distance("Smith", "Washington") > 0
    assert Soundex.normalized_similarity("Smith", "Smythe") == 1.0
