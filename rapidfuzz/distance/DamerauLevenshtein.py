"""rapidfuzz.distance.DamerauLevenshtein"""

from __future__ import annotations

from rapidfuzz._rapidfuzz import (
    damerau_levenshtein_distance as distance,
    damerau_levenshtein_normalized_distance as normalized_distance,
    damerau_levenshtein_normalized_similarity as normalized_similarity,
    damerau_levenshtein_similarity as similarity,
)

__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
]
