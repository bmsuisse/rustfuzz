"""rapidfuzz.distance.JaroWinkler"""

from __future__ import annotations

from rapidfuzz._rapidfuzz import (
    jaro_winkler_distance as distance,
    jaro_winkler_normalized_distance as normalized_distance,
    jaro_winkler_normalized_similarity as normalized_similarity,
    jaro_winkler_similarity as similarity,
)

__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
]
