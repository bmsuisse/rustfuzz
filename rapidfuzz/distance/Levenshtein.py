"""rapidfuzz.distance.Levenshtein"""

from __future__ import annotations

from rapidfuzz._rapidfuzz import (
    levenshtein_distance as distance,
    levenshtein_editops as editops,
    levenshtein_normalized_distance as normalized_distance,
    levenshtein_normalized_similarity as normalized_similarity,
    levenshtein_opcodes as opcodes,
    levenshtein_similarity as similarity,
)

__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
    "editops",
    "opcodes",
]
