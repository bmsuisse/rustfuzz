"""rustfuzz.distance.Hamming"""

from __future__ import annotations

from rustfuzz._rustfuzz import (
    hamming_distance as distance,
    hamming_editops as editops,
    hamming_normalized_distance as normalized_distance,
    hamming_normalized_similarity as normalized_similarity,
    hamming_opcodes as opcodes,
    hamming_similarity as similarity,
)

__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
    "editops",
    "opcodes",
]
