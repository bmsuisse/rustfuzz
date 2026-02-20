"""rustfuzz.distance.Indel"""

from __future__ import annotations

from rustfuzz._rustfuzz import (
    indel_distance as distance,
    indel_editops as editops,
    indel_normalized_distance as normalized_distance,
    indel_normalized_similarity as normalized_similarity,
    indel_opcodes as opcodes,
    indel_similarity as similarity,
)

__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
    "editops",
    "opcodes",
]
