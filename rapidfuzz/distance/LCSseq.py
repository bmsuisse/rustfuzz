"""rapidfuzz.distance.LCSseq"""

from __future__ import annotations

from rapidfuzz._rapidfuzz import (
    lcs_seq_distance as distance,
    lcs_seq_editops as editops,
    lcs_seq_normalized_distance as normalized_distance,
    lcs_seq_normalized_similarity as normalized_similarity,
    lcs_seq_opcodes as opcodes,
    lcs_seq_similarity as similarity,
)

__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
    "editops",
    "opcodes",
]
