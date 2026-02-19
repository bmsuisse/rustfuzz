"""rapidfuzz.distance.OSA"""

from __future__ import annotations

from rapidfuzz._rapidfuzz import (
    osa_distance as distance,
    osa_normalized_distance as normalized_distance,
    osa_normalized_similarity as normalized_similarity,
    osa_similarity as similarity,
)

__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
]
