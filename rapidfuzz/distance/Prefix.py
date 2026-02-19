"""rapidfuzz.distance.Prefix"""

from __future__ import annotations

from rapidfuzz._rapidfuzz import (
    prefix_distance as distance,
    prefix_normalized_distance as normalized_distance,
    prefix_normalized_similarity as normalized_similarity,
    prefix_similarity as similarity,
)

__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
]
