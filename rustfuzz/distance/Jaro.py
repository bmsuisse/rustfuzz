"""rustfuzz.distance.Jaro"""

from __future__ import annotations

from rustfuzz._rustfuzz import (
    jaro_distance as distance,
)
from rustfuzz._rustfuzz import (
    jaro_normalized_distance as normalized_distance,
)
from rustfuzz._rustfuzz import (
    jaro_normalized_similarity as normalized_similarity,
)
from rustfuzz._rustfuzz import (
    jaro_similarity as similarity,
)

__all__ = [
    "distance",
    "similarity",
    "normalized_distance",
    "normalized_similarity",
]
