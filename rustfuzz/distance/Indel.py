"""rustfuzz.distance.Indel"""

from __future__ import annotations

from rustfuzz._rustfuzz import (
    indel_distance as distance,
)
from rustfuzz._rustfuzz import (
    indel_editops as editops,
)
from rustfuzz._rustfuzz import (
    indel_normalized_distance as normalized_distance,
)
from rustfuzz._rustfuzz import (
    indel_normalized_similarity as normalized_similarity,
)
from rustfuzz._rustfuzz import (
    indel_opcodes as opcodes,
)
from rustfuzz._rustfuzz import (
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
