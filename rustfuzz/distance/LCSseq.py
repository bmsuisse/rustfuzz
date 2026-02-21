"""rustfuzz.distance.LCSseq"""

from __future__ import annotations

from rustfuzz._rustfuzz import (
    lcs_seq_distance as distance,
)
from rustfuzz._rustfuzz import (
    lcs_seq_editops as editops,
)
from rustfuzz._rustfuzz import (
    lcs_seq_normalized_distance as normalized_distance,
)
from rustfuzz._rustfuzz import (
    lcs_seq_normalized_similarity as normalized_similarity,
)
from rustfuzz._rustfuzz import (
    lcs_seq_opcodes as opcodes,
)
from rustfuzz._rustfuzz import (
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
