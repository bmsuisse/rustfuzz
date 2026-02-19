"""
rapidfuzz.distance._initialize — edit operation data types.
Re-exported from the Rust extension module.
"""

from __future__ import annotations

from rapidfuzz._rapidfuzz import (
    Editop,
    Editops,
    MatchingBlock,
    Opcode,
    Opcodes,
    ScoreAlignment,
)

__all__ = [
    "Editop",
    "Editops",
    "MatchingBlock",
    "Opcode",
    "Opcodes",
    "ScoreAlignment",
]
