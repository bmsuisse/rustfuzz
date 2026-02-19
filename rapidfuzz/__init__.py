"""
rapidfuzz — rapid fuzzy string matching, powered by Rust.
"""

from __future__ import annotations

from . import distance, fuzz, process, utils
from ._rapidfuzz import (  # noqa: F401
    Editop,
    Editops,
    MatchingBlock,
    Opcode,
    Opcodes,
    ScoreAlignment,
)

__version__: str = "3.14.3"
__author__: str = "BM Suisse"

__all__ = [
    "distance",
    "fuzz",
    "process",
    "utils",
    "Editop",
    "Editops",
    "MatchingBlock",
    "Opcode",
    "Opcodes",
    "ScoreAlignment",
    "__version__",
]
