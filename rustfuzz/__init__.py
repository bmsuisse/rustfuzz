"""
rustfuzz — rapid fuzzy string matching, powered by Rust.
"""

from __future__ import annotations

from . import distance, fuzz, join, process, search, utils
from ._rustfuzz import (  # noqa: F401
    Editop,
    Editops,
    MatchingBlock,
    Opcode,
    Opcodes,
    ScoreAlignment,
)

__version__: str = "0.1.4"
__author__: str = "BM Suisse"

__all__ = [
    "distance",
    "fuzz",
    "join",
    "process",
    "search",
    "spark",
    "utils",
    "Editop",
    "Editops",
    "MatchingBlock",
    "Opcode",
    "Opcodes",
    "ScoreAlignment",
    "__version__",
]
