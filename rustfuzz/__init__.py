"""
rustfuzz — rapid fuzzy string matching, powered by Rust.
"""

from __future__ import annotations

from . import distance, filter, fuzz, join, process, query, search, sort, utils
from ._rustfuzz import (  # noqa: F401
    Editop,
    Editops,
    MatchingBlock,
    Opcode,
    Opcodes,
    ScoreAlignment,
)

__version__: str = "0.1.14"
__author__: str = "BM Suisse"

__all__ = [
    "distance",
    "filter",
    "fuzz",
    "join",
    "process",
    "query",
    "search",
    "sort",
    "utils",
    "Editop",
    "Editops",
    "MatchingBlock",
    "Opcode",
    "Opcodes",
    "ScoreAlignment",
    "__version__",
]
