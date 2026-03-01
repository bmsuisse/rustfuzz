"""
rustfuzz — rapid fuzzy string matching, powered by Rust.
"""

from __future__ import annotations

from . import (
    distance,
    document,
    engine,
    filter,
    fuzz,
    join,
    process,
    query,
    search,
    sort,
    utils,
)
from ._rustfuzz import (  # noqa: F401
    Editop,
    Editops,
    MatchingBlock,
    Opcode,
    Opcodes,
    ScoreAlignment,
)
from .document import Document
from .engine import Retriever, RetrieverConfig

__version__: str = "0.1.21"
__author__: str = "BM Suisse"

__all__ = [
    "distance",
    "document",
    "engine",
    "filter",
    "fuzz",
    "join",
    "process",
    "query",
    "search",
    "sort",
    "utils",
    "Document",
    "Retriever",
    "RetrieverConfig",
    "Editop",
    "Editops",
    "MatchingBlock",
    "Opcode",
    "Opcodes",
    "ScoreAlignment",
    "__version__",
]
