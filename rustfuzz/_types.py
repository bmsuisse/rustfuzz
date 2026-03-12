"""
rustfuzz._types — Shared type aliases used across rustfuzz modules.

Centralises type definitions to avoid duplication across search, engine, and query.
"""

from __future__ import annotations

from typing import Any

# Result tuples returned by BM25 / HybridSearch / Retriever search methods
Result = tuple[str, float]
MetaResult = tuple[str, float, Any]
