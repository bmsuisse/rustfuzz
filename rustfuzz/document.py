"""
rustfuzz.document — Lightweight document container with metadata.

Compatible with LangChain ``Document`` objects — ``HybridSearch`` and
``MultiJoiner`` accept both interchangeably.
"""

from __future__ import annotations

from typing import Any


class Document:
    """
    A lightweight document with content and metadata.

    Compatible with LangChain Document objects — HybridSearch accepts both.

    Parameters
    ----------
    content : str
        The document text content.
    metadata : dict[str, Any] | None, default None
        Optional metadata dict attached to this document.
    _vector : list[float] | None, default None
        Optional vector embedding for this document.

    Examples
    --------
    >>> doc = Document("Apple iPhone 15 Pro", metadata={"category": "phones", "price": 999})
    >>> doc.content
    'Apple iPhone 15 Pro'
    >>> doc.metadata
    {'category': 'phones', 'price': 999}
    """

    __slots__ = ("content", "metadata", "_vector")

    def __init__(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        _vector: list[float] | None = None,
    ) -> None:
        self.content = content
        self.metadata = metadata or {}
        self._vector = _vector

    def __repr__(self) -> str:
        meta_preview = f", metadata={self.metadata!r}" if self.metadata else ""
        vec_preview = ", _vector=[...]" if self._vector else ""
        text = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f'Document("{text}"{meta_preview}{vec_preview})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        return (
            self.content == other.content
            and self.metadata == other.metadata
            and self._vector == other._vector
        )


__all__ = ["Document"]
