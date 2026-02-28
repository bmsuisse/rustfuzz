"""
rustfuzz.sort — Meilisearch-compatible sort evaluator for search results.

Supports Meilisearch sort syntax: ``["attribute:asc", "attribute:desc"]``
or a comma-separated string ``"attribute:asc, attribute:desc"``.

Dot-notation is supported for nested attributes.

Usage::

    from rustfuzz.sort import apply_sort

    results = [("doc1", 0.9, {"price": 100}), ("doc2", 0.8, {"price": 50})]
    sorted_results = apply_sort(results, ["price:asc"])
"""

from __future__ import annotations

from typing import Any


def _resolve_attr(meta: dict[str, Any], attr: str) -> tuple[bool, Any]:
    """Resolve a dot-separated attribute path against a metadata dict."""
    parts = attr.split(".")
    current: Any = meta
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _parse_sort_keys(sort_expr: list[str] | str) -> list[tuple[str, bool]]:
    """
    Parse sort expressions into (attribute, reverse) pairs.

    Parameters
    ----------
    sort_expr : list[str] | str
        Either a list like ``["price:asc", "name:desc"]``
        or a comma-separated string ``"price:asc, name:desc"``.

    Returns
    -------
    list[tuple[str, bool]]
        List of (attribute_name, reverse) where reverse=True means descending.
    """
    if isinstance(sort_expr, str):
        parts = [s.strip() for s in sort_expr.split(",") if s.strip()]
    else:
        parts = sort_expr

    keys: list[tuple[str, bool]] = []
    for part in parts:
        if ":" in part:
            attr, direction = part.rsplit(":", 1)
            attr = attr.strip()
            direction = direction.strip().lower()
            reverse = direction == "desc"
        else:
            attr = part.strip()
            reverse = False
        keys.append((attr, reverse))
    return keys


def apply_sort(
    results: list[Any],
    sort_expr: list[str] | str | None,
) -> list[Any]:
    """
    Sort search results by metadata attributes (Meilisearch-style).

    Operates on ``(text, score, metadata)`` triples.  Results without
    metadata are placed at the end of the sorted list.

    Parameters
    ----------
    results : list
        Search results — either ``(text, score)`` or ``(text, score, metadata)``.
    sort_expr : list[str] | str | None
        Sort expression(s) like ``["price:asc"]``, ``"price:desc, name:asc"``,
        or ``None`` (no-op, preserves relevance order).

    Returns
    -------
    list
        The sorted results, preserving original tuple structure.
    """
    if sort_expr is None:
        return results
    if not results:
        return results

    keys = _parse_sort_keys(sort_expr)
    if not keys:
        return results

    def _sort_key(item: Any) -> tuple[Any, ...]:
        """Build a composite sort key for multi-attribute sorting."""
        key_parts: list[Any] = []
        # Check if item has metadata
        has_meta = len(item) >= 3 and isinstance(item[2], dict)

        for attr, reverse in keys:
            if has_meta:
                found, val = _resolve_attr(item[2], attr)
            else:
                found, val = False, None

            if not found or val is None:
                # Push missing/null values to end regardless of sort direction
                key_parts.append((1,))
            else:
                # Normalise types for comparison:
                # - numbers compare as numbers
                # - strings compare as lowercase strings
                if isinstance(val, (int, float)):
                    sort_val: Any = (0, -val if reverse else val)
                elif isinstance(val, str):
                    if reverse:
                        sort_val = (0, _ReverseStr(val.lower()))
                    else:
                        sort_val = (0, val.lower())
                else:
                    if reverse:
                        sort_val = (0, _ReverseStr(str(val)))
                    else:
                        sort_val = (0, str(val))
                key_parts.append(sort_val)

        return tuple(key_parts)

    return sorted(results, key=_sort_key)


class _ReverseStr:
    """Helper for reverse-sorting strings."""

    __slots__ = ("val",)

    def __init__(self, val: str) -> None:
        self.val = val

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _ReverseStr):
            return self.val > other.val
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, _ReverseStr):
            return self.val >= other.val
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, _ReverseStr):
            return self.val < other.val
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, _ReverseStr):
            return self.val <= other.val
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ReverseStr):
            return self.val == other.val
        return NotImplemented

    def __repr__(self) -> str:
        return f"_ReverseStr({self.val!r})"


__all__ = ["apply_sort"]
