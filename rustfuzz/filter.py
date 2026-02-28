"""
rustfuzz.filter — Meilisearch-compatible filter expression parser & evaluator.

Supports the full Meilisearch filter expression syntax:
  =, !=, >, <, >=, <=, TO, EXISTS, IS NULL, IS EMPTY,
  IN, CONTAINS, STARTS WITH, NOT, AND, OR, parentheses.

Usage::

    from rustfuzz.filter import parse_filter, evaluate_filter, apply_filter

    # Parse once, evaluate many times
    node = parse_filter('price > 100 AND category IN ["phones", "tablets"]')
    assert evaluate_filter(node, {"price": 150, "category": "phones"})

    # Or use the one-shot helper on search results
    results = bm25.get_top_n("query", n=10)
    filtered = apply_filter(results, 'brand = "Apple"')
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Union

# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

FilterNode = Union[
    "ComparisonNode",
    "RangeNode",
    "ExistsNode",
    "IsNullNode",
    "IsEmptyNode",
    "InNode",
    "ContainsNode",
    "StartsWithNode",
    "NotNode",
    "AndNode",
    "OrNode",
]


@dataclass(slots=True)
class ComparisonNode:
    """``attribute OP value``  where OP is =, !=, >, <, >=, <=."""

    attribute: str
    op: str
    value: Any


@dataclass(slots=True)
class RangeNode:
    """``attribute low TO high``  (inclusive both ends)."""

    attribute: str
    low: float | int
    high: float | int


@dataclass(slots=True)
class ExistsNode:
    """``attribute EXISTS``."""

    attribute: str


@dataclass(slots=True)
class IsNullNode:
    """``attribute IS NULL``."""

    attribute: str


@dataclass(slots=True)
class IsEmptyNode:
    """``attribute IS EMPTY``."""

    attribute: str


@dataclass(slots=True)
class InNode:
    """``attribute IN [v1, v2, ...]``."""

    attribute: str
    values: list[Any]


@dataclass(slots=True)
class ContainsNode:
    """``attribute CONTAINS value``."""

    attribute: str
    value: str


@dataclass(slots=True)
class StartsWithNode:
    """``attribute STARTS WITH value``."""

    attribute: str
    value: str


@dataclass(slots=True)
class NotNode:
    """``NOT child``."""

    child: FilterNode


@dataclass(slots=True)
class AndNode:
    """``left AND right``."""

    left: FilterNode
    right: FilterNode


@dataclass(slots=True)
class OrNode:
    """``left OR right``."""

    left: FilterNode
    right: FilterNode


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

# Token patterns (order matters — longer matches first)
_TOKEN_RE = re.compile(
    r"""
    \s+                         |  # whitespace (skip)
    "(?:[^"\\]|\\.)*"           |  # double-quoted string
    '(?:[^'\\]|\\.)*'           |  # single-quoted string
    -?[0-9]+(?:\.[0-9]+)?       |  # number (int or float, optional negative)
    [a-zA-Z_][a-zA-Z0-9_.]*    |  # identifier (including dot-path)
    [()[\],]                    |  # punctuation
    !=|>=|<=|>|<|=                 # comparison operators
    """,
    re.VERBOSE,
)


def _tokenise(expr: str) -> list[str]:
    """Split a filter expression into tokens."""
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(expr):
        tok = m.group()
        if tok.strip():
            tokens.append(tok)
    return tokens


def _parse_value(tok: str) -> Any:
    """Convert a token string into a typed Python value."""
    # Quoted string
    if (tok.startswith('"') and tok.endswith('"')) or (
        tok.startswith("'") and tok.endswith("'")
    ):
        return tok[1:-1].replace('\\"', '"').replace("\\'", "'")
    # Boolean-like
    upper = tok.upper()
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    if upper == "NULL":
        return None
    # Number
    try:
        if "." in tok:
            return float(tok)
        return int(tok)
    except ValueError:
        pass
    # Bare word → treat as string
    return tok


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------


class _Parser:
    """Recursive-descent parser for Meilisearch filter expressions."""

    __slots__ = ("tokens", "pos")

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -- helpers --

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, expected: str) -> str:
        tok = self._peek()
        if tok is None or tok.upper() != expected.upper():
            raise ValueError(
                f"Expected {expected!r} at position {self.pos}, got {tok!r}"
            )
        return self._advance()

    def _peek_upper(self) -> str | None:
        tok = self._peek()
        return tok.upper() if tok else None

    # -- grammar --

    def parse(self) -> FilterNode:
        node = self._parse_or()
        if self.pos < len(self.tokens):
            raise ValueError(
                f"Unexpected token at position {self.pos}: {self.tokens[self.pos]!r}"
            )
        return node

    def _parse_or(self) -> FilterNode:
        left = self._parse_and()
        while self._peek_upper() == "OR":
            self._advance()
            right = self._parse_and()
            left = OrNode(left, right)
        return left

    def _parse_and(self) -> FilterNode:
        left = self._parse_not()
        while self._peek_upper() == "AND":
            self._advance()
            right = self._parse_not()
            left = AndNode(left, right)
        return left

    def _parse_not(self) -> FilterNode:
        if self._peek_upper() == "NOT":
            self._advance()
            child = self._parse_not()
            return NotNode(child)
        return self._parse_primary()

    def _parse_primary(self) -> FilterNode:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        # Parenthesised sub-expression
        if tok == "(":
            self._advance()
            node = self._parse_or()
            self._expect(")")
            return node

        # Must be an attribute-based condition
        return self._parse_condition()

    def _parse_condition(self) -> FilterNode:
        """Parse: attribute OP value | attribute EXISTS | attribute IS ... etc."""
        attr = self._advance()

        next_tok = self._peek_upper()
        if next_tok is None:
            raise ValueError(f"Unexpected end after attribute {attr!r}")

        # attribute EXISTS
        if next_tok == "EXISTS":
            self._advance()
            return ExistsNode(attr)

        # attribute NOT EXISTS / NOT IN / NOT CONTAINS / NOT STARTS
        if next_tok == "NOT":
            self._advance()
            after_not = self._peek_upper()
            if after_not == "EXISTS":
                self._advance()
                return NotNode(ExistsNode(attr))
            if after_not == "IN":
                self._advance()
                values = self._parse_in_list()
                return NotNode(InNode(attr, values))
            if after_not == "CONTAINS":
                self._advance()
                val = self._advance()
                return NotNode(ContainsNode(attr, str(_parse_value(val))))
            if after_not == "STARTS":
                self._advance()
                self._expect("WITH")
                val = self._advance()
                return NotNode(StartsWithNode(attr, str(_parse_value(val))))
            if after_not == "NULL":
                self._advance()
                return NotNode(IsNullNode(attr))
            if after_not == "EMPTY":
                self._advance()
                return NotNode(IsEmptyNode(attr))
            raise ValueError(f"Unexpected token after NOT: {after_not!r}")

        # attribute IS NULL / IS NOT NULL / IS EMPTY / IS NOT EMPTY
        if next_tok == "IS":
            self._advance()
            after_is = self._peek_upper()
            if after_is == "NULL":
                self._advance()
                return IsNullNode(attr)
            if after_is == "EMPTY":
                self._advance()
                return IsEmptyNode(attr)
            if after_is == "NOT":
                self._advance()
                after_not = self._peek_upper()
                if after_not == "NULL":
                    self._advance()
                    return NotNode(IsNullNode(attr))
                if after_not == "EMPTY":
                    self._advance()
                    return NotNode(IsEmptyNode(attr))
                raise ValueError(
                    f"Expected NULL or EMPTY after IS NOT, got {after_not!r}"
                )
            raise ValueError(f"Expected NULL, EMPTY, or NOT after IS, got {after_is!r}")

        # attribute IN [...]
        if next_tok == "IN":
            self._advance()
            values = self._parse_in_list()
            return InNode(attr, values)

        # attribute CONTAINS value
        if next_tok == "CONTAINS":
            self._advance()
            val = self._advance()
            return ContainsNode(attr, str(_parse_value(val)))

        # attribute STARTS WITH value
        if next_tok == "STARTS":
            self._advance()
            self._expect("WITH")
            val = self._advance()
            return StartsWithNode(attr, str(_parse_value(val)))

        # Comparison operators: =, !=, >, <, >=, <=
        if next_tok in ("=", "!=", ">", "<", ">=", "<="):
            op = self._advance()
            val_tok = self._advance()
            return ComparisonNode(attr, op, _parse_value(val_tok))

        # attribute low TO high  (range)
        # The next token is a value, followed by TO, followed by another value
        val_tok = self._peek()
        if val_tok is not None:
            saved_pos = self.pos
            try:
                low_val = _parse_value(self._advance())
                if self._peek_upper() == "TO":
                    self._advance()
                    high_tok = self._advance()
                    high_val = _parse_value(high_tok)
                    return RangeNode(attr, low_val, high_val)
            except (ValueError, IndexError):
                pass
            # Backtrack
            self.pos = saved_pos

        raise ValueError(f"Cannot parse condition starting with {attr!r} {next_tok!r}")

    def _parse_in_list(self) -> list[Any]:
        """Parse ``[v1, v2, ...]``."""
        self._expect("[")
        values: list[Any] = []
        while self._peek() != "]":
            if values:
                self._expect(",")
            val_tok = self._advance()
            values.append(_parse_value(val_tok))
        self._expect("]")
        return values


# ---------------------------------------------------------------------------
# Public: parse
# ---------------------------------------------------------------------------


def parse_filter(expression: str) -> FilterNode:
    """
    Parse a Meilisearch-style filter expression into an AST node.

    Parameters
    ----------
    expression : str
        Filter expression, e.g. ``'price > 100 AND brand = "Apple"'``

    Returns
    -------
    FilterNode
        The parsed AST root.

    Raises
    ------
    ValueError
        If the expression is syntactically invalid.
    """
    tokens = _tokenise(expression)
    if not tokens:
        raise ValueError("Empty filter expression")
    return _Parser(tokens).parse()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _resolve_attr(meta: dict[str, Any], attr: str) -> tuple[bool, Any]:
    """
    Resolve a dot-separated attribute path against a metadata dict.

    Returns (found, value).
    """
    parts = attr.split(".")
    current: Any = meta
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _compare(a: Any, b: Any, op: str) -> bool:
    """Compare two values with the given operator."""
    if a is None or b is None:
        if op == "=":
            return a is None and b is None
        if op == "!=":
            return not (a is None and b is None)
        return False
    try:
        if op == "=":
            return a == b  # type: ignore[no-any-return]
        if op == "!=":
            return a != b  # type: ignore[no-any-return]
        if op == ">":
            return a > b  # type: ignore[no-any-return]
        if op == "<":
            return a < b  # type: ignore[no-any-return]
        if op == ">=":
            return a >= b  # type: ignore[no-any-return]
        if op == "<=":
            return a <= b  # type: ignore[no-any-return]
    except TypeError:
        return False
    return False


def evaluate_filter(node: FilterNode, metadata: dict[str, Any]) -> bool:
    """
    Evaluate a parsed filter expression against a metadata dict.

    Parameters
    ----------
    node : FilterNode
        Parsed filter AST (from :func:`parse_filter`).
    metadata : dict[str, Any]
        The metadata dict to evaluate against.

    Returns
    -------
    bool
        True if the metadata matches the filter.
    """
    if isinstance(node, ComparisonNode):
        found, val = _resolve_attr(metadata, node.attribute)
        if not found:
            return False
        # If the metadata value is a list, check membership for =
        if isinstance(val, list) and node.op == "=":
            return node.value in val
        if isinstance(val, list) and node.op == "!=":
            return node.value not in val
        return _compare(val, node.value, node.op)

    if isinstance(node, RangeNode):
        found, val = _resolve_attr(metadata, node.attribute)
        if not found or val is None:
            return False
        try:
            return node.low <= val <= node.high  # type: ignore[no-any-return]
        except TypeError:
            return False

    if isinstance(node, ExistsNode):
        found, _ = _resolve_attr(metadata, node.attribute)
        return found

    if isinstance(node, IsNullNode):
        found, val = _resolve_attr(metadata, node.attribute)
        return found and val is None

    if isinstance(node, IsEmptyNode):
        found, val = _resolve_attr(metadata, node.attribute)
        if not found:
            return False
        if val is None:
            return False
        if isinstance(val, (str, list, dict)):
            return len(val) == 0
        return False

    if isinstance(node, InNode):
        found, val = _resolve_attr(metadata, node.attribute)
        if not found:
            return False
        # If metadata value is a list, check intersection
        if isinstance(val, list):
            return any(v in node.values for v in val)
        return val in node.values

    if isinstance(node, ContainsNode):
        found, val = _resolve_attr(metadata, node.attribute)
        if not found or not isinstance(val, str):
            return False
        return node.value in val

    if isinstance(node, StartsWithNode):
        found, val = _resolve_attr(metadata, node.attribute)
        if not found or not isinstance(val, str):
            return False
        return val.startswith(node.value)

    if isinstance(node, NotNode):
        return not evaluate_filter(node.child, metadata)

    if isinstance(node, AndNode):
        return evaluate_filter(node.left, metadata) and evaluate_filter(
            node.right, metadata
        )

    if isinstance(node, OrNode):
        return evaluate_filter(node.left, metadata) or evaluate_filter(
            node.right, metadata
        )

    raise TypeError(f"Unknown filter node type: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Public: apply_filter  (works on search result tuples)
# ---------------------------------------------------------------------------


def apply_filter(
    results: list[Any],
    filter_expr: str | FilterNode | None,
) -> list[Any]:
    """
    Filter search results by a Meilisearch-style filter expression.

    Operates on ``(text, score, metadata)`` triples.  Results without
    metadata (plain ``(text, score)`` pairs) are dropped when a filter
    is active.

    Parameters
    ----------
    results : list
        Search results — either ``(text, score)`` or ``(text, score, metadata)``.
    filter_expr : str | FilterNode | None
        A filter expression string, a pre-parsed :class:`FilterNode`,
        or ``None`` (no-op).

    Returns
    -------
    list
        The filtered results, preserving original tuple structure.
    """
    if filter_expr is None:
        return results
    if not results:
        return results

    node = parse_filter(filter_expr) if isinstance(filter_expr, str) else filter_expr

    filtered: list[Any] = []
    for r in results:
        if len(r) < 3:
            # No metadata — cannot filter, skip
            continue
        meta = r[2]
        if meta is None:
            continue
        if not isinstance(meta, dict):
            continue
        if evaluate_filter(node, meta):
            filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# JSON serialization — bridge to Rust-side filter evaluator
# ---------------------------------------------------------------------------


def _filter_node_to_dict(node: FilterNode) -> dict[str, Any]:
    """Convert a filter AST node to a JSON-serializable dict."""
    if isinstance(node, ComparisonNode):
        return {
            "type": "comparison",
            "attribute": node.attribute,
            "op": node.op,
            "value": node.value,
        }
    if isinstance(node, RangeNode):
        return {
            "type": "range",
            "attribute": node.attribute,
            "low": float(node.low),
            "high": float(node.high),
        }
    if isinstance(node, ExistsNode):
        return {"type": "exists", "attribute": node.attribute}
    if isinstance(node, IsNullNode):
        return {"type": "is_null", "attribute": node.attribute}
    if isinstance(node, IsEmptyNode):
        return {"type": "is_empty", "attribute": node.attribute}
    if isinstance(node, InNode):
        return {
            "type": "in",
            "attribute": node.attribute,
            "values": node.values,
        }
    if isinstance(node, ContainsNode):
        return {
            "type": "contains",
            "attribute": node.attribute,
            "value": node.value,
        }
    if isinstance(node, StartsWithNode):
        return {
            "type": "starts_with",
            "attribute": node.attribute,
            "value": node.value,
        }
    if isinstance(node, NotNode):
        return {"type": "not", "child": _filter_node_to_dict(node.child)}
    if isinstance(node, AndNode):
        return {
            "type": "and",
            "left": _filter_node_to_dict(node.left),
            "right": _filter_node_to_dict(node.right),
        }
    if isinstance(node, OrNode):
        return {
            "type": "or",
            "left": _filter_node_to_dict(node.left),
            "right": _filter_node_to_dict(node.right),
        }
    raise TypeError(f"Unknown filter node type: {type(node).__name__}")


def filter_to_json(node: FilterNode) -> str:
    """
    Serialize a parsed filter AST to a JSON string.

    Used to pass filter expressions to the Rust-side evaluator for
    blazing-fast filter mask computation.

    Parameters
    ----------
    node : FilterNode
        Parsed filter AST (from :func:`parse_filter`).

    Returns
    -------
    str
        JSON string representing the filter AST.
    """
    import json

    return json.dumps(_filter_node_to_dict(node))


def filters_to_json(nodes: list[FilterNode]) -> str:
    """
    Serialize multiple filter ASTs combined with AND to a single JSON string.

    Parameters
    ----------
    nodes : list[FilterNode]
        Parsed filter ASTs to combine.

    Returns
    -------
    str
        JSON string representing the combined filter AST.
    """
    import json

    if not nodes:
        raise ValueError("No filter nodes provided")
    if len(nodes) == 1:
        return json.dumps(_filter_node_to_dict(nodes[0]))
    # Combine with AND
    combined = nodes[0]
    for n in nodes[1:]:
        combined = AndNode(combined, n)
    return json.dumps(_filter_node_to_dict(combined))


__all__ = [
    "FilterNode",
    "ComparisonNode",
    "RangeNode",
    "ExistsNode",
    "IsNullNode",
    "IsEmptyNode",
    "InNode",
    "ContainsNode",
    "StartsWithNode",
    "NotNode",
    "AndNode",
    "OrNode",
    "parse_filter",
    "evaluate_filter",
    "apply_filter",
    "filter_to_json",
    "filters_to_json",
]

