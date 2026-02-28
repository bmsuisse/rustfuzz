from typing import Any, Union

# AST Node types
class ComparisonNode:
    attribute: str
    op: str
    value: Any

class RangeNode:
    attribute: str
    low: float | int
    high: float | int

class ExistsNode:
    attribute: str

class IsNullNode:
    attribute: str

class IsEmptyNode:
    attribute: str

class InNode:
    attribute: str
    values: list[Any]

class ContainsNode:
    attribute: str
    value: str

class StartsWithNode:
    attribute: str
    value: str

class NotNode:
    child: FilterNode

class AndNode:
    left: FilterNode
    right: FilterNode

class OrNode:
    left: FilterNode
    right: FilterNode

FilterNode = Union[
    ComparisonNode,
    RangeNode,
    ExistsNode,
    IsNullNode,
    IsEmptyNode,
    InNode,
    ContainsNode,
    StartsWithNode,
    NotNode,
    AndNode,
    OrNode,
]

def parse_filter(expression: str) -> FilterNode: ...
def evaluate_filter(node: FilterNode, metadata: dict[str, Any]) -> bool: ...
def apply_filter(
    results: list[Any],
    filter_expr: str | FilterNode | None,
) -> list[Any]: ...
