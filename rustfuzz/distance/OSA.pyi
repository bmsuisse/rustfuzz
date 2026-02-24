from collections.abc import Callable
from typing import Any

def distance(
    s1: Any,
    s2: Any,
    *,
    processor: Callable[[Any], Any] | None = None,
    score_cutoff: int | float | None = None,
) -> int | float: ...
def similarity(
    s1: Any,
    s2: Any,
    *,
    processor: Callable[[Any], Any] | None = None,
    score_cutoff: int | float | None = None,
) -> int | float: ...
def normalized_distance(
    s1: Any,
    s2: Any,
    *,
    processor: Callable[[Any], Any] | None = None,
    score_cutoff: float | None = None,
) -> float: ...
def normalized_similarity(
    s1: Any,
    s2: Any,
    *,
    processor: Callable[[Any], Any] | None = None,
    score_cutoff: float | None = None,
) -> float: ...
