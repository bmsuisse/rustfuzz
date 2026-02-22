from collections.abc import Callable
from typing import Any

def sorensen_dice(s1: Any, s2: Any, *, processor: Callable[[Any], Any] | None = None, score_cutoff: float | None = None, n: int = 2) -> float:
    ...

def jaccard(s1: Any, s2: Any, *, processor: Callable[[Any], Any] | None = None, score_cutoff: float | None = None, n: int = 2) -> float:
    ...

