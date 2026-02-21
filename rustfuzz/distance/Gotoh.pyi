from typing import Any, Callable

def distance(s1: Any, s2: Any, *, processor: Callable[[Any], Any] | None = None, score_cutoff: int | float | None = None, open_penalty: int | float = ..., extend_penalty: int | float = ...) -> int | float:
    ...

