from collections.abc import Callable, Iterable, Iterator
from typing import Any

def extract(
    query: Any,
    choices: Iterable[Any],
    *,
    processor: Callable[[Any], Any] | None = None,
    scorer: Callable[..., float] | None = None,
    limit: int | None = 5,
    score_cutoff: float | None = None,
) -> list[tuple[Any, float, int]]: ...
def extractBests(
    query: Any,
    choices: Iterable[Any],
    *,
    processor: Callable[[Any], Any] | None = None,
    scorer: Callable[..., float] | None = None,
    limit: int | None = 5,
    score_cutoff: float | None = 0.0,
) -> list[tuple[Any, float, int]]: ...
def extractOne(
    query: Any,
    choices: Iterable[Any],
    *,
    processor: Callable[[Any], Any] | None = None,
    scorer: Callable[..., float] | None = None,
    score_cutoff: float | None = None,
) -> tuple[Any, float, int] | None: ...
def extract_iter(
    query: Any,
    choices: Iterable[Any],
    *,
    processor: Callable[[Any], Any] | None = None,
    scorer: Callable[..., float] | None = None,
    score_cutoff: float | None = None,
) -> Iterator[tuple[Any, float, int]]: ...
def cdist(
    queries: Iterable[Any],
    choices: Iterable[Any],
    *,
    scorer: Callable[..., float] | None = None,
    processor: Callable[[Any], Any] | None = None,
    score_cutoff: float | None = None,
    dtype: Any = None,
    workers: int = 1,
) -> Any: ...
def dedupe(
    choices: Iterable[Any],
    *,
    max_edits: int = 2,
    scorer: Callable[..., float] | None = None,
    processor: Callable[[Any], Any] | None = None,
) -> list[Any]: ...
