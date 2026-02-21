from typing import Any, Callable, Sequence, Iterable, Hashable, TypeVar
import numpy.typing as npt

T = TypeVar('T')
TScore = TypeVar('TScore', int, float)

def extract(query: Any, choices: Iterable[Any], *, processor: Callable[[Any], Any] | None = None, scorer: Callable[..., float] | None = None, limit: int | None = 5, score_cutoff: float | None = None) -> list[tuple[Any, float, int]]:
    ...

def extractBests(*args, **kwargs) -> Any:
    """Returns all matches with score >= score_cutoff, sorted by score."""
    ...

def extractOne(*args, **kwargs) -> Any:
    ...

def extract_iter(query: Any, choices: Iterable[Any], *, processor: Callable[[Any], Any] | None = None, scorer: Callable[..., float] | None = None, score_cutoff: float | None = None) -> Iterable[tuple[Any, float, int]]:
    ...

def cdist(*args, **kwargs) -> Any:
    """Compute a pairwise distance matrix. Requires numpy."""
    ...

def dedupe(*args, **kwargs) -> Any:
    """
    Deduplicate a list of choices using a BK-Tree.
    Note: Threshold is absolute Levenshtein distance (max allowed edits),
    default 2. Not a percentage score.
    """
    ...

