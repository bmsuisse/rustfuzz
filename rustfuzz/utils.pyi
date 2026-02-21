from typing import Any, Callable, Sequence, Iterable, Hashable, TypeVar
import numpy.typing as npt

T = TypeVar('T')
TScore = TypeVar('TScore', int, float)

def default_process(*args, **kwargs) -> Any:
    ...

