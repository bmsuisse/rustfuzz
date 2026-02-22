from typing import Any, TypeVar

T = TypeVar('T')
TScore = TypeVar('TScore', int, float)

def default_process(*args, **kwargs) -> Any:
    ...

