from collections.abc import Callable
from typing import Any


def default_process(s: Any, *, processor: Callable[..., Any] | None = None) -> str: ...
