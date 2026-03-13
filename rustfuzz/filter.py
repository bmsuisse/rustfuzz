"""Backward-compatible shim — filter module has moved to ``rustfuzz.search.filter``.

All public symbols are re-exported here so existing imports continue to work.
"""

from rustfuzz.search.filter import *  # noqa: F401, F403
from rustfuzz.search.filter import __all__  # noqa: F401
