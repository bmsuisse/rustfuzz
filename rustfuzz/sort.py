"""Backward-compatible shim — sort module has moved to ``rustfuzz.search.sort``.

All public symbols are re-exported here so existing imports continue to work.
"""

from rustfuzz.search.sort import *  # noqa: F401, F403
from rustfuzz.search.sort import __all__  # noqa: F401
