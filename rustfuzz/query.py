"""Backward-compatible shim — query module has moved to ``rustfuzz.search.query``.

All public symbols are re-exported here so existing imports continue to work.
"""

from rustfuzz.search.query import *  # noqa: F401, F403
