"""
rustfuzz.fuzz — fuzzy string similarity scorers.

All functions accept optional `processor` and `score_cutoff` keyword arguments.
"""

from __future__ import annotations

from ._rustfuzz import (
    fuzz_partial_ratio as partial_ratio,
    fuzz_partial_ratio_alignment as partial_ratio_alignment,
    fuzz_partial_token_ratio as partial_token_ratio,
    fuzz_partial_token_set_ratio as partial_token_set_ratio,
    fuzz_partial_token_sort_ratio as partial_token_sort_ratio,
    fuzz_qratio as QRatio,
    fuzz_ratio as ratio,
    fuzz_token_ratio as token_ratio,
    fuzz_token_set_ratio as token_set_ratio,
    fuzz_token_sort_ratio as token_sort_ratio,
    fuzz_wratio as WRatio,
)

__all__ = [
    "ratio",
    "partial_ratio",
    "partial_ratio_alignment",
    "token_sort_ratio",
    "token_set_ratio",
    "token_ratio",
    "partial_token_sort_ratio",
    "partial_token_set_ratio",
    "partial_token_ratio",
    "WRatio",
    "QRatio",
]
