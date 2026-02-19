# SPDX-License-Identifier: MIT
# fuzz_cpp: Python wrappers over the Rust fuzz scorers.
# Provides _RF_ScorerPy attribute so the test infrastructure accepts these.
from __future__ import annotations

import rapidfuzz._rapidfuzz as _rf
from rapidfuzz._utils import ScorerFlag, add_scorer_attrs
from rapidfuzz.fuzz_py import (
    QRatio as _QRatio_py,
    WRatio as _WRatio_py,
    partial_ratio as _partial_ratio_py,
    partial_ratio_alignment as _partial_ratio_alignment_py,
    partial_token_ratio as _partial_token_ratio_py,
    partial_token_set_ratio as _partial_token_set_ratio_py,
    partial_token_sort_ratio as _partial_token_sort_ratio_py,
    ratio as _ratio_py,
    token_ratio as _token_ratio_py,
    token_set_ratio as _token_set_ratio_py,
    token_sort_ratio as _token_sort_ratio_py,
)


def _wrap(rust_fn, py_fn):
    """Return a Python function forwarding to rust_fn with py_fn's metadata."""
    def wrapper(*args, **kwargs):
        return rust_fn(*args, **kwargs)
    wrapper.__name__ = py_fn.__name__
    wrapper.__qualname__ = py_fn.__qualname__
    wrapper.__doc__ = py_fn.__doc__
    wrapper.__wrapped__ = rust_fn
    return wrapper


def _get_scorer_flags_fuzz(**_kwargs):
    return {
        "optimal_score": 100,
        "worst_score": 0,
        "flags": ScorerFlag.RESULT_F64 | ScorerFlag.SYMMETRIC,
    }


_fuzz_attrs = {"get_scorer_flags": _get_scorer_flags_fuzz}

ratio = _wrap(_rf.fuzz_ratio, _ratio_py)
partial_ratio = _wrap(_rf.fuzz_partial_ratio, _partial_ratio_py)
partial_ratio_alignment = _wrap(_rf.fuzz_partial_ratio_alignment, _partial_ratio_alignment_py)
token_sort_ratio = _wrap(_rf.fuzz_token_sort_ratio, _token_sort_ratio_py)
token_set_ratio = _wrap(_rf.fuzz_token_set_ratio, _token_set_ratio_py)
token_ratio = _wrap(_rf.fuzz_token_ratio, _token_ratio_py)
partial_token_sort_ratio = _wrap(_rf.fuzz_partial_token_sort_ratio, _partial_token_sort_ratio_py)
partial_token_set_ratio = _wrap(_rf.fuzz_partial_token_set_ratio, _partial_token_set_ratio_py)
partial_token_ratio = _wrap(_rf.fuzz_partial_token_ratio, _partial_token_ratio_py)
WRatio = _wrap(_rf.fuzz_wratio, _WRatio_py)
QRatio = _wrap(_rf.fuzz_qratio, _QRatio_py)

add_scorer_attrs(ratio, _fuzz_attrs)
add_scorer_attrs(partial_ratio, _fuzz_attrs)
add_scorer_attrs(partial_ratio_alignment, _fuzz_attrs)
add_scorer_attrs(token_sort_ratio, _fuzz_attrs)
add_scorer_attrs(token_set_ratio, _fuzz_attrs)
add_scorer_attrs(token_ratio, _fuzz_attrs)
add_scorer_attrs(partial_token_sort_ratio, _fuzz_attrs)
add_scorer_attrs(partial_token_set_ratio, _fuzz_attrs)
add_scorer_attrs(partial_token_ratio, _fuzz_attrs)
add_scorer_attrs(WRatio, _fuzz_attrs)
add_scorer_attrs(QRatio, _fuzz_attrs)

__all__ = [
    "QRatio",
    "WRatio",
    "partial_ratio",
    "partial_ratio_alignment",
    "partial_token_ratio",
    "partial_token_set_ratio",
    "partial_token_sort_ratio",
    "ratio",
    "token_ratio",
    "token_set_ratio",
    "token_sort_ratio",
]
