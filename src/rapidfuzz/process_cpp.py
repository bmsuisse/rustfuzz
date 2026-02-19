# SPDX-License-Identifier: MIT
# process_cpp: delegates to process_py since process functions call Python scorers.
# Correctness guaranteed; further parallelism can be added to Rust later.
from __future__ import annotations

from rapidfuzz.process_py import (
    cdist,
    cpdist,
    extract,
    extract_iter,
    extractOne,
)

__all__ = ["cdist", "cpdist", "extract", "extract_iter", "extractOne"]
