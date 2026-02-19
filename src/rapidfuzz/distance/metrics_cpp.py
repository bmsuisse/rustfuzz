# SPDX-License-Identifier: MIT
# metrics_cpp: Python wrappers over the Rust _rapidfuzz extension.
# Provides _RF_ScorerPy and _RF_Scorer attributes so that the existing
# test infrastructure (tests/common.py:GenericScorer) accepts these as
# cpp_scorers.
from __future__ import annotations

import rapidfuzz._rapidfuzz as _rf
from rapidfuzz._utils import (
    ScorerFlag,
    add_scorer_attrs,
    default_distance_attribute as _dist_attr,
    default_normalized_distance_attribute as _norm_dist_attr,
    default_normalized_similarity_attribute as _norm_sim_attr,
    default_similarity_attribute as _sim_attr,
)
from rapidfuzz.distance import metrics_py as _py

# ---------------------------------------------------------------------------
# Helper: create a thin wrapper that forwards to a Rust built-in function.
# Copies __name__, __qualname__, and __doc__ from the corresponding py func.
# ---------------------------------------------------------------------------

def _wrap(rust_fn, py_fn):
    """Return a Python function forwarding to rust_fn with py_fn's metadata."""
    def wrapper(*args, **kwargs):
        return rust_fn(*args, **kwargs)
    wrapper.__name__ = py_fn.__name__
    wrapper.__qualname__ = py_fn.__qualname__
    wrapper.__doc__ = py_fn.__doc__
    wrapper.__wrapped__ = rust_fn
    return wrapper


# ---------------------------------------------------------------------------
# OSA
# ---------------------------------------------------------------------------
osa_distance = _wrap(_rf.osa_distance, _py.osa_distance)
osa_similarity = _wrap(_rf.osa_similarity, _py.osa_similarity)
osa_normalized_distance = _wrap(_rf.osa_normalized_distance, _py.osa_normalized_distance)
osa_normalized_similarity = _wrap(_rf.osa_normalized_similarity, _py.osa_normalized_similarity)

add_scorer_attrs(osa_distance, _dist_attr)
add_scorer_attrs(osa_similarity, _sim_attr)
add_scorer_attrs(osa_normalized_distance, _norm_dist_attr)
add_scorer_attrs(osa_normalized_similarity, _norm_sim_attr)
osa_distance._RF_Scorer = True
osa_similarity._RF_Scorer = True
osa_normalized_distance._RF_Scorer = True
osa_normalized_similarity._RF_Scorer = True

# ---------------------------------------------------------------------------
# Prefix
# ---------------------------------------------------------------------------
prefix_distance = _wrap(_rf.prefix_distance, _py.prefix_distance)
prefix_similarity = _wrap(_rf.prefix_similarity, _py.prefix_similarity)
prefix_normalized_distance = _wrap(_rf.prefix_normalized_distance, _py.prefix_normalized_distance)
prefix_normalized_similarity = _wrap(_rf.prefix_normalized_similarity, _py.prefix_normalized_similarity)

add_scorer_attrs(prefix_distance, _dist_attr)
add_scorer_attrs(prefix_similarity, _sim_attr)
add_scorer_attrs(prefix_normalized_distance, _norm_dist_attr)
add_scorer_attrs(prefix_normalized_similarity, _norm_sim_attr)
prefix_distance._RF_Scorer = True
prefix_similarity._RF_Scorer = True
prefix_normalized_distance._RF_Scorer = True
prefix_normalized_similarity._RF_Scorer = True

# ---------------------------------------------------------------------------
# Postfix
# ---------------------------------------------------------------------------
postfix_distance = _wrap(_rf.postfix_distance, _py.postfix_distance)
postfix_similarity = _wrap(_rf.postfix_similarity, _py.postfix_similarity)
postfix_normalized_distance = _wrap(_rf.postfix_normalized_distance, _py.postfix_normalized_distance)
postfix_normalized_similarity = _wrap(_rf.postfix_normalized_similarity, _py.postfix_normalized_similarity)

add_scorer_attrs(postfix_distance, _dist_attr)
add_scorer_attrs(postfix_similarity, _sim_attr)
add_scorer_attrs(postfix_normalized_distance, _norm_dist_attr)
add_scorer_attrs(postfix_normalized_similarity, _norm_sim_attr)
postfix_distance._RF_Scorer = True
postfix_similarity._RF_Scorer = True
postfix_normalized_distance._RF_Scorer = True
postfix_normalized_similarity._RF_Scorer = True

# ---------------------------------------------------------------------------
# Jaro
# ---------------------------------------------------------------------------
jaro_distance = _wrap(_rf.jaro_distance, _py.jaro_distance)
jaro_similarity = _wrap(_rf.jaro_similarity, _py.jaro_similarity)
jaro_normalized_distance = _wrap(_rf.jaro_normalized_distance, _py.jaro_normalized_distance)
jaro_normalized_similarity = _wrap(_rf.jaro_normalized_similarity, _py.jaro_normalized_similarity)

add_scorer_attrs(jaro_distance, _norm_dist_attr)
add_scorer_attrs(jaro_similarity, _norm_sim_attr)
add_scorer_attrs(jaro_normalized_distance, _norm_dist_attr)
add_scorer_attrs(jaro_normalized_similarity, _norm_sim_attr)
jaro_distance._RF_Scorer = True
jaro_similarity._RF_Scorer = True
jaro_normalized_distance._RF_Scorer = True
jaro_normalized_similarity._RF_Scorer = True

# ---------------------------------------------------------------------------
# JaroWinkler
# ---------------------------------------------------------------------------
jaro_winkler_distance = _wrap(_rf.jaro_winkler_distance, _py.jaro_winkler_distance)
jaro_winkler_similarity = _wrap(_rf.jaro_winkler_similarity, _py.jaro_winkler_similarity)
jaro_winkler_normalized_distance = _wrap(_rf.jaro_winkler_normalized_distance, _py.jaro_winkler_normalized_distance)
jaro_winkler_normalized_similarity = _wrap(_rf.jaro_winkler_normalized_similarity, _py.jaro_winkler_normalized_similarity)

add_scorer_attrs(jaro_winkler_distance, _norm_dist_attr)
add_scorer_attrs(jaro_winkler_similarity, _norm_sim_attr)
add_scorer_attrs(jaro_winkler_normalized_distance, _norm_dist_attr)
add_scorer_attrs(jaro_winkler_normalized_similarity, _norm_sim_attr)
jaro_winkler_distance._RF_Scorer = True
jaro_winkler_similarity._RF_Scorer = True
jaro_winkler_normalized_distance._RF_Scorer = True
jaro_winkler_normalized_similarity._RF_Scorer = True

# ---------------------------------------------------------------------------
# DamerauLevenshtein
# ---------------------------------------------------------------------------
damerau_levenshtein_distance = _wrap(_rf.damerau_levenshtein_distance, _py.damerau_levenshtein_distance)
damerau_levenshtein_similarity = _wrap(_rf.damerau_levenshtein_similarity, _py.damerau_levenshtein_similarity)
damerau_levenshtein_normalized_distance = _wrap(_rf.damerau_levenshtein_normalized_distance, _py.damerau_levenshtein_normalized_distance)
damerau_levenshtein_normalized_similarity = _wrap(_rf.damerau_levenshtein_normalized_similarity, _py.damerau_levenshtein_normalized_similarity)

add_scorer_attrs(damerau_levenshtein_distance, _dist_attr)
add_scorer_attrs(damerau_levenshtein_similarity, _sim_attr)
add_scorer_attrs(damerau_levenshtein_normalized_distance, _norm_dist_attr)
add_scorer_attrs(damerau_levenshtein_normalized_similarity, _norm_sim_attr)
damerau_levenshtein_distance._RF_Scorer = True
damerau_levenshtein_similarity._RF_Scorer = True
damerau_levenshtein_normalized_distance._RF_Scorer = True
damerau_levenshtein_normalized_similarity._RF_Scorer = True

# ---------------------------------------------------------------------------
# Levenshtein (weight-dependent scorer flags)
# ---------------------------------------------------------------------------
from rapidfuzz.distance.metrics_py import (  # noqa: E402
    _get_scorer_flags_levenshtein_distance as _lev_dist_flags,
    _get_scorer_flags_levenshtein_normalized_distance as _lev_norm_dist_flags,
    _get_scorer_flags_levenshtein_normalized_similarity as _lev_norm_sim_flags,
    _get_scorer_flags_levenshtein_similarity as _lev_sim_flags,
)

levenshtein_distance = _wrap(_rf.levenshtein_distance, _py.levenshtein_distance)
levenshtein_similarity = _wrap(_rf.levenshtein_similarity, _py.levenshtein_similarity)
levenshtein_normalized_distance = _wrap(_rf.levenshtein_normalized_distance, _py.levenshtein_normalized_distance)
levenshtein_normalized_similarity = _wrap(_rf.levenshtein_normalized_similarity, _py.levenshtein_normalized_similarity)
levenshtein_editops = _wrap(_rf.levenshtein_editops, _py.levenshtein_editops)
levenshtein_opcodes = _wrap(_rf.levenshtein_opcodes, _py.levenshtein_opcodes)

add_scorer_attrs(levenshtein_distance, {"get_scorer_flags": _lev_dist_flags})
add_scorer_attrs(levenshtein_similarity, {"get_scorer_flags": _lev_sim_flags})
add_scorer_attrs(levenshtein_normalized_distance, {"get_scorer_flags": _lev_norm_dist_flags})
add_scorer_attrs(levenshtein_normalized_similarity, {"get_scorer_flags": _lev_norm_sim_flags})
levenshtein_distance._RF_Scorer = True
levenshtein_similarity._RF_Scorer = True
levenshtein_normalized_distance._RF_Scorer = True
levenshtein_normalized_similarity._RF_Scorer = True

# ---------------------------------------------------------------------------
# LCSseq
# ---------------------------------------------------------------------------
lcs_seq_distance = _wrap(_rf.lcs_seq_distance, _py.lcs_seq_distance)
lcs_seq_similarity = _wrap(_rf.lcs_seq_similarity, _py.lcs_seq_similarity)
lcs_seq_normalized_distance = _wrap(_rf.lcs_seq_normalized_distance, _py.lcs_seq_normalized_distance)
lcs_seq_normalized_similarity = _wrap(_rf.lcs_seq_normalized_similarity, _py.lcs_seq_normalized_similarity)
lcs_seq_editops = _wrap(_rf.lcs_seq_editops, _py.lcs_seq_editops)
lcs_seq_opcodes = _wrap(_rf.lcs_seq_opcodes, _py.lcs_seq_opcodes)

add_scorer_attrs(lcs_seq_distance, _dist_attr)
add_scorer_attrs(lcs_seq_similarity, _sim_attr)
add_scorer_attrs(lcs_seq_normalized_distance, _norm_dist_attr)
add_scorer_attrs(lcs_seq_normalized_similarity, _norm_sim_attr)
lcs_seq_distance._RF_Scorer = True
lcs_seq_similarity._RF_Scorer = True
lcs_seq_normalized_distance._RF_Scorer = True
lcs_seq_normalized_similarity._RF_Scorer = True

# ---------------------------------------------------------------------------
# Indel
# ---------------------------------------------------------------------------
indel_distance = _wrap(_rf.indel_distance, _py.indel_distance)
indel_similarity = _wrap(_rf.indel_similarity, _py.indel_similarity)
indel_normalized_distance = _wrap(_rf.indel_normalized_distance, _py.indel_normalized_distance)
indel_normalized_similarity = _wrap(_rf.indel_normalized_similarity, _py.indel_normalized_similarity)
indel_editops = _wrap(_rf.indel_editops, _py.indel_editops)
indel_opcodes = _wrap(_rf.indel_opcodes, _py.indel_opcodes)

add_scorer_attrs(indel_distance, _dist_attr)
add_scorer_attrs(indel_similarity, _sim_attr)
add_scorer_attrs(indel_normalized_distance, _norm_dist_attr)
add_scorer_attrs(indel_normalized_similarity, _norm_sim_attr)
indel_distance._RF_Scorer = True
indel_similarity._RF_Scorer = True
indel_normalized_distance._RF_Scorer = True
indel_normalized_similarity._RF_Scorer = True

# ---------------------------------------------------------------------------
# Hamming
# ---------------------------------------------------------------------------
hamming_distance = _wrap(_rf.hamming_distance, _py.hamming_distance)
hamming_similarity = _wrap(_rf.hamming_similarity, _py.hamming_similarity)
hamming_normalized_distance = _wrap(_rf.hamming_normalized_distance, _py.hamming_normalized_distance)
hamming_normalized_similarity = _wrap(_rf.hamming_normalized_similarity, _py.hamming_normalized_similarity)
hamming_editops = _wrap(_rf.hamming_editops, _py.hamming_editops)
hamming_opcodes = _wrap(_rf.hamming_opcodes, _py.hamming_opcodes)

add_scorer_attrs(hamming_distance, _dist_attr)
add_scorer_attrs(hamming_similarity, _sim_attr)
add_scorer_attrs(hamming_normalized_distance, _norm_dist_attr)
add_scorer_attrs(hamming_normalized_similarity, _norm_sim_attr)
hamming_distance._RF_Scorer = True
hamming_similarity._RF_Scorer = True
hamming_normalized_distance._RF_Scorer = True
hamming_normalized_similarity._RF_Scorer = True

__all__ = [
    "damerau_levenshtein_distance",
    "damerau_levenshtein_normalized_distance",
    "damerau_levenshtein_normalized_similarity",
    "damerau_levenshtein_similarity",
    "hamming_distance",
    "hamming_editops",
    "hamming_normalized_distance",
    "hamming_normalized_similarity",
    "hamming_opcodes",
    "hamming_similarity",
    "indel_distance",
    "indel_editops",
    "indel_normalized_distance",
    "indel_normalized_similarity",
    "indel_opcodes",
    "indel_similarity",
    "jaro_distance",
    "jaro_normalized_distance",
    "jaro_normalized_similarity",
    "jaro_similarity",
    "jaro_winkler_distance",
    "jaro_winkler_normalized_distance",
    "jaro_winkler_normalized_similarity",
    "jaro_winkler_similarity",
    "lcs_seq_distance",
    "lcs_seq_editops",
    "lcs_seq_normalized_distance",
    "lcs_seq_normalized_similarity",
    "lcs_seq_opcodes",
    "lcs_seq_similarity",
    "levenshtein_distance",
    "levenshtein_editops",
    "levenshtein_normalized_distance",
    "levenshtein_normalized_similarity",
    "levenshtein_opcodes",
    "levenshtein_similarity",
    "osa_distance",
    "osa_normalized_distance",
    "osa_normalized_similarity",
    "osa_similarity",
    "postfix_distance",
    "postfix_normalized_distance",
    "postfix_normalized_similarity",
    "postfix_similarity",
    "prefix_distance",
    "prefix_normalized_distance",
    "prefix_normalized_similarity",
    "prefix_similarity",
]
