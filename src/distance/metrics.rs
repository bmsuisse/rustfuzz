// SPDX-License-Identifier: MIT
// PyO3 wrappers for all distance metric functions.
// Each function handles: processor, score_cutoff, None/NaN inputs.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::algorithms as alg;
use crate::distance::initialize::{Editops, Opcodes, raw_editops_to_opcodes_py, raw_editops_to_py};
use crate::types::{extract_sequences, is_none, Seq};

// ---------------------------------------------------------------------------
// Macro to extract two sequences, returning early if either is None/NaN
// ---------------------------------------------------------------------------
macro_rules! get_seqs {
    ($py:expr, $s1:expr, $s2:expr, $proc:expr, $none_val:expr) => {{
        if is_none($s1) || is_none($s2) {
            return Ok($none_val);
        }
        extract_sequences($py, $s1, $s2, $proc)?
    }};
}

macro_rules! get_seqs_editops {
    ($py:expr, $s1:expr, $s2:expr, $proc:expr) => {{
        if is_none($s1) || is_none($s2) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "expected str, bytes, or sequence",
            ));
        }
        extract_sequences($py, $s1, $s2, $proc)?
    }};
}

// ---------------------------------------------------------------------------
// Score cutoff helpers
// ---------------------------------------------------------------------------

fn check_dist_cutoff(dist: usize, cutoff: Option<usize>) -> usize {
    if let Some(c) = cutoff {
        if dist > c { c + 1 } else { dist }
    } else {
        dist
    }
}

fn check_sim_f64_cutoff(sim: f64, cutoff: Option<f64>) -> f64 {
    if let Some(c) = cutoff {
        if sim < c { 0.0 } else { sim }
    } else {
        sim
    }
}

// ===========================================================================
// LEVENSHTEIN
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, weights=None, processor=None, score_cutoff=None))]
pub fn levenshtein_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    weights: Option<(usize, usize, usize)>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let av = a.as_i64();
    let bv = b.as_i64();
    let w = weights.unwrap_or((1, 1, 1));
    let dist = alg::levenshtein(&av, &bv, w);
    Ok(check_dist_cutoff(dist, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, weights=None, processor=None, score_cutoff=None))]
pub fn levenshtein_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    weights: Option<(usize, usize, usize)>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let av = a.as_i64();
    let bv = b.as_i64();
    let w = weights.unwrap_or((1, 1, 1));
    let (ins, del, rep) = w;
    let len1 = av.len();
    let len2 = bv.len();
    // Match Python's _levenshtein_maximum formula
    let mut max_v = len1 * del + len2 * ins;
    if len1 >= len2 {
        max_v = max_v.min(len2 * rep + (len1 - len2) * del);
    } else {
        max_v = max_v.min(len1 * rep + (len2 - len1) * ins);
    }
    let dist = alg::levenshtein(&av, &bv, w);
    let sim = max_v.saturating_sub(dist);
    Ok(sim)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, weights=None, processor=None, score_cutoff=None))]
pub fn levenshtein_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    weights: Option<(usize, usize, usize)>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let w = weights.unwrap_or((1, 1, 1));
    let (ins, del, rep) = w;
    let len1 = av.len();
    let len2 = bv.len();
    // Match Python's _levenshtein_maximum formula
    let mut max_v = len1 * del + len2 * ins;
    if len1 >= len2 {
        max_v = max_v.min(len2 * rep + (len1 - len2) * del);
    } else {
        max_v = max_v.min(len1 * rep + (len2 - len1) * ins);
    }
    let dist = alg::levenshtein(&av, &bv, w);
    let nd = alg::normalized_distance(dist, max_v);
    Ok(check_sim_f64_cutoff(nd, score_cutoff.map(|c| 1.0 - c)).min(1.0))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, weights=None, processor=None, score_cutoff=None))]
pub fn levenshtein_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    weights: Option<(usize, usize, usize)>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let w = weights.unwrap_or((1, 1, 1));
    let (ins, del, rep) = w;
    let len1 = av.len();
    let len2 = bv.len();
    // Match Python's _levenshtein_maximum formula
    let mut max_v = len1 * del + len2 * ins;
    if len1 >= len2 {
        max_v = max_v.min(len2 * rep + (len1 - len2) * del);
    } else {
        max_v = max_v.min(len1 * rep + (len2 - len1) * ins);
    }
    let dist = alg::levenshtein(&av, &bv, w);
    let ns = alg::normalized_similarity(dist, max_v);
    Ok(check_sim_f64_cutoff(ns, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None))]
pub fn levenshtein_editops(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
) -> PyResult<Editops> {
    let (a, b) = get_seqs_editops!(py, s1, s2, &processor);
    let av = a.as_i64();
    let bv = b.as_i64();
    let src_len = av.len();
    let dest_len = bv.len();
    let ops = alg::levenshtein_editops_trace(&av, &bv);
    raw_editops_to_py(py, ops, src_len, dest_len)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None))]
pub fn levenshtein_opcodes(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
) -> PyResult<Opcodes> {
    let (a, b) = get_seqs_editops!(py, s1, s2, &processor);
    let av = a.as_i64();
    let bv = b.as_i64();
    let src_len = av.len();
    let dest_len = bv.len();
    let ops = alg::levenshtein_editops_trace(&av, &bv);
    raw_editops_to_opcodes_py(py, ops, src_len, dest_len)
}

// ===========================================================================
// HAMMING
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, pad=true, processor=None, score_cutoff=None))]
pub fn hamming_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    pad: bool,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let av = a.as_i64();
    let bv = b.as_i64();
    if !pad && av.len() != bv.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sequences are not the same length.",
        ));
    }
    let dist = alg::hamming_distance(&av, &bv, pad);
    Ok(check_dist_cutoff(dist, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, pad=true, processor=None, score_cutoff=None))]
pub fn hamming_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    pad: bool,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let av = a.as_i64();
    let bv = b.as_i64();
    if !pad && av.len() != bv.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sequences are not the same length.",
        ));
    }
    let dist = alg::hamming_distance(&av, &bv, pad);
    let max_v = av.len().max(bv.len());
    Ok(max_v.saturating_sub(dist))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, pad=true, processor=None, score_cutoff=None))]
pub fn hamming_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    pad: bool,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    if !pad && av.len() != bv.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sequences are not the same length.",
        ));
    }
    let dist = alg::hamming_distance(&av, &bv, pad);
    let max_v = av.len().max(bv.len());
    let nd = alg::normalized_distance(dist, max_v);
    Ok(check_sim_f64_cutoff(nd, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, pad=true, processor=None, score_cutoff=None))]
pub fn hamming_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    pad: bool,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    if !pad && av.len() != bv.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sequences are not the same length.",
        ));
    }
    let dist = alg::hamming_distance(&av, &bv, pad);
    let max_v = av.len().max(bv.len());
    let ns = alg::normalized_similarity(dist, max_v);
    Ok(check_sim_f64_cutoff(ns, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, pad=true, processor=None))]
pub fn hamming_editops(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    pad: bool,
    processor: Option<PyObject>,
) -> PyResult<Editops> {
    let (a, b) = get_seqs_editops!(py, s1, s2, &processor);
    let av = a.as_i64();
    let bv = b.as_i64();
    if !pad && av.len() != bv.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sequences are not the same length.",
        ));
    }
    let src_len = av.len();
    let dest_len = bv.len();
    let ops = alg::hamming_editops_trace(&av, &bv);
    raw_editops_to_py(py, ops, src_len, dest_len)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, pad=true, processor=None))]
pub fn hamming_opcodes(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    pad: bool,
    processor: Option<PyObject>,
) -> PyResult<Opcodes> {
    let (a, b) = get_seqs_editops!(py, s1, s2, &processor);
    let av = a.as_i64();
    let bv = b.as_i64();
    if !pad && av.len() != bv.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sequences are not the same length.",
        ));
    }
    let src_len = av.len();
    let dest_len = bv.len();
    let ops = alg::hamming_editops_trace(&av, &bv);
    raw_editops_to_opcodes_py(py, ops, src_len, dest_len)
}

// ===========================================================================
// INDEL
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn indel_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::indel_distance(&av, &bv);
    Ok(check_dist_cutoff(dist, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn indel_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::indel_distance(&av, &bv);
    let max_v = av.len() + bv.len();
    let sim = max_v.saturating_sub(dist);
    Ok(sim)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn indel_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::indel_distance(&av, &bv);
    let max_v = av.len() + bv.len();
    let nd = alg::normalized_distance(dist, max_v);
    Ok(check_sim_f64_cutoff(nd, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn indel_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::indel_distance(&av, &bv);
    let max_v = av.len() + bv.len();
    let ns = alg::normalized_similarity(dist, max_v);
    Ok(check_sim_f64_cutoff(ns, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None))]
pub fn indel_editops(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
) -> PyResult<Editops> {
    let (a, b) = get_seqs_editops!(py, s1, s2, &processor);
    let av = a.as_i64();
    let bv = b.as_i64();
    let src_len = av.len();
    let dest_len = bv.len();
    let ops = alg::indel_editops_trace(&av, &bv);
    raw_editops_to_py(py, ops, src_len, dest_len)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None))]
pub fn indel_opcodes(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
) -> PyResult<Opcodes> {
    let (a, b) = get_seqs_editops!(py, s1, s2, &processor);
    let av = a.as_i64();
    let bv = b.as_i64();
    let src_len = av.len();
    let dest_len = bv.len();
    let ops = alg::indel_editops_trace(&av, &bv);
    raw_editops_to_opcodes_py(py, ops, src_len, dest_len)
}

// ===========================================================================
// JARO
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn jaro_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let sim = alg::jaro(&a.as_i64(), &b.as_i64());
    let dist = 1.0 - sim;
    Ok(check_sim_f64_cutoff(dist, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn jaro_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let sim = alg::jaro(&a.as_i64(), &b.as_i64());
    Ok(check_sim_f64_cutoff(sim, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn jaro_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let sim = alg::jaro(&a.as_i64(), &b.as_i64());
    let dist = 1.0 - sim;
    Ok(check_sim_f64_cutoff(dist, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn jaro_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    jaro_similarity(py, s1, s2, processor, score_cutoff)
}

// ===========================================================================
// JARO-WINKLER
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=None))]
pub fn jaro_winkler_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    prefix_weight: f64,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let sim = alg::jaro_winkler(&a.as_i64(), &b.as_i64(), prefix_weight);
    let dist = 1.0 - sim;
    Ok(check_sim_f64_cutoff(dist, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=None))]
pub fn jaro_winkler_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    prefix_weight: f64,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let sim = alg::jaro_winkler(&a.as_i64(), &b.as_i64(), prefix_weight);
    Ok(check_sim_f64_cutoff(sim, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=None))]
pub fn jaro_winkler_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    prefix_weight: f64,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let sim = alg::jaro_winkler(&a.as_i64(), &b.as_i64(), prefix_weight);
    let dist = 1.0 - sim;
    Ok(check_sim_f64_cutoff(dist, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=None))]
pub fn jaro_winkler_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    prefix_weight: f64,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    jaro_winkler_similarity(py, s1, s2, prefix_weight, processor, score_cutoff)
}

// ===========================================================================
// LCSseq
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn lcs_seq_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let dist = alg::lcs_seq_distance(&a.as_i64(), &b.as_i64());
    Ok(check_dist_cutoff(dist, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn lcs_seq_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    Ok(alg::lcs_seq_similarity(&a.as_i64(), &b.as_i64()))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn lcs_seq_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::lcs_seq_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let nd = alg::normalized_distance(dist, max_v);
    Ok(check_sim_f64_cutoff(nd, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn lcs_seq_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::lcs_seq_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let ns = alg::normalized_similarity(dist, max_v);
    Ok(check_sim_f64_cutoff(ns, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None))]
pub fn lcs_seq_editops(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
) -> PyResult<Editops> {
    let (a, b) = get_seqs_editops!(py, s1, s2, &processor);
    let av = a.as_i64();
    let bv = b.as_i64();
    let src_len = av.len();
    let dest_len = bv.len();
    let ops = alg::lcs_seq_editops_trace(&av, &bv);
    raw_editops_to_py(py, ops, src_len, dest_len)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None))]
pub fn lcs_seq_opcodes(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
) -> PyResult<Opcodes> {
    let (a, b) = get_seqs_editops!(py, s1, s2, &processor);
    let av = a.as_i64();
    let bv = b.as_i64();
    let src_len = av.len();
    let dest_len = bv.len();
    let ops = alg::lcs_seq_editops_trace(&av, &bv);
    raw_editops_to_opcodes_py(py, ops, src_len, dest_len)
}

// ===========================================================================
// OSA
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn osa_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let dist = alg::osa_distance(&a.as_i64(), &b.as_i64());
    Ok(check_dist_cutoff(dist, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn osa_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::osa_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    Ok(max_v.saturating_sub(dist))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn osa_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::osa_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let nd = alg::normalized_distance(dist, max_v);
    Ok(check_sim_f64_cutoff(nd, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn osa_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::osa_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let ns = alg::normalized_similarity(dist, max_v);
    Ok(check_sim_f64_cutoff(ns, score_cutoff))
}

// ===========================================================================
// DAMERAU-LEVENSHTEIN
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn damerau_levenshtein_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let dist = alg::damerau_levenshtein_distance(&a.as_i64(), &b.as_i64());
    Ok(check_dist_cutoff(dist, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn damerau_levenshtein_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::damerau_levenshtein_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    Ok(max_v.saturating_sub(dist))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn damerau_levenshtein_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::damerau_levenshtein_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let nd = alg::normalized_distance(dist, max_v);
    Ok(check_sim_f64_cutoff(nd, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn damerau_levenshtein_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::damerau_levenshtein_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let ns = alg::normalized_similarity(dist, max_v);
    Ok(check_sim_f64_cutoff(ns, score_cutoff))
}

// ===========================================================================
// PREFIX
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn prefix_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let dist = alg::prefix_distance(&a.as_i64(), &b.as_i64());
    Ok(check_dist_cutoff(dist, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn prefix_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    Ok(alg::prefix_similarity(&a.as_i64(), &b.as_i64()))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn prefix_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::prefix_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let nd = alg::normalized_distance(dist, max_v);
    Ok(check_sim_f64_cutoff(nd, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn prefix_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::prefix_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let ns = alg::normalized_similarity(dist, max_v);
    Ok(check_sim_f64_cutoff(ns, score_cutoff))
}

// ===========================================================================
// POSTFIX
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn postfix_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    let dist = alg::postfix_distance(&a.as_i64(), &b.as_i64());
    Ok(check_dist_cutoff(dist, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn postfix_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<usize>,
) -> PyResult<usize> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0usize);
    Ok(alg::postfix_similarity(&a.as_i64(), &b.as_i64()))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn postfix_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 1.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::postfix_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let nd = alg::normalized_distance(dist, max_v);
    Ok(check_sim_f64_cutoff(nd, score_cutoff.map(|c| 1.0 - c)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn postfix_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (a, b) = get_seqs!(py, s1, s2, &processor, 0.0f64);
    let av = a.as_i64();
    let bv = b.as_i64();
    let dist = alg::postfix_distance(&av, &bv);
    let max_v = av.len().max(bv.len());
    let ns = alg::normalized_similarity(dist, max_v);
    Ok(check_sim_f64_cutoff(ns, score_cutoff))
}
