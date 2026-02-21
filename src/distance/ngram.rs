use pyo3::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

use crate::types::{extract_single, get_processed_args, Seq};

// ---------------------------------------------------------------------------
// N-gram Similarity (Sorensen-Dice & Jaccard)
// ---------------------------------------------------------------------------

fn get_ngrams<T: Hash + Eq + Clone>(seq: &[T], n: usize) -> HashMap<Vec<T>, usize> {
    let mut counts = HashMap::new();
    if seq.len() < n || n == 0 {
        return counts;
    }
    for window in seq.windows(n) {
        *counts.entry(window.to_vec()).or_insert(0) += 1;
    }
    counts
}

pub fn sorensen_dice_similarity<T: Hash + Eq + Clone>(
    s1: &[T],
    s2: &[T],
    n: usize,
) -> f64 {
    if s1.is_empty() && s2.is_empty() { return 1.0; }
    if s1.len() < n || s2.len() < n { return 0.0; }
    
    let ngrams1 = get_ngrams(s1, n);
    let ngrams2 = get_ngrams(s2, n);
    
    let mut intersection = 0;
    for (k, v1) in &ngrams1 {
        if let Some(v2) = ngrams2.get(k) {
            intersection += v1.min(v2);
        }
    }
    
    let total_ngrams1 = s1.len() - n + 1;
    let total_ngrams2 = s2.len() - n + 1;
    
    (2.0 * intersection as f64) / ((total_ngrams1 + total_ngrams2) as f64)
}

pub fn jaccard_similarity<T: Hash + Eq + Clone>(
    s1: &[T],
    s2: &[T],
    n: usize,
) -> f64 {
    if s1.is_empty() && s2.is_empty() { return 1.0; }
    if s1.len() < n || s2.len() < n { return 0.0; }
    
    let ngrams1 = get_ngrams(s1, n);
    let ngrams2 = get_ngrams(s2, n);
    
    let mut intersection = 0;
    for (k, v1) in &ngrams1 {
        if let Some(v2) = ngrams2.get(k) {
            intersection += v1.min(v2);
        }
    }
    
    let total_ngrams1 = s1.len() - n + 1;
    let total_ngrams2 = s2.len() - n + 1;
    let union = total_ngrams1 + total_ngrams2 - intersection;
    
    if union == 0 { 1.0 } else { (intersection as f64) / (union as f64) }
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, n=2, processor=None, score_cutoff=None))]
pub fn sorensen_dice(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    n: usize,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (p1, p2) = get_processed_args(py, s1, s2, &processor)?;
    let v1 = extract_single(&p1)?;
    let v2 = extract_single(&p2)?;

    let sim = match (&v1, &v2) {
        (Seq::Ascii(a), Seq::Ascii(b)) => sorensen_dice_similarity(*a, *b, n),
        (Seq::U32(a), Seq::U32(b)) => sorensen_dice_similarity(a, b, n),
        (Seq::U64(a), Seq::U64(b)) => sorensen_dice_similarity(a, b, n),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Mismatched sequence types")),
    };

    if let Some(cutoff) = score_cutoff {
        if sim < cutoff {
            return Ok(0.0);
        }
    }
    
    Ok(sim)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, n=2, processor=None, score_cutoff=None))]
pub fn jaccard(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    n: usize,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (p1, p2) = get_processed_args(py, s1, s2, &processor)?;
    let v1 = extract_single(&p1)?;
    let v2 = extract_single(&p2)?;

    let sim = match (&v1, &v2) {
        (Seq::Ascii(a), Seq::Ascii(b)) => jaccard_similarity(*a, *b, n),
        (Seq::U32(a), Seq::U32(b)) => jaccard_similarity(a, b, n),
        (Seq::U64(a), Seq::U64(b)) => jaccard_similarity(a, b, n),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Mismatched sequence types")),
    };

    if let Some(cutoff) = score_cutoff {
        if sim < cutoff {
            return Ok(0.0);
        }
    }
    
    Ok(sim)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sorensen_dice, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard, m)?)?;
    Ok(())
}
