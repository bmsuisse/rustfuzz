// SPDX-License-Identifier: MIT
// Batch processing utilities using native Rust implementations.

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::fuzz::{
    fuzz_ratio, fuzz_qratio, fuzz_wratio, fuzz_partial_ratio,
    fuzz_token_sort_ratio, fuzz_partial_token_sort_ratio,
    fuzz_token_set_ratio, fuzz_partial_token_set_ratio,
    fuzz_token_ratio, fuzz_partial_token_ratio
};
// removed unused imports

// We will use a fast native match for known scorers
#[derive(Clone, Copy)]
pub enum ScorerType {
    Ratio,
    QRatio,
    WRatio,
    PartialRatio,
    TokenSortRatio,
    PartialTokenSortRatio,
    TokenSetRatio,
    PartialTokenSetRatio,
    TokenRatio,
    PartialTokenRatio,
    Unknown,
}

impl ScorerType {
    pub fn from_str(name: &str) -> Self {
        match name {
            "ratio" => ScorerType::Ratio,
            "qratio" => ScorerType::QRatio,
            "wratio" => ScorerType::WRatio,
            "partial_ratio" => ScorerType::PartialRatio,
            "token_sort_ratio" => ScorerType::TokenSortRatio,
            "partial_token_sort_ratio" => ScorerType::PartialTokenSortRatio,
            "token_set_ratio" => ScorerType::TokenSetRatio,
            "partial_token_set_ratio" => ScorerType::PartialTokenSetRatio,
            "token_ratio" => ScorerType::TokenRatio,
            "partial_token_ratio" => ScorerType::PartialTokenRatio,
            _ => ScorerType::Unknown,
        }
    }
}

pub fn execute_scorer(
    py: Python<'_>,
    stype: ScorerType,
    scorer_obj: Option<&PyObject>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    match stype {
        ScorerType::Ratio => fuzz_ratio(py, s1, s2, processor, score_cutoff),
        ScorerType::QRatio => fuzz_qratio(py, s1, s2, processor, score_cutoff),
        ScorerType::WRatio => fuzz_wratio(py, s1, s2, processor, score_cutoff),
        ScorerType::PartialRatio => fuzz_partial_ratio(py, s1, s2, processor, score_cutoff),
        ScorerType::TokenSortRatio => fuzz_token_sort_ratio(py, s1, s2, processor, score_cutoff),
        ScorerType::PartialTokenSortRatio => fuzz_partial_token_sort_ratio(py, s1, s2, processor, score_cutoff),
        ScorerType::TokenSetRatio => fuzz_token_set_ratio(py, s1, s2, processor, score_cutoff),
        ScorerType::PartialTokenSetRatio => fuzz_partial_token_set_ratio(py, s1, s2, processor, score_cutoff),
        ScorerType::TokenRatio => fuzz_token_ratio(py, s1, s2, processor, score_cutoff),
        ScorerType::PartialTokenRatio => fuzz_partial_token_ratio(py, s1, s2, processor, score_cutoff),
        ScorerType::Unknown => {
            if let Some(func) = scorer_obj {
                // Call python function dynamically
                let kwargs = pyo3::types::PyDict::new(py);
                if let Some(proc) = processor {
                    kwargs.set_item("processor", proc)?;
                }
                if let Some(sc) = score_cutoff {
                    kwargs.set_item("score_cutoff", sc)?;
                }
                let res = func.call(py, (s1, s2), Some(&kwargs))?;
                res.extract::<f64>(py)
            } else {
                Ok(0.0)
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (query, choices, scorer_name, scorer_obj, processor=None, limit=Some(5), score_cutoff=None))]
pub fn extract(
    py: Python<'_>,
    query: &Bound<'_, PyAny>,
    choices: &Bound<'_, PyAny>,
    scorer_name: &str,
    scorer_obj: Option<PyObject>,
    processor: Option<PyObject>,
    limit: Option<usize>,
    score_cutoff: Option<f64>,
) -> PyResult<Vec<(PyObject, f64, usize)>> {
    let mut results = Vec::new();
    let iter = choices.try_iter()?;
    let stype = ScorerType::from_str(scorer_name);
    
    // Process query once if we can.
    // However, execute_scorer expects raw s1, s2 so processor gets called repeatedly
    // inside `execute_scorer`. RapidFuzz optimizes this by pre-processing query!
    
    // Pre-process query if processor is provided
    let processed_query = if let Some(ref proc) = processor {
        proc.call1(py, (query,))?.into_bound(py)
    } else {
        query.clone()
    };





    // Pre-calculate L1 Character bounds for early rejection natively
    let mut q_hist = [0i32; 256];
    let mut q_is_ascii = false;
    let mut q_len = 0;
    
    if let Ok(q_seq) = crate::types::extract_single(&processed_query) {
        if let crate::types::Seq::Ascii(slice) = q_seq {
            q_is_ascii = true;
            q_len = slice.len();
            for &c in slice {
                q_hist[c as usize] += 1;
            }
        }
    }
    for (idx, choice_res) in iter.enumerate() {

        let choice = choice_res?;

        // Fast L1 Bounds Rejection (Only for ASCII matching)
        if q_is_ascii && score_cutoff.is_some() {
            let cutoff = score_cutoff.unwrap();
            if let Ok(crate::types::Seq::Ascii(slice)) = crate::types::extract_single(&choice) {
                // Determine `max_dist` dynamically from `cutoff` assuming Ratio
                // length sum
                let lensum = q_len + slice.len();
                if lensum > 0 {
                    let mut max_dist = lensum as f64 * (1.0 - cutoff / 100.0);
                    if max_dist < 0.0 { max_dist = 0.0; }
                    let allowed_edits = max_dist.floor() as usize;

                    // Compute L1 diff
                    let mut hist_diff = 0;
                    let mut c_hist = [0i32; 256];
                    for &c in slice {
                        c_hist[c as usize] += 1;
                    }
                    for i in 0..256 {
                        hist_diff += (q_hist[i] - c_hist[i]).abs();
                    }
                    
                    // If the character disparity is strictly larger than total errors allowed,
                    // we can mathematically skip this string completely!
                    if (hist_diff as usize) > allowed_edits {
                        continue;
                    }
                }
            }
        }
        
        let score = execute_scorer(
            py,
            stype,
            scorer_obj.as_ref(),
            &processed_query,
            &choice,
            processor.as_ref().map(|p| p.clone_ref(py)), 
            score_cutoff,
        )?;
        
        if score_cutoff.is_none() || score >= score_cutoff.unwrap() {
            results.push((choice.into(), score, idx));
        }
    }

    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(l) = limit {
        results.truncate(l);
    }

    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (query, choices, scorer_name, scorer_obj, processor=None, score_cutoff=None))]
pub fn extract_one(
    py: Python<'_>,
    query: &Bound<'_, PyAny>,
    choices: &Bound<'_, PyAny>,
    scorer_name: &str,
    scorer_obj: Option<PyObject>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<Option<(PyObject, f64, usize)>> {
    let results = extract(
        py,
        query,
        choices,
        scorer_name,
        scorer_obj,
        processor,
        Some(1),
        score_cutoff,
    )?;
    Ok(results.into_iter().next())
}

#[pyfunction]
#[pyo3(signature = (query, choices, scorer_name, scorer_obj, processor=None, score_cutoff=None))]
pub fn extract_iter(
    py: Python<'_>,
    query: &Bound<'_, PyAny>,
    choices: &Bound<'_, PyAny>,
    scorer_name: &str,
    scorer_obj: Option<PyObject>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<Vec<(PyObject, f64, usize)>> {
    extract(py, query, choices, scorer_name, scorer_obj, processor, None, score_cutoff)
}



