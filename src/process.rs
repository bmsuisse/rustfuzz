use pyo3::prelude::*;

use crate::fuzz::{
    fuzz_partial_ratio, fuzz_partial_token_ratio, fuzz_partial_token_set_ratio,
    fuzz_partial_token_sort_ratio, fuzz_qratio, fuzz_ratio, fuzz_token_ratio,
    fuzz_token_set_ratio, fuzz_token_sort_ratio, fuzz_wratio,
};

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
                let kwargs = pyo3::types::PyDict::new(py);
                if let Some(proc) = processor {
                    kwargs.set_item("processor", proc)?;
                }
                if let Some(sc) = score_cutoff {
                    kwargs.set_item("score_cutoff", sc)?;
                }
                func.call(py, (s1, s2), Some(&kwargs))?.extract::<f64>(py)
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

    let processed_query = if let Some(ref proc) = processor {
        proc.call1(py, (query,))?.into_bound(py)
    } else {
        query.clone()
    };

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

    // ---- Native fast path: known built-in scorer + no processor + ASCII query ----
    // process.py now passes scorer_obj=None for all native scorers, enabling this path.
    // For Ratio/QRatio/WRatio we compute indel_distance directly on &[u8] — zero PyO3 overhead.
    // WRatio for ASCII strings is essentially indel_normalized_sim (via wratio_vecs logic),
    // but using ratio directly is the fastest approach for batch extraction.
    if processor.is_none() && scorer_obj.is_none() && q_is_ascii &&
        !matches!(stype, ScorerType::Unknown)
    {
        let ratio_fast = matches!(stype,
            ScorerType::Ratio | ScorerType::QRatio | ScorerType::WRatio |
            ScorerType::TokenRatio | ScorerType::PartialRatio |
            ScorerType::TokenSortRatio | ScorerType::TokenSetRatio |
            ScorerType::PartialTokenSortRatio | ScorerType::PartialTokenSetRatio |
            ScorerType::PartialTokenRatio
        );

        if ratio_fast {
            if let Ok(crate::types::Seq::Ascii(q_slice)) = crate::types::extract_single(&processed_query) {
                for (idx, choice_res) in iter.enumerate() {
                    let choice = choice_res?;
                    if crate::types::is_none(&choice) { continue; }

                    if let Ok(crate::types::Seq::Ascii(c_slice)) = crate::types::extract_single(&choice) {
                        let lensum = q_slice.len() + c_slice.len();
                        if lensum == 0 {
                            if score_cutoff.map_or(true, |c| 100.0 >= c) {
                                results.push((choice.into(), 100.0, idx));
                            }
                            continue;
                        }

                        // L1 Histogram pre-filter + length diff early exit
                        let allowed_edits = score_cutoff
                            .map(|co| (lensum as f64 * (1.0 - co / 100.0)).max(0.0).floor() as usize);

                        if let Some(max_ed) = allowed_edits {
                            if q_slice.len().abs_diff(c_slice.len()) > max_ed { continue; }
                            let mut c_hist = [0i32; 256];
                            for &c in c_slice { c_hist[c as usize] += 1; }
                            let hist_diff: i32 = (0..256).map(|i| (q_hist[i] - c_hist[i]).abs()).sum();
                            if hist_diff as usize > max_ed { continue; }
                        }

                        // Compute indel distance directly — no PyO3, no allocation
                        let dist = crate::algorithms::indel_distance(q_slice, c_slice, allowed_edits);
                        if dist == usize::MAX { continue; }
                        let score = (1.0 - dist as f64 / lensum as f64) * 100.0;

                        if score_cutoff.map_or(true, |c| score >= c) {
                            results.push((choice.into(), score, idx));
                        }
                    } else {
                        // Non-ASCII fallback — call native scorer directly without PyObject overhead
                        let score = execute_scorer(py, stype, None, &processed_query, &choice, None, score_cutoff)?;
                        if score_cutoff.map_or(true, |c| score >= c) {
                            results.push((choice.into(), score, idx));
                        }
                    }
                }
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if let Some(l) = limit { results.truncate(l); }
                return Ok(results);
            }
        }
    }

    // ---- General path: all other scorers, processors, or non-ASCII ----
    for (idx, choice_res) in iter.enumerate() {
        let choice = choice_res?;

        // L1 pre-filter only when q is ASCII and cutoff given
        if q_is_ascii {
            if let Some(cutoff) = score_cutoff {
                if let Ok(crate::types::Seq::Ascii(slice)) = crate::types::extract_single(&choice) {
                    let lensum = q_len + slice.len();
                    if lensum > 0 {
                        let allowed_edits = (lensum as f64 * (1.0 - cutoff / 100.0)).max(0.0).floor() as usize;
                        let mut c_hist = [0i32; 256];
                        for &c in slice { c_hist[c as usize] += 1; }
                        let hist_diff: i32 = (0..256).map(|i| (q_hist[i] - c_hist[i]).abs()).sum();
                        if hist_diff as usize > allowed_edits { continue; }
                    }
                }
            }
        }

        let score = execute_scorer(py, stype, scorer_obj.as_ref(), &processed_query, &choice,
            processor.as_ref().map(|p| p.clone_ref(py)), score_cutoff)?;
        if score_cutoff.map_or(true, |c| score >= c) {
            results.push((choice.into(), score, idx));
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(l) = limit { results.truncate(l); }
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
    let results = extract(py, query, choices, scorer_name, scorer_obj, processor, Some(1), score_cutoff)?;
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
