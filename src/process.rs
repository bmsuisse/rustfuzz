use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::BinaryHeap;

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct ScoreItem {
    pub score: f64,
    pub idx: usize,
}

impl Eq for ScoreItem {}

impl PartialOrd for ScoreItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for ScoreItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

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
        let name = name.strip_prefix("fuzz_").unwrap_or(name);
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

/// Score a raw *mut PyObject against a pre-built query (q_slice, q_hist, q_pm).
/// Returns None if item should be skipped (filtered or non-string).
///
/// SAFETY: raw must be a valid borrowed ptr (GIL held, object alive).
#[inline(always)]
unsafe fn score_raw(
    stype: ScorerType,
    raw: *mut pyo3::ffi::PyObject,
    q_slice: &[u8],
    q_hist: &[i32; 256],
    q_pm: &crate::algorithms::PatternMask64<u8>,
    use_pm: bool,
    score_cutoff: Option<f64>,
    cache: Option<&crate::fuzz::QueryTokenCache>,
) -> Option<(f64, &'static [u8])> {
    // Fast None / null guard
    if raw.is_null() || raw == pyo3::ffi::Py_None() { return None; }

    let c_slice: &[u8];

    if pyo3::ffi::PyUnicode_Check(raw) != 0 {
        let mut length: isize = 0;
        let ptr = pyo3::ffi::PyUnicode_AsUTF8AndSize(raw, &mut length);
        if ptr.is_null() { return None; }
        let s = std::slice::from_raw_parts(ptr as *const u8, length as usize);
        if !s.is_ascii() { return None; }
        c_slice = s;
    } else if pyo3::ffi::PyBytes_Check(raw) != 0 {
        let s = std::slice::from_raw_parts(
            pyo3::ffi::PyBytes_AsString(raw) as *const u8,
            pyo3::ffi::PyBytes_Size(raw) as usize,
        );
        c_slice = s;
    } else {
        return None;
    }

    if let Some(score) = score_bytes_parallel(stype, c_slice, q_slice, q_hist, q_pm, use_pm, score_cutoff, cache) {
        let static_slice: &'static [u8] = std::mem::transmute(c_slice);
        Some((score, static_slice))
    } else {
        None
    }
}

/// Parallel scorer for a pre-extracted byte slice (no GIL needed).
/// Used by the Rayon parallel path in extract_parallel.
#[inline(always)]
fn score_bytes_parallel(
    stype: ScorerType,
    c_slice: &[u8],
    q_slice: &[u8],
    q_hist: &[i32; 256],
    q_pm: &crate::algorithms::PatternMask64<u8>,
    use_pm: bool,
    score_cutoff: Option<f64>,
    cache: Option<&crate::fuzz::QueryTokenCache>,
) -> Option<f64> {
    if matches!(stype, ScorerType::Ratio | ScorerType::QRatio) {
        let q_len = q_slice.len();
        let lensum = q_len + c_slice.len();
        if lensum == 0 {
            return if score_cutoff.map_or(true, |co| 100.0 >= co) { Some(100.0) } else { None };
        }

        let allowed_edits = score_cutoff
            .map(|co| (lensum as f64 * (1.0 - co / 100.0)).max(0.0).floor() as usize);

        if let Some(max_ed) = allowed_edits {
            if q_slice.len().abs_diff(c_slice.len()) > max_ed { return None; }
            let mut c_hist = [0i32; 256];
            for &c in c_slice { c_hist[c as usize] += 1; }
            let diff: i32 = q_hist.iter().zip(c_hist.iter()).map(|(&q, &c)| (q - c).abs()).sum();
            if diff as usize > max_ed { return None; }
        }

        let dist = if use_pm {
            let lcs = crate::algorithms::lcs_from_pm64(q_pm, q_len, c_slice, allowed_edits);
            (q_len + c_slice.len()) - 2 * lcs
        } else {
            crate::algorithms::indel_distance(q_slice, c_slice, allowed_edits)
        };

        if dist == usize::MAX { return None; }
        let score = (1.0 - dist as f64 / lensum as f64) * 100.0;
        return if score_cutoff.map_or(true, |c| score >= c) { Some(score) } else { None };
    }

    let score = match stype {
         ScorerType::WRatio => crate::fuzz::wratio_bytes(q_slice, c_slice, score_cutoff, cache),
         ScorerType::PartialRatio => crate::fuzz::partial_ratio_bytes(q_slice, c_slice, score_cutoff, cache),
         ScorerType::TokenSortRatio => crate::fuzz::token_sort_ratio_bytes(q_slice, c_slice, score_cutoff, cache),
         ScorerType::PartialTokenSortRatio => crate::fuzz::partial_token_sort_ratio_bytes(q_slice, c_slice, score_cutoff, cache),
         ScorerType::TokenSetRatio => crate::fuzz::token_set_ratio_bytes(q_slice, c_slice, score_cutoff, cache),
         ScorerType::PartialTokenSetRatio => crate::fuzz::partial_token_set_ratio_bytes(q_slice, c_slice, score_cutoff, cache),
         ScorerType::TokenRatio => crate::fuzz::token_ratio_bytes(q_slice, c_slice, score_cutoff, cache),
         ScorerType::PartialTokenRatio => crate::fuzz::partial_token_ratio_bytes(q_slice, c_slice, score_cutoff, cache),
         _ => 0.0,
    };
    
    if score_cutoff.map_or(true, |c| score >= c) { Some(score) } else { None }
}

/// Minimum batch size to justify Rayon parallelism (avoids overhead on small batches)
const PARALLEL_THRESHOLD: usize = 32;

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
    let ratio_fast = processor.is_none()
        && scorer_obj.is_none()
        && q_is_ascii
        && !matches!(stype, ScorerType::Unknown);

    if ratio_fast {
        if let Ok(crate::types::Seq::Ascii(q_slice)) = crate::types::extract_single(&processed_query) {
            let use_pm = q_len <= 64 && q_len > 0;
            let mut q_pm = crate::algorithms::PatternMask64::<u8>::new();
            if use_pm {
                for (i, &c) in q_slice.iter().enumerate() {
                    q_pm.insert(c, 1u64 << i);
                }
            }
            
            let token_cache_val = if matches!(stype, ScorerType::WRatio | ScorerType::TokenSortRatio | ScorerType::TokenSetRatio | ScorerType::TokenRatio | ScorerType::PartialTokenSortRatio | ScorerType::PartialTokenSetRatio | ScorerType::PartialTokenRatio) {
                let q_str = unsafe { std::str::from_utf8_unchecked(q_slice) };
                Some(crate::fuzz::QueryTokenCache::new(q_str))
            } else { None };
            let token_cache = token_cache_val.as_ref();

            if let Ok(list) = choices.downcast::<pyo3::types::PyList>() {
                let n = list.len();

                // --- Rayon parallel path for large lists ---
                if n >= PARALLEL_THRESHOLD {
                    // Phase 1: extract byte slices without GIL refcount changes
                    let list_ptr = list.as_ptr();
                    let mut slices: Vec<Option<&'static [u8]>> = Vec::with_capacity(n);
                    for idx in 0..n {
                        let raw = unsafe { pyo3::ffi::PyList_GetItem(list_ptr, idx as isize) };
                        if raw.is_null() || raw == unsafe { pyo3::ffi::Py_None() } {
                            slices.push(None);
                            continue;
                        }
                        let s: Option<&'static [u8]> = if unsafe { pyo3::ffi::PyUnicode_Check(raw) } != 0 {
                            let mut length: isize = 0;
                            let ptr = unsafe { pyo3::ffi::PyUnicode_AsUTF8AndSize(raw, &mut length) };
                            if ptr.is_null() { None } else {
                                let s = unsafe { std::slice::from_raw_parts(ptr as *const u8, length as usize) };
                                if s.is_ascii() { Some(unsafe { std::mem::transmute(s) }) } else { None }
                            }
                        } else if unsafe { pyo3::ffi::PyBytes_Check(raw) } != 0 {
                            let len = unsafe { pyo3::ffi::PyBytes_Size(raw) } as usize;
                            let ptr = unsafe { pyo3::ffi::PyBytes_AsString(raw) } as *const u8;
                            Some(unsafe { std::mem::transmute(std::slice::from_raw_parts(ptr, len)) })
                        } else { None };
                        slices.push(s);
                    }

                    // Phase 2: parallel scoring — GIL released
                    let q_hist_ref = &q_hist;
                    let q_pm_ref = &q_pm;
                    
                    let init_cutoff = score_cutoff.unwrap_or(0.0).max(0.0);
                    let global_cutoff = AtomicU64::new(init_cutoff.to_bits());
                    let limit_val = limit.unwrap_or(usize::MAX);

                    let local_heaps: Vec<BinaryHeap<Reverse<ScoreItem>>> = py.allow_threads(|| {
                        slices.par_iter().enumerate().fold(
                            || BinaryHeap::with_capacity(limit_val.min(100)),
                            |mut heap, (idx, opt)| {
                                let current_cutoff = f64::from_bits(global_cutoff.load(Ordering::Relaxed));
                                if let Some(c_slice) = opt {
                                    if let Some(score) = score_bytes_parallel(stype, *c_slice, q_slice, q_hist_ref, q_pm_ref, use_pm, Some(current_cutoff), token_cache) {
                                        if heap.len() < limit_val {
                                            heap.push(Reverse(ScoreItem { score, idx }));
                                            if heap.len() == limit_val {
                                                if let Some(Reverse(min_item)) = heap.peek() {
                                                    global_cutoff.fetch_max(min_item.score.to_bits(), Ordering::Relaxed);
                                                }
                                            }
                                        } else {
                                            if let Some(mut peek) = heap.peek_mut() {
                                                if score > peek.0.score {
                                                    peek.0 = ScoreItem { score, idx };
                                                }
                                            }
                                            if let Some(Reverse(min_item)) = heap.peek() {
                                                global_cutoff.fetch_max(min_item.score.to_bits(), Ordering::Relaxed);
                                            }
                                        }
                                    }
                                }
                                heap
                            }
                        ).collect()
                    });

                    // Phase 3: re-acquire GIL, INCREF winners
                    let mut primitive_results: Vec<(usize, f64)> = local_heaps
                        .into_iter()
                        .flat_map(|heap| heap.into_iter().map(|Reverse(item)| (item.idx, item.score)))
                        .collect();
                    primitive_results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    if let Some(l) = limit {
                        primitive_results.truncate(l);
                    }
                    for (idx, score) in primitive_results {
                        let raw = unsafe { pyo3::ffi::PyList_GetItem(list_ptr, idx as isize) };
                        let obj = unsafe { pyo3::ffi::Py_INCREF(raw); PyObject::from_owned_ptr(py, raw) };
                        results.push((obj, score, idx));
                    }
                    return Ok(results);
                }

                // --- Sequential list path ---
                let list_ptr = list.as_ptr();
                
                let init_cutoff = score_cutoff.unwrap_or(0.0).max(0.0);
                let mut current_cutoff = init_cutoff;
                let limit_val = limit.unwrap_or(usize::MAX);

                // Fast path for extractOne / limit=1
                if limit_val == 1 {
                    let mut best_score: f64 = -1.0;
                    let mut best_idx: usize = usize::MAX;

                    for idx in 0..n {
                        let raw = unsafe { pyo3::ffi::PyList_GetItem(list_ptr, idx as isize) };
                        if let Some((score, _)) = unsafe { score_raw(stype,raw, q_slice, &q_hist, &q_pm, use_pm, Some(current_cutoff), token_cache) } {
                            if score > best_score {
                                best_score = score;
                                best_idx = idx;
                                current_cutoff = current_cutoff.max(score);
                            }
                        }
                    }

                    if best_idx != usize::MAX && best_score >= init_cutoff {
                        let raw = unsafe { pyo3::ffi::PyList_GetItem(list_ptr, best_idx as isize) };
                        let obj = unsafe { pyo3::ffi::Py_INCREF(raw); PyObject::from_owned_ptr(py, raw) };
                        results.push((obj, best_score, best_idx));
                    }
                    return Ok(results);
                }

                // Fast path for small arrays without BinaryHeap overhead
                if n <= 256 || limit_val >= n {
                    let mut items: Vec<ScoreItem> = Vec::with_capacity(n.min(256));
                    for idx in 0..n {
                        let raw = unsafe { pyo3::ffi::PyList_GetItem(list_ptr, idx as isize) };
                        if let Some((score, _)) = unsafe { score_raw(stype,raw, q_slice, &q_hist, &q_pm, use_pm, Some(init_cutoff), token_cache) } {
                            items.push(ScoreItem { score, idx });
                        }
                    }
                    items.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    if let Some(l) = limit { items.truncate(l); }
                    
                    for item in items {
                        let raw = unsafe { pyo3::ffi::PyList_GetItem(list_ptr, item.idx as isize) };
                        let obj = unsafe { pyo3::ffi::Py_INCREF(raw); PyObject::from_owned_ptr(py, raw) };
                        results.push((obj, item.score, item.idx));
                    }
                    return Ok(results);
                }

                let mut heap = BinaryHeap::with_capacity(limit_val.min(100));

                for idx in 0..n {
                    let raw = unsafe { pyo3::ffi::PyList_GetItem(list_ptr, idx as isize) };
                    if let Some((score, _)) = unsafe { score_raw(stype,raw, q_slice, &q_hist, &q_pm, use_pm, Some(current_cutoff), token_cache) } {
                        if heap.len() < limit_val {
                            heap.push(Reverse(ScoreItem { score, idx }));
                        } else {
                            if let Some(mut peek) = heap.peek_mut() {
                                if score > peek.0.score {
                                    peek.0 = ScoreItem { score, idx };
                                }
                            }
                        }
                        if heap.len() == limit_val {
                            if let Some(Reverse(min_item)) = heap.peek() {
                                current_cutoff = current_cutoff.max(min_item.score);
                            }
                        }
                    }
                }
                
                let mut primitive_results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
                primitive_results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if let Some(l) = limit { primitive_results.truncate(l); }
                for (idx, score) in primitive_results {
                    let raw = unsafe { pyo3::ffi::PyList_GetItem(list_ptr, idx as isize) };
                    let obj = unsafe { pyo3::ffi::Py_INCREF(raw); PyObject::from_owned_ptr(py, raw) };
                    results.push((obj, score, idx));
                }
                return Ok(results);
            } else if let Ok(tup) = choices.downcast::<pyo3::types::PyTuple>() {
                let tup_ptr = tup.as_ptr();
                let n = tup.len();

                let init_cutoff = score_cutoff.unwrap_or(0.0).max(0.0);
                let mut current_cutoff = init_cutoff;
                let limit_val = limit.unwrap_or(usize::MAX);

                // Fast path for extractOne / limit=1
                if limit_val == 1 {
                    let mut best_score: f64 = -1.0;
                    let mut best_idx: usize = usize::MAX;

                    for idx in 0..n {
                        let raw = unsafe { pyo3::ffi::PyTuple_GetItem(tup_ptr, idx as isize) };
                        if let Some((score, _)) = unsafe { score_raw(stype,raw, q_slice, &q_hist, &q_pm, use_pm, Some(current_cutoff), token_cache) } {
                            if score > best_score {
                                best_score = score;
                                best_idx = idx;
                                current_cutoff = current_cutoff.max(score);
                            }
                        }
                    }

                    if best_idx != usize::MAX && best_score >= init_cutoff {
                        let raw = unsafe { pyo3::ffi::PyTuple_GetItem(tup_ptr, best_idx as isize) };
                        let obj = unsafe { pyo3::ffi::Py_INCREF(raw); PyObject::from_owned_ptr(py, raw) };
                        results.push((obj, best_score, best_idx));
                    }
                    return Ok(results);
                }

                // Fast path for small arrays without BinaryHeap overhead
                if n <= 256 || limit_val >= n {
                    let mut items: Vec<ScoreItem> = Vec::with_capacity(n.min(256));
                    for idx in 0..n {
                        let raw = unsafe { pyo3::ffi::PyTuple_GetItem(tup_ptr, idx as isize) };
                        if let Some((score, _)) = unsafe { score_raw(stype,raw, q_slice, &q_hist, &q_pm, use_pm, Some(init_cutoff), token_cache) } {
                            items.push(ScoreItem { score, idx });
                        }
                    }
                    items.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    if let Some(l) = limit { items.truncate(l); }
                    
                    for item in items {
                        let raw = unsafe { pyo3::ffi::PyTuple_GetItem(tup_ptr, item.idx as isize) };
                        let obj = unsafe { pyo3::ffi::Py_INCREF(raw); PyObject::from_owned_ptr(py, raw) };
                        results.push((obj, item.score, item.idx));
                    }
                    return Ok(results);
                }

                let mut heap = BinaryHeap::with_capacity(limit_val.min(100));

                for idx in 0..n {
                    let raw = unsafe { pyo3::ffi::PyTuple_GetItem(tup_ptr, idx as isize) };
                    if let Some((score, _)) = unsafe { score_raw(stype,raw, q_slice, &q_hist, &q_pm, use_pm, Some(current_cutoff), token_cache) } {
                        if heap.len() < limit_val {
                            heap.push(Reverse(ScoreItem { score, idx }));
                        } else {
                            if let Some(mut peek) = heap.peek_mut() {
                                if score > peek.0.score {
                                    peek.0 = ScoreItem { score, idx };
                                }
                            }
                        }
                        if heap.len() == limit_val {
                            if let Some(Reverse(min_item)) = heap.peek() {
                                current_cutoff = current_cutoff.max(min_item.score);
                            }
                        }
                    }
                }

                let mut primitive_results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
                primitive_results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if let Some(l) = limit { primitive_results.truncate(l); }
                for (idx, score) in primitive_results {
                    let raw = unsafe { pyo3::ffi::PyTuple_GetItem(tup_ptr, idx as isize) };
                    let obj = unsafe { pyo3::ffi::Py_INCREF(raw); PyObject::from_owned_ptr(py, raw) };
                    results.push((obj, score, idx));
                }
                return Ok(results);
            } else if let Ok(dict) = choices.downcast::<pyo3::types::PyDict>() {
                // Dict fast path: score values, return (value, score, key_index) — key unused, value is the string
                let dict_ptr = dict.as_ptr();
                let mut ppos: pyo3::ffi::Py_ssize_t = 0;
                let mut key_ptr: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
                let mut val_ptr: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
                let mut idx = 0usize;
                
                let init_cutoff = score_cutoff.unwrap_or(0.0).max(0.0);
                let mut current_cutoff = init_cutoff;
                let limit_val = limit.unwrap_or(usize::MAX);

                // Fast path for extractOne
                if limit_val == 1 {
                    let mut best_score: f64 = -1.0;
                    let mut best_idx: usize = usize::MAX;
                    let mut best_val_ptr: *mut pyo3::ffi::PyObject = std::ptr::null_mut();

                    loop {
                        let has_next = unsafe { pyo3::ffi::PyDict_Next(dict_ptr, &mut ppos, &mut key_ptr, &mut val_ptr) };
                        if has_next == 0 { break; }
                        if let Some((score, _)) = unsafe { score_raw(stype,val_ptr, q_slice, &q_hist, &q_pm, use_pm, Some(current_cutoff), token_cache) } {
                            if score > best_score {
                                best_score = score;
                                best_idx = idx;
                                best_val_ptr = val_ptr;
                                current_cutoff = current_cutoff.max(score);
                            }
                        }
                        idx += 1;
                    }

                    if best_idx != usize::MAX && best_score >= init_cutoff && !best_val_ptr.is_null() {
                        let obj = unsafe { pyo3::ffi::Py_INCREF(best_val_ptr); PyObject::from_owned_ptr(py, best_val_ptr) };
                        results.push((obj, best_score, best_idx));
                    }
                    return Ok(results);
                }

                // If not limit 1, collect all valid items - we typically don't heap dictionary traversal due to Python's API footprint
                loop {
                    let has_next = unsafe { pyo3::ffi::PyDict_Next(dict_ptr, &mut ppos, &mut key_ptr, &mut val_ptr) };
                    if has_next == 0 { break; }
                    if let Some((score, _)) = unsafe { score_raw(stype,val_ptr, q_slice, &q_hist, &q_pm, use_pm, score_cutoff, token_cache) } {
                        let obj = unsafe { pyo3::ffi::Py_INCREF(val_ptr); PyObject::from_owned_ptr(py, val_ptr) };
                        results.push((obj, score, idx));
                    }
                    idx += 1;
                }
            } else {
                // Generic iterator fallback
                for (idx, item_res) in choices.try_iter()?.enumerate() {
                    let item = item_res?;
                    let raw = item.as_ptr();
                    if let Some((score, _)) = unsafe { score_raw(stype,raw, q_slice, &q_hist, &q_pm, use_pm, score_cutoff, token_cache) } {
                        results.push((item.clone().unbind(), score, idx));
                    }
                }
            }

            results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(l) = limit { results.truncate(l); }
            return Ok(results);
        }
    }

    // ---- General path: custom scorers, processors, or non-ASCII ----
    let iter = choices.try_iter()?;
    for (idx, choice_res) in iter.enumerate() {
        let choice = choice_res?;

        if q_is_ascii {
            if let Some(cutoff) = score_cutoff {
                if let Ok(crate::types::Seq::Ascii(slice)) = crate::types::extract_single(&choice) {
                    let lensum = q_len + slice.len();
                    if lensum > 0 {
                        let allowed_edits = (lensum as f64 * (1.0 - cutoff / 100.0)).max(0.0).floor() as usize;
                        let mut c_hist = [0i32; 256];
                        for &c in slice { c_hist[c as usize] += 1; }
                        let hist_diff: i32 = q_hist.iter().zip(c_hist.iter()).map(|(&q, &c)| (q - c).abs()).sum();
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

/// Compute a full N×M score matrix between two lists of strings.
/// Returns a flat Vec<f64> of length N*M, row-major (queries × choices).
/// Rayon-parallel over query rows, so this scales linearly with CPU cores.
#[pyfunction]
#[pyo3(signature = (queries, choices, scorer_name, _scorer_obj, processor=None, score_cutoff=None))]
pub fn cdist(
    py: Python<'_>,
    queries: &Bound<'_, PyAny>,
    choices: &Bound<'_, PyAny>,
    scorer_name: &str,
    _scorer_obj: Option<PyObject>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<(Vec<f64>, usize, usize)> {
    let stype = ScorerType::from_str(scorer_name);

    // Collect and process queries
    let q_list: Vec<PyObject> = queries.try_iter()?
        .map(|r| r.map(|o| o.unbind()))
        .collect::<PyResult<_>>()?;
    let c_list: Vec<PyObject> = choices.try_iter()?
        .map(|r| r.map(|o| o.unbind()))
        .collect::<PyResult<_>>()?;

    let nq = q_list.len();
    let nc = c_list.len();

    // Extract ASCII byte slices with optional processor.
    // Instead of cloning into Vec<u8>, we keep the PyObjects alive and borrow their memory.
    // Returns None for non-ASCII / non-string items.
    let extract_bytes = |obj: &PyObject| -> Option<&'static [u8]> {
        let bound = obj.bind(py);
        let processed = if let Some(ref proc) = processor {
            match proc.call1(py, (bound,)) {
                Ok(r) => r.into_bound(py),
                Err(_) => return None,
            }
        } else {
            bound.clone()
        };
        match crate::types::extract_single(&processed) {
            Ok(crate::types::Seq::Ascii(s)) => Some(unsafe { std::mem::transmute(s) }),
            _ => None,
        }
    };

    let q_bytes: Vec<Option<&'static [u8]>> = q_list.iter().map(|o| extract_bytes(o)).collect();
    let c_bytes: Vec<Option<&'static [u8]>> = c_list.iter().map(|o| extract_bytes(o)).collect();

    // Build per-query histogram and PM for fast scoring
    struct QueryCache {
        bytes: &'static [u8],
        hist: [i32; 256],
        pm: crate::algorithms::PatternMask64<u8>,
        use_pm: bool,
        token_cache: Option<crate::fuzz::QueryTokenCache>,
    }

    let is_token_scorer = matches!(stype, ScorerType::WRatio | ScorerType::TokenSortRatio | ScorerType::TokenSetRatio | ScorerType::TokenRatio | ScorerType::PartialTokenSortRatio | ScorerType::PartialTokenSetRatio | ScorerType::PartialTokenRatio);

    let q_caches: Vec<Option<QueryCache>> = q_bytes.iter().map(|opt| {
        opt.map(|b| {
            let mut hist = [0i32; 256];
            for &c in b { hist[c as usize] += 1; }
            let use_pm = b.len() <= 64 && !b.is_empty();
            let mut pm = crate::algorithms::PatternMask64::<u8>::new();
            if use_pm {
                for (i, &c) in b.iter().enumerate() {
                    pm.insert(c, 1u64 << i);
                }
            }
            let token_cache = if is_token_scorer {
                let s = unsafe { std::str::from_utf8_unchecked(b) };
                Some(crate::fuzz::QueryTokenCache::new(s))
            } else { None };
            QueryCache { bytes: b, hist, pm, use_pm, token_cache }
        })
    }).collect();

    let use_parallel = (nq * nc) >= 1000;

    let matrix: Vec<f64> = py.allow_threads(|| {
        if use_parallel {
            // Parallel computation: each row (query) is a Rayon task
            q_caches.par_iter().flat_map(|q_opt| {
                (0..nc).map(|j| {
                    let (qc, cb) = match (q_opt, &c_bytes[j]) {
                        (Some(q), Some(c)) => (q, c),
                        _ => return 0.0,
                    };
                    let score = score_bytes_parallel(
                        stype, *cb, qc.bytes, &qc.hist, &qc.pm, qc.use_pm, score_cutoff, qc.token_cache.as_ref()
                    );
                    score.unwrap_or(0.0)
                }).collect::<Vec<f64>>()
            }).collect()
        } else {
            // Sequential fast path for small data
            q_caches.iter().flat_map(|q_opt| {
                (0..nc).map(|j| {
                    let (qc, cb) = match (q_opt, &c_bytes[j]) {
                        (Some(q), Some(c)) => (q, c),
                        _ => return 0.0,
                    };
                    let score = score_bytes_parallel(
                        stype, *cb, qc.bytes, &qc.hist, &qc.pm, qc.use_pm, score_cutoff, qc.token_cache.as_ref()
                    );
                    score.unwrap_or(0.0)
                }).collect::<Vec<f64>>()
            }).collect()
        }
    });

    Ok((matrix, nq, nc))
}
