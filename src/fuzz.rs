// SPDX-License-Identifier: MIT
// Fuzz scorer implementations in Rust.
// All functions return 0.0 when score_cutoff is not met.
// Algorithms mirror fuzz_py.py exactly.

use pyo3::prelude::*;

use crate::algorithms as alg;
use crate::distance::initialize::ScoreAlignment;
use crate::types::{get_processed_args, extract_single, is_none};
use crate::dispatch_metric;

// ---------------------------------------------------------------------------
// Internal: normalized indel similarity (same as fuzz.ratio / 100)
// ---------------------------------------------------------------------------

fn indel_normalized_sim(av: &crate::types::Seq<'_>, bv: &crate::types::Seq<'_>, score_cutoff: Option<f64>) -> f64 {
    let lensum = av.len() + bv.len();
    if lensum == 0 {
        return 1.0;
    }
    
    let max_dist = score_cutoff.map(|mut c| {
        // e.g. cutoff = 90.0
        // max_dist = lensum - (cutoff / 100.0 * lensum)
        if c > 100.0 { c = 100.0; }
        let allowed = lensum as f64 * (1.0 - c / 100.0);
        // We can safely floor or ceil depending on bounds.
        // We floor it (or cast to usize directly which truncates).
        allowed.floor() as usize
    });
    
    // dispatch_metric passes exactly args. 
    // We updated `inject_bounds.py` to allow passing score_cutoff to `indel_distance`.
    let dist = dispatch_metric!(alg::indel_distance, av, bv, max_dist);
    
    if dist == usize::MAX {
        return 0.0; // Early exit cutoff fallback
    }
    
    1.0 - (dist as f64 / lensum as f64)
}

fn norm_sim_to_score(sim: f64) -> f64 {
    sim * 100.0
}

fn score_cutoff_check(score: f64, cutoff: Option<f64>) -> f64 {
    match cutoff {
        Some(c) if score < c => 0.0,
        _ => score,
    }
}

// ---------------------------------------------------------------------------
// Split string into tokens (mirrors fuzz_py._split_sequence)
// ---------------------------------------------------------------------------

fn split_tokens(s: &str) -> Vec<String> {
    s.split_whitespace().map(str::to_string).collect()
}

fn tokens_sort_key(s: &str) -> String {
    let mut tokens = split_tokens(s);
    tokens.sort_unstable();
    tokens.join(" ")
}

fn tokens_to_set_intersection_diff(
    s1: &str,
    s2: &str,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let tokens1: std::collections::HashSet<&str> = s1.split_whitespace().collect();
    let tokens2: std::collections::HashSet<&str> = s2.split_whitespace().collect();

    let mut intersection: Vec<String> =
        tokens1.intersection(&tokens2).map(|s| s.to_string()).collect();
    let mut diff1: Vec<String> = tokens1
        .difference(&tokens2)
        .map(|s| s.to_string())
        .collect();
    let mut diff2: Vec<String> = tokens2
        .difference(&tokens1)
        .map(|s| s.to_string())
        .collect();
    intersection.sort_unstable();
    diff1.sort_unstable();
    diff2.sort_unstable();
    (intersection, diff1, diff2)
}

fn str_to_i64_vec(s: &str) -> Vec<u64> {
    s.chars().map(|c| c as u64).collect()
}

fn indel_score_100(s1: &str, s2: &str) -> f64 {
    let av = str_to_i64_vec(s1);
    let bv = str_to_i64_vec(s2);
    norm_sim_to_score(indel_normalized_sim(&crate::types::Seq::U64(av), &crate::types::Seq::U64(bv), None))
}

// ===========================================================================
// ratio — normalized Indel similarity × 100
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;
    let score = norm_sim_to_score(indel_normalized_sim(&av, &bv, score_cutoff));
    Ok(score_cutoff_check(score, score_cutoff))
}

// ===========================================================================
// partial_ratio
// ===========================================================================

/// Find indel similarity of shorter within the best aligned window of longer.
/// Mirrors Python's _partial_ratio_impl with three loops:
///   Loop 1: prefix overhangs (shorter vs longer[:i] for i < s_len)
///   Loop 2: exact-size windows (shorter vs longer[i:i+s_len])
///   Loop 3: suffix overhangs (shorter vs longer[i:] for i > l_len-s_len)
fn partial_ratio_short_long(shorter: &[u64], longer: &[u64]) -> (f64, usize, usize) {
    let s_len = shorter.len();
    let l_len = longer.len();
    if s_len == 0 {
        // Both empty → perfect match (100); only shorter empty → no match (0)
        return if l_len == 0 { (100.0, 0, 0) } else { (0.0, 0, 0) };
    }
    if l_len == 0 {
        return (0.0, 0, 0);
    }

    let mut best_score = 0.0f64;
    let mut best_start = 0usize;
    let mut best_end = 0usize;

    let score_fn = |shorter: &[u64], window: &[u64]| -> f64 {
        let dist = alg::indel_distance(shorter, window, None);
        let lensum = shorter.len() + window.len();
        if lensum == 0 { 100.0 } else { (1.0 - dist as f64 / lensum as f64) * 100.0 }
    };

    // Loop 1: prefix overhangs — shorter vs longer[:i] for i in 1..s_len
    for i in 1..s_len.min(l_len) {
        let window = &longer[..i];
        let score = score_fn(shorter, window);
        if score > best_score {
            best_score = score; best_start = 0; best_end = i;
            if best_score == 100.0 { return (best_score, best_start, best_end); }
        }
    }

    // Loop 2: exact-size windows — shorter vs longer[i:i+s_len]
    let exact_end = if l_len >= s_len { l_len - s_len } else { 0 };
    for start in 0..=exact_end {
        let end = (start + s_len).min(l_len);
        let window = &longer[start..end];
        let score = score_fn(shorter, window);
        if score > best_score {
            best_score = score; best_start = start; best_end = end;
            if best_score == 100.0 { return (best_score, best_start, best_end); }
        }
    }

    // Loop 3: suffix overhangs — shorter vs longer[i:] for i in (l_len-s_len+1)..l_len
    let suffix_from = if l_len >= s_len { l_len - s_len + 1 } else { 0 };
    for i in suffix_from..l_len {
        let window = &longer[i..];
        if window.is_empty() { break; }
        let score = score_fn(shorter, window);
        if score > best_score {
            best_score = score; best_start = i; best_end = l_len;
            if best_score == 100.0 { return (best_score, best_start, best_end); }
        }
    }

    (best_score, best_start, best_end)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_partial_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;

    // Both empty → 100.0 (Python returns 100 for equal empty strings)
    if av.is_empty() && bv.is_empty() {
        return Ok(score_cutoff_check(100.0, score_cutoff));
    }

    let score = if av.len() <= bv.len() {
        let mut s = partial_ratio_short_long(&av.to_u64(), &bv.to_u64()).0;
        // When equal length, Python also tries the reversed order
        if s != 100.0 && av.len() == bv.len() {
            let s2_rev = partial_ratio_short_long(&bv.to_u64(), &av.to_u64()).0;
            if s2_rev > s { s = s2_rev; }
        }
        s
    } else {
        partial_ratio_short_long(&bv.to_u64(), &av.to_u64()).0
    };
    Ok(score_cutoff_check(score, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_partial_ratio_alignment(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<Option<ScoreAlignment>> {
    if is_none(s1) || is_none(s2) {
        return Ok(None);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;
    let s1_is_shorter = av.len() <= bv.len();

    let (score, src_start, src_end, dest_start, dest_end) = if s1_is_shorter {
        let (s, ds, de) = partial_ratio_short_long(&av.to_u64(), &bv.to_u64());
        // When equal length, Python also tries reversed order
        if s != 100.0 && av.len() == bv.len() {
            let (s2, ds2, de2) = partial_ratio_short_long(&bv.to_u64(), &av.to_u64());
            if s2 > s {
                // res2 uses longer=av as dest, shorter=bv as src
                // Python returns: ScoreAlignment(res2.score, res2.dest_start, res2.dest_end, res2.src_start, res2.src_end)
                return Ok(if let Some(c) = score_cutoff {
                    if s2 < c { None } else { Some(ScoreAlignment { score: s2, src_start: ds2, src_end: de2, dest_start: 0, dest_end: bv.len() }) }
                } else {
                    Some(ScoreAlignment { score: s2, src_start: ds2, src_end: de2, dest_start: 0, dest_end: bv.len() })
                });
            }
        }
        (s, 0, av.len(), ds, de)
    } else {
        let (s, ds, de) = partial_ratio_short_long(&bv.to_u64(), &av.to_u64());
        (s, ds, de, 0, bv.len())
    };

    if let Some(c) = score_cutoff {
        if score < c {
            return Ok(None);
        }
    }
    Ok(Some(ScoreAlignment {
        score,
        src_start,
        src_end,
        dest_start,
        dest_end,
    }))
}

// ===========================================================================
// token_sort_ratio
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_token_sort_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let a = extract_single(&a_obj)?;
    let b = extract_single(&b_obj)?;
    // Requires string input: convert back via chars
    let s1_str = seq_to_string(&a);
    let s2_str = seq_to_string(&b);
    let sorted1 = tokens_sort_key(&s1_str);
    let sorted2 = tokens_sort_key(&s2_str);
    let score = indel_score_100(&sorted1, &sorted2);
    Ok(score_cutoff_check(score, score_cutoff))
}

fn seq_to_string(seq: &crate::types::Seq<'_>) -> String {
    seq.to_string_lossy()
}

// ===========================================================================
// token_set_ratio
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_token_set_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let a = extract_single(&a_obj)?;
    let b = extract_single(&b_obj)?;
    let s1_str = seq_to_string(&a);
    let s2_str = seq_to_string(&b);

    // If both empty after tokenizing, return 0
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(&s1_str, &s2_str);
    if intersect.is_empty() && diff1.is_empty() && diff2.is_empty() {
        return Ok(0.0);
    }

    let t0 = intersect.join(" ");
    let t1 = if diff1.is_empty() {
        t0.clone()
    } else if t0.is_empty() {
        diff1.join(" ")
    } else {
        format!("{} {}", t0, diff1.join(" "))
    };
    let t2 = if diff2.is_empty() {
        t0.clone()
    } else if t0.is_empty() {
        diff2.join(" ")
    } else {
        format!("{} {}", t0, diff2.join(" "))
    };

    if intersect.is_empty() {
        // No intersection: compare the two unique-word sets directly
        let score = indel_score_100(&t1, &t2);
        return Ok(score_cutoff_check(score, score_cutoff));
    }

    let score = indel_score_100(&t0, &t1)
        .max(indel_score_100(&t0, &t2))
        .max(indel_score_100(&t1, &t2));
    Ok(score_cutoff_check(score, score_cutoff))
}

// ===========================================================================
// token_ratio
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_token_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let a = extract_single(&a_obj)?;
    let b = extract_single(&b_obj)?;
    let s1_str = seq_to_string(&a);
    let s2_str = seq_to_string(&b);

    // token_sort_ratio
    let sorted1 = tokens_sort_key(&s1_str);
    let sorted2 = tokens_sort_key(&s2_str);
    let tsr = indel_score_100(&sorted1, &sorted2);

    // token_set_ratio
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(&s1_str, &s2_str);
    let t0 = intersect.join(" ");
    let t1 = if diff1.is_empty() {
        t0.clone()
    } else if t0.is_empty() {
        diff1.join(" ")
    } else {
        format!("{} {}", t0, diff1.join(" "))
    };
    let t2 = if diff2.is_empty() {
        t0.clone()
    } else if t0.is_empty() {
        diff2.join(" ")
    } else {
        format!("{} {}", t0, diff2.join(" "))
    };
    let tset = if intersect.is_empty() {
        // No intersection: compare unique-word sets directly
        indel_score_100(&t1, &t2)
    } else {
        indel_score_100(&t0, &t1)
            .max(indel_score_100(&t0, &t2))
            .max(indel_score_100(&t1, &t2))
    };

    let score = tsr.max(tset);
    Ok(score_cutoff_check(score, score_cutoff))
}

// ===========================================================================
// partial_token_sort_ratio
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_partial_token_sort_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let a = extract_single(&a_obj)?;
    let b = extract_single(&b_obj)?;
    let s1_str = seq_to_string(&a);
    let s2_str = seq_to_string(&b);
    let sorted1 = tokens_sort_key(&s1_str);
    let sorted2 = tokens_sort_key(&s2_str);
    let sv1 = str_to_i64_vec(&sorted1);
    let sv2 = str_to_i64_vec(&sorted2);
    let score = if sv1.len() <= sv2.len() { partial_ratio_short_long(&sv1, &sv2).0 } else { partial_ratio_short_long(&sv2, &sv1).0 };
    Ok(score_cutoff_check(score, score_cutoff))
}

// ===========================================================================
// partial_token_set_ratio
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_partial_token_set_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let a = extract_single(&a_obj)?;
    let b = extract_single(&b_obj)?;
    let s1_str = seq_to_string(&a);
    let s2_str = seq_to_string(&b);

    // Build token sets
    let tokens_a: std::collections::HashSet<&str> = s1_str.split_whitespace().collect();
    let tokens_b: std::collections::HashSet<&str> = s2_str.split_whitespace().collect();

    // In FuzzyWuzzy: empty tokens → return 0
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return Ok(0.0);
    }

    // If there is ANY common word, return 100 immediately (Python behavior)
    if tokens_a.intersection(&tokens_b).next().is_some() {
        return Ok(score_cutoff_check(100.0, score_cutoff));
    }

    // No common words: compare sorted unique-to-A vs sorted unique-to-B via partial_ratio
    let mut diff_ab: Vec<&str> = tokens_a.iter().copied().collect();
    diff_ab.sort_unstable();
    let mut diff_ba: Vec<&str> = tokens_b.iter().copied().collect();
    diff_ba.sort_unstable();

    let diff_ab_str = diff_ab.join(" ");
    let diff_ba_str = diff_ba.join(" ");

    let tv_ab = str_to_i64_vec(&diff_ab_str);
    let tv_ba = str_to_i64_vec(&diff_ba_str);

    let score = if tv_ab.len() <= tv_ba.len() {
        partial_ratio_short_long(&tv_ab, &tv_ba).0
    } else {
        partial_ratio_short_long(&tv_ba, &tv_ab).0
    };

    Ok(score_cutoff_check(score, score_cutoff))
}

// ===========================================================================
// partial_token_ratio
// ===========================================================================


#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_partial_token_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let a = extract_single(&a_obj)?;
    let b = extract_single(&b_obj)?;
    let s1_str = seq_to_string(&a);
    let s2_str = seq_to_string(&b);

    // partial_token_sort_ratio
    let sorted1 = tokens_sort_key(&s1_str);
    let sorted2 = tokens_sort_key(&s2_str);
    let sv1 = str_to_i64_vec(&sorted1);
    let sv2 = str_to_i64_vec(&sorted2);
    let ptsr = if sv1.len() <= sv2.len() { partial_ratio_short_long(&sv1, &sv2).0 } else { partial_ratio_short_long(&sv2, &sv1).0 };

    // partial_token_set_ratio
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(&s1_str, &s2_str);
    let ptset = if intersect.is_empty() && diff1.is_empty() && diff2.is_empty() {
        0.0
    } else if !intersect.is_empty() {
        // exit early when there is a common word in both sequences
        100.0
    } else {
        let dab = diff1.join(" ");
        let dba = diff2.join(" ");
        let tv1 = str_to_i64_vec(&dab);
        let tv2 = str_to_i64_vec(&dba);
        if tv1.len() <= tv2.len() {
            partial_ratio_short_long(&tv1, &tv2).0
        } else {
            partial_ratio_short_long(&tv2, &tv1).0
        }
    };

    let score = ptsr.max(ptset);
    Ok(score_cutoff_check(score, score_cutoff))
}

// ===========================================================================
// WRatio
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_wratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;

    if av.is_empty() || bv.is_empty() {
        return Ok(0.0);
    }

    let unbase_scale: f64 = 0.95;
    let sc = score_cutoff.unwrap_or(0.0);

    // basic ratio
    let base = norm_sim_to_score(indel_normalized_sim(&av, &bv, score_cutoff));

    let len1 = av.len();
    let len2 = bv.len();
    let len_ratio = if len1 > len2 {
        len1 as f64 / len2 as f64
    } else {
        len2 as f64 / len1 as f64
    };

    let s1_str = seq_to_string(&av);
    let s2_str = seq_to_string(&bv);

    let end_ratio;

    if len_ratio < 1.5 {
        // token_ratio branch
        let sorted1 = tokens_sort_key(&s1_str);
        let sorted2 = tokens_sort_key(&s2_str);
        let tsr = indel_score_100(&sorted1, &sorted2);
        let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(&s1_str, &s2_str);
        let t0 = intersect.join(" ");
        let t1 = if diff1.is_empty() { t0.clone() } else if t0.is_empty() { diff1.join(" ") } else { format!("{} {}", t0, diff1.join(" ")) };
        let t2 = if diff2.is_empty() { t0.clone() } else if t0.is_empty() { diff2.join(" ") } else { format!("{} {}", t0, diff2.join(" ")) };
        let tset = if intersect.is_empty() { 0.0 } else { indel_score_100(&t0, &t1).max(indel_score_100(&t0, &t2)).max(indel_score_100(&t1, &t2)) };
        let tr = tsr.max(tset);
        end_ratio = base.max(tr * unbase_scale);
    } else {
        // partial ratio branch
        let partial_scale: f64 = if len_ratio <= 8.0 { 0.9 } else { 0.6 };
        let pr = if av.len() <= bv.len() { partial_ratio_short_long(&av.to_u64(), &bv.to_u64()).0 } else { partial_ratio_short_long(&bv.to_u64(), &av.to_u64()).0 };
        let mut er = base.max(pr * partial_scale);

        // partial_token_ratio branch
        let sorted1 = tokens_sort_key(&s1_str);
        let sorted2 = tokens_sort_key(&s2_str);
        let sv1 = str_to_i64_vec(&sorted1);
        let sv2 = str_to_i64_vec(&sorted2);
        let ptsr = if sv1.len() <= sv2.len() {
            partial_ratio_short_long(&sv1, &sv2).0
        } else {
            partial_ratio_short_long(&sv2, &sv1).0
        };
        let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(&s1_str, &s2_str);
        let ptset = if intersect.is_empty() && diff1.is_empty() && diff2.is_empty() {
            0.0
        } else if !intersect.is_empty() {
            100.0
        } else {
            let dab = diff1.join(" ");
            let dba = diff2.join(" ");
            let tv1 = str_to_i64_vec(&dab);
            let tv2 = str_to_i64_vec(&dba);
            if tv1.len() <= tv2.len() { partial_ratio_short_long(&tv1, &tv2).0 } else { partial_ratio_short_long(&tv2, &tv1).0 }
        };
        let ptr = ptsr.max(ptset);
        er = er.max(ptr * unbase_scale * partial_scale);
        end_ratio = er;
    }

    Ok(score_cutoff_check(end_ratio, Some(sc)))
}

// ===========================================================================
// QRatio (same as ratio but returns 0 for empty strings)
// ===========================================================================

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_qratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) {
        return Ok(0.0);
    }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;

    if av.is_empty() || bv.is_empty() {
        return Ok(0.0);
    }

    let score = norm_sim_to_score(indel_normalized_sim(&av, &bv, score_cutoff));
    Ok(score_cutoff_check(score, score_cutoff))
}

// Expose the raw execution block
pub fn execute_scorer_raw(
    stype: crate::process::ScorerType,
    s1: &crate::types::Seq<'_>,
    s2: &crate::types::Seq<'_>,
    score_cutoff: Option<f64>,
) -> f64 {
    use crate::process::ScorerType::*;
    match stype {
        Ratio => norm_sim_to_score(score_cutoff_check(indel_normalized_sim(s1, s2, score_cutoff), score_cutoff)),
        _ => 0.0 // Placeholder
    }
}
