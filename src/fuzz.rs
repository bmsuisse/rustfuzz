use pyo3::prelude::*;

use crate::algorithms as alg;
use crate::distance::initialize::ScoreAlignment;
use crate::types::{extract_single, get_processed_args, is_none, Seq};
use crate::dispatch_metric;

fn indel_normalized_sim(av: &Seq<'_>, bv: &Seq<'_>, score_cutoff: Option<f64>) -> f64 {
    let lensum = av.len() + bv.len();
    if lensum == 0 {
        return 1.0;
    }
    let max_dist = score_cutoff.map(|c| {
        let c = c.min(100.0);
        (lensum as f64 * (1.0 - c / 100.0)).floor() as usize
    });
    let dist = dispatch_metric!(alg::indel_distance, av, bv, max_dist);
    if dist == usize::MAX {
        return 0.0;
    }
    1.0 - (dist as f64 / lensum as f64)
}

fn score_cutoff_check(score: f64, cutoff: Option<f64>) -> f64 {
    match cutoff {
        Some(c) if score < c => 0.0,
        _ => score,
    }
}

fn indel_score_100(s1: &str, s2: &str) -> f64 {
    let av: Vec<u64> = s1.chars().map(|c| c as u64).collect();
    let bv: Vec<u64> = s2.chars().map(|c| c as u64).collect();
    indel_normalized_sim(&Seq::U64(av), &Seq::U64(bv), None) * 100.0
}

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
    let mut intersection: Vec<String> = tokens1.intersection(&tokens2).map(|s| s.to_string()).collect();
    let mut diff1: Vec<String> = tokens1.difference(&tokens2).map(|s| s.to_string()).collect();
    let mut diff2: Vec<String> = tokens2.difference(&tokens1).map(|s| s.to_string()).collect();
    intersection.sort_unstable();
    diff1.sort_unstable();
    diff2.sort_unstable();
    (intersection, diff1, diff2)
}

fn build_token_set_strings(t0: &str, diff: &[String]) -> String {
    if diff.is_empty() {
        t0.to_string()
    } else if t0.is_empty() {
        diff.join(" ")
    } else {
        format!("{} {}", t0, diff.join(" "))
    }
}

/// Returns (token_sort_score, token_set_score).
fn token_sort_and_set(s1: &str, s2: &str) -> (f64, f64) {
    let tsr = indel_score_100(&tokens_sort_key(s1), &tokens_sort_key(s2));
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(s1, s2);
    let t0 = intersect.join(" ");
    let t1 = build_token_set_strings(&t0, &diff1);
    let t2 = build_token_set_strings(&t0, &diff2);
    let tset = if intersect.is_empty() {
        indel_score_100(&t1, &t2)
    } else {
        indel_score_100(&t0, &t1)
            .max(indel_score_100(&t0, &t2))
            .max(indel_score_100(&t1, &t2))
    };
    (tsr, tset)
}

fn partial_ratio_short_long(shorter: &[u64], longer: &[u64]) -> (f64, usize, usize) {
    let s_len = shorter.len();
    let l_len = longer.len();
    if s_len == 0 {
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

    for i in 1..s_len.min(l_len) {
        let score = score_fn(shorter, &longer[..i]);
        if score > best_score {
            (best_score, best_start, best_end) = (score, 0, i);
            if best_score == 100.0 { return (best_score, best_start, best_end); }
        }
    }

    let exact_end = if l_len >= s_len { l_len - s_len } else { 0 };
    for start in 0..=exact_end {
        let end = (start + s_len).min(l_len);
        let score = score_fn(shorter, &longer[start..end]);
        if score > best_score {
            (best_score, best_start, best_end) = (score, start, end);
            if best_score == 100.0 { return (best_score, best_start, best_end); }
        }
    }

    let suffix_from = if l_len >= s_len { l_len - s_len + 1 } else { 0 };
    for i in suffix_from..l_len {
        let window = &longer[i..];
        if window.is_empty() { break; }
        let score = score_fn(shorter, window);
        if score > best_score {
            (best_score, best_start, best_end) = (score, i, l_len);
            if best_score == 100.0 { return (best_score, best_start, best_end); }
        }
    }

    (best_score, best_start, best_end)
}

fn partial_ratio_ordered(shorter: &[u64], longer: &[u64]) -> f64 {
    partial_ratio_short_long(shorter, longer).0
}

fn partial_ratio_vecs(av: &Seq<'_>, bv: &Seq<'_>) -> f64 {
    // ASCII fast path — bypass Vec<u64> entirely
    if let (Seq::Ascii(a), Seq::Ascii(b)) = (av, bv) {
        let (needle, haystack) = if a.len() <= b.len() { (*a, *b) } else { (*b, *a) };
        if needle.len() <= 64 {
            return crate::algorithms::partial_ratio_ascii_fast(needle, haystack);
        }
        // Large needle — fall through to Vec<u64> path
    }
    if av.len() <= bv.len() {
        let mut s = partial_ratio_ordered(&av.to_u64(), &bv.to_u64());
        if s != 100.0 && av.len() == bv.len() {
            let s2 = partial_ratio_ordered(&bv.to_u64(), &av.to_u64());
            if s2 > s { s = s2; }
        }
        s
    } else {
        partial_ratio_ordered(&bv.to_u64(), &av.to_u64())
    }
}

fn partial_ratio_str(s1: &str, s2: &str) -> f64 {
    // ASCII fast path: BitPal sliding window, no Vec<u64> allocation
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        // Guaranteed: needle <= 64 for typical token strings
        let (needle, haystack) = if b1.len() <= b2.len() { (b1, b2) } else { (b2, b1) };
        if needle.len() <= 64 {
            return crate::algorithms::partial_ratio_ascii_fast(needle, haystack);
        }
    }
    let sv1: Vec<u64> = s1.chars().map(|c| c as u64).collect();
    let sv2: Vec<u64> = s2.chars().map(|c| c as u64).collect();
    if sv1.len() <= sv2.len() { partial_ratio_ordered(&sv1, &sv2) } else { partial_ratio_ordered(&sv2, &sv1) }
}

fn partial_token_set_score(s1: &str, s2: &str) -> f64 {
    let tokens_a: std::collections::HashSet<&str> = s1.split_whitespace().collect();
    let tokens_b: std::collections::HashSet<&str> = s2.split_whitespace().collect();
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }
    if tokens_a.intersection(&tokens_b).next().is_some() {
        return 100.0;
    }
    let mut diff_ab: Vec<&str> = tokens_a.iter().copied().collect();
    diff_ab.sort_unstable();
    let mut diff_ba: Vec<&str> = tokens_b.iter().copied().collect();
    diff_ba.sort_unstable();
    partial_ratio_str(&diff_ab.join(" "), &diff_ba.join(" "))
}


#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;
    let score = indel_normalized_sim(&av, &bv, score_cutoff) * 100.0;
    Ok(score_cutoff_check(score, score_cutoff))
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
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;
    if av.is_empty() && bv.is_empty() {
        return Ok(score_cutoff_check(100.0, score_cutoff));
    }

    // ASCII fast path: BitPal sliding window — O(N) total bit-ops, no Vec<u64> allocation
    let score = if let (Seq::Ascii(a_bytes), Seq::Ascii(b_bytes)) = (&av, &bv) {
        let (needle, haystack) = if a_bytes.len() <= b_bytes.len() {
            (*a_bytes, *b_bytes)
        } else {
            (*b_bytes, *a_bytes)
        };
        if needle.len() <= 64 {
            crate::algorithms::partial_ratio_ascii_fast(needle, haystack)
        } else {
            partial_ratio_vecs(&av, &bv)
        }
    } else {
        partial_ratio_vecs(&av, &bv)
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
    if is_none(s1) || is_none(s2) { return Ok(None); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;
    let s1_is_shorter = av.len() <= bv.len();

    let (score, src_start, src_end, dest_start, dest_end) = if s1_is_shorter {
        let (s, ds, de) = partial_ratio_short_long(&av.to_u64(), &bv.to_u64());
        if s != 100.0 && av.len() == bv.len() {
            let (s2, ds2, de2) = partial_ratio_short_long(&bv.to_u64(), &av.to_u64());
            if s2 > s {
                return Ok(if score_cutoff.map_or(false, |c| s2 < c) {
                    None
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

    if score_cutoff.map_or(false, |c| score < c) {
        return Ok(None);
    }
    Ok(Some(ScoreAlignment { score, src_start, src_end, dest_start, dest_end }))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_token_sort_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let s1_str = extract_single(&a_obj)?.to_string_lossy();
    let s2_str = extract_single(&b_obj)?.to_string_lossy();
    let score = indel_score_100(&tokens_sort_key(&s1_str), &tokens_sort_key(&s2_str));
    Ok(score_cutoff_check(score, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_token_set_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let s1_str = extract_single(&a_obj)?.to_string_lossy();
    let s2_str = extract_single(&b_obj)?.to_string_lossy();
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(&s1_str, &s2_str);
    if intersect.is_empty() && diff1.is_empty() && diff2.is_empty() {
        return Ok(0.0);
    }
    let t0 = intersect.join(" ");
    let t1 = build_token_set_strings(&t0, &diff1);
    let t2 = build_token_set_strings(&t0, &diff2);
    let score = if intersect.is_empty() {
        indel_score_100(&t1, &t2)
    } else {
        indel_score_100(&t0, &t1)
            .max(indel_score_100(&t0, &t2))
            .max(indel_score_100(&t1, &t2))
    };
    Ok(score_cutoff_check(score, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_token_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let s1_str = extract_single(&a_obj)?.to_string_lossy();
    let s2_str = extract_single(&b_obj)?.to_string_lossy();
    let (tsr, tset) = token_sort_and_set(&s1_str, &s2_str);
    Ok(score_cutoff_check(tsr.max(tset), score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_partial_token_sort_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let s1_str = extract_single(&a_obj)?.to_string_lossy();
    let s2_str = extract_single(&b_obj)?.to_string_lossy();
    let score = partial_ratio_str(&tokens_sort_key(&s1_str), &tokens_sort_key(&s2_str));
    Ok(score_cutoff_check(score, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_partial_token_set_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let s1_str = extract_single(&a_obj)?.to_string_lossy();
    let s2_str = extract_single(&b_obj)?.to_string_lossy();
    let score = partial_token_set_score(&s1_str, &s2_str);
    Ok(score_cutoff_check(score, score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_partial_token_ratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let s1_str = extract_single(&a_obj)?.to_string_lossy();
    let s2_str = extract_single(&b_obj)?.to_string_lossy();
    let ptsr = partial_ratio_str(&tokens_sort_key(&s1_str), &tokens_sort_key(&s2_str));
    let ptset = partial_token_set_score(&s1_str, &s2_str);
    Ok(score_cutoff_check(ptsr.max(ptset), score_cutoff))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_wratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;
    if av.is_empty() || bv.is_empty() { return Ok(0.0); }

    let unbase_scale: f64 = 0.95;
    let sc = score_cutoff.unwrap_or(0.0);
    let base = indel_normalized_sim(&av, &bv, score_cutoff) * 100.0;
    let len_ratio = av.len().max(bv.len()) as f64 / av.len().min(bv.len()) as f64;
    let s1_str = av.to_string_lossy();
    let s2_str = bv.to_string_lossy();

    let end_ratio = if len_ratio < 1.5 {
        let (tsr, tset) = token_sort_and_set(&s1_str, &s2_str);
        base.max(tsr.max(tset) * unbase_scale)
    } else {
        let partial_scale: f64 = if len_ratio <= 8.0 { 0.9 } else { 0.6 };
        let pr = partial_ratio_vecs(&av, &bv);
        let mut er = base.max(pr * partial_scale);
        let ptsr = partial_ratio_str(&tokens_sort_key(&s1_str), &tokens_sort_key(&s2_str));
        let ptset = partial_token_set_score(&s1_str, &s2_str);
        er = er.max(ptsr.max(ptset) * unbase_scale * partial_scale);
        er
    };

    Ok(score_cutoff_check(end_ratio, Some(sc)))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, processor=None, score_cutoff=None))]
pub fn fuzz_qratio(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    if is_none(s1) || is_none(s2) { return Ok(0.0); }
    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
    let av = extract_single(&a_obj)?;
    let bv = extract_single(&b_obj)?;
    if av.is_empty() || bv.is_empty() { return Ok(0.0); }
    let score = indel_normalized_sim(&av, &bv, score_cutoff) * 100.0;
    Ok(score_cutoff_check(score, score_cutoff))
}
