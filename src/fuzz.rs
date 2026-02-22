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
    // ASCII fast path: use Seq::Ascii to avoid Vec<u64> allocation
    if s1.is_ascii() && s2.is_ascii() {
        let av = Seq::Ascii(s1.as_bytes());
        let bv = Seq::Ascii(s2.as_bytes());
        return indel_normalized_sim(&av, &bv, None) * 100.0;
    }
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

fn tokens_to_set_intersection_diff<'a>(
    s1: &'a str,
    s2: &'a str,
    cache: Option<&'a QueryTokenCache>,
) -> (Vec<&'a str>, Vec<&'a str>, Vec<&'a str>) {
    let t1: Vec<&str> = if let Some(c) = cache {
        c.tokens.iter().map(|s| s.as_str()).collect()
    } else {
        let mut t: Vec<&str> = s1.split_whitespace().collect();
        t.sort_unstable();
        t.dedup();
        t
    };
    let mut t2: Vec<&str> = s2.split_whitespace().collect();
    t2.sort_unstable();
    t2.dedup();

    let mut intersection = Vec::new();
    let mut diff1 = Vec::new();
    let mut diff2 = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < t1.len() && j < t2.len() {
        match t1[i].cmp(t2[j]) {
            std::cmp::Ordering::Equal => { intersection.push(t1[i]); i += 1; j += 1; }
            std::cmp::Ordering::Less  => { diff1.push(t1[i]); i += 1; }
            std::cmp::Ordering::Greater => { diff2.push(t2[j]); j += 1; }
        }
    }
    diff1.extend_from_slice(&t1[i..]);
    diff2.extend_from_slice(&t2[j..]);
    (intersection, diff1, diff2)
}

fn join_tokens(tokens: &[&str]) -> String {
    tokens.join(" ")
}


fn build_token_set_strings_borrow(t0: &str, diff: &[&str]) -> String {
    if diff.is_empty() {
        t0.to_string()
    } else if t0.is_empty() {
        diff.join(" ")
    } else {
        format!("{} {}", t0, diff.join(" "))
    }
}

/// Returns (token_sort_score, token_set_score). Uncutoff variant.
fn token_sort_and_set(s1: &str, s2: &str) -> (f64, f64) {
    let tsr = indel_score_100(&tokens_sort_key(s1), &tokens_sort_key(s2));
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(s1, s2, None);
    let t0 = join_tokens(&intersect);
    let t1 = build_token_set_strings_borrow(&t0, &diff1);
    let t2 = build_token_set_strings_borrow(&t0, &diff2);
    let tset = if intersect.is_empty() {
        indel_score_100(&t1, &t2)
    } else {
        indel_score_100(&t0, &t1)
            .max(indel_score_100(&t0, &t2))
            .max(indel_score_100(&t1, &t2))
    };
    (tsr, tset)
}

/// Score-cutoff-aware token_sort_and_set: mirrors rf's token_ratio which accepts
/// a score_cutoff. Skips indel_score_100 calls that can't possibly beat cutoff.
fn token_sort_and_set_cutoff(s1: &str, s2: &str, score_cutoff: f64, cache: Option<&QueryTokenCache>) -> (f64, f64) {
    let s1k = if let Some(c) = cache { c.sort_key.clone() } else { tokens_sort_key(s1) };
    let s2k = tokens_sort_key(s2);
    // Pass cutoff so indel_normalized_sim can short-circuit
    let av = Seq::Ascii(s1k.as_bytes());
    let bv = Seq::Ascii(s2k.as_bytes());
    let tsr = if s1k.is_ascii() && s2k.is_ascii() {
        indel_normalized_sim(&av, &bv, Some(score_cutoff)) * 100.0
    } else {
        indel_score_100(&s1k, &s2k)
    };
    if tsr == 100.0 { return (tsr, 100.0); }
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(s1, s2, cache);
    let t0 = join_tokens(&intersect);
    let t1 = build_token_set_strings_borrow(&t0, &diff1);
    let t2 = build_token_set_strings_borrow(&t0, &diff2);
    let cutoff = score_cutoff.max(tsr);
    let tset = if intersect.is_empty() {
        let av = Seq::Ascii(t1.as_bytes());
        let bv = Seq::Ascii(t2.as_bytes());
        if t1.is_ascii() && t2.is_ascii() {
            indel_normalized_sim(&av, &bv, Some(cutoff)) * 100.0
        } else {
            indel_score_100(&t1, &t2)
        }
    } else {
        // With shared intersection tokens are always ASCII
        let r01 = indel_score_100(&t0, &t1);
        if r01 == 100.0 { return (tsr, 100.0); }
        let r02 = indel_score_100(&t0, &t2);
        if r02 == 100.0 { return (tsr, 100.0); }
        let r12 = indel_score_100(&t1, &t2);
        r01.max(r02).max(r12)
    };
    (tsr, tset)
}

/// partial_ratio_vecs with score_cutoff — enables early exit in BitPal inner loop
fn partial_ratio_vecs_sc(av: &Seq<'_>, bv: &Seq<'_>, score_cutoff: f64) -> f64 {
    // ASCII fast path — cutoff-aware
    if let (Seq::Ascii(a), Seq::Ascii(b)) = (av, bv) {
        let (needle, haystack) = if a.len() <= b.len() { (*a, *b) } else { (*b, *a) };
        if needle.len() <= 64 {
            return crate::algorithms::partial_ratio_ascii_fast(needle, haystack);
        }
    }
    // Fallback: Vec<u64> path without cutoff (cutoff would require per-window tracking)
    partial_ratio_vecs(av, bv)
}

/// partial_token_ratio with score_cutoff — mirrors rf's partial_token_ratio:
/// avoids computing the second partial_ratio if sorted tokens equal diff tokens,
/// and propagates score_cutoff between the two partial_ratio calls.
fn partial_token_ratio_sc(s1: &str, s2: &str, score_cutoff: f64, cache: Option<&QueryTokenCache>) -> f64 {
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(s1, s2, cache);
    // Early exit: common word found
    if !intersect.is_empty() { return 100.0; }
    if diff1.is_empty() && diff2.is_empty() { return 0.0; }

    let s1_sorted: Vec<&str> = if let Some(c) = cache {
        c.tokens.iter().map(|s| s.as_str()).collect()
    } else {
        let mut t: Vec<&str> = s1.split_whitespace().collect();
        t.sort_unstable(); t.dedup(); t
    };
    let s2_sorted: Vec<&str> = {
        let mut t: Vec<&str> = s2.split_whitespace().collect();
        t.sort_unstable(); t.dedup(); t
    };

    let s1j = join_tokens(&s1_sorted);
    let s2j = join_tokens(&s2_sorted);
    let result = partial_ratio_str(&s1j, &s2j);
    if result == 100.0 { return 100.0; }

    // Only compute second call if diffs differ from sorted (rf optimization)
    let same1 = s1_sorted.len() == diff1.len();
    let same2 = s2_sorted.len() == diff2.len();
    if same1 && same2 { return result; }

    let sc2 = score_cutoff.max(result);
    let d1j = join_tokens(&diff1);
    let d2j = join_tokens(&diff2);
    result.max(partial_ratio_str_cutoff(&d1j, &d2j, sc2))
}

/// partial_ratio_str with score_cutoff threshold — returns 0 if result < cutoff
fn partial_ratio_str_cutoff(s1: &str, s2: &str, score_cutoff: f64) -> f64 {
    let result = partial_ratio_str(s1, s2);
    if result < score_cutoff { 0.0 } else { result }
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
    // Use sort-merge instead of HashSet — same interface, no hash table
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(s1, s2, None);
    if intersect.is_empty() && diff1.is_empty() && diff2.is_empty() {
        return 0.0;
    }
    if !intersect.is_empty() {
        return 100.0;
    }
    // No intersection — compare sorted diff1 vs diff2
    partial_ratio_str(&join_tokens(&diff1), &join_tokens(&diff2))
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
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(&s1_str, &s2_str, None);
    if intersect.is_empty() && diff1.is_empty() && diff2.is_empty() {
        return Ok(0.0);
    }
    let t0 = join_tokens(&intersect);
    let t1 = build_token_set_strings_borrow(&t0, &diff1);
    let t2 = build_token_set_strings_borrow(&t0, &diff2);
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

    const UNBASE_SCALE: f64 = 0.95;
    let mut sc = score_cutoff.unwrap_or(0.0);

    let len_ratio = av.len().max(bv.len()) as f64 / av.len().min(bv.len()) as f64;

    // Stage 1: base ratio — always computed first
    let mut end_ratio = indel_normalized_sim(&av, &bv, score_cutoff) * 100.0;

    if end_ratio == 100.0 { return Ok(end_ratio); }

    if len_ratio < 1.5 {
        // Short strings: token_ratio can beat base — propagate cutoff
        let s1_str = av.to_string_lossy();
        let s2_str = bv.to_string_lossy();
        sc = sc.max(end_ratio) / UNBASE_SCALE;
        let (tsr, tset) = token_sort_and_set_cutoff(&s1_str, &s2_str, sc, None);
        let tr = tsr.max(tset);
        if tr > 0.0 {
            end_ratio = end_ratio.max(tr * UNBASE_SCALE);
        }
    } else {
        let partial_scale: f64 = if len_ratio <= 8.0 { 0.9 } else { 0.6 };
        let s1_str = av.to_string_lossy();
        let s2_str = bv.to_string_lossy();

        // Stage 2: partial_ratio — propagate cutoff so it can skip
        sc = sc.max(end_ratio) / partial_scale;
        let pr = partial_ratio_vecs_sc(&av, &bv, sc);
        end_ratio = end_ratio.max(pr * partial_scale);

        if end_ratio == 100.0 { return Ok(end_ratio); }

        // Stage 3: partial_token_ratio — propagate cutoff again
        sc = sc.max(end_ratio) / UNBASE_SCALE;
        let ptr = partial_token_ratio_sc(&s1_str, &s2_str, sc, None);
        end_ratio = end_ratio.max(ptr * UNBASE_SCALE * partial_scale);
    }

    Ok(score_cutoff_check(end_ratio, score_cutoff))
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

// --- Raw bytes dispatchers for process.rs Rayon multi-threading ---
pub(crate) fn ratio_bytes(q: &[u8], c: &[u8], cutoff: Option<f64>) -> f64 {
    let av = Seq::Ascii(q);
    let bv = Seq::Ascii(c);
    indel_normalized_sim(&av, &bv, cutoff) * 100.0
}


pub struct QueryTokenCache {
    pub sort_key: String,
    pub tokens: Vec<String>,
}

impl QueryTokenCache {
    pub fn new(s: &str) -> Self {
        let sort_key = tokens_sort_key(s);
        let mut t: Vec<&str> = s.split_whitespace().collect();
        t.sort_unstable();
        t.dedup();
        let tokens = t.into_iter().map(String::from).collect();
        Self { sort_key, tokens }
    }
}

pub(crate) fn wratio_bytes(q: &[u8], c: &[u8], cutoff: Option<f64>, cache: Option<&QueryTokenCache>) -> f64 {
    let av = Seq::Ascii(q);
    let bv = Seq::Ascii(c);
    if av.is_empty() || bv.is_empty() { return 0.0; }
    
    const UNBASE_SCALE: f64 = 0.95;
    let mut sc = cutoff.unwrap_or(0.0);
    let len_ratio = av.len().max(bv.len()) as f64 / av.len().min(bv.len()) as f64;
    let mut end_ratio = ratio_bytes(q, c, cutoff);
    if end_ratio == 100.0 { return end_ratio; }

    let q_str = unsafe { std::str::from_utf8_unchecked(q) };
    let c_str = unsafe { std::str::from_utf8_unchecked(c) };

    if len_ratio < 1.5 {
        sc = sc.max(end_ratio) / UNBASE_SCALE;
        let (tsr, tset) = token_sort_and_set_cutoff(q_str, c_str, sc, cache);
        let tr = tsr.max(tset);
        if tr > 0.0 { end_ratio = end_ratio.max(tr * UNBASE_SCALE); }
    } else {
        let partial_scale: f64 = if len_ratio <= 8.0 { 0.9 } else { 0.6 };
        sc = sc.max(end_ratio) / partial_scale;
        let pr = partial_ratio_vecs_sc(&av, &bv, sc);
        end_ratio = end_ratio.max(pr * partial_scale);
        if end_ratio == 100.0 { return end_ratio; }

        sc = sc.max(end_ratio) / UNBASE_SCALE;
        let ptr = partial_token_ratio_sc(q_str, c_str, sc, cache);
        end_ratio = end_ratio.max(ptr * UNBASE_SCALE * partial_scale);
    }
    score_cutoff_check(end_ratio, cutoff)
}

pub(crate) fn partial_ratio_bytes(q: &[u8], c: &[u8], cutoff: Option<f64>, _cache: Option<&QueryTokenCache>) -> f64 {
    let av = Seq::Ascii(q);
    let bv = Seq::Ascii(c);
    if q.is_empty() || c.is_empty() { return 0.0; }
    let score = if q.len() <= c.len() {
        if q.len() <= 64 { crate::algorithms::partial_ratio_ascii_fast(q, c) } else { partial_ratio_vecs(&av, &bv) }
    } else {
        if c.len() <= 64 { crate::algorithms::partial_ratio_ascii_fast(c, q) } else { partial_ratio_vecs(&av, &bv) }
    };
    score_cutoff_check(score, cutoff)
}

pub(crate) fn token_sort_ratio_bytes(q: &[u8], c: &[u8], cutoff: Option<f64>, cache: Option<&QueryTokenCache>) -> f64 {
    let sq = unsafe { std::str::from_utf8_unchecked(q) };
    let sc = unsafe { std::str::from_utf8_unchecked(c) };
    let qk = if let Some(ca) = cache { ca.sort_key.clone() } else { tokens_sort_key(sq) };
    let score = indel_score_100(&qk, &tokens_sort_key(sc));
    score_cutoff_check(score, cutoff)
}

pub(crate) fn token_set_ratio_bytes(q: &[u8], c: &[u8], cutoff: Option<f64>, cache: Option<&QueryTokenCache>) -> f64 {
    let sq = unsafe { std::str::from_utf8_unchecked(q) };
    let sc = unsafe { std::str::from_utf8_unchecked(c) };
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(sq, sc, cache);
    if intersect.is_empty() && diff1.is_empty() && diff2.is_empty() { return 0.0; }
    let t0 = join_tokens(&intersect);
    let t1 = build_token_set_strings_borrow(&t0, &diff1);
    let t2 = build_token_set_strings_borrow(&t0, &diff2);
    let score = if intersect.is_empty() {
        indel_score_100(&t1, &t2)
    } else {
        indel_score_100(&t0, &t1)
            .max(indel_score_100(&t0, &t2))
            .max(indel_score_100(&t1, &t2))
    };
    score_cutoff_check(score, cutoff)
}

pub(crate) fn token_ratio_bytes(q: &[u8], c: &[u8], cutoff: Option<f64>, cache: Option<&QueryTokenCache>) -> f64 {
    let sq = unsafe { std::str::from_utf8_unchecked(q) };
    let sc = unsafe { std::str::from_utf8_unchecked(c) };
    let (tsr, tset) = token_sort_and_set_cutoff(sq, sc, cutoff.unwrap_or(0.0), cache);
    score_cutoff_check(tsr.max(tset), cutoff)
}

pub(crate) fn partial_token_sort_ratio_bytes(q: &[u8], c: &[u8], cutoff: Option<f64>, cache: Option<&QueryTokenCache>) -> f64 {
    let sq = unsafe { std::str::from_utf8_unchecked(q) };
    let sc = unsafe { std::str::from_utf8_unchecked(c) };
    let qk = if let Some(ca) = cache { ca.sort_key.clone() } else { tokens_sort_key(sq) };
    let score = partial_ratio_str(&qk, &tokens_sort_key(sc));
    score_cutoff_check(score, cutoff)
}

pub(crate) fn partial_token_set_ratio_bytes(q: &[u8], c: &[u8], cutoff: Option<f64>, _cache: Option<&QueryTokenCache>) -> f64 {
    let sq = unsafe { std::str::from_utf8_unchecked(q) };
    let sc = unsafe { std::str::from_utf8_unchecked(c) };
    let score = partial_token_set_score(sq, sc);
    score_cutoff_check(score, cutoff)
}

pub(crate) fn partial_token_ratio_bytes(q: &[u8], c: &[u8], cutoff: Option<f64>, cache: Option<&QueryTokenCache>) -> f64 {
    let sq = unsafe { std::str::from_utf8_unchecked(q) };
    let sc = unsafe { std::str::from_utf8_unchecked(c) };
    let qk = if let Some(ca) = cache { ca.sort_key.clone() } else { tokens_sort_key(sq) };
    let ptsr = partial_ratio_str(&qk, &tokens_sort_key(sc));
    // Let's optimize partial_token_set_score internally if we want, but for now just call it since we don't pass cache down this branch inside this specific function without rewriting it.
    // Actually, we can rewrite it here easily:
    let (intersect, diff1, diff2) = tokens_to_set_intersection_diff(sq, sc, cache);
    let ptset = if intersect.is_empty() && diff1.is_empty() && diff2.is_empty() {
        0.0
    } else if !intersect.is_empty() {
        100.0
    } else {
        partial_ratio_str(&join_tokens(&diff1), &join_tokens(&diff2))
    };
    score_cutoff_check(ptsr.max(ptset), cutoff)
}
