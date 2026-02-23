use pyo3::prelude::*;

use crate::types::{extract_single, get_processed_args, Seq};

// ---------------------------------------------------------------------------
// Distance & Similarity using Gotoh (affine gap penalties)
// ---------------------------------------------------------------------------

/// Compute the raw Gotoh distance as an internal scaled usize.
/// Returns (distance_scaled, max_cost_scaled) for similarity/normalisation.
fn gotoh_raw(v1: &Seq<'_>, v2: &Seq<'_>, op: usize, ep: usize) -> PyResult<(usize, usize)> {
    let dist = match (v1, v2) {
        (Seq::Ascii(a), Seq::Ascii(b)) => crate::algorithms::gotoh_distance(a, b, op, ep),
        (Seq::U32(a), Seq::U32(b)) => crate::algorithms::gotoh_distance(a, b, op, ep),
        (Seq::U64(a), Seq::U64(b)) => crate::algorithms::gotoh_distance(a, b, op, ep),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Mismatched sequence types")),
    };

    // Max cost: replace every character (cost 100 per mismatch) for min(m,n)
    // plus gap-open + gap-extend for the remaining |m-n| characters.
    let (m, n) = match (v1, v2) {
        (Seq::Ascii(a), Seq::Ascii(b)) => (a.len(), b.len()),
        (Seq::U32(a), Seq::U32(b)) => (a.len(), b.len()),
        (Seq::U64(a), Seq::U64(b)) => (a.len(), b.len()),
        _ => (0, 0),
    };

    let max_cost = if m == 0 && n == 0 {
        0
    } else if m == 0 || n == 0 {
        // All gaps: one open + max(m,n) extends
        let longer = m.max(n);
        op + longer * ep
    } else {
        // Worst case: delete all of s1, insert all of s2
        // = (open + m*extend) + (open + n*extend) = 2*open + (m+n)*extend
        // But we must also compare against replacing all min(m,n) chars + gapping the rest
        let gap_all = 2 * op + (m + n) * ep;
        let replace_min = m.min(n) * 100; // 100 = mismatch cost in scaled units
        let diff = if m > n { m - n } else { n - m };
        let replace_and_gap = if diff == 0 {
            replace_min
        } else {
            replace_min + op + diff * ep
        };
        gap_all.min(replace_and_gap).max(dist)
    };

    Ok((dist, max_cost))
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, open_penalty=1.0, extend_penalty=0.5, processor=None, score_cutoff=None))]
pub fn gotoh_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    open_penalty: f64,
    extend_penalty: f64,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (p1, p2) = get_processed_args(py, s1, s2, &processor)?;
    let v1 = extract_single(&p1)?;
    let v2 = extract_single(&p2)?;

    let op = (open_penalty * 100.0) as usize;
    let ep = (extend_penalty * 100.0) as usize;

    let (dist, _) = gotoh_raw(&v1, &v2, op, ep)?;
    let float_dist = (dist as f64) / 100.0;

    if let Some(cutoff) = score_cutoff {
        if float_dist > cutoff {
            return Ok(cutoff + 1.0);
        }
    }

    Ok(float_dist)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, open_penalty=1.0, extend_penalty=0.5, processor=None, score_cutoff=None))]
pub fn gotoh_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    open_penalty: f64,
    extend_penalty: f64,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (p1, p2) = get_processed_args(py, s1, s2, &processor)?;
    let v1 = extract_single(&p1)?;
    let v2 = extract_single(&p2)?;

    let op = (open_penalty * 100.0) as usize;
    let ep = (extend_penalty * 100.0) as usize;

    let (dist, max_cost) = gotoh_raw(&v1, &v2, op, ep)?;
    let sim = if max_cost == 0 {
        0.0
    } else {
        ((max_cost as f64) - (dist as f64)) / 100.0
    };

    if let Some(cutoff) = score_cutoff {
        if sim < cutoff {
            return Ok(0.0);
        }
    }

    Ok(sim)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, open_penalty=1.0, extend_penalty=0.5, processor=None, score_cutoff=None))]
pub fn gotoh_normalized_distance(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    open_penalty: f64,
    extend_penalty: f64,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (p1, p2) = get_processed_args(py, s1, s2, &processor)?;
    let v1 = extract_single(&p1)?;
    let v2 = extract_single(&p2)?;

    let op = (open_penalty * 100.0) as usize;
    let ep = (extend_penalty * 100.0) as usize;

    let (dist, max_cost) = gotoh_raw(&v1, &v2, op, ep)?;
    let norm = if max_cost == 0 {
        0.0
    } else {
        (dist as f64) / (max_cost as f64)
    };

    if let Some(cutoff) = score_cutoff {
        if norm > cutoff {
            return Ok(1.0);
        }
    }

    Ok(norm)
}

#[pyfunction]
#[pyo3(signature = (s1, s2, *, open_penalty=1.0, extend_penalty=0.5, processor=None, score_cutoff=None))]
pub fn gotoh_normalized_similarity(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    open_penalty: f64,
    extend_penalty: f64,
    processor: Option<PyObject>,
    score_cutoff: Option<f64>,
) -> PyResult<f64> {
    let (p1, p2) = get_processed_args(py, s1, s2, &processor)?;
    let v1 = extract_single(&p1)?;
    let v2 = extract_single(&p2)?;

    let op = (open_penalty * 100.0) as usize;
    let ep = (extend_penalty * 100.0) as usize;

    let (dist, max_cost) = gotoh_raw(&v1, &v2, op, ep)?;
    let norm_sim = if max_cost == 0 {
        1.0
    } else {
        1.0 - (dist as f64) / (max_cost as f64)
    };

    if let Some(cutoff) = score_cutoff {
        if norm_sim < cutoff {
            return Ok(0.0);
        }
    }

    Ok(norm_sim)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gotoh_distance, m)?)?;
    m.add_function(wrap_pyfunction!(gotoh_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(gotoh_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(gotoh_normalized_similarity, m)?)?;
    Ok(())
}
