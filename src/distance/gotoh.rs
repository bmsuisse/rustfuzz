use pyo3::prelude::*;

use crate::types::{extract_single, get_processed_args, Seq};

// ---------------------------------------------------------------------------
// Distance & Similarity using Gotoh
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (s1, s2, *, open_penalty=1.0, extend_penalty=0.5, processor=None, score_cutoff=None))]
pub fn distance(
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
    
    // Scale penalties to internal integer representation (x100 for precision if we wanted, 
    // but the underlying algorithm currently takes usize.
    // For affine gaps, we often use floats in python but rust typically wants ints.
    // For simplicity we will cast them to usize for the rust level, but typically we'd multiply 
    // to preserve some precision if they were floats.
    let op = (open_penalty * 100.0) as usize;
    let ep = (extend_penalty * 100.0) as usize;

    let dist = match (&v1, &v2) {
        (Seq::Ascii(a), Seq::Ascii(b)) => crate::algorithms::gotoh_distance(a, b, op, ep),
        (Seq::U32(a), Seq::U32(b)) => crate::algorithms::gotoh_distance(a, b, op, ep),
        (Seq::U64(a), Seq::U64(b)) => crate::algorithms::gotoh_distance(a, b, op, ep),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Mismatched sequence types")),
    };

    let float_dist = (dist as f64) / 100.0;
    
    if let Some(cutoff) = score_cutoff {
        if float_dist > cutoff {
            return Ok(cutoff + 1.0);
        }
    }
    
    Ok(float_dist)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    Ok(())
}
