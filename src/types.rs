// SPDX-License-Identifier: MIT
// Input type extraction: converts Python objects to Vec<i64> for algorithm use.
//
// Follows conv_sequences semantics from _common_py.py:
// - str -> list of ord() codepoints (i64)
// - bytes -> list of byte values (i64)
// - array('u'/'w') -> list of ord() codepoints (i64)
// - other sequences -> list of hash values (i64)
// - str+str or bytes+bytes are handled natively as slices of equal-len chars

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString, PyTuple};

/// Opaque sequence type used across all algorithm functions.
#[derive(Clone)]
pub enum Seq {
    Str(Vec<u32>),   // codepoints from str or char-like array
    Bytes(Vec<u8>),  // raw bytes
    Hash(Vec<i64>),  // arbitrary hashable elements
}

impl Seq {
    pub fn len(&self) -> usize {
        match self {
            Seq::Str(v) => v.len(),
            Seq::Bytes(v) => v.len(),
            Seq::Hash(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to Vec<i64> for generic algorithm use
    pub fn as_i64(&self) -> Vec<i64> {
        match self {
            Seq::Str(v) => v.iter().map(|&c| c as i64).collect(),
            Seq::Bytes(v) => v.iter().map(|&b| b as i64).collect(),
            Seq::Hash(v) => v.clone(),
        }
    }

    /// Compare individual elements
    pub fn elem(&self, i: usize) -> i64 {
        match self {
            Seq::Str(v) => v[i] as i64,
            Seq::Bytes(v) => v[i] as i64,
            Seq::Hash(v) => v[i],
        }
    }
}

fn extract_single(obj: &Bound<'_, PyAny>) -> PyResult<Seq> {
    // str -> codepoints
    if let Ok(s) = obj.downcast::<PyString>() {
        let chars: Vec<u32> = s.to_str()?.chars().map(|c| c as u32).collect();
        return Ok(Seq::Str(chars));
    }
    // bytes -> raw bytes
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Seq::Bytes(b.as_bytes().to_vec()));
    }
    // Try to iterate as a sequence
    if let Ok(seq) = obj.try_iter() {
        let mut result: Vec<i64> = Vec::new();
        for item in seq {
            let item = item?;
            // Single character string -> use ord()
            if let Ok(s) = item.downcast::<PyString>() {
                let st = s.to_str()?;
                let chars: Vec<char> = st.chars().collect();
                if chars.len() == 1 {
                    result.push(chars[0] as i64);
                    continue;
                }
            }
            // integer -> use directly
            if let Ok(i) = item.extract::<i64>() {
                result.push(i);
                continue;
            }
            // fallback: hash
            result.push(item.hash()? as i64);
        }
        return Ok(Seq::Hash(result));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected str, bytes, or sequence",
    ))
}

/// Extract both sequences, applying conv_sequences semantics:
/// str+str stays as str, bytes+bytes stays as bytes, mixed -> normalize both.
pub fn extract_sequences(
    py: Python<'_>,
    s1: &Bound<'_, PyAny>,
    s2: &Bound<'_, PyAny>,
    processor: &Option<PyObject>,
) -> PyResult<(Seq, Seq)> {
    let (s1_obj, s2_obj) = if let Some(proc) = processor {
        let p1 = proc.call1(py, (s1,))?.into_bound(py);
        let p2 = proc.call1(py, (s2,))?.into_bound(py);
        (p1, p2)
    } else {
        (s1.clone(), s2.clone())
    };

    // str+str: keep as codepoints
    if s1_obj.downcast::<PyString>().is_ok() && s2_obj.downcast::<PyString>().is_ok() {
        let a = extract_single(&s1_obj)?;
        let b = extract_single(&s2_obj)?;
        return Ok((a, b));
    }
    // bytes+bytes: keep as bytes
    if s1_obj.downcast::<PyBytes>().is_ok() && s2_obj.downcast::<PyBytes>().is_ok() {
        let a = extract_single(&s1_obj)?;
        let b = extract_single(&s2_obj)?;
        return Ok((a, b));
    }
    // mixed or other: normalize both to i64 sequences
    let a = extract_single(&s1_obj)?.as_i64();
    let b = extract_single(&s2_obj)?.as_i64();
    Ok((Seq::Hash(a), Seq::Hash(b)))
}

/// Quick check if a Python object is None, NaN, or pandas.NA
pub fn is_none(obj: &Bound<'_, PyAny>) -> bool {
    if obj.is_none() {
        return true;
    }
    if let Ok(f) = obj.extract::<f64>() {
        return f.is_nan();
    }
    // pandas.NA: check by repr
    if let Ok(r) = obj.str() {
        return r.to_str().map(|s| s == "<NA>").unwrap_or(false);
    }
    false
}
