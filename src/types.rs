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
use pyo3::types::{PyBytes, PyString};

/// We just alias Seq to Vec<i64> to avoid changing every single algorithm signature right now.
/// This prevents the double-allocation of `Vec<u32>` to `Vec<i64>`.
pub type Seq = Vec<i64>;

fn extract_single(obj: &Bound<'_, PyAny>) -> PyResult<Seq> {
    // str -> codepoints directly to i64
    if let Ok(s) = obj.downcast::<PyString>() {
        // FAST PATH: PyString can expose internal ASCII/utf8 buffers without allocation
        // PyO3's safe `.to_str()` checks UTF-8 validity which is slow.
        // We can unsafely grab the bytes and map them. Since Python strings are valid.
        unsafe {
            let py_str = s.as_ptr();
            let mut length: isize = 0;
            // PyUnicode_AsUTF8AndSize is fast and returns a pointer to the UTF-8 buffer
            let ptr = pyo3::ffi::PyUnicode_AsUTF8AndSize(py_str, &mut length);
            if !ptr.is_null() {
                let slice = std::slice::from_raw_parts(ptr as *const u8, length as usize);
                if slice.is_ascii() {
                    let chars: Vec<i64> = slice.iter().map(|&c| c as i64).collect();
                    return Ok(chars);
                } else {
                    let st = std::str::from_utf8_unchecked(slice);
                    let chars: Vec<i64> = st.chars().map(|c| c as i64).collect();
                    return Ok(chars);
                }
            }
        }
    }
    // bytes -> raw bytes directly to i64
    if let Ok(b) = obj.downcast::<PyBytes>() {
        let bytes: Vec<i64> = b.as_bytes().iter().map(|&x| x as i64).collect();
        return Ok(bytes);
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
        return Ok(result);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected str, bytes, or sequence",
    ))
}

/// Extract both sequences, applying conv_sequences semantics:
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

    let a = extract_single(&s1_obj)?;
    let b = extract_single(&s2_obj)?;
    
    Ok((a, b))
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
