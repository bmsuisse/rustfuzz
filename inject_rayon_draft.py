import re

# To execute without the GIL, we need a standalone enum that isn't tied to the PyO3 lifetime 'a.
# We can define an owned/raw fallback in process.rs.
# RapidFuzz works fast because it doesn't clone.
# We can use PyO3's `PyString` but extract the raw `*const u8` (UTF-8 buffer) using `PyUnicode_AsUTF8AndSize`.
# As long as we hold the references to the Python strings in a `Vec<PyObject>`, the strings won't be GC'd.
# So `unsafe { std::slice::from_raw_parts }` is valid while those objects live!

rust_code = """
use pyo3::prelude::*;
use pyo3::types::{PyString, PyFloat, PyInt, PyBool};
use std::str::FromStr;
use rayon::prelude::*;
use crate::fuzz::*;
use crate::types::{extract_single, ScorerType, Seq, is_none};

// We define a raw struct that can be Sent across threads safely.
// We guarantee lifetime because `Vec<PyObject>` holds the python strings alive
// during the `allow_threads` execution scope.
struct RawChoice {
    idx: usize,
    obj: PyObject, 
    // Is it ascii or u32 array?
    is_ascii: bool,
    ascii_ptr: *const u8,
    ascii_len: usize,
    u32_vec: Option<Vec<u32>>, 
}
unsafe impl Send for RawChoice {}
unsafe impl Sync for RawChoice {}

fn process_choice(py: Python<'_>, idx: usize, choice_res: PyResult<Bound<'_, PyAny>>) -> PyResult<Option<RawChoice>> {
    let choice = choice_res?;
    if is_none(&choice) {
        return Ok(None);
    }
    
    // Extract using our standard zero-copy logic
    let temp_seq = extract_single(&choice)?;
    let (is_ascii, ascii_ptr, ascii_len, u32_vec) = match temp_seq {
        Seq::Ascii(slice) => {
            (true, slice.as_ptr(), slice.len(), None)
        }
        Seq::U32(vec) => {
            (false, std::ptr::null(), 0, Some(vec))
        }
        _ => {
            // For now, assume process.rs only extracts strings.
            // If it hits anything else, just clone it to U32 for safety
            (false, std::ptr::null(), 0, Some(match temp_seq {
                Seq::U64(v) => v.into_iter().map(|x| x as u32).collect(),
                Seq::U32(v) => v,
                _ => vec![]
            }))
        }
    };
    
    Ok(Some(RawChoice {
        idx,
        obj: choice.into(),
        is_ascii,
        ascii_ptr,
        ascii_len,
        u32_vec,
    }))
}

// ... execute_scorer_raw which takes Seq directly to avoid Gil ...
"""
print("Rust design defined. Since `execute_scorer` dynamically dispatches based on `s1` and `s2` (which uses Bound<'_, PyAny>), we have to refactor that as well. But wait! Both processor and scorer_obj natively require python callbacks. Parallel threading ONLY works if NO processor and NO custom scorer is used.")
print("This makes sense; we can only go fast natively.")
