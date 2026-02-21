use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

#[derive(Clone, Debug)]
pub enum Seq<'a> {
    Ascii(&'a [u8]),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

pub fn extract_single<'a>(obj: &'a Bound<'a, PyAny>) -> PyResult<Seq<'a>> {
    if let Ok(s) = obj.downcast::<PyString>() {
        unsafe {
            let py_str = s.as_ptr();
            let mut length: isize = 0;
            let ptr = pyo3::ffi::PyUnicode_AsUTF8AndSize(py_str, &mut length);
            if !ptr.is_null() {
                let slice = std::slice::from_raw_parts(ptr as *const u8, length as usize);
                if slice.is_ascii() {
                    return Ok(Seq::Ascii(slice));
                }
            }
        }
        let st = s.to_str()?;
        return Ok(Seq::U32(st.chars().map(|c| c as u32).collect()));
    }

    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Seq::Ascii(b.as_bytes()));
    }

    if let Ok(seq) = obj.try_iter() {
        let mut result: Vec<u64> = Vec::new();
        for item in seq {
            let item = item?;
            if let Ok(s) = item.downcast::<PyString>() {
                let st = s.to_str()?;
                let mut iter = st.chars();
                if let (Some(ch), None) = (iter.next(), iter.next()) {
                    result.push(ch as u64);
                    continue;
                }
            }
            if let Ok(i) = item.extract::<u64>() {
                result.push(i);
                continue;
            }
            result.push(item.hash()? as u64);
        }
        return Ok(Seq::U64(result));
    }

    Err(pyo3::exceptions::PyTypeError::new_err("expected str, bytes, or sequence"))
}

pub fn get_processed_args<'py>(
    py: Python<'py>,
    s1: &Bound<'py, PyAny>,
    s2: &Bound<'py, PyAny>,
    processor: &Option<PyObject>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if let Some(proc) = processor {
        let p1 = proc.call1(py, (s1,))?.into_bound(py);
        let p2 = proc.call1(py, (s2,))?.into_bound(py);
        Ok((p1, p2))
    } else {
        Ok((s1.clone(), s2.clone()))
    }
}

pub fn is_none(obj: &Bound<'_, PyAny>) -> bool {
    if obj.is_none() {
        return true;
    }
    if let Ok(f) = obj.extract::<f64>() {
        return f.is_nan();
    }
    if let Ok(r) = obj.str() {
        return r.to_str().map(|s| s == "<NA>").unwrap_or(false);
    }
    false
}

impl<'a> Seq<'a> {
    pub fn is_empty(&self) -> bool {
        match self {
            Seq::Ascii(v) => v.is_empty(),
            Seq::U32(v) => v.is_empty(),
            Seq::U64(v) => v.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Seq::Ascii(v) => v.len(),
            Seq::U32(v) => v.len(),
            Seq::U64(v) => v.len(),
        }
    }

    pub fn to_u64(&self) -> Vec<u64> {
        match self {
            Seq::Ascii(v) => v.iter().map(|&c| c as u64).collect(),
            Seq::U32(v) => v.iter().map(|&c| c as u64).collect(),
            Seq::U64(v) => v.clone(),
        }
    }

    pub fn to_string_lossy(&self) -> String {
        match self {
            Seq::Ascii(v) => String::from_utf8_lossy(v).into_owned(),
            Seq::U32(v) => v.iter().filter_map(|&c| char::from_u32(c)).collect(),
            Seq::U64(_) => String::new(),
        }
    }
}
