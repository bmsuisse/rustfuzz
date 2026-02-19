// SPDX-License-Identifier: MIT
// PyO3 utils module: default_process

use pyo3::prelude::*;

/// Preprocesses a string by converting to lowercase and replacing
/// non-alphanumeric characters with spaces (then stripping leading/trailing).
/// Mirrors utils_py.default_process behavior.
#[pyfunction]
#[pyo3(signature = (s, *, processor=None))]
pub fn default_process(s: &Bound<'_, PyAny>, processor: Option<PyObject>) -> PyResult<String> {
    let _ = processor;
    use pyo3::types::PyBytes;

    let text = if s.is_none() {
        return Ok(String::new());
    } else if let Ok(bytes) = s.downcast::<PyBytes>() {
        // decode bytes as latin-1 (byte values 0-255 map to unicode codepoints)
        bytes
            .as_bytes()
            .iter()
            .map(|&b| b as char)
            .collect::<String>()
    } else {
        s.extract::<String>()?
    };

    let result: String = text
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .to_lowercase();

    Ok(result.trim().to_string())
}
