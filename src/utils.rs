use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
#[pyo3(signature = (s, *, processor=None))]
pub fn default_process(s: &Bound<'_, PyAny>, processor: Option<PyObject>) -> PyResult<String> {
    let _ = processor;
    let text = if s.is_none() {
        return Ok(String::new());
    } else if let Ok(bytes) = s.downcast::<PyBytes>() {
        bytes.as_bytes().iter().map(|&b| b as char).collect::<String>()
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
