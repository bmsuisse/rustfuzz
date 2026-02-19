// SPDX-License-Identifier: MIT
// PyO3 classes: Editop, Editops, Opcode, Opcodes, MatchingBlock, ScoreAlignment
// Mirrors the Python _initialize_py.py interface exactly.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

// ---------------------------------------------------------------------------
// Editop
// ---------------------------------------------------------------------------

#[pyclass(module = "rapidfuzz.distance._initialize_cpp")]
#[derive(Clone, Debug, PartialEq)]
pub struct Editop {
    #[pyo3(get, set)]
    pub tag: String,
    #[pyo3(get, set)]
    pub src_pos: usize,
    #[pyo3(get, set)]
    pub dest_pos: usize,
}

#[pymethods]
impl Editop {
    #[new]
    fn new(tag: String, src_pos: usize, dest_pos: usize) -> Self {
        Editop { tag, src_pos, dest_pos }
    }

    fn __repr__(&self) -> String {
        format!(
            "Editop(tag='{}', src_pos={}, dest_pos={})",
            self.tag, self.src_pos, self.dest_pos
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(op) = other.extract::<PyRef<Editop>>() {
            return self.tag == op.tag && self.src_pos == op.src_pos && self.dest_pos == op.dest_pos;
        }
        // Also accept 3-tuples
        if let Ok(tup) = other.downcast::<PyTuple>() {
            if tup.len() == 3 {
                if let (Ok(tag), Ok(sp), Ok(dp)) = (
                    tup.get_item(0).and_then(|v| v.extract::<String>()),
                    tup.get_item(1).and_then(|v| v.extract::<usize>()),
                    tup.get_item(2).and_then(|v| v.extract::<usize>()),
                ) {
                    return self.tag == tag && self.src_pos == sp && self.dest_pos == dp;
                }
            }
        }
        // Also accept lists
        if let Ok(lst) = other.downcast::<pyo3::types::PyList>() {
            if lst.len() == 3 {
                if let (Ok(tag), Ok(sp), Ok(dp)) = (
                    lst.get_item(0).and_then(|v| v.extract::<String>()),
                    lst.get_item(1).and_then(|v| v.extract::<usize>()),
                    lst.get_item(2).and_then(|v| v.extract::<usize>()),
                ) {
                    return self.tag == tag && self.src_pos == sp && self.dest_pos == dp;
                }
            }
        }
        // Duck-type: accept any object with tag, src_pos, dest_pos attributes (e.g. Python's Editop)
        if let (Ok(tag), Ok(sp), Ok(dp)) = (
            other.getattr("tag").and_then(|v| v.extract::<String>()),
            other.getattr("src_pos").and_then(|v| v.extract::<usize>()),
            other.getattr("dest_pos").and_then(|v| v.extract::<usize>()),
        ) {
            return self.tag == tag && self.src_pos == sp && self.dest_pos == dp;
        }
        false
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.tag.hash(&mut h);
        self.src_pos.hash(&mut h);
        self.dest_pos.hash(&mut h);
        h.finish()
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items: Vec<PyObject> = vec![
            self.tag.clone().into_pyobject(py)?.into_any().unbind(),
            self.src_pos.into_pyobject(py)?.into_any().unbind(),
            self.dest_pos.into_pyobject(py)?.into_any().unbind(),
        ];
        let list = PyList::new(py, items)?;
        Ok(list.as_any().call_method0("__iter__")?.unbind())
    }

    fn __getitem__(&self, idx: i64, py: Python<'_>) -> PyResult<PyObject> {
        let items: [PyObject; 3] = [
            self.tag.clone().into_pyobject(py)?.into_any().unbind(),
            self.src_pos.into_pyobject(py)?.into_any().unbind(),
            self.dest_pos.into_pyobject(py)?.into_any().unbind(),
        ];
        let i = if idx < 0 { 3 + idx } else { idx };
        if i < 0 || i >= 3 {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
        }
        Ok(items[i as usize].clone_ref(py))
    }
}

// ---------------------------------------------------------------------------
// Opcode
// ---------------------------------------------------------------------------

#[pyclass(module = "rapidfuzz.distance._initialize_cpp")]
#[derive(Clone, Debug, PartialEq)]
pub struct Opcode {
    #[pyo3(get, set)]
    pub tag: String,
    #[pyo3(get, set)]
    pub src_start: usize,
    #[pyo3(get, set)]
    pub src_end: usize,
    #[pyo3(get, set)]
    pub dest_start: usize,
    #[pyo3(get, set)]
    pub dest_end: usize,
}

#[pymethods]
impl Opcode {
    #[new]
    #[pyo3(signature = (tag="equal", src_start=0, src_end=0, dest_start=0, dest_end=0))]
    fn new(tag: &str, src_start: usize, src_end: usize, dest_start: usize, dest_end: usize) -> Self {
        Opcode {
            tag: tag.to_string(),
            src_start,
            src_end,
            dest_start,
            dest_end,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Opcode(tag='{}', src_start={}, src_end={}, dest_start={}, dest_end={})",
            self.tag, self.src_start, self.src_end, self.dest_start, self.dest_end
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(op) = other.extract::<PyRef<Opcode>>() {
            return self.tag == op.tag
                && self.src_start == op.src_start
                && self.src_end == op.src_end
                && self.dest_start == op.dest_start
                && self.dest_end == op.dest_end;
        }
        // Also accept 5-tuples
        if let Ok(tup) = other.downcast::<PyTuple>() {
            if tup.len() == 5 {
                if let (Ok(tag), Ok(ss), Ok(se), Ok(ds), Ok(de)) = (
                    tup.get_item(0).and_then(|v| v.extract::<String>()),
                    tup.get_item(1).and_then(|v| v.extract::<usize>()),
                    tup.get_item(2).and_then(|v| v.extract::<usize>()),
                    tup.get_item(3).and_then(|v| v.extract::<usize>()),
                    tup.get_item(4).and_then(|v| v.extract::<usize>()),
                ) {
                    return self.tag == tag
                        && self.src_start == ss
                        && self.src_end == se
                        && self.dest_start == ds
                        && self.dest_end == de;
                }
            }
        }
        // Also accept lists
        if let Ok(lst) = other.downcast::<pyo3::types::PyList>() {
            if lst.len() == 5 {
                if let (Ok(tag), Ok(ss), Ok(se), Ok(ds), Ok(de)) = (
                    lst.get_item(0).and_then(|v| v.extract::<String>()),
                    lst.get_item(1).and_then(|v| v.extract::<usize>()),
                    lst.get_item(2).and_then(|v| v.extract::<usize>()),
                    lst.get_item(3).and_then(|v| v.extract::<usize>()),
                    lst.get_item(4).and_then(|v| v.extract::<usize>()),
                ) {
                    return self.tag == tag
                        && self.src_start == ss
                        && self.src_end == se
                        && self.dest_start == ds
                        && self.dest_end == de;
                }
            }
        }
        false
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.tag.hash(&mut h);
        self.src_start.hash(&mut h);
        self.src_end.hash(&mut h);
        self.dest_start.hash(&mut h);
        self.dest_end.hash(&mut h);
        h.finish()
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items = vec![
            self.tag.clone().into_pyobject(py)?.into_any().unbind(),
            self.src_start.into_pyobject(py)?.into_any().unbind(),
            self.src_end.into_pyobject(py)?.into_any().unbind(),
            self.dest_start.into_pyobject(py)?.into_any().unbind(),
            self.dest_end.into_pyobject(py)?.into_any().unbind(),
        ];
        let list = PyList::new(py, items)?;
        Ok(list.as_any().call_method0("__iter__")?.unbind())
    }

    fn __getitem__(&self, idx: i64, py: Python<'_>) -> PyResult<PyObject> {
        let items: [PyObject; 5] = [
            self.tag.clone().into_pyobject(py)?.into_any().unbind(),
            self.src_start.into_pyobject(py)?.into_any().unbind(),
            self.src_end.into_pyobject(py)?.into_any().unbind(),
            self.dest_start.into_pyobject(py)?.into_any().unbind(),
            self.dest_end.into_pyobject(py)?.into_any().unbind(),
        ];
        let i = if idx < 0 { 5 + idx } else { idx };
        if i < 0 || i >= 5 {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
        }
        Ok(items[i as usize].clone_ref(py))
    }
}

// ---------------------------------------------------------------------------
// MatchingBlock
// ---------------------------------------------------------------------------

#[pyclass(module = "rapidfuzz.distance._initialize_cpp")]
#[derive(Clone, Debug, PartialEq)]
pub struct MatchingBlock {
    #[pyo3(get, set)]
    pub a: usize,
    #[pyo3(get, set)]
    pub b: usize,
    #[pyo3(get, set)]
    pub size: usize,
}

#[pymethods]
impl MatchingBlock {
    #[new]
    fn new(a: usize, b: usize, size: usize) -> Self {
        MatchingBlock { a, b, size }
    }

    fn __repr__(&self) -> String {
        format!("MatchingBlock(a={}, b={}, size={})", self.a, self.b, self.size)
    }

    fn __eq__(&self, other: &MatchingBlock) -> bool {
        self == other
    }
}

// ---------------------------------------------------------------------------
// ScoreAlignment
// ---------------------------------------------------------------------------

#[pyclass(module = "rapidfuzz.distance._initialize_cpp")]
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreAlignment {
    #[pyo3(get, set)]
    pub score: f64,
    #[pyo3(get, set)]
    pub src_start: usize,
    #[pyo3(get, set)]
    pub src_end: usize,
    #[pyo3(get, set)]
    pub dest_start: usize,
    #[pyo3(get, set)]
    pub dest_end: usize,
}

#[pymethods]
impl ScoreAlignment {
    #[new]
    fn new(score: f64, src_start: usize, src_end: usize, dest_start: usize, dest_end: usize) -> Self {
        ScoreAlignment { score, src_start, src_end, dest_start, dest_end }
    }

    fn __repr__(&self) -> String {
        format!(
            "ScoreAlignment(score={}, src_start={}, src_end={}, dest_start={}, dest_end={})",
            self.score, self.src_start, self.src_end, self.dest_start, self.dest_end
        )
    }

    fn __eq__(&self, other: &ScoreAlignment) -> bool {
        self == other
    }

    fn __getitem__(&self, idx: i64, py: Python<'_>) -> PyResult<PyObject> {
        let items: [PyObject; 5] = [
            self.score.into_pyobject(py)?.into_any().unbind(),
            self.src_start.into_pyobject(py)?.into_any().unbind(),
            self.src_end.into_pyobject(py)?.into_any().unbind(),
            self.dest_start.into_pyobject(py)?.into_any().unbind(),
            self.dest_end.into_pyobject(py)?.into_any().unbind(),
        ];
        let i = if idx < 0 { 5 + idx } else { idx };
        if i < 0 || i >= 5 {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
        }
        Ok(items[i as usize].clone_ref(py))
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items = vec![
            self.score.into_pyobject(py)?.into_any().unbind(),
            self.src_start.into_pyobject(py)?.into_any().unbind(),
            self.src_end.into_pyobject(py)?.into_any().unbind(),
            self.dest_start.into_pyobject(py)?.into_any().unbind(),
            self.dest_end.into_pyobject(py)?.into_any().unbind(),
        ];
        let list = PyList::new(py, items)?;
        Ok(list.as_any().call_method0("__iter__")?.unbind())
    }
}

// ---------------------------------------------------------------------------
// Helper: convert raw editop tuples to MatchingBlock list
// ---------------------------------------------------------------------------

pub fn ops_to_matching_blocks(
    ops: &[(String, usize, usize)],
    src_len: usize,
    dest_len: usize,
) -> Vec<MatchingBlock> {
    let mut blocks = Vec::new();
    let mut src_pos = 0usize;
    let mut dest_pos = 0usize;

    for (tag, sp, dp) in ops {
        if *sp > src_pos || *dp > dest_pos {
            // there's a matching block before this op
            let block_len = dp.min(&(dest_pos + (sp - src_pos))) - &dest_pos;
            if block_len > 0 {
                blocks.push(MatchingBlock {
                    a: src_pos,
                    b: dest_pos,
                    size: block_len,
                });
            }
        }
        match tag.as_str() {
            "delete" => src_pos = sp + 1,
            "insert" => dest_pos = dp + 1,
            "replace" => {
                src_pos = sp + 1;
                dest_pos = dp + 1;
            }
            _ => {}
        }
    }
    // trailing matching block
    let trailing = (src_len - src_pos).min(dest_len - dest_pos);
    if trailing > 0 {
        blocks.push(MatchingBlock {
            a: src_pos,
            b: dest_pos,
            size: trailing,
        });
    }
    blocks.push(MatchingBlock {
        a: src_len,
        b: dest_len,
        size: 0,
    });
    blocks
}

// ---------------------------------------------------------------------------
// Editops
// ---------------------------------------------------------------------------

#[pyclass(module = "rapidfuzz.distance._initialize_cpp")]
#[derive(Clone)]
pub struct Editops {
    pub ops: Vec<(String, usize, usize)>, // (tag, src_pos, dest_pos)
    #[pyo3(get, set)]
    pub src_len: usize,
    #[pyo3(get, set)]
    pub dest_len: usize,
}

fn parse_editop(item: &Bound<'_, PyAny>) -> PyResult<(String, usize, usize)> {
    // Accept Editop objects
    if let Ok(op) = item.extract::<PyRef<Editop>>() {
        return Ok((op.tag.clone(), op.src_pos, op.dest_pos));
    }
    // Try as tuple (tag, src_pos, dest_pos) - MUST be exactly 3 elements
    let tup = item.downcast::<PyTuple>()?;
    if tup.len() != 3 {
        return Err(pyo3::exceptions::PyTypeError::new_err("expected 3-element tuple for Editop"));
    }
    let tag: String = tup.get_item(0)?.extract()?;
    let sp: usize = tup.get_item(1)?.extract()?;
    let dp: usize = tup.get_item(2)?.extract()?;
    Ok((tag, sp, dp))
}

fn parse_opcode(item: &Bound<'_, PyAny>) -> PyResult<(String, usize, usize, usize, usize)> {
    if let Ok(op) = item.extract::<PyRef<Opcode>>() {
        return Ok((op.tag.clone(), op.src_start, op.src_end, op.dest_start, op.dest_end));
    }
    let tup = item.downcast::<PyTuple>()?;
    let tag: String = tup.get_item(0)?.extract()?;
    let ss: usize = tup.get_item(1)?.extract()?;
    let se: usize = tup.get_item(2)?.extract()?;
    let ds: usize = tup.get_item(3)?.extract()?;
    let de: usize = tup.get_item(4)?.extract()?;
    Ok((tag, ss, se, ds, de))
}

/// Accepts tuples, Editop objects, or Opcodes-like objects and produces editops
fn editops_from_list(
    py: Python<'_>,
    ops_obj: &Bound<'_, PyAny>,
    src_len: usize,
    dest_len: usize,
) -> PyResult<Editops> {
    let mut ops: Vec<(String, usize, usize)> = Vec::new();
    for item in ops_obj.try_iter()? {
        let item = item?;
        // Try editop format first
        if let Ok(t) = parse_editop(&item) {
            ops.push(t);
        } else if let Ok((tag, ss, se, ds, de)) = parse_opcode(&item) {
            // Convert opcode block to editops
            if tag == "equal" {
                // no editops for equal blocks
            } else if tag == "replace" {
                for i in 0..(se - ss) {
                    ops.push(("replace".to_string(), ss + i, ds + i));
                }
            } else if tag == "delete" {
                for i in 0..(se - ss) {
                    ops.push(("delete".to_string(), ss + i, ds));
                }
            } else if tag == "insert" {
                for j in 0..(de - ds) {
                    ops.push(("insert".to_string(), ss, ds + j));
                }
            }
        }
    }
    Ok(Editops { ops, src_len, dest_len })
}

fn resolve_slice(idx: i64, len: usize) -> usize {
    if idx < 0 {
        (len as i64 + idx).max(0) as usize
    } else {
        (idx as usize).min(len)
    }
}

#[pymethods]
impl Editops {
    #[new]
    #[pyo3(signature = (ops, src_len, dest_len))]
    fn new(py: Python<'_>, ops: &Bound<'_, PyAny>, src_len: usize, dest_len: usize) -> PyResult<Self> {
        editops_from_list(py, ops, src_len, dest_len)
    }

    fn __repr__(&self) -> String {
        format!(
            "Editops([{}], src_len={}, dest_len={})",
            self.ops
                .iter()
                .map(|(t, s, d)| format!("Editop(tag='{}', src_pos={}, dest_pos={})", t, s, d))
                .collect::<Vec<_>>()
                .join(", "),
            self.src_len,
            self.dest_len,
        )
    }

    fn __len__(&self) -> usize {
        self.ops.len()
    }

    fn __eq__(&self, other: &Editops) -> bool {
        self.ops == other.ops && self.src_len == other.src_len && self.dest_len == other.dest_len
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Ok(idx) = key.extract::<i64>() {
            let n = self.ops.len() as i64;
            let i = if idx < 0 { n + idx } else { idx };
            if i < 0 || i >= n {
                return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
            }
            let (t, s, d) = &self.ops[i as usize];
            let op = Editop { tag: t.clone(), src_pos: *s, dest_pos: *d };
            return Ok(op.into_pyobject(py)?.into_any().unbind());
        }
        // slice
        if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            let len = self.ops.len();
            let indices = slice.indices(len as isize)?;
            if indices.step <= 0 {
                if indices.step == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err("slice step cannot be zero"));
                }
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "step sizes below 0 lead to an invalid order of editops",
                ));
            }
            let step = indices.step as usize;
            let mut new_ops = Vec::new();
            let mut i = indices.start as usize;
            while i < indices.stop as usize {
                new_ops.push(self.ops[i].clone());
                i += step;
            }
            let result = Editops {
                ops: new_ops,
                src_len: self.src_len,
                dest_len: self.dest_len,
            };
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }
        Err(pyo3::exceptions::PyTypeError::new_err("invalid index type"))
    }

    fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        // Handle integer index
        if let Ok(idx) = key.extract::<i64>() {
            let len = self.ops.len() as i64;
            let i = if idx < 0 { len + idx } else { idx };
            if i < 0 || i >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
            }
            self.ops.remove(i as usize);
            return Ok(());
        }
        if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            let len = self.ops.len();
            let indices = slice.indices(len as isize)?;
            // Collect indices to remove
            let mut to_remove: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let step = indices.step as usize;
            let mut i = indices.start as usize;
            while i < indices.stop as usize && i < len {
                to_remove.insert(i);
                i += step;
            }
            self.ops = self
                .ops
                .iter()
                .enumerate()
                .filter(|(i, _)| !to_remove.contains(i))
                .map(|(_, v)| v.clone())
                .collect();
            return Ok(());
        }
        Err(pyo3::exceptions::PyTypeError::new_err("unsupported key type"))
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items: Vec<PyObject> = self
            .ops
            .iter()
            .map(|(t, s, d)| {
                let op = Editop { tag: t.clone(), src_pos: *s, dest_pos: *d };
                op.into_pyobject(py).map(|v| v.into_any().unbind())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let list = PyList::new(py, items)?;
        Ok(list.as_any().call_method0("__iter__")?.unbind())
    }

    fn copy(&self) -> Editops {
        self.clone()
    }

    fn as_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items: Vec<PyObject> = self
            .ops
            .iter()
            .map(|(t, s, d)| {
                PyTuple::new(
                    py,
                    [
                        t.as_str().into_pyobject(py).unwrap().into_any().unbind(),
                        s.into_pyobject(py).unwrap().into_any().unbind(),
                        d.into_pyobject(py).unwrap().into_any().unbind(),
                    ],
                )
                .map(|v| v.into_any().unbind())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PyList::new(py, items)?.into_any().unbind())
    }

    fn inverse(&self) -> Editops {
        let ops: Vec<(String, usize, usize)> = self
            .ops
            .iter()
            .map(|(tag, sp, dp)| {
                let new_tag = match tag.as_str() {
                    "insert" => "delete",
                    "delete" => "insert",
                    other => other,
                };
                (new_tag.to_string(), *dp, *sp)
            })
            .collect();
        Editops {
            ops,
            src_len: self.dest_len,
            dest_len: self.src_len,
        }
    }

    fn as_matching_blocks(&self, py: Python<'_>) -> PyResult<PyObject> {
        let blocks = editops_to_matching_blocks(&self.ops, self.src_len, self.dest_len);
        let items: Vec<PyObject> = blocks
            .iter()
            .map(|b| {
                MatchingBlock { a: b.0, b: b.1, size: b.2 }
                    .into_pyobject(py)
                    .map(|v| v.into_any().unbind())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PyList::new(py, items)?.into_any().unbind())
    }

    fn as_opcodes(&self, py: Python<'_>) -> PyResult<Opcodes> {
        let opcodes = editops_to_opcodes(&self.ops, self.src_len, self.dest_len);
        Ok(Opcodes {
            ops: opcodes,
            src_len: self.src_len,
            dest_len: self.dest_len,
        })
    }

    fn remove_subsequence(&self, other: &Editops) -> Editops {
        // Remove operations in `other` from self
        let mut result_ops = self.ops.clone();
        for op in &other.ops {
            if let Some(pos) = result_ops.iter().position(|x| x == op) {
                result_ops.remove(pos);
            }
        }
        Editops {
            ops: result_ops,
            src_len: self.src_len,
            dest_len: self.dest_len,
        }
    }
}

// ---------------------------------------------------------------------------
// Opcodes
// ---------------------------------------------------------------------------

#[pyclass(module = "rapidfuzz.distance._initialize_cpp")]
#[derive(Clone)]
pub struct Opcodes {
    pub ops: Vec<(String, usize, usize, usize, usize)>, // (tag, ss, se, ds, de)
    #[pyo3(get, set)]
    pub src_len: usize,
    #[pyo3(get, set)]
    pub dest_len: usize,
}

fn opcodes_from_list(
    py: Python<'_>,
    ops_obj: &Bound<'_, PyAny>,
    src_len: usize,
    dest_len: usize,
) -> PyResult<Opcodes> {
    let mut opcode_ops: Vec<(String, usize, usize, usize, usize)> = Vec::new();
    let mut editop_ops: Vec<(String, usize, usize)> = Vec::new();
    let mut has_opcodes = false;
    let mut has_editops = false;

    for item in ops_obj.try_iter()? {
        let item = item?;
        if let Ok((tag, ss, se, ds, de)) = parse_opcode(&item) {
            opcode_ops.push((tag, ss, se, ds, de));
            has_opcodes = true;
        } else if let Ok((tag, sp, dp)) = parse_editop(&item) {
            editop_ops.push((tag, sp, dp));
            has_editops = true;
        }
    }

    // When list is empty, generate sentinel equal block if any content
    if !has_opcodes && !has_editops {
        if src_len == 0 && dest_len == 0 {
            return Ok(Opcodes { ops: vec![], src_len, dest_len });
        }
        let eq_block = ("equal".to_string(), 0, src_len, 0, dest_len);
        return Ok(Opcodes { ops: vec![eq_block], src_len, dest_len });
    }

    if has_editops && !has_opcodes {
        // Convert editops to opcodes using the proper algorithm (merges consecutive same-type ops)
        let ops = editops_to_opcodes(&editop_ops, src_len, dest_len);
        return Ok(Opcodes { ops, src_len, dest_len });
    }

    // Input is already in opcode 5-tuple format, normalize gaps
    let ops = merge_opcodes(opcode_ops, src_len, dest_len);
    Ok(Opcodes { ops, src_len, dest_len })
}

/// Normalize opcodes: fill gaps with "equal", merge adjacent equal blocks
fn merge_opcodes(
    input: Vec<(String, usize, usize, usize, usize)>,
    src_len: usize,
    dest_len: usize,
) -> Vec<(String, usize, usize, usize, usize)> {
    let mut result: Vec<(String, usize, usize, usize, usize)> = Vec::new();
    let mut src_pos = 0usize;
    let mut dest_pos = 0usize;

    for (tag, ss, se, ds, de) in &input {
        if src_pos < *ss || dest_pos < *ds {
            // Insert equal block to fill gap
            let gap = (ss - src_pos).min(ds - dest_pos);
            if gap > 0 {
                push_opcode(&mut result, "equal", src_pos, src_pos + gap, dest_pos, dest_pos + gap);
                src_pos += gap;
                dest_pos += gap;
            }
        }
        push_opcode(&mut result, tag, *ss, *se, *ds, *de);
        src_pos = *se;
        dest_pos = *de;
    }

    // Trailing equal block
    if src_pos < src_len || dest_pos < dest_len {
        let gap = (src_len - src_pos).min(dest_len - dest_pos);
        if gap > 0 {
            push_opcode(&mut result, "equal", src_pos, src_pos + gap, dest_pos, dest_pos + gap);
        }
    }
    result
}

fn push_opcode(
    result: &mut Vec<(String, usize, usize, usize, usize)>,
    tag: &str,
    ss: usize,
    se: usize,
    ds: usize,
    de: usize,
) {
    if let Some(last) = result.last_mut() {
        if last.0 == tag {
            // Merge with last
            last.2 = se;
            last.4 = de;
            return;
        }
    }
    result.push((tag.to_string(), ss, se, ds, de));
}

#[pymethods]
impl Opcodes {
    #[new]
    fn new(py: Python<'_>, ops: &Bound<'_, PyAny>, src_len: usize, dest_len: usize) -> PyResult<Self> {
        opcodes_from_list(py, ops, src_len, dest_len)
    }

    fn __repr__(&self) -> String {
        format!(
            "Opcodes([{}], src_len={}, dest_len={})",
            self.ops
                .iter()
                .map(|(t, ss, se, ds, de)| format!(
                    "Opcode(tag='{}', src_start={}, src_end={}, dest_start={}, dest_end={})",
                    t, ss, se, ds, de
                ))
                .collect::<Vec<_>>()
                .join(", "),
            self.src_len,
            self.dest_len,
        )
    }

    fn __len__(&self) -> usize {
        self.ops.len()
    }

    fn __eq__(&self, other: &Opcodes) -> bool {
        self.ops == other.ops && self.src_len == other.src_len && self.dest_len == other.dest_len
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Ok(idx) = key.extract::<i64>() {
            let n = self.ops.len() as i64;
            let i = if idx < 0 { n + idx } else { idx };
            if i < 0 || i >= n {
                return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
            }
            let (t, ss, se, ds, de) = &self.ops[i as usize];
            let op = Opcode {
                tag: t.clone(),
                src_start: *ss,
                src_end: *se,
                dest_start: *ds,
                dest_end: *de,
            };
            return Ok(op.into_pyobject(py)?.into_any().unbind());
        }
        Err(pyo3::exceptions::PyTypeError::new_err("invalid key"))
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items: Vec<PyObject> = self
            .ops
            .iter()
            .map(|(t, ss, se, ds, de)| {
                let op = Opcode {
                    tag: t.clone(),
                    src_start: *ss,
                    src_end: *se,
                    dest_start: *ds,
                    dest_end: *de,
                };
                op.into_pyobject(py).map(|v| v.into_any().unbind())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let list = PyList::new(py, items)?;
        Ok(list.as_any().call_method0("__iter__")?.unbind())
    }

    fn copy(&self) -> Opcodes {
        self.clone()
    }

    fn as_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items: Vec<PyObject> = self
            .ops
            .iter()
            .map(|(t, ss, se, ds, de)| {
                PyTuple::new(
                    py,
                    [
                        t.as_str().into_pyobject(py).unwrap().into_any().unbind(),
                        ss.into_pyobject(py).unwrap().into_any().unbind(),
                        se.into_pyobject(py).unwrap().into_any().unbind(),
                        ds.into_pyobject(py).unwrap().into_any().unbind(),
                        de.into_pyobject(py).unwrap().into_any().unbind(),
                    ],
                )
                .map(|v| v.into_any().unbind())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PyList::new(py, items)?.into_any().unbind())
    }

    fn inverse(&self) -> Opcodes {
        let ops: Vec<(String, usize, usize, usize, usize)> = self
            .ops
            .iter()
            .map(|(tag, ss, se, ds, de)| {
                let new_tag = match tag.as_str() {
                    "insert" => "delete",
                    "delete" => "insert",
                    other => other,
                };
                (new_tag.to_string(), *ds, *de, *ss, *se)
            })
            .collect();
        Opcodes {
            ops,
            src_len: self.dest_len,
            dest_len: self.src_len,
        }
    }

    fn as_editops(&self, py: Python<'_>) -> PyResult<Editops> {
        let mut ops: Vec<(String, usize, usize)> = Vec::new();
        for (tag, ss, se, ds, de) in &self.ops {
            match tag.as_str() {
                "equal" => {}
                "replace" => {
                    for i in 0..(se - ss) {
                        ops.push(("replace".to_string(), ss + i, ds + i));
                    }
                }
                "delete" => {
                    for i in 0..(se - ss) {
                        ops.push(("delete".to_string(), ss + i, *ds));
                    }
                }
                "insert" => {
                    for j in 0..(de - ds) {
                        ops.push(("insert".to_string(), *ss, ds + j));
                    }
                }
                _ => {}
            }
        }
        Ok(Editops {
            ops,
            src_len: self.src_len,
            dest_len: self.dest_len,
        })
    }

    fn as_matching_blocks(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut blocks: Vec<PyObject> = Vec::new();
        for (tag, ss, se, ds, de) in &self.ops {
            if tag == "equal" {
                let size = (se - ss).min(de - ds);
                if size > 0 {
                    let b = MatchingBlock { a: *ss, b: *ds, size };
                    blocks.push(b.into_pyobject(py)?.into_any().unbind());
                }
            }
        }
        blocks.push(
            MatchingBlock { a: self.src_len, b: self.dest_len, size: 0 }
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        Ok(PyList::new(py, blocks)?.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers for editops <-> opcodes conversion
// ---------------------------------------------------------------------------

/// Convert editops to opcodes, exactly mirroring Python's _initialize_py.py Editops.as_opcodes()
fn editops_to_opcodes(
    ops: &[(String, usize, usize)],
    src_len: usize,
    dest_len: usize,
) -> Vec<(String, usize, usize, usize, usize)> {
    let mut result: Vec<(String, usize, usize, usize, usize)> = Vec::new();
    let mut src_pos = 0usize;
    let mut dest_pos = 0usize;
    let n = ops.len();
    let mut i = 0;
    while i < n {
        let (tag, sp, dp) = &ops[i];
        // Insert equal block if there's a gap
        if *sp > src_pos || *dp > dest_pos {
            result.push(("equal".to_string(), src_pos, *sp, dest_pos, *dp));
            src_pos = *sp;
            dest_pos = *dp;
        }
        // Group consecutive editops of same tag where positions advance correctly
        let src_begin = src_pos;
        let dest_begin = dest_pos;
        let cur_tag = tag.clone();
        while i < n {
            let (cur, csp, cdp) = &ops[i];
            if cur != &cur_tag {
                break;
            }
            if *csp != src_pos || *cdp != dest_pos {
                break;
            }
            match cur_tag.as_str() {
                "replace" => { src_pos += 1; dest_pos += 1; }
                "insert" => { dest_pos += 1; }
                "delete" => { src_pos += 1; }
                _ => {}
            }
            i += 1;
        }
        // Merge with previous if same tag
        if let Some(last) = result.last_mut() {
            if last.0 == cur_tag && last.2 == src_begin && last.4 == dest_begin {
                last.2 = src_pos;
                last.4 = dest_pos;
                continue;
            }
        }
        result.push((cur_tag, src_begin, src_pos, dest_begin, dest_pos));
    }

    // Trailing equal block
    if src_pos < src_len || dest_pos < dest_len {
        result.push(("equal".to_string(), src_pos, src_len, dest_pos, dest_len));
    }
    result
}

fn editops_to_matching_blocks(
    ops: &[(String, usize, usize)],
    src_len: usize,
    dest_len: usize,
) -> Vec<(usize, usize, usize)> {
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
    let mut src_pos = 0usize;
    let mut dest_pos = 0usize;

    for (tag, sp, dp) in ops {
        if *sp > src_pos && *dp > dest_pos {
            let eq_len = (sp - src_pos).min(dp - dest_pos);
            if eq_len > 0 {
                blocks.push((src_pos, dest_pos, eq_len));
            }
        }
        match tag.as_str() {
            "replace" => { src_pos = sp + 1; dest_pos = dp + 1; }
            "delete" => { src_pos = sp + 1; }
            "insert" => { dest_pos = dp + 1; }
            _ => {}
        }
    }
    if src_pos < src_len && dest_pos < dest_len {
        let eq_len = (src_len - src_pos).min(dest_len - dest_pos);
        if eq_len > 0 {
            blocks.push((src_pos, dest_pos, eq_len));
        }
    }
    blocks.push((src_len, dest_len, 0));
    blocks
}

pub fn raw_editops_to_py(
    py: Python<'_>,
    ops: Vec<(String, usize, usize)>,
    src_len: usize,
    dest_len: usize,
) -> PyResult<Editops> {
    Ok(Editops { ops, src_len, dest_len })
}

pub fn raw_editops_to_opcodes_py(
    py: Python<'_>,
    ops: Vec<(String, usize, usize)>,
    src_len: usize,
    dest_len: usize,
) -> PyResult<Opcodes> {
    let opcode_list = editops_to_opcodes(&ops, src_len, dest_len);
    Ok(Opcodes { ops: opcode_list, src_len, dest_len })
}
