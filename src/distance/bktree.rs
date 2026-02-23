use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple, PyType};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BK-Tree
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BKNode {
    value: String,
    children: HashMap<usize, usize>,
}

#[pyclass]
pub struct BKTree {
    nodes: Vec<BKNode>,
}

#[pymethods]
impl BKTree {
    #[new]
    pub fn new() -> Self {
        BKTree { nodes: Vec::new() }
    }

    pub fn insert(&mut self, item: String) {
        if self.nodes.is_empty() {
            self.nodes.push(BKNode {
                value: item,
                children: HashMap::new(),
            });
            return;
        }

        let mut curr = 0;
        loop {
            // Using Levenshtein distance for BKTree metric (must be a metric space)
            let dist = crate::algorithms::levenshtein_uniform(
                self.nodes[curr].value.as_bytes(),
                item.as_bytes(),
            );

            if dist == 0 {
                return; // Already exists
            }

            if let Some(&next_node) = self.nodes[curr].children.get(&dist) {
                curr = next_node;
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(BKNode {
                    value: item,
                    children: HashMap::new(),
                });
                self.nodes[curr].children.insert(dist, new_idx);
                break;
            }
        }
    }

    pub fn dedupe(&mut self, items: Vec<String>, threshold: usize) -> Vec<String> {
        let mut unique = Vec::new();
        for item in items {
            if self.nodes.is_empty() {
                self.insert(item.clone());
                unique.push(item);
                continue;
            }

            let mut candidates = vec![0];
            let mut found_close = false;

            while let Some(curr) = candidates.pop() {
                let node = &self.nodes[curr];
                let dist = crate::algorithms::levenshtein_uniform(
                    node.value.as_bytes(),
                    item.as_bytes(),
                );

                if dist <= threshold {
                    found_close = true;
                    break;
                }

                let min_dist = dist.saturating_sub(threshold);
                let max_dist = dist + threshold;

                for (&child_dist, &child_idx) in &node.children {
                    if child_dist >= min_dist && child_dist <= max_dist {
                        candidates.push(child_idx);
                    }
                }
            }

            if !found_close {
                self.insert(item.clone());
                unique.push(item);
            }
        }
        unique
    }

    /// Collect all values in the tree via DFS.
    fn all_values(&self) -> Vec<String> {
        let mut result = Vec::new();
        if self.nodes.is_empty() {
            return result;
        }
        let mut stack = vec![0usize];
        while let Some(idx) = stack.pop() {
            result.push(self.nodes[idx].value.clone());
            for &child_idx in self.nodes[idx].children.values() {
                stack.push(child_idx);
            }
        }
        result
    }

    /// Pickle support: serialise as (cls, args, state) — pickle calls cls(), then __setstate__(state).
    fn __reduce__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
        let cls = PyType::new::<BKTree>(py);
        let new_args = PyTuple::empty(py);
        let state = slf.__getstate__(py)?;
        // 3-tuple: (callable, args, state)
        Ok(PyTuple::new(py, [
            cls.into_any().unbind(),
            new_args.into_any().unbind(),
            state,
        ])?.into_any().unbind())
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let values = self.all_values();
        let py_values: Vec<PyObject> = values.iter()
            .map(|s| s.clone().into_pyobject(py).map(|v| v.into_any().unbind()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PyList::new(py, py_values)?.into_any().unbind())
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.nodes.clear();
        for item in state.try_iter()? {
            let item = item?;
            let s: String = item.extract()?;
            self.insert(s);
        }
        Ok(())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BKTree>()?;
    Ok(())
}
