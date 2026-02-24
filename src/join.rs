// SPDX-License-Identifier: MIT
// Multi-array fuzzy full join.
//
// Each array element may carry: text, sparse vector, and/or dense vector.
// Channels are fused with weighted Reciprocal Rank Fusion (RRF).
// All heavy work runs outside the GIL via Rayon.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// Tokeniser (mirror of search.rs — kept local to avoid cross-module coupling)
// ---------------------------------------------------------------------------
#[inline]
fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// Lightweight in-module BM25 (no pyclass overhead — plain Rust struct)
// ---------------------------------------------------------------------------
struct Bm25Index {
    tf_norm: Vec<FxHashMap<String, f64>>,
    idf: FxHashMap<String, f64>,
}

impl Bm25Index {
    fn build(docs: &[Option<String>], k1: f64, b: f64) -> Self {
        let n = docs.len() as f64;
        let tokenised: Vec<Vec<String>> = docs
            .iter()
            .map(|d| d.as_deref().map(tokenise).unwrap_or_default())
            .collect();

        let total: usize = tokenised.iter().map(|d| d.len()).sum();
        let avgdl = if n > 0.0 { total as f64 / n } else { 1.0 };

        // Document frequency
        let mut df: FxHashMap<String, usize> = FxHashMap::default();
        for doc in &tokenised {
            let mut seen: FxHashMap<&str, bool> = FxHashMap::default();
            for term in doc {
                if seen.insert(term.as_str(), true).is_none() {
                    *df.entry(term.clone()).or_insert(0) += 1;
                }
            }
        }

        // IDF (Robertson BM25+)
        let idf: FxHashMap<String, f64> = df
            .iter()
            .map(|(term, &dft)| {
                let dft_f = dft as f64;
                let val = ((n - dft_f + 0.5) / (dft_f + 0.5) + 1.0).ln();
                (term.clone(), val)
            })
            .collect();

        // Normalised TF per document
        let tf_norm: Vec<FxHashMap<String, f64>> = tokenised
            .iter()
            .map(|doc| {
                let dl = doc.len() as f64;
                let mut raw: FxHashMap<String, usize> = FxHashMap::default();
                for term in doc {
                    *raw.entry(term.clone()).or_insert(0) += 1;
                }
                raw.into_iter()
                    .map(|(term, tf)| {
                        let tf_f = tf as f64;
                        let norm =
                            tf_f * (k1 + 1.0) / (tf_f + k1 * (1.0 - b + b * dl / avgdl));
                        (term, norm)
                    })
                    .collect()
            })
            .collect();

        Bm25Index { tf_norm, idf }
    }

    /// Score `query` against document `doc_idx`.
    fn score(&self, query_terms: &[String], doc_idx: usize) -> f64 {
        let doc_tf = &self.tf_norm[doc_idx];
        query_terms
            .iter()
            .map(|term| {
                let idf = self.idf.get(term).copied().unwrap_or(0.0);
                let tf = doc_tf.get(term).copied().unwrap_or(0.0);
                idf * tf
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Sparse dot product
// ---------------------------------------------------------------------------
#[inline]
fn sparse_dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    // Both slices must be sorted by key (ascending) for merge-join.
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut acc = 0.0f32;
    while ia < a.len() && ib < b.len() {
        match a[ia].0.cmp(&b[ib].0) {
            std::cmp::Ordering::Equal => {
                acc += a[ia].1 * b[ib].1;
                ia += 1;
                ib += 1;
            }
            std::cmp::Ordering::Less => ia += 1,
            std::cmp::Ordering::Greater => ib += 1,
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Dense cosine (dot product on pre-normalised vectors)
// ---------------------------------------------------------------------------
#[inline]
fn dense_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Internal entry
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct Entry {
    text: Option<String>,
    sparse: Option<Vec<(u32, f32)>>,
    dense: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Result row
// ---------------------------------------------------------------------------
struct JoinRow {
    src_array: String,
    src_idx: usize,
    src_text: Option<String>,
    tgt_array: String,
    tgt_idx: usize,
    tgt_text: Option<String>,
    score: f64,
    text_score: Option<f64>,
    sparse_score: Option<f64>,
    dense_score: Option<f64>,
}

// ---------------------------------------------------------------------------
// MultiJoiner pyclass
// ---------------------------------------------------------------------------

/// Fuzzy full-join across multiple named arrays.
///
/// Each array element may carry a text string, a sparse vector, and/or a dense vector.
/// All active channels are fused with weighted Reciprocal Rank Fusion (RRF).
///
/// Parameters
/// ----------
/// text_weight : float, default 1.0
/// sparse_weight : float, default 1.0
/// dense_weight : float, default 1.0
/// bm25_k1 : float, default 1.5
/// bm25_b : float, default 0.75
/// rrf_k : int, default 60
///     RRF smoothing constant.
#[pyclass]
pub struct MultiJoiner {
    arrays: Vec<(String, Vec<Entry>)>,
    text_weight: f64,
    sparse_weight: f64,
    dense_weight: f64,
    bm25_k1: f64,
    bm25_b: f64,
    rrf_k: usize,
}

#[pymethods]
impl MultiJoiner {
    #[new]
    #[pyo3(signature = (
        text_weight = 1.0,
        sparse_weight = 1.0,
        dense_weight = 1.0,
        bm25_k1 = 1.5,
        bm25_b = 0.75,
        rrf_k = 60,
    ))]
    pub fn new(
        text_weight: f64,
        sparse_weight: f64,
        dense_weight: f64,
        bm25_k1: f64,
        bm25_b: f64,
        rrf_k: usize,
    ) -> Self {
        MultiJoiner {
            arrays: Vec::new(),
            text_weight,
            sparse_weight,
            dense_weight,
            bm25_k1,
            bm25_b,
            rrf_k,
        }
    }

    /// Add a named array.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Unique array name.
    /// texts : list[str | None] | None
    ///     Text per element. None disables the text channel for this array.
    /// sparse : list[dict[int, float]] | None
    ///     Sparse vectors as {token_id: weight}. Sorted by key internally.
    /// dense : list[list[float]] | None
    ///     Dense embedding vectors. Assumed pre-normalised for cosine similarity.
    #[pyo3(signature = (name, texts=None, sparse=None, dense=None))]
    pub fn add_array(
        &mut self,
        name: String,
        texts: Option<Vec<Option<String>>>,
        sparse: Option<Vec<Option<FxHashMap<u32, f32>>>>,
        dense: Option<Vec<Option<Vec<f32>>>>,
    ) -> PyResult<()> {
        // Determine array length from whichever channel is provided
        let n = texts
            .as_ref()
            .map(|v| v.len())
            .or_else(|| sparse.as_ref().map(|v| v.len()))
            .or_else(|| dense.as_ref().map(|v| v.len()))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "add_array: at least one of texts, sparse, dense must be provided",
                )
            })?;

        // Validate lengths match
        if let Some(ref t) = texts {
            if t.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "add_array: texts length mismatch",
                ));
            }
        }
        if let Some(ref s) = sparse {
            if s.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "add_array: sparse length mismatch",
                ));
            }
        }
        if let Some(ref d) = dense {
            if d.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "add_array: dense length mismatch",
                ));
            }
        }

        let entries: Vec<Entry> = (0..n)
            .map(|i| {
                let text = texts.as_ref().and_then(|t| t[i].clone());
                let sparse_vec = sparse.as_ref().and_then(|s| {
                    s[i].as_ref().map(|m| {
                        let mut pairs: Vec<(u32, f32)> =
                            m.iter().map(|(&k, &v)| (k, v)).collect();
                        pairs.sort_unstable_by_key(|p| p.0);
                        pairs
                    })
                });
                let dense_vec = dense.as_ref().and_then(|d| d[i].clone());
                Entry { text, sparse: sparse_vec, dense: dense_vec }
            })
            .collect();

        self.arrays.push((name, entries));
        Ok(())
    }

    /// Number of arrays currently registered.
    #[getter]
    pub fn num_arrays(&self) -> usize {
        self.arrays.len()
    }

    /// Run the full join across every ordered pair of arrays (src ≠ tgt).
    ///
    /// Parameters
    /// ----------
    /// n : int, default 1
    ///     Number of top target matches to return per source element.
    ///
    /// Returns
    /// -------
    /// list[dict]
    ///     Each row: src_array, src_idx, src_text, tgt_array, tgt_idx, tgt_text,
    ///     score, text_score, sparse_score, dense_score.
    #[pyo3(signature = (n=1))]
    pub fn join(&self, py: Python<'_>, n: usize) -> PyResult<Vec<PyObject>> {
        // Collect all pairs
        let mut all_rows: Vec<JoinRow> = Vec::new();
        for (si, (src_name, src_entries)) in self.arrays.iter().enumerate() {
            for (ti, (tgt_name, tgt_entries)) in self.arrays.iter().enumerate() {
                if si == ti {
                    continue;
                }
                let rows = self.join_arrays_inner(
                    src_name,
                    src_entries,
                    tgt_name,
                    tgt_entries,
                    n,
                );
                all_rows.extend(rows);
            }
        }
        rows_to_py(py, all_rows)
    }

    /// Run the join for one specific ordered pair.
    #[pyo3(signature = (src_name, tgt_name, n=1))]
    pub fn join_pair(
        &self,
        py: Python<'_>,
        src_name: &str,
        tgt_name: &str,
        n: usize,
    ) -> PyResult<Vec<PyObject>> {
        let src = self
            .arrays
            .iter()
            .find(|(name, _)| name == src_name)
            .map(|(_, e)| e)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "unknown array '{src_name}'"
                ))
            })?;
        let tgt = self
            .arrays
            .iter()
            .find(|(name, _)| name == tgt_name)
            .map(|(_, e)| e)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "unknown array '{tgt_name}'"
                ))
            })?;
        let rows =
            self.join_arrays_inner(src_name, src, tgt_name, tgt, n);
        rows_to_py(py, rows)
    }
}

// ---------------------------------------------------------------------------
// Core join logic (pure Rust, no GIL)
// ---------------------------------------------------------------------------

impl MultiJoiner {
    fn join_arrays_inner(
        &self,
        src_name: &str,
        src_entries: &[Entry],
        tgt_name: &str,
        tgt_entries: &[Entry],
        n: usize,
    ) -> Vec<JoinRow> {
        let rrf_k = self.rrf_k;

        // ── Text channel setup ────────────────────────────────────────────
        // Build BM25 over tgt texts once, reuse for every src query.
        let tgt_texts: Vec<Option<String>> =
            tgt_entries.iter().map(|e| e.text.clone()).collect();
        let has_any_text = src_entries.iter().any(|e| e.text.is_some())
            && tgt_entries.iter().any(|e| e.text.is_some());
        // Build BM25 whenever text is available and either the text or sparse
        // channel is active (sparse auto-falls back to BM25 when no explicit
        // sparse vectors are provided).
        let bm25 = if has_any_text
            && (self.text_weight > 0.0 || self.sparse_weight > 0.0)
        {
            Some(Bm25Index::build(&tgt_texts, self.bm25_k1, self.bm25_b))
        } else {
            None
        };

        // ── Rayon-parallel over source entries ────────────────────────────
        let rows: Vec<JoinRow> = src_entries
            .par_iter()
            .enumerate()
            .flat_map(|(src_idx, src)| {
                // Per-target scores for each active channel
                let n_tgt = tgt_entries.len();

                // Text channel: BM25 rank + indel fuzzy rank → RRF
                let text_final_scores: Option<Vec<f64>> =
                    if self.text_weight > 0.0 {
                        if let (Some(bm25_idx), Some(src_text)) =
                            (&bm25, &src.text)
                        {
                        let q_terms = tokenise(src_text);

                        // BM25 scores for all tgt docs
                        let bm25_scores: Vec<(usize, f64)> = (0..n_tgt)
                            .map(|j| (j, bm25_idx.score(&q_terms, j)))
                            .collect();

                        // Sort BM25 for ranking
                        let mut bm25_ranked = bm25_scores.clone();
                        bm25_ranked.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });

                        // Indel fuzzy scores for all tgt docs
                        let q_bytes = src_text.as_bytes();
                        let q_is_ascii = src_text.is_ascii();
                        let fuzzy_scores: Vec<(usize, f64)> = tgt_entries
                            .iter()
                            .enumerate()
                            .map(|(j, tgt)| {
                                let sim = tgt
                                    .text
                                    .as_deref()
                                    .map(|d| {
                                        if q_is_ascii && d.is_ascii() {
                                            let d_b = d.as_bytes();
                                            let ls =
                                                q_bytes.len() + d_b.len();
                                            if ls == 0 {
                                                1.0
                                            } else {
                                                use crate::algorithms::indel_distance;
                                                1.0 - indel_distance(
                                                    q_bytes, d_b, None,
                                                )
                                                    as f64
                                                    / ls as f64
                                            }
                                        } else {
                                            let qv: Vec<u32> = src_text
                                                .chars()
                                                .map(|c| c as u32)
                                                .collect();
                                            let dv: Vec<u32> = d
                                                .chars()
                                                .map(|c| c as u32)
                                                .collect();
                                            let ls = qv.len() + dv.len();
                                            if ls == 0 {
                                                1.0
                                            } else {
                                                use crate::algorithms::indel_distance;
                                                1.0 - indel_distance(
                                                    &qv, &dv, None,
                                                )
                                                    as f64
                                                    / ls as f64
                                            }
                                        }
                                    })
                                    .unwrap_or(0.0);
                                (j, sim)
                            })
                            .collect();

                        let mut fuzzy_ranked = fuzzy_scores.clone();
                        fuzzy_ranked.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });

                        // RRF fusion of BM25 rank + fuzzy rank
                        let mut rrf = vec![0.0f64; n_tgt];
                        for (rank, (j, _)) in bm25_ranked.iter().enumerate() {
                            rrf[*j] += 1.0 / (rrf_k + rank + 1) as f64;
                        }
                        for (rank, (j, _)) in fuzzy_ranked.iter().enumerate() {
                            rrf[*j] += 1.0 / (rrf_k + rank + 1) as f64;
                        }
                        Some(rrf)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                // Sparse channel: explicit dot product, or BM25 auto-fallback.
                // When no sparse vectors are provided but text + BM25 are
                // available, pure BM25 scores are used as the sparse signal.
                // This means the sparse channel captures term-overlap
                // independently from the BM25+fuzzy RRF in the text channel.
                let sparse_scores: Option<Vec<f64>> =
                    if self.sparse_weight > 0.0 {
                        if let Some(sv) = src.sparse.as_ref() {
                            // Explicit sparse vectors — merge-join dot product
                            Some(
                                tgt_entries
                                    .iter()
                                    .map(|tgt| {
                                        tgt.sparse
                                            .as_ref()
                                            .map(|tv| sparse_dot(sv, tv) as f64)
                                            .unwrap_or(0.0)
                                    })
                                    .collect(),
                            )
                        } else if let (Some(bm25_idx), Some(src_text)) =
                            (&bm25, &src.text)
                        {
                            // BM25 auto-fallback: score query against every
                            // target doc using pure BM25 (no fuzzy rerank).
                            let q_terms = tokenise(src_text);
                            Some(
                                (0..n_tgt)
                                    .map(|j| bm25_idx.score(&q_terms, j))
                                    .collect(),
                            )
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                // Dense channel: cosine (dot on normalised) per tgt
                let dense_scores: Option<Vec<f64>> =
                    if self.dense_weight > 0.0 {
                        src.dense.as_ref().map(|sv| {
                            tgt_entries
                                .iter()
                                .map(|tgt| {
                                    tgt.dense
                                        .as_ref()
                                        .map(|tv| dense_dot(sv, tv) as f64)
                                        .unwrap_or(0.0)
                                })
                                .collect()
                        })
                    } else {
                        None
                    };

                // RRF-rank sparse and dense channels
                let sparse_rrf: Option<Vec<f64>> =
                    sparse_scores.as_ref().map(|scores| {
                        let mut ranked: Vec<(usize, f64)> =
                            scores.iter().cloned().enumerate().collect();
                        ranked.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let mut rrf = vec![0.0f64; n_tgt];
                        for (rank, (j, _)) in ranked.iter().enumerate() {
                            rrf[*j] += 1.0 / (rrf_k + rank + 1) as f64;
                        }
                        rrf
                    });

                let dense_rrf: Option<Vec<f64>> =
                    dense_scores.as_ref().map(|scores| {
                        let mut ranked: Vec<(usize, f64)> =
                            scores.iter().cloned().enumerate().collect();
                        ranked.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let mut rrf = vec![0.0f64; n_tgt];
                        for (rank, (j, _)) in ranked.iter().enumerate() {
                            rrf[*j] += 1.0 / (rrf_k + rank + 1) as f64;
                        }
                        rrf
                    });

                // Combine channel RRF scores with weights
                let total_weight = {
                    let mut w = 0.0f64;
                    if text_final_scores.is_some() {
                        w += self.text_weight;
                    }
                    if sparse_rrf.is_some() {
                        w += self.sparse_weight;
                    }
                    if dense_rrf.is_some() {
                        w += self.dense_weight;
                    }
                    w.max(1e-10)
                };

                let combined_scores: Vec<f64> = (0..n_tgt)
                    .map(|j| {
                        let t = text_final_scores
                            .as_ref()
                            .map(|v| self.text_weight * v[j])
                            .unwrap_or(0.0);
                        let s = sparse_rrf
                            .as_ref()
                            .map(|v| self.sparse_weight * v[j])
                            .unwrap_or(0.0);
                        let d = dense_rrf
                            .as_ref()
                            .map(|v| self.dense_weight * v[j])
                            .unwrap_or(0.0);
                        (t + s + d) / total_weight
                    })
                    .collect();

                // Pick top-n tgt by combined score
                let mut ranked_tgt: Vec<usize> = (0..n_tgt).collect();
                ranked_tgt.sort_by(|&a, &b| {
                    combined_scores[b]
                        .partial_cmp(&combined_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                ranked_tgt.truncate(n);

                ranked_tgt
                    .into_iter()
                    .map(|tgt_idx| JoinRow {
                        src_array: src_name.to_string(),
                        src_idx,
                        src_text: src.text.clone(),
                        tgt_array: tgt_name.to_string(),
                        tgt_idx,
                        tgt_text: tgt_entries[tgt_idx].text.clone(),
                        score: combined_scores[tgt_idx],
                        text_score: text_final_scores
                            .as_ref()
                            .map(|v| v[tgt_idx]),
                        sparse_score: sparse_scores
                            .as_ref()
                            .map(|v| v[tgt_idx]),
                        dense_score: dense_scores
                            .as_ref()
                            .map(|v| v[tgt_idx]),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        rows
    }
}

// ---------------------------------------------------------------------------
// Helper: convert Vec<JoinRow> → Vec<PyObject> (list of dicts)
// ---------------------------------------------------------------------------
fn rows_to_py(py: Python<'_>, rows: Vec<JoinRow>) -> PyResult<Vec<PyObject>> {
    rows.into_iter()
        .map(|r| {
            let d = PyDict::new(py);
            d.set_item("src_array", &r.src_array)?;
            d.set_item("src_idx", r.src_idx)?;
            d.set_item(
                "src_text",
                r.src_text.as_deref().map(|s| s.to_string()),
            )?;
            d.set_item("tgt_array", &r.tgt_array)?;
            d.set_item("tgt_idx", r.tgt_idx)?;
            d.set_item(
                "tgt_text",
                r.tgt_text.as_deref().map(|s| s.to_string()),
            )?;
            d.set_item("score", r.score)?;
            d.set_item("text_score", r.text_score)?;
            d.set_item("sparse_score", r.sparse_score)?;
            d.set_item("dense_score", r.dense_score)?;
            Ok(d.into_any().unbind())
        })
        .collect()
}
