// SPDX-License-Identifier: MIT
// BM25Okapi full-text search index + cosine similarity utilities.
// All heavy computation runs outside the GIL via Rayon.

use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// Tokeniser: simple whitespace split, lowercase
// ---------------------------------------------------------------------------
#[inline]
fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// BM25 Index
// ---------------------------------------------------------------------------

/// BM25Okapi index over a text corpus.
///
/// Parameters
/// ----------
/// corpus : list[str]
///     Documents to index.
/// k1 : float, default 1.5
///     Term frequency saturation parameter.
/// b : float, default 0.75
///     Length normalisation factor.
#[pyclass]
pub struct BM25Index {
    /// The original corpus strings (for retrieval)
    corpus: Vec<String>,
    /// Tokenised corpus
    #[allow(dead_code)]
    tokenised: Vec<Vec<String>>,
    /// IDF per term
    idf: FxHashMap<String, f64>,
    /// Normalised TF per document: tf_norm[doc][term] = raw_tf * (k1+1) / (raw_tf + k1*(1-b+b*dl/avgdl))
    tf_norm: Vec<FxHashMap<String, f64>>,
    /// Average document length (in tokens)
    #[allow(dead_code)]
    avgdl: f64,
    #[allow(dead_code)]
    k1: f64,
    #[allow(dead_code)]
    b: f64,
}

#[pymethods]
impl BM25Index {
    #[new]
    #[pyo3(signature = (corpus, k1=1.5, b=0.75))]
    pub fn new(corpus: Vec<String>, k1: f64, b: f64) -> Self {
        let n = corpus.len() as f64;
        let tokenised: Vec<Vec<String>> = corpus.iter().map(|d| tokenise(d)).collect();

        // Average document length
        let total_tokens: usize = tokenised.iter().map(|d| d.len()).sum();
        let avgdl = if n > 0.0 { total_tokens as f64 / n } else { 1.0 };

        // Document frequency per term
        let mut df: FxHashMap<String, usize> = FxHashMap::default();
        for doc in &tokenised {
            let mut seen: FxHashMap<&str, bool> = FxHashMap::default();
            for term in doc {
                if seen.insert(term.as_str(), true).is_none() {
                    *df.entry(term.clone()).or_insert(0) += 1;
                }
            }
        }

        // IDF: ln((N - df + 0.5) / (df + 0.5) + 1)  [Robertson BM25+]
        let idf: FxHashMap<String, f64> = df.iter().map(|(term, &dft)| {
            let dft_f = dft as f64;
            let idf_val = ((n - dft_f + 0.5) / (dft_f + 0.5) + 1.0).ln();
            (term.clone(), idf_val)
        }).collect();

        // Per-document normalised TF
        let tf_norm: Vec<FxHashMap<String, f64>> = tokenised.iter().map(|doc| {
            let dl = doc.len() as f64;
            let mut raw_tf: FxHashMap<String, usize> = FxHashMap::default();
            for term in doc {
                *raw_tf.entry(term.clone()).or_insert(0) += 1;
            }
            raw_tf.into_iter().map(|(term, tf)| {
                let tf_f = tf as f64;
                let norm = tf_f * (k1 + 1.0) / (tf_f + k1 * (1.0 - b + b * dl / avgdl));
                (term, norm)
            }).collect()
        }).collect();

        BM25Index { corpus, tokenised, idf, tf_norm, avgdl, k1, b }
    }

    /// Pickle support: serialise as (corpus, k1, b), rebuild on unpickle.
    fn __reduce__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
        let cls = PyType::new::<BM25Index>(py);
        let corpus_list: Vec<PyObject> = slf.corpus.iter()
            .map(|s| s.clone().into_pyobject(py).map(|v| v.into_any().unbind()))
            .collect::<Result<Vec<_>, _>>()?;
        let py_corpus = pyo3::types::PyList::new(py, corpus_list)?;
        let args = PyTuple::new(py, [
            py_corpus.into_any().unbind(),
            slf.k1.into_pyobject(py)?.into_any().unbind(),
            slf.b.into_pyobject(py)?.into_any().unbind(),
        ])?;
        Ok(PyTuple::new(py, [cls.into_any().unbind(), args.into_any().unbind()])?.into_any().unbind())
    }

    /// Number of documents in the index.
    #[getter]
    pub fn num_docs(&self) -> usize {
        self.corpus.len()
    }

    /// Return BM25 score for `query` against every document.
    pub fn get_scores(&self, query: &str) -> Vec<f64> {
        let q_terms = tokenise(query);
        let n = self.corpus.len();
        let tf_norm = &self.tf_norm;
        let idf = &self.idf;

        (0..n).into_par_iter().map(|i| {
            let doc_tf = &tf_norm[i];
            q_terms.iter().map(|term| {
                let idf_val = idf.get(term).copied().unwrap_or(0.0);
                let tf_val = doc_tf.get(term).copied().unwrap_or(0.0);
                idf_val * tf_val
            }).sum::<f64>()
        }).collect()
    }

    /// Return top-N `(document, score)` pairs for `query`.
    #[pyo3(signature = (query, n=5))]
    pub fn get_top_n(&self, query: &str, n: usize) -> Vec<(String, f64)> {
        let scores = self.get_scores(query);
        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate()
            .filter(|(_, s)| *s > 0.0)
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(n);
        indexed.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect()
    }

    /// BM25 scores for a batch of queries — Rayon-parallel over queries.
    pub fn get_batch_scores(&self, queries: Vec<String>) -> Vec<Vec<f64>> {
        queries.par_iter().map(|q| self.get_scores(q)).collect()
    }

    /// BM25 + fuzzy hybrid: run BM25, take top `bm25_candidates`, then re-rank
    /// with WRatio fuzzy similarity. Final score = bm25_weight * norm_bm25 + fuzzy_weight * fuzzy.
    #[pyo3(signature = (query, n=5, bm25_candidates=50, fuzzy_weight=0.3))]
    pub fn get_top_n_fuzzy(
        &self,
        _py: Python<'_>,
        query: &str,
        n: usize,
        bm25_candidates: usize,
        fuzzy_weight: f64,
    ) -> PyResult<Vec<(String, f64)>> {
        use crate::algorithms::indel_distance;

        let bm25_weight = 1.0 - fuzzy_weight;
        let candidates = self.get_top_n(query, bm25_candidates.max(n * 10));
        if candidates.is_empty() {
            // Fallback: no BM25 hits — pure fuzzy over all docs
            return self.fuzzy_only(query, n);
        }

        // Normalise BM25 scores
        let max_bm25 = candidates.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
        let min_bm25 = candidates.iter().map(|(_, s)| *s).fold(f64::MAX, f64::min);
        let bm25_range = (max_bm25 - min_bm25).max(1e-10);

        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        let mut scored: Vec<(String, f64)> = candidates.into_iter().map(|(doc, bm25_raw)| {
            let norm_bm25 = (bm25_raw - min_bm25) / bm25_range;
            let fuzzy = if q_is_ascii && doc.is_ascii() {
                // Use indel (ratio) as the fuzzy metric — fast and allocation-free for short strings
                let d_bytes = doc.as_bytes();
                let lensum = q_bytes.len() + d_bytes.len();
                if lensum == 0 { 1.0 } else {
                    let dist = indel_distance(q_bytes, d_bytes, None);
                    1.0 - dist as f64 / lensum as f64
                }
            } else {
                // Unicode fallback
                let qv: Vec<u32> = query.chars().map(|c| c as u32).collect();
                let dv: Vec<u32> = doc.chars().map(|c| c as u32).collect();
                let lensum = qv.len() + dv.len();
                if lensum == 0 { 1.0 } else {
                    let dist = indel_distance(&qv, &dv, None);
                    1.0 - dist as f64 / lensum as f64
                }
            };
            let combined = bm25_weight * norm_bm25 + fuzzy_weight * fuzzy;
            (doc, combined)
        }).collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        Ok(scored)
    }

    /// Reciprocal Rank Fusion of BM25 + fuzzy ranks.
    /// More robust than linear interpolation — insensitive to score scale differences.
    /// k=60 is the standard RRF constant from (Cormack et al., 2009).
    #[pyo3(signature = (query, n=5, bm25_candidates=100, rrf_k=60))]
    pub fn get_top_n_rrf(
        &self,
        query: &str,
        n: usize,
        bm25_candidates: usize,
        rrf_k: usize,
    ) -> Vec<(String, f64)> {
        use crate::algorithms::indel_distance;

        let candidates_n = bm25_candidates.max(n * 10);

        // BM25 ranking
        let bm25_results = self.get_top_n(query, candidates_n);
        
        // If BM25 yields nothing (e.g. completely misspelled), we must rank the entire corpus
        // using fuzzy match, effectively treating the BM25 rank as tied at the bottom for all docs.
        let is_bm25_empty = bm25_results.is_empty();
        let target_docs: Vec<(usize, &String)> = if is_bm25_empty {
            self.corpus.iter().enumerate().collect()
        } else {
            bm25_results.iter().map(|(doc, _)| {
                let idx = self.corpus.iter().position(|d| d == doc).unwrap();
                (idx, doc)
            }).collect()
        };

        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        // Fuzzy ranking over the candidate set
        let mut fuzzy_ranked: Vec<(usize, f64)> = target_docs.iter().map(|(i, doc)| {
            let fuzzy = if q_is_ascii && doc.is_ascii() {
                let d_bytes = doc.as_bytes();
                let lensum = q_bytes.len() + d_bytes.len();
                if lensum == 0 { 1.0 } else {
                    let dist = indel_distance(q_bytes, d_bytes, None);
                    1.0 - dist as f64 / lensum as f64
                }
            } else {
                let qv: Vec<u32> = query.chars().map(|c| c as u32).collect();
                let dv: Vec<u32> = doc.chars().map(|c| c as u32).collect();
                let lensum = qv.len() + dv.len();
                if lensum == 0 { 1.0 } else {
                    let dist = indel_distance(&qv, &dv, None);
                    1.0 - dist as f64 / lensum as f64
                }
            };
            (*i, fuzzy)
        }).collect();
        fuzzy_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // RRF score per candidate index
        // Use a hashmap or a vec sized to corpus
        let mut rrf_scores: Vec<f64> = vec![0.0; self.corpus.len()];
        
        if !is_bm25_empty {
            for (rank, (doc, _)) in bm25_results.iter().enumerate() {
                let idx = self.corpus.iter().position(|d| d == doc).unwrap();
                rrf_scores[idx] += 1.0 / (rrf_k + rank + 1) as f64;
            }
        }
        
        for (rank, (idx, _)) in fuzzy_ranked.iter().enumerate() {
            rrf_scores[*idx] += 1.0 / (rrf_k + rank + 1) as f64;
        }

        // Collect, sort by RRF score
        let mut results: Vec<(String, f64)> = target_docs.into_iter()
            .map(|(idx, doc)| (doc.clone(), rrf_scores[idx]))
            .collect();
            
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(n);
        results
    }

    fn fuzzy_only(&self, query: &str, n: usize) -> PyResult<Vec<(String, f64)>> {
        use crate::algorithms::indel_distance;
        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();
        let mut scored: Vec<(String, f64)> = self.corpus.iter().map(|doc| {
            let sim = if q_is_ascii && doc.is_ascii() {
                let d = doc.as_bytes();
                let ls = q_bytes.len() + d.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(q_bytes, d, None) as f64 / ls as f64 }
            } else {
                let qv: Vec<u32> = query.chars().map(|c| c as u32).collect();
                let dv: Vec<u32> = doc.chars().map(|c| c as u32).collect();
                let ls = qv.len() + dv.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(&qv, &dv, None) as f64 / ls as f64 }
            };
            (doc.clone(), sim)
        }).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        Ok(scored)
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity matrix (for semantic hybrid search)
// Accepts two 2D arrays as flat Vec<f32> with shape info.
// Rows are unit-normalised embeddings. Result: dot product matrix.
// ---------------------------------------------------------------------------

/// Compute dot product (cosine sim on unit vectors) between two embedding matrices.
///
/// Parameters
/// ----------
/// a : list[list[float]]
///     Query embeddings, shape (nq, dim)
/// b : list[list[float]]
///     Corpus embeddings, shape (nc, dim)
///
/// Returns
/// -------
/// (flat_scores: list[float], nq: int, nc: int)
///     Row-major matrix of shape (nq, nc)
#[pyfunction]
pub fn cosine_similarity_matrix(
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
) -> (Vec<f32>, usize, usize) {
    let nq = a.len();
    let nc = b.len();
    if nq == 0 || nc == 0 {
        return (Vec::new(), nq, nc);
    }

    // Parallel over query rows
    let flat: Vec<f32> = a.par_iter().flat_map(|q_row| {
        b.iter().map(|c_row| {
            q_row.iter().zip(c_row.iter()).map(|(x, y)| x * y).sum::<f32>()
        }).collect::<Vec<f32>>()
    }).collect();

    (flat, nq, nc)
}
