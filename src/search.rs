// SPDX-License-Identifier: MIT
// BM25Okapi full-text search index + cosine similarity utilities.
// All heavy computation runs outside the GIL via Rayon.

use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::Value as JsonValue;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
struct ScoreItem {
    score: f64,
    idx: usize,
}
impl PartialEq for ScoreItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for ScoreItem {}
impl PartialOrd for ScoreItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for ScoreItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Tokeniser: simple whitespace split, lowercase, or charabia normalization
// ---------------------------------------------------------------------------
#[inline]
fn tokenise(text: &str, normalize: bool) -> Vec<String> {
    if normalize {
        use charabia::Tokenize;
        text.tokenize()
            .filter(|t| t.is_word())
            .map(|t| t.lemma().to_string())
            .collect()
    } else {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .collect()
    }
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
    /// Reverse lookup: document text → index
    corpus_index: FxHashMap<String, usize>,
    /// Tokenised corpus
    #[allow(dead_code)]
    tokenised: Vec<Vec<String>>,
    /// IDF per term
    idf: FxHashMap<String, f64>,
    /// Normalised TF per document: tf_norm[doc][term] = raw_tf * (k1+1) / (raw_tf + k1*(1-b+b*dl/avgdl))
    tf_norm: Vec<FxHashMap<String, f64>>,
    /// Token positions per document: positions[doc][term] = [pos0, pos1, ...]
    positions: Vec<FxHashMap<String, Vec<usize>>>,
    /// Average document length (in tokens)
    #[allow(dead_code)]
    avgdl: f64,
    #[allow(dead_code)]
    k1: f64,
    #[allow(dead_code)]
    b: f64,
    #[allow(dead_code)]
    normalize: bool,
}

#[pymethods]
impl BM25Index {
    #[new]
    #[pyo3(signature = (corpus, k1=1.5, b=0.75, normalize=false))]
    pub fn new(corpus: Vec<String>, k1: f64, b: f64, normalize: bool) -> Self {
        let n = corpus.len() as f64;
        let tokenised: Vec<Vec<String>> = corpus.iter().map(|d| tokenise(d, normalize)).collect();

        // Build reverse index: doc text → index
        let mut corpus_index: FxHashMap<String, usize> = FxHashMap::default();
        for (i, doc) in corpus.iter().enumerate() {
            corpus_index.insert(doc.clone(), i);
        }

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

        // Per-document normalised TF + positions
        let mut positions: Vec<FxHashMap<String, Vec<usize>>> = Vec::with_capacity(tokenised.len());
        let tf_norm: Vec<FxHashMap<String, f64>> = tokenised.iter().map(|doc| {
            let dl = doc.len() as f64;
            let mut raw_tf: FxHashMap<String, usize> = FxHashMap::default();
            let mut doc_positions: FxHashMap<String, Vec<usize>> = FxHashMap::default();
            for (pos, term) in doc.iter().enumerate() {
                *raw_tf.entry(term.clone()).or_insert(0) += 1;
                doc_positions.entry(term.clone()).or_default().push(pos);
            }
            positions.push(doc_positions);
            raw_tf.into_iter().map(|(term, tf)| {
                let tf_f = tf as f64;
                let norm = tf_f * (k1 + 1.0) / (tf_f + k1 * (1.0 - b + b * dl / avgdl));
                (term, norm)
            }).collect()
        }).collect();

        BM25Index { corpus, corpus_index, tokenised, idf, tf_norm, positions, avgdl, k1, b, normalize }
    }

    /// Pickle support: serialise as (corpus, k1, b, normalize), rebuild on unpickle.
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
            pyo3::types::PyBool::new(py, slf.normalize).as_any().clone().unbind(),
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
        let q_terms = tokenise(query, self.normalize);
        let n = self.corpus.len();
        let tf_norm = &self.tf_norm;
        let idf = &self.idf;

        if n >= 10_000 {
            (0..n).into_par_iter().map(|i| {
                let doc_tf = &tf_norm[i];
                q_terms.iter().map(|term| {
                    let idf_val = idf.get(term).copied().unwrap_or(0.0);
                    let tf_val = doc_tf.get(term).copied().unwrap_or(0.0);
                    idf_val * tf_val
                }).sum::<f64>()
            }).collect()
        } else {
            (0..n).map(|i| {
                let doc_tf = &tf_norm[i];
                q_terms.iter().map(|term| {
                    let idf_val = idf.get(term).copied().unwrap_or(0.0);
                    let tf_val = doc_tf.get(term).copied().unwrap_or(0.0);
                    idf_val * tf_val
                }).sum::<f64>()
            }).collect()
        }
    }

    /// Return BM25 scores with an optional boolean mask.
    /// Documents where `allowed[i]` is false get a score of 0.0.
    #[pyo3(signature = (query, allowed=None))]
    pub fn get_scores_filtered(&self, query: &str, allowed: Option<Vec<bool>>) -> Vec<f64> {
        let mut scores = self.get_scores(query);
        if let Some(ref mask) = allowed {
            for (i, s) in scores.iter_mut().enumerate() {
                if i < mask.len() && !mask[i] {
                    *s = 0.0;
                }
            }
        }
        scores
    }

    /// Return top-N `(document, score)` pairs for `query`.
    #[pyo3(signature = (query, n=5))]
    pub fn get_top_n(&self, query: &str, n: usize) -> Vec<(String, f64)> {
        self.get_top_n_filtered(query, n, None)
    }

    /// Return top-N with an optional allowed mask.
    #[pyo3(signature = (query, n=5, allowed=None))]
    pub fn get_top_n_filtered(&self, query: &str, n: usize, allowed: Option<Vec<bool>>) -> Vec<(String, f64)> {
        let scores = self.get_scores_filtered(query, allowed);
        
        let valid_count = scores.len();
        if valid_count <= 256 || n >= valid_count {
            let mut items: Vec<(usize, f64)> = scores.into_iter().enumerate().filter(|(_, s)| *s > 0.0).collect();
            items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(n);
            return items.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect();
        }

        let mut heap: BinaryHeap<Reverse<ScoreItem>> = BinaryHeap::with_capacity(n.min(100));
        for (idx, score) in scores.into_iter().enumerate() {
            if score > 0.0 {
                if heap.len() < n {
                    heap.push(Reverse(ScoreItem { score, idx }));
                } else if let Some(mut peek) = heap.peek_mut() {
                    if score > peek.0.score {
                        peek.0 = ScoreItem { score, idx };
                    }
                }
            }
        }

        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect()
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

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
        self.get_top_n_rrf_filtered(query, n, bm25_candidates, rrf_k, None)
    }

    /// RRF with optional allowed mask.
    #[pyo3(signature = (query, n=5, bm25_candidates=100, rrf_k=60, allowed=None))]
    pub fn get_top_n_rrf_filtered(
        &self,
        query: &str,
        n: usize,
        bm25_candidates: usize,
        rrf_k: usize,
        allowed: Option<Vec<bool>>,
    ) -> Vec<(String, f64)> {
        use crate::algorithms::indel_distance;

        let candidates_n = bm25_candidates.max(n * 10);

        // BM25 ranking (with filter)
        let bm25_results = self.get_top_n_filtered(query, candidates_n, allowed.clone());
        
        let is_bm25_empty = bm25_results.is_empty();
        let target_docs: Vec<(usize, &String)> = if is_bm25_empty {
            // Fall back to all allowed docs
            self.corpus.iter().enumerate()
                .filter(|(i, _)| allowed.as_ref().is_none_or(|m| i < &m.len() && m[*i]))
                .collect()
        } else {
            bm25_results.iter().map(|(doc, _)| {
                let idx = *self.corpus_index.get(doc).unwrap();
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
        fuzzy_ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut rrf_scores: Vec<f64> = vec![0.0; self.corpus.len()];
        
        if !is_bm25_empty {
            for (rank, (doc, _)) in bm25_results.iter().enumerate() {
                let idx = *self.corpus_index.get(doc).unwrap();
                rrf_scores[idx] += 1.0 / (rrf_k + rank + 1) as f64;
            }
        }
        
        for (rank, (idx, _)) in fuzzy_ranked.iter().enumerate() {
            rrf_scores[*idx] += 1.0 / (rrf_k + rank + 1) as f64;
        }

        let mut results: Vec<(String, f64)> = target_docs.into_iter()
            .map(|(idx, doc)| (doc.clone(), rrf_scores[idx]))
            .collect();
            
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(n);
        results
    }

    fn fuzzy_only(&self, query: &str, n: usize) -> PyResult<Vec<(String, f64)>> {
        use crate::algorithms::indel_distance;
        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        let scores: Vec<f64> = self.corpus.iter().map(|doc| {
            if q_is_ascii && doc.is_ascii() {
                let d = doc.as_bytes();
                let ls = q_bytes.len() + d.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(q_bytes, d, None) as f64 / ls as f64 }
            } else {
                let qv: Vec<u32> = query.chars().map(|c| c as u32).collect();
                let dv: Vec<u32> = doc.chars().map(|c| c as u32).collect();
                let ls = qv.len() + dv.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(&qv, &dv, None) as f64 / ls as f64 }
            }
        }).collect();

        let corpus_len = self.corpus.len();
        if corpus_len <= 256 || n >= corpus_len {
            let mut items: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
            items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(n);
            return Ok(items.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect());
        }

        let mut heap: BinaryHeap<Reverse<ScoreItem>> = BinaryHeap::with_capacity(n.min(100));
        for (idx, score) in scores.into_iter().enumerate() {
            if heap.len() < n {
                heap.push(Reverse(ScoreItem { score, idx }));
            } else if let Some(mut peek) = heap.peek_mut() {
                if score > peek.0.score {
                    peek.0 = ScoreItem { score, idx };
                }
            }
        }

        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect())
    }

    // ── New methods ──────────────────────────────────────────────

    /// Return the IDF map: term → idf value.
    pub fn get_idf_map(&self) -> FxHashMap<String, f64> {
        self.idf.clone()
    }

    /// Return the TF-norm vector for a specific document.
    #[pyo3(signature = (doc_idx))]
    pub fn get_document_vector(&self, doc_idx: usize) -> PyResult<FxHashMap<String, f64>> {
        if doc_idx >= self.corpus.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("doc_idx {} out of range (corpus has {} docs)", doc_idx, self.corpus.len())
            ));
        }
        Ok(self.tf_norm[doc_idx].clone())
    }

    /// Per-term score breakdown for a query against a specific document.
    /// Returns Vec<(term, idf, tf_norm, term_score)>.
    #[pyo3(signature = (query, doc_idx))]
    pub fn explain(&self, query: &str, doc_idx: usize) -> PyResult<Vec<(String, f64, f64, f64)>> {
        if doc_idx >= self.corpus.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("doc_idx {} out of range (corpus has {} docs)", doc_idx, self.corpus.len())
            ));
        }
        let q_terms = tokenise(query, self.normalize);
        let doc_tf = &self.tf_norm[doc_idx];
        let mut breakdown = Vec::new();
        for term in q_terms {
            let idf_val = self.idf.get(&term).copied().unwrap_or(0.0);
            let tf_val = doc_tf.get(&term).copied().unwrap_or(0.0);
            let score = idf_val * tf_val;
            breakdown.push((term, idf_val, tf_val, score));
        }
        Ok(breakdown)
    }

    /// BM25 scoring with phrase proximity boost.
    /// Documents where query terms appear adjacent or within `window` positions
    /// get a multiplicative boost.
    #[pyo3(signature = (query, n=5, proximity_window=3, phrase_boost=2.0))]
    pub fn get_top_n_phrase(
        &self,
        query: &str,
        n: usize,
        proximity_window: usize,
        phrase_boost: f64,
    ) -> Vec<(String, f64)> {
        let bm25_scores = self.get_scores(query);
        let q_terms = tokenise(query, self.normalize);

        let mut scored: Vec<(usize, f64)> = bm25_scores.into_iter().enumerate()
            .filter(|(_, s)| *s > 0.0)
            .map(|(i, bm25)| {
                let boost = phrase_proximity_boost(&q_terms, &self.positions[i], proximity_window);
                let final_score = bm25 * (1.0 + (phrase_boost - 1.0) * boost);
                (i, final_score)
            })
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect()
    }
}

// ---------------------------------------------------------------------------
// Phrase proximity boost helper
// ---------------------------------------------------------------------------

/// Compute a phrase proximity boost in [0.0, 1.0].
/// Returns 1.0 if all consecutive query term pairs appear within `window`
/// positions of each other. Returns 0.0 if no pair is within range.
fn phrase_proximity_boost(
    q_terms: &[String],
    doc_positions: &FxHashMap<String, Vec<usize>>,
    window: usize,
) -> f64 {
    if q_terms.len() <= 1 {
        return if q_terms.len() == 1 && doc_positions.contains_key(&q_terms[0]) { 1.0 } else { 0.0 };
    }
    let mut hits = 0usize;
    let pairs = q_terms.len() - 1;
    for pair in q_terms.windows(2) {
        let (t1, t2) = (&pair[0], &pair[1]);
        if let (Some(p1), Some(p2)) = (doc_positions.get(t1), doc_positions.get(t2)) {
            // Check if any position pair is within window
            'outer: for &a in p1 {
                for &b in p2 {
                    let dist = a.abs_diff(b);
                    if dist <= window {
                        hits += 1;
                        break 'outer;
                    }
                }
            }
        }
    }
    hits as f64 / pairs as f64
}

// ============================================================================
// BM25L
// ============================================================================

#[pyclass]
pub struct BM25L {
    corpus: Vec<String>,
    corpus_index: FxHashMap<String, usize>,
    #[allow(dead_code)]
    tokenised: Vec<Vec<String>>,
    idf: FxHashMap<String, f64>,
    tf_norm: Vec<FxHashMap<String, f64>>,
    positions: Vec<FxHashMap<String, Vec<usize>>>,
    #[allow(dead_code)]
    avgdl: f64,
    #[allow(dead_code)]
    k1: f64,
    #[allow(dead_code)]
    b: f64,
    #[allow(dead_code)]
    delta: f64,
    #[allow(dead_code)]
    normalize: bool,
}

#[pymethods]
impl BM25L {
    #[new]
    #[pyo3(signature = (corpus, k1=1.5, b=0.75, delta=0.5, normalize=false))]
    pub fn new(corpus: Vec<String>, k1: f64, b: f64, delta: f64, normalize: bool) -> Self {
        let n = corpus.len() as f64;
        let tokenised: Vec<Vec<String>> = corpus.iter().map(|d| tokenise(d, normalize)).collect();

        let mut corpus_index: FxHashMap<String, usize> = FxHashMap::default();
        for (i, doc) in corpus.iter().enumerate() {
            corpus_index.insert(doc.clone(), i);
        }

        let total_tokens: usize = tokenised.iter().map(|d| d.len()).sum();
        let avgdl = if n > 0.0 { total_tokens as f64 / n } else { 1.0 };

        let mut df: FxHashMap<String, usize> = FxHashMap::default();
        for doc in &tokenised {
            let mut seen: FxHashMap<&str, bool> = FxHashMap::default();
            for term in doc {
                if seen.insert(term.as_str(), true).is_none() {
                    *df.entry(term.clone()).or_insert(0) += 1;
                }
            }
        }

        let idf: FxHashMap<String, f64> = df.iter().map(|(term, &dft)| {
            let dft_f = dft as f64;
            let idf_val = ((n + 1.0) / (dft_f + 0.5)).ln();
            (term.clone(), idf_val)
        }).collect();

        let mut positions: Vec<FxHashMap<String, Vec<usize>>> = Vec::with_capacity(tokenised.len());
        let tf_norm: Vec<FxHashMap<String, f64>> = tokenised.iter().map(|doc| {
            let dl = doc.len() as f64;
            let mut raw_tf: FxHashMap<String, usize> = FxHashMap::default();
            let mut doc_positions: FxHashMap<String, Vec<usize>> = FxHashMap::default();
            for (pos, term) in doc.iter().enumerate() {
                *raw_tf.entry(term.clone()).or_insert(0) += 1;
                doc_positions.entry(term.clone()).or_default().push(pos);
            }
            positions.push(doc_positions);
            raw_tf.into_iter().map(|(term, tf)| {
                let tf_f = tf as f64;
                let ctd = tf_f / (1.0 - b + b * dl / avgdl);
                let norm = (k1 + 1.0) * (ctd + delta) / (k1 + ctd + delta);
                (term, norm)
            }).collect()
        }).collect();

        BM25L { corpus, corpus_index, tokenised, idf, tf_norm, positions, avgdl, k1, b, delta, normalize }
    }

    #[getter]
    pub fn num_docs(&self) -> usize {
        self.corpus.len()
    }

    // `get_scores` and `get_top_n` have the exact same logic structure as BM25Okapi, since the formulas were pre-computed into `tf_norm`
    pub fn get_scores(&self, query: &str) -> Vec<f64> {
        let q_terms = tokenise(query, self.normalize);
        let n = self.corpus.len();
        let tf_norm = &self.tf_norm;
        let idf = &self.idf;

        if n >= 10_000 {
            (0..n).into_par_iter().map(|i| {
                let doc_tf = &tf_norm[i];
                q_terms.iter().map(|term| {
                    let idf_val = idf.get(term).copied().unwrap_or(0.0);
                    let tf_val = doc_tf.get(term).copied().unwrap_or(0.0);
                    idf_val * tf_val
                }).sum::<f64>()
            }).collect()
        } else {
            (0..n).map(|i| {
                let doc_tf = &tf_norm[i];
                q_terms.iter().map(|term| {
                    let idf_val = idf.get(term).copied().unwrap_or(0.0);
                    let tf_val = doc_tf.get(term).copied().unwrap_or(0.0);
                    idf_val * tf_val
                }).sum::<f64>()
            }).collect()
        }
    }

    #[pyo3(signature = (query, allowed=None))]
    pub fn get_scores_filtered(&self, query: &str, allowed: Option<Vec<bool>>) -> Vec<f64> {
        let mut scores = self.get_scores(query);
        if let Some(ref mask) = allowed {
            for (i, s) in scores.iter_mut().enumerate() {
                if i < mask.len() && !mask[i] { *s = 0.0; }
            }
        }
        scores
    }

    #[pyo3(signature = (query, n=5))]
    pub fn get_top_n(&self, query: &str, n: usize) -> Vec<(String, f64)> {
        self.get_top_n_filtered(query, n, None)
    }

    #[pyo3(signature = (query, n=5, allowed=None))]
    pub fn get_top_n_filtered(&self, query: &str, n: usize, allowed: Option<Vec<bool>>) -> Vec<(String, f64)> {
        let scores = self.get_scores_filtered(query, allowed);
        
        let valid_count = scores.len();
        if valid_count <= 256 || n >= valid_count {
            let mut items: Vec<(usize, f64)> = scores.into_iter().enumerate().filter(|(_, s)| *s > 0.0).collect();
            items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(n);
            return items.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect();
        }

        let mut heap: BinaryHeap<Reverse<ScoreItem>> = BinaryHeap::with_capacity(n.min(100));
        for (idx, score) in scores.into_iter().enumerate() {
            if score > 0.0 {
                if heap.len() < n {
                    heap.push(Reverse(ScoreItem { score, idx }));
                } else if let Some(mut peek) = heap.peek_mut() {
                    if score > peek.0.score {
                        peek.0 = ScoreItem { score, idx };
                    }
                }
            }
        }

        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect()
    }

    /// BM25 scores for a batch of queries — Rayon-parallel over queries.
    pub fn get_batch_scores(&self, queries: Vec<String>) -> Vec<Vec<f64>> {
        queries.par_iter().map(|q| self.get_scores(q)).collect()
    }

    /// BM25 + fuzzy hybrid: run BM25, take top `bm25_candidates`, then re-rank
    /// with indel fuzzy similarity.
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
            return self.fuzzy_only(query, n);
        }

        let max_bm25 = candidates.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
        let min_bm25 = candidates.iter().map(|(_, s)| *s).fold(f64::MAX, f64::min);
        let bm25_range = (max_bm25 - min_bm25).max(1e-10);

        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        let mut scored: Vec<(String, f64)> = candidates.into_iter().map(|(doc, bm25_raw)| {
            let norm_bm25 = (bm25_raw - min_bm25) / bm25_range;
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
            let combined = bm25_weight * norm_bm25 + fuzzy_weight * fuzzy;
            (doc, combined)
        }).collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        Ok(scored)
    }

    /// Reciprocal Rank Fusion of BM25 + fuzzy ranks.
    #[pyo3(signature = (query, n=5, bm25_candidates=100, rrf_k=60))]
    pub fn get_top_n_rrf(&self, query: &str, n: usize, bm25_candidates: usize, rrf_k: usize) -> Vec<(String, f64)> {
        self.get_top_n_rrf_filtered(query, n, bm25_candidates, rrf_k, None)
    }

    #[pyo3(signature = (query, n=5, bm25_candidates=100, rrf_k=60, allowed=None))]
    pub fn get_top_n_rrf_filtered(&self, query: &str, n: usize, bm25_candidates: usize, rrf_k: usize, allowed: Option<Vec<bool>>) -> Vec<(String, f64)> {
        use crate::algorithms::indel_distance;

        let candidates_n = bm25_candidates.max(n * 10);
        let bm25_results = self.get_top_n_filtered(query, candidates_n, allowed.clone());
        
        let is_bm25_empty = bm25_results.is_empty();
        let target_docs: Vec<(usize, &String)> = if is_bm25_empty {
            self.corpus.iter().enumerate()
                .filter(|(i, _)| allowed.as_ref().is_none_or(|m| i < &m.len() && m[*i]))
                .collect()
        } else {
            bm25_results.iter().map(|(doc, _)| {
                let idx = *self.corpus_index.get(doc).unwrap();
                (idx, doc)
            }).collect()
        };

        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

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
        fuzzy_ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut rrf_scores: Vec<f64> = vec![0.0; self.corpus.len()];
        
        if !is_bm25_empty {
            for (rank, (doc, _)) in bm25_results.iter().enumerate() {
                let idx = *self.corpus_index.get(doc).unwrap();
                rrf_scores[idx] += 1.0 / (rrf_k + rank + 1) as f64;
            }
        }
        
        for (rank, (idx, _)) in fuzzy_ranked.iter().enumerate() {
            rrf_scores[*idx] += 1.0 / (rrf_k + rank + 1) as f64;
        }

        let mut results: Vec<(String, f64)> = target_docs.into_iter()
            .map(|(idx, doc)| (doc.clone(), rrf_scores[idx]))
            .collect();
            
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(n);
        results
    }

    fn fuzzy_only(&self, query: &str, n: usize) -> PyResult<Vec<(String, f64)>> {
        use crate::algorithms::indel_distance;
        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        let scores: Vec<f64> = self.corpus.iter().map(|doc| {
            if q_is_ascii && doc.is_ascii() {
                let d = doc.as_bytes();
                let ls = q_bytes.len() + d.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(q_bytes, d, None) as f64 / ls as f64 }
            } else {
                let qv: Vec<u32> = query.chars().map(|c| c as u32).collect();
                let dv: Vec<u32> = doc.chars().map(|c| c as u32).collect();
                let ls = qv.len() + dv.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(&qv, &dv, None) as f64 / ls as f64 }
            }
        }).collect();

        let corpus_len = self.corpus.len();
        if corpus_len <= 256 || n >= corpus_len {
            let mut items: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
            items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(n);
            return Ok(items.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect());
        }

        let mut heap: BinaryHeap<Reverse<ScoreItem>> = BinaryHeap::with_capacity(n.min(100));
        for (idx, score) in scores.into_iter().enumerate() {
            if heap.len() < n {
                heap.push(Reverse(ScoreItem { score, idx }));
            } else if let Some(mut peek) = heap.peek_mut() {
                if score > peek.0.score {
                    peek.0 = ScoreItem { score, idx };
                }
            }
        }

        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect())
    }

    pub fn get_idf_map(&self) -> FxHashMap<String, f64> {
        self.idf.clone()
    }

    #[pyo3(signature = (doc_idx))]
    pub fn get_document_vector(&self, doc_idx: usize) -> PyResult<FxHashMap<String, f64>> {
        if doc_idx >= self.corpus.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("doc_idx {} out of range (corpus has {} docs)", doc_idx, self.corpus.len())
            ));
        }
        Ok(self.tf_norm[doc_idx].clone())
    }

    #[pyo3(signature = (query, doc_idx))]
    pub fn explain(&self, query: &str, doc_idx: usize) -> PyResult<Vec<(String, f64, f64, f64)>> {
        if doc_idx >= self.corpus.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("doc_idx {} out of range (corpus has {} docs)", doc_idx, self.corpus.len())
            ));
        }
        let q_terms = tokenise(query, self.normalize);
        let doc_tf = &self.tf_norm[doc_idx];
        let mut breakdown = Vec::new();
        for term in q_terms {
            let idf_val = self.idf.get(&term).copied().unwrap_or(0.0);
            let tf_val = doc_tf.get(&term).copied().unwrap_or(0.0);
            let score = idf_val * tf_val;
            breakdown.push((term, idf_val, tf_val, score));
        }
        Ok(breakdown)
    }

    #[pyo3(signature = (query, n=5, proximity_window=3, phrase_boost=2.0))]
    pub fn get_top_n_phrase(&self, query: &str, n: usize, proximity_window: usize, phrase_boost: f64) -> Vec<(String, f64)> {
        let bm25_scores = self.get_scores(query);
        let q_terms = tokenise(query, self.normalize);
        let mut scored: Vec<(usize, f64)> = bm25_scores.into_iter().enumerate()
            .filter(|(_, s)| *s > 0.0)
            .map(|(i, bm25)| {
                let boost = phrase_proximity_boost(&q_terms, &self.positions[i], proximity_window);
                let final_score = bm25 * (1.0 + (phrase_boost - 1.0) * boost);
                (i, final_score)
            }).collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect()
    }
}

// ============================================================================
// BM25+
// ============================================================================

#[pyclass]
pub struct BM25Plus {
    corpus: Vec<String>,
    corpus_index: FxHashMap<String, usize>,
    #[allow(dead_code)]
    tokenised: Vec<Vec<String>>,
    idf: FxHashMap<String, f64>,
    tf_norm: Vec<FxHashMap<String, f64>>,
    positions: Vec<FxHashMap<String, Vec<usize>>>,
    #[allow(dead_code)]
    avgdl: f64,
    #[allow(dead_code)]
    k1: f64,
    #[allow(dead_code)]
    b: f64,
    #[allow(dead_code)]
    delta: f64,
    #[allow(dead_code)]
    normalize: bool,
}

#[pymethods]
impl BM25Plus {
    #[new]
    #[pyo3(signature = (corpus, k1=1.5, b=0.75, delta=1.0, normalize=false))]
    pub fn new(corpus: Vec<String>, k1: f64, b: f64, delta: f64, normalize: bool) -> Self {
        let n = corpus.len() as f64;
        let tokenised: Vec<Vec<String>> = corpus.iter().map(|d| tokenise(d, normalize)).collect();

        let total_tokens: usize = tokenised.iter().map(|d| d.len()).sum();
        let avgdl = if n > 0.0 { total_tokens as f64 / n } else { 1.0 };

        let mut df: FxHashMap<String, usize> = FxHashMap::default();
        for doc in &tokenised {
            let mut seen: FxHashMap<&str, bool> = FxHashMap::default();
            for term in doc {
                if seen.insert(term.as_str(), true).is_none() {
                    *df.entry(term.clone()).or_insert(0) += 1;
                }
            }
        }

        // IDF: ln((N + 1) / df)
        let idf: FxHashMap<String, f64> = df.iter().map(|(term, &dft)| {
            let dft_f = dft as f64;
            let idf_val = ((n + 1.0) / dft_f).ln();
            (term.clone(), idf_val)
        }).collect();

        let mut corpus_index: FxHashMap<String, usize> = FxHashMap::default();
        for (i, doc) in corpus.iter().enumerate() {
            corpus_index.insert(doc.clone(), i);
        }

        let mut positions: Vec<FxHashMap<String, Vec<usize>>> = Vec::with_capacity(tokenised.len());
        let tf_norm: Vec<FxHashMap<String, f64>> = tokenised.iter().map(|doc| {
            let dl = doc.len() as f64;
            let mut raw_tf: FxHashMap<String, usize> = FxHashMap::default();
            let mut doc_positions: FxHashMap<String, Vec<usize>> = FxHashMap::default();
            for (pos, term) in doc.iter().enumerate() {
                *raw_tf.entry(term.clone()).or_insert(0) += 1;
                doc_positions.entry(term.clone()).or_default().push(pos);
            }
            positions.push(doc_positions);
            raw_tf.into_iter().map(|(term, tf)| {
                let tf_f = tf as f64;
                let norm = delta + (tf_f * (k1 + 1.0)) / (k1 * (1.0 - b + b * dl / avgdl) + tf_f);
                (term, norm)
            }).collect()
        }).collect();

        BM25Plus { corpus, corpus_index, tokenised, idf, tf_norm, positions, avgdl, k1, b, delta, normalize }
    }

    #[getter]
    pub fn num_docs(&self) -> usize {
        self.corpus.len()
    }

    pub fn get_scores(&self, query: &str) -> Vec<f64> {
        let q_terms = tokenise(query, self.normalize);
        let n = self.corpus.len();
        let tf_norm = &self.tf_norm;
        let idf = &self.idf;

        if n >= 10_000 {
            (0..n).into_par_iter().map(|i| {
                let doc_tf = &tf_norm[i];
                q_terms.iter().map(|term| {
                    let idf_val = idf.get(term).copied().unwrap_or(0.0);
                    let tf_val = doc_tf.get(term).copied().unwrap_or(0.0);
                    idf_val * tf_val
                }).sum::<f64>()
            }).collect()
        } else {
            (0..n).map(|i| {
                let doc_tf = &tf_norm[i];
                q_terms.iter().map(|term| {
                    let idf_val = idf.get(term).copied().unwrap_or(0.0);
                    let tf_val = doc_tf.get(term).copied().unwrap_or(0.0);
                    idf_val * tf_val
                }).sum::<f64>()
            }).collect()
        }
    }

    #[pyo3(signature = (query, allowed=None))]
    pub fn get_scores_filtered(&self, query: &str, allowed: Option<Vec<bool>>) -> Vec<f64> {
        let mut scores = self.get_scores(query);
        if let Some(ref mask) = allowed {
            for (i, s) in scores.iter_mut().enumerate() {
                if i < mask.len() && !mask[i] { *s = 0.0; }
            }
        }
        scores
    }

    #[pyo3(signature = (query, n=5))]
    pub fn get_top_n(&self, query: &str, n: usize) -> Vec<(String, f64)> {
        self.get_top_n_filtered(query, n, None)
    }

    #[pyo3(signature = (query, n=5, allowed=None))]
    pub fn get_top_n_filtered(&self, query: &str, n: usize, allowed: Option<Vec<bool>>) -> Vec<(String, f64)> {
        let scores = self.get_scores_filtered(query, allowed);
        let valid_count = scores.len();
        if valid_count <= 256 || n >= valid_count {
            let mut items: Vec<(usize, f64)> = scores.into_iter().enumerate().filter(|(_, s)| *s > 0.0).collect();
            items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(n);
            return items.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect();
        }

        let mut heap: BinaryHeap<Reverse<ScoreItem>> = BinaryHeap::with_capacity(n.min(100));
        for (idx, score) in scores.into_iter().enumerate() {
            if score > 0.0 {
                if heap.len() < n {
                    heap.push(Reverse(ScoreItem { score, idx }));
                } else if let Some(mut peek) = heap.peek_mut() {
                    if score > peek.0.score {
                        peek.0 = ScoreItem { score, idx };
                    }
                }
            }
        }
        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect()
    }

    /// BM25 scores for a batch of queries — Rayon-parallel over queries.
    pub fn get_batch_scores(&self, queries: Vec<String>) -> Vec<Vec<f64>> {
        queries.par_iter().map(|q| self.get_scores(q)).collect()
    }

    /// BM25 + fuzzy hybrid: run BM25, take top `bm25_candidates`, then re-rank
    /// with indel fuzzy similarity.
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
            return self.fuzzy_only(query, n);
        }

        let max_bm25 = candidates.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
        let min_bm25 = candidates.iter().map(|(_, s)| *s).fold(f64::MAX, f64::min);
        let bm25_range = (max_bm25 - min_bm25).max(1e-10);

        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        let mut scored: Vec<(String, f64)> = candidates.into_iter().map(|(doc, bm25_raw)| {
            let norm_bm25 = (bm25_raw - min_bm25) / bm25_range;
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
            let combined = bm25_weight * norm_bm25 + fuzzy_weight * fuzzy;
            (doc, combined)
        }).collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        Ok(scored)
    }

    #[pyo3(signature = (query, n=5, bm25_candidates=100, rrf_k=60))]
    pub fn get_top_n_rrf(&self, query: &str, n: usize, bm25_candidates: usize, rrf_k: usize) -> Vec<(String, f64)> {
        self.get_top_n_rrf_filtered(query, n, bm25_candidates, rrf_k, None)
    }

    #[pyo3(signature = (query, n=5, bm25_candidates=100, rrf_k=60, allowed=None))]
    pub fn get_top_n_rrf_filtered(&self, query: &str, n: usize, bm25_candidates: usize, rrf_k: usize, allowed: Option<Vec<bool>>) -> Vec<(String, f64)> {
        use crate::algorithms::indel_distance;

        let candidates_n = bm25_candidates.max(n * 10);
        let bm25_results = self.get_top_n_filtered(query, candidates_n, allowed.clone());
        
        let is_bm25_empty = bm25_results.is_empty();
        let target_docs: Vec<(usize, &String)> = if is_bm25_empty {
            self.corpus.iter().enumerate()
                .filter(|(i, _)| allowed.as_ref().is_none_or(|m| i < &m.len() && m[*i]))
                .collect()
        } else {
            bm25_results.iter().map(|(doc, _)| {
                let idx = *self.corpus_index.get(doc).unwrap();
                (idx, doc)
            }).collect()
        };

        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

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
        fuzzy_ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut rrf_scores: Vec<f64> = vec![0.0; self.corpus.len()];
        
        if !is_bm25_empty {
            for (rank, (doc, _)) in bm25_results.iter().enumerate() {
                let idx = *self.corpus_index.get(doc).unwrap();
                rrf_scores[idx] += 1.0 / (rrf_k + rank + 1) as f64;
            }
        }
        
        for (rank, (idx, _)) in fuzzy_ranked.iter().enumerate() {
            rrf_scores[*idx] += 1.0 / (rrf_k + rank + 1) as f64;
        }

        let mut results: Vec<(String, f64)> = target_docs.into_iter()
            .map(|(idx, doc)| (doc.clone(), rrf_scores[idx]))
            .collect();
            
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(n);
        results
    }

    fn fuzzy_only(&self, query: &str, n: usize) -> PyResult<Vec<(String, f64)>> {
        use crate::algorithms::indel_distance;
        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        let scores: Vec<f64> = self.corpus.iter().map(|doc| {
            if q_is_ascii && doc.is_ascii() {
                let d = doc.as_bytes();
                let ls = q_bytes.len() + d.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(q_bytes, d, None) as f64 / ls as f64 }
            } else {
                let qv: Vec<u32> = query.chars().map(|c| c as u32).collect();
                let dv: Vec<u32> = doc.chars().map(|c| c as u32).collect();
                let ls = qv.len() + dv.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(&qv, &dv, None) as f64 / ls as f64 }
            }
        }).collect();

        let corpus_len = self.corpus.len();
        if corpus_len <= 256 || n >= corpus_len {
            let mut items: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
            items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(n);
            return Ok(items.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect());
        }

        let mut heap: BinaryHeap<Reverse<ScoreItem>> = BinaryHeap::with_capacity(n.min(100));
        for (idx, score) in scores.into_iter().enumerate() {
            if heap.len() < n {
                heap.push(Reverse(ScoreItem { score, idx }));
            } else if let Some(mut peek) = heap.peek_mut() {
                if score > peek.0.score {
                    peek.0 = ScoreItem { score, idx };
                }
            }
        }

        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect())
    }

    pub fn get_idf_map(&self) -> FxHashMap<String, f64> {
        self.idf.clone()
    }

    #[pyo3(signature = (doc_idx))]
    pub fn get_document_vector(&self, doc_idx: usize) -> PyResult<FxHashMap<String, f64>> {
        if doc_idx >= self.corpus.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("doc_idx {} out of range (corpus has {} docs)", doc_idx, self.corpus.len())
            ));
        }
        Ok(self.tf_norm[doc_idx].clone())
    }

    #[pyo3(signature = (query, doc_idx))]
    pub fn explain(&self, query: &str, doc_idx: usize) -> PyResult<Vec<(String, f64, f64, f64)>> {
        if doc_idx >= self.corpus.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("doc_idx {} out of range (corpus has {} docs)", doc_idx, self.corpus.len())
            ));
        }
        let q_terms = tokenise(query, self.normalize);
        let doc_tf = &self.tf_norm[doc_idx];
        let mut breakdown = Vec::new();
        for term in q_terms {
            let idf_val = self.idf.get(&term).copied().unwrap_or(0.0);
            let tf_val = doc_tf.get(&term).copied().unwrap_or(0.0);
            let score = idf_val * tf_val;
            breakdown.push((term, idf_val, tf_val, score));
        }
        Ok(breakdown)
    }

    #[pyo3(signature = (query, n=5, proximity_window=3, phrase_boost=2.0))]
    pub fn get_top_n_phrase(&self, query: &str, n: usize, proximity_window: usize, phrase_boost: f64) -> Vec<(String, f64)> {
        let bm25_scores = self.get_scores(query);
        let q_terms = tokenise(query, self.normalize);
        let mut scored: Vec<(usize, f64)> = bm25_scores.into_iter().enumerate()
            .filter(|(_, s)| *s > 0.0)
            .map(|(i, bm25)| {
                let boost = phrase_proximity_boost(&q_terms, &self.positions[i], proximity_window);
                let final_score = bm25 * (1.0 + (phrase_boost - 1.0) * boost);
                (i, final_score)
            }).collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect()
    }
}
// ============================================================================

#[pyclass]
pub struct BM25T {
    corpus: Vec<String>,
    corpus_index: FxHashMap<String, usize>,
    #[allow(dead_code)]
    tokenised: Vec<Vec<String>>,
    idf: FxHashMap<String, f64>,
    tf_norm: Vec<FxHashMap<String, f64>>,
    positions: Vec<FxHashMap<String, Vec<usize>>>,
    #[allow(dead_code)]
    avgdl: f64,
    #[allow(dead_code)]
    k1: f64,
    #[allow(dead_code)]
    b: f64,
    #[allow(dead_code)]
    normalize: bool,
}

#[pymethods]
impl BM25T {
    #[new]
    #[pyo3(signature = (corpus, k1=1.5, b=0.75, normalize=false))]
    pub fn new(corpus: Vec<String>, k1: f64, b: f64, normalize: bool) -> Self {
        let n = corpus.len() as f64;
        let tokenised: Vec<Vec<String>> = corpus.iter().map(|d| tokenise(d, normalize)).collect();

        let total_tokens: usize = tokenised.iter().map(|d| d.len()).sum();
        let avgdl = if n > 0.0 { total_tokens as f64 / n } else { 1.0 };

        let mut df: FxHashMap<String, usize> = FxHashMap::default();
        let mut term_postings: FxHashMap<String, Vec<(usize, usize)>> = FxHashMap::default(); // term -> list of (doc_idx, tf)

        for (i, doc) in tokenised.iter().enumerate() {
            let mut raw_tf: FxHashMap<&str, usize> = FxHashMap::default();
            for term in doc {
                *raw_tf.entry(term.as_str()).or_insert(0) += 1;
            }
            for (term, tf) in raw_tf {
                *df.entry(term.to_string()).or_insert(0) += 1;
                term_postings.entry(term.to_string()).or_default().push((i, tf));
            }
        }

        // IDF: ln((N + 1) / df)
        // Wait, rank_bm25 uses ln((N + 1) / df) for BM25+ but rank_bm25 PR #9 for BM25T uses original Okapi IDF: ln((N - df + 0.5) / (df + 0.5) + 1).
        // Let's use standard Okapi IDF here to match rank_bm25 base implementation exactly if possible, or keep the existing BM25T PR idf:
        // Actually, PR #9 inherits from BM25 which uses math.log(self.corpus_size + 1) usually or standard okapi. 
        // We will stick to the ln((N + 1) / df) used in rank_bm25 plus for now, but rank_bm25 BM25 base class uses ln(N - qf + 0.5) / (qf + 0.5) + 1
        // Let's use standard Okapi IDF from our Okapi code just to be safe, which is: ln(((N - df + 0.5) / (df + 0.5)) + 1.0)
        let idf: FxHashMap<String, f64> = df.iter().map(|(term, &dft)| {
            let dft_f = dft as f64;
            // ln( (N - df + 0.5) / (df + 0.5) + 1.0 )
            let idf_val = ((n - dft_f + 0.5) / (dft_f + 0.5) + 1.0).ln();
            (term.clone(), idf_val)
        }).collect();

        let base_gk1 = if (k1 - 1.0).abs() < f64::EPSILON {
            1.0
        } else {
            (k1 / (k1 - 1.0)) * k1.ln()
        };

        // Pre-compute k1dash for all terms
        let mut term_k1dash: FxHashMap<String, f64> = FxHashMap::default();
        for (term, postings) in &term_postings {
            let dft_f = postings.len() as f64;
            let mut k1dash_num = 0.0;
            for &(doc_idx, tf) in postings {
                let dl = tokenised[doc_idx].len() as f64;
                let ctd = (tf as f64) / (1.0 - b + b * dl / avgdl);
                k1dash_num += (ctd + 1.0).ln();
            }
            let k1dash = (base_gk1 - k1dash_num / dft_f).powi(2);
            term_k1dash.insert(term.clone(), k1dash);
        }

        // Compute final tf_norm
        let mut tf_norm: Vec<FxHashMap<String, f64>> = vec![FxHashMap::default(); corpus.len()];
        for (term, postings) in term_postings {
            let k1dash = *term_k1dash.get(&term).unwrap();
            for (doc_idx, tf) in postings {
                let dl = tokenised[doc_idx].len() as f64;
                let ctd = (tf as f64) / (1.0 - b + b * dl / avgdl);
                let norm = (ctd * (k1dash + 1.0)) / (k1dash + ctd);
                tf_norm[doc_idx].insert(term.clone(), norm);
            }
        }

        let mut corpus_index: FxHashMap<String, usize> = FxHashMap::default();
        for (i, doc) in corpus.iter().enumerate() {
            corpus_index.insert(doc.clone(), i);
        }

        // Build positions
        let mut positions: Vec<FxHashMap<String, Vec<usize>>> = Vec::with_capacity(tokenised.len());
        for doc in &tokenised {
            let mut doc_positions: FxHashMap<String, Vec<usize>> = FxHashMap::default();
            for (pos, term) in doc.iter().enumerate() {
                doc_positions.entry(term.clone()).or_default().push(pos);
            }
            positions.push(doc_positions);
        }

        BM25T { corpus, corpus_index, tokenised, idf, tf_norm, positions, avgdl, k1, b, normalize }
    }

    #[getter]
    pub fn num_docs(&self) -> usize {
        self.corpus.len()
    }

    pub fn get_scores(&self, query: &str) -> Vec<f64> {
        let q_terms = tokenise(query, self.normalize);
        let n = self.corpus.len();
        let tf_norm = &self.tf_norm;
        let idf = &self.idf;

        if n >= 10_000 {
            (0..n).into_par_iter().map(|i| {
                let doc_tf = &tf_norm[i];
                q_terms.iter().map(|term| {
                    let idf_val = idf.get(term).copied().unwrap_or(0.0);
                    let tf_val = doc_tf.get(term).copied().unwrap_or(0.0);
                    idf_val * tf_val
                }).sum::<f64>()
            }).collect()
        } else {
            (0..n).map(|i| {
                let doc_tf = &tf_norm[i];
                q_terms.iter().map(|term| {
                    let idf_val = idf.get(term).copied().unwrap_or(0.0);
                    let tf_val = doc_tf.get(term).copied().unwrap_or(0.0);
                    idf_val * tf_val
                }).sum::<f64>()
            }).collect()
        }
    }

    #[pyo3(signature = (query, allowed=None))]
    pub fn get_scores_filtered(&self, query: &str, allowed: Option<Vec<bool>>) -> Vec<f64> {
        let mut scores = self.get_scores(query);
        if let Some(ref mask) = allowed {
            for (i, s) in scores.iter_mut().enumerate() {
                if i < mask.len() && !mask[i] { *s = 0.0; }
            }
        }
        scores
    }

    #[pyo3(signature = (query, n=5))]
    pub fn get_top_n(&self, query: &str, n: usize) -> Vec<(String, f64)> {
        self.get_top_n_filtered(query, n, None)
    }

    #[pyo3(signature = (query, n=5, allowed=None))]
    pub fn get_top_n_filtered(&self, query: &str, n: usize, allowed: Option<Vec<bool>>) -> Vec<(String, f64)> {
        let scores = self.get_scores_filtered(query, allowed);
        let valid_count = scores.len();
        if valid_count <= 256 || n >= valid_count {
            let mut items: Vec<(usize, f64)> = scores.into_iter().enumerate().filter(|(_, s)| *s > 0.0).collect();
            items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(n);
            return items.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect();
        }

        let mut heap: BinaryHeap<Reverse<ScoreItem>> = BinaryHeap::with_capacity(n.min(100));
        for (idx, score) in scores.into_iter().enumerate() {
            if score > 0.0 {
                if heap.len() < n {
                    heap.push(Reverse(ScoreItem { score, idx }));
                } else if let Some(mut peek) = heap.peek_mut() {
                    if score > peek.0.score {
                        peek.0 = ScoreItem { score, idx };
                    }
                }
            }
        }
        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect()
    }

    /// BM25 scores for a batch of queries — Rayon-parallel over queries.
    pub fn get_batch_scores(&self, queries: Vec<String>) -> Vec<Vec<f64>> {
        queries.par_iter().map(|q| self.get_scores(q)).collect()
    }

    /// BM25 + fuzzy hybrid: run BM25, take top `bm25_candidates`, then re-rank
    /// with indel fuzzy similarity.
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
            return self.fuzzy_only(query, n);
        }

        let max_bm25 = candidates.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
        let min_bm25 = candidates.iter().map(|(_, s)| *s).fold(f64::MAX, f64::min);
        let bm25_range = (max_bm25 - min_bm25).max(1e-10);

        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        let mut scored: Vec<(String, f64)> = candidates.into_iter().map(|(doc, bm25_raw)| {
            let norm_bm25 = (bm25_raw - min_bm25) / bm25_range;
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
            let combined = bm25_weight * norm_bm25 + fuzzy_weight * fuzzy;
            (doc, combined)
        }).collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        Ok(scored)
    }

    #[pyo3(signature = (query, n=5, bm25_candidates=100, rrf_k=60))]
    pub fn get_top_n_rrf(&self, query: &str, n: usize, bm25_candidates: usize, rrf_k: usize) -> Vec<(String, f64)> {
        self.get_top_n_rrf_filtered(query, n, bm25_candidates, rrf_k, None)
    }

    #[pyo3(signature = (query, n=5, bm25_candidates=100, rrf_k=60, allowed=None))]
    pub fn get_top_n_rrf_filtered(&self, query: &str, n: usize, bm25_candidates: usize, rrf_k: usize, allowed: Option<Vec<bool>>) -> Vec<(String, f64)> {
        use crate::algorithms::indel_distance;

        let candidates_n = bm25_candidates.max(n * 10);
        let bm25_results = self.get_top_n_filtered(query, candidates_n, allowed.clone());
        
        let is_bm25_empty = bm25_results.is_empty();
        let target_docs: Vec<(usize, &String)> = if is_bm25_empty {
            self.corpus.iter().enumerate()
                .filter(|(i, _)| allowed.as_ref().is_none_or(|m| i < &m.len() && m[*i]))
                .collect()
        } else {
            bm25_results.iter().map(|(doc, _)| {
                let idx = *self.corpus_index.get(doc).unwrap();
                (idx, doc)
            }).collect()
        };

        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

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
        fuzzy_ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut rrf_scores: Vec<f64> = vec![0.0; self.corpus.len()];
        
        if !is_bm25_empty {
            for (rank, (doc, _)) in bm25_results.iter().enumerate() {
                let idx = *self.corpus_index.get(doc).unwrap();
                rrf_scores[idx] += 1.0 / (rrf_k + rank + 1) as f64;
            }
        }
        
        for (rank, (idx, _)) in fuzzy_ranked.iter().enumerate() {
            rrf_scores[*idx] += 1.0 / (rrf_k + rank + 1) as f64;
        }

        let mut results: Vec<(String, f64)> = target_docs.into_iter()
            .map(|(idx, doc)| (doc.clone(), rrf_scores[idx]))
            .collect();
            
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(n);
        results
    }

    fn fuzzy_only(&self, query: &str, n: usize) -> PyResult<Vec<(String, f64)>> {
        use crate::algorithms::indel_distance;
        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        let scores: Vec<f64> = self.corpus.iter().map(|doc| {
            if q_is_ascii && doc.is_ascii() {
                let d = doc.as_bytes();
                let ls = q_bytes.len() + d.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(q_bytes, d, None) as f64 / ls as f64 }
            } else {
                let qv: Vec<u32> = query.chars().map(|c| c as u32).collect();
                let dv: Vec<u32> = doc.chars().map(|c| c as u32).collect();
                let ls = qv.len() + dv.len();
                if ls == 0 { 1.0 } else { 1.0 - indel_distance(&qv, &dv, None) as f64 / ls as f64 }
            }
        }).collect();

        let corpus_len = self.corpus.len();
        if corpus_len <= 256 || n >= corpus_len {
            let mut items: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
            items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            items.truncate(n);
            return Ok(items.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect());
        }

        let mut heap: BinaryHeap<Reverse<ScoreItem>> = BinaryHeap::with_capacity(n.min(100));
        for (idx, score) in scores.into_iter().enumerate() {
            if heap.len() < n {
                heap.push(Reverse(ScoreItem { score, idx }));
            } else if let Some(mut peek) = heap.peek_mut() {
                if score > peek.0.score {
                    peek.0 = ScoreItem { score, idx };
                }
            }
        }

        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|Reverse(item)| (item.idx, item.score)).collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect())
    }

    pub fn get_idf_map(&self) -> FxHashMap<String, f64> {
        self.idf.clone()
    }

    #[pyo3(signature = (doc_idx))]
    pub fn get_document_vector(&self, doc_idx: usize) -> PyResult<FxHashMap<String, f64>> {
        if doc_idx >= self.corpus.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("doc_idx {} out of range (corpus has {} docs)", doc_idx, self.corpus.len())
            ));
        }
        Ok(self.tf_norm[doc_idx].clone())
    }

    #[pyo3(signature = (query, doc_idx))]
    pub fn explain(&self, query: &str, doc_idx: usize) -> PyResult<Vec<(String, f64, f64, f64)>> {
        if doc_idx >= self.corpus.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("doc_idx {} out of range (corpus has {} docs)", doc_idx, self.corpus.len())
            ));
        }
        let q_terms = tokenise(query, self.normalize);
        let doc_tf = &self.tf_norm[doc_idx];
        let mut breakdown = Vec::new();
        for term in q_terms {
            let idf_val = self.idf.get(&term).copied().unwrap_or(0.0);
            let tf_val = doc_tf.get(&term).copied().unwrap_or(0.0);
            let score = idf_val * tf_val;
            breakdown.push((term, idf_val, tf_val, score));
        }
        Ok(breakdown)
    }

    #[pyo3(signature = (query, n=5, proximity_window=3, phrase_boost=2.0))]
    pub fn get_top_n_phrase(&self, query: &str, n: usize, proximity_window: usize, phrase_boost: f64) -> Vec<(String, f64)> {
        let bm25_scores = self.get_scores(query);
        let q_terms = tokenise(query, self.normalize);
        let mut scored: Vec<(usize, f64)> = bm25_scores.into_iter().enumerate()
            .filter(|(_, s)| *s > 0.0)
            .map(|(i, bm25)| {
                let boost = phrase_proximity_boost(&q_terms, &self.positions[i], proximity_window);
                let final_score = bm25 * (1.0 + (phrase_boost - 1.0) * boost);
                (i, final_score)
            }).collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored.into_iter().map(|(i, s)| (self.corpus[i].clone(), s)).collect()
    }
}

// ============================================================================
// HybridSearchIndex — 3-way RRF (BM25 + Fuzzy + Dense) entirely in Rust
// ============================================================================

/// Hybrid search index combining BM25 text retrieval, fuzzy string matching,
/// and dense vector (embedding) similarity via 3-way Reciprocal Rank Fusion.
///
/// All heavy computation runs in Rust outside the Python GIL.
/// Designed for million-scale corpora.
#[pyclass]
pub struct HybridSearchIndex {
    /// The BM25 index (reuses existing implementation)
    bm25: BM25Variant,
    /// The original corpus strings
    corpus: Vec<String>,
    /// Reverse lookup: doc text → corpus index
    corpus_index: FxHashMap<String, usize>,
    /// Optional dense embeddings, shape (num_docs, dim), stored as f32
    embeddings: Option<Vec<Vec<f32>>>,
    /// Embedding dimensionality (0 if no embeddings)
    dim: usize,
    /// BM25 parameters (for pickle)
    k1: f64,
    b: f64,
    algorithm: String,
    delta: Option<f64>,
    /// Optional per-document metadata stored as serde_json::Value for fast filter evaluation
    metadata: Option<Vec<JsonValue>>,
}

pub enum BM25Variant {
    Okapi(BM25Index),
    L(BM25L),
    Plus(BM25Plus),
    T(BM25T),
}

impl BM25Variant {
    pub fn get_top_n_filtered(
        &self,
        query: &str,
        n: usize,
        allowed: Option<Vec<bool>>,
    ) -> Vec<(String, f64)> {
        match self {
            Self::Okapi(idx) => idx.get_top_n_filtered(query, n, allowed),
            Self::L(idx) => idx.get_top_n_filtered(query, n, allowed),
            Self::Plus(idx) => idx.get_top_n_filtered(query, n, allowed),
            Self::T(idx) => idx.get_top_n_filtered(query, n, allowed),
        }
    }
}

#[pymethods]
impl HybridSearchIndex {
    /// Create a new HybridSearchIndex.
    ///
    /// Parameters
    /// ----------
    /// corpus : list[str]
    ///     Documents to index.
    /// embeddings : list[list[float]] | None
    ///     Optional dense embeddings, one per document. Shape (N, D).
    /// k1 : float
    ///     BM25 term frequency saturation.
    /// b : float
    ///     BM25 length normalisation.
    #[new]
    #[pyo3(signature = (corpus, embeddings=None, k1=1.5, b=0.75, algorithm=None, delta=None))]
    pub fn new(
        corpus: Vec<String>,
        embeddings: Option<Vec<Vec<f32>>>,
        k1: f64,
        b: f64,
        algorithm: Option<&str>,
        delta: Option<f64>,
    ) -> PyResult<Self> {
        let n = corpus.len();
        let algo_str = algorithm.unwrap_or("bm25");

        // Validate embeddings
        let (emb, dim) = if let Some(ref e) = embeddings {
            if e.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("embeddings length ({}) must match corpus length ({})", e.len(), n)
                ));
            }
            let d = if e.is_empty() { 0 } else { e[0].len() };
            // Validate all rows have same dimension
            for (i, row) in e.iter().enumerate() {
                if row.len() != d {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("embedding {} has {} dims, expected {}", i, row.len(), d)
                    ));
                }
            }
            (embeddings, d)
        } else {
            (None, 0)
        };

        // Build reverse index
        let mut corpus_index: FxHashMap<String, usize> = FxHashMap::default();
        for (i, doc) in corpus.iter().enumerate() {
            corpus_index.insert(doc.clone(), i);
        }

        let bm25_variant = match algo_str {
            "bm25" => BM25Variant::Okapi(BM25Index::new(corpus.clone(), k1, b, false)),
            "bm25l" => BM25Variant::L(BM25L::new(corpus.clone(), k1, b, delta.unwrap_or(0.5), false)),
            "bm25+" => BM25Variant::Plus(BM25Plus::new(corpus.clone(), k1, b, delta.unwrap_or(1.0), false)),
            "bm25t" => BM25Variant::T(BM25T::new(corpus.clone(), k1, b, false)),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown BM25 algorithm '{}'. Expected 'bm25', 'bm25l', 'bm25+', or 'bm25t'", algo_str)
            )),
        };

        Ok(HybridSearchIndex {
            bm25: bm25_variant,
            corpus,
            corpus_index,
            embeddings: emb,
            dim,
            k1,
            b,
            algorithm: algo_str.to_string(),
            delta,
            metadata: None,
        })
    }

    /// Number of documents in the index.
    #[getter]
    pub fn num_docs(&self) -> usize {
        self.corpus.len()
    }

    /// Whether the index has dense vector embeddings.
    #[getter]
    pub fn has_vectors(&self) -> bool {
        self.embeddings.is_some()
    }

    /// Embedding dimensionality (0 if no embeddings).
    #[getter]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 3-way hybrid search: BM25 + Fuzzy + Dense via Reciprocal Rank Fusion.
    ///
    /// Parameters
    /// ----------
    /// query : str
    ///     Text query.
    /// query_embedding : list[float] | None
    ///     Dense embedding for the query. If None, falls back to 2-way RRF.
    /// n : int
    ///     Number of results to return.
    /// rrf_k : int
    ///     RRF smoothing constant (default 60, from Cormack et al. 2009).
    /// bm25_candidates : int
    ///     Number of BM25 candidates to pre-filter before fuzzy+dense scoring.
    ///     This is the key to million-scale performance.
    #[pyo3(signature = (query, query_embedding=None, n=5, rrf_k=60, bm25_candidates=100))]
    pub fn search(
        &self,
        query: &str,
        query_embedding: Option<Vec<f32>>,
        n: usize,
        rrf_k: usize,
        bm25_candidates: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        self.search_filtered(query, query_embedding, n, rrf_k, bm25_candidates, None)
    }

    /// 3-way hybrid search with an optional allowed mask.
    #[pyo3(signature = (query, query_embedding=None, n=5, rrf_k=60, bm25_candidates=100, allowed=None))]
    pub fn search_filtered(
        &self,
        query: &str,
        query_embedding: Option<Vec<f32>>,
        n: usize,
        rrf_k: usize,
        bm25_candidates: usize,
        allowed: Option<Vec<bool>>,
    ) -> PyResult<Vec<(String, f64)>> {
        

        let candidates_n = bm25_candidates.max(n * 10);

        // ── Step 1: BM25 ranking (with filter) ──────────────────
        let bm25_results = self.bm25.get_top_n_filtered(query, candidates_n, allowed.clone());

        let is_bm25_empty = bm25_results.is_empty();

        // Candidate doc indices + refs
        let target_docs: Vec<(usize, &String)> = if is_bm25_empty {
            // BM25 returned nothing — fall back to all allowed docs
            self.corpus.iter().enumerate()
                .filter(|(i, _)| allowed.as_ref().is_none_or(|m| i < &m.len() && m[*i]))
                .collect()
        } else {
            bm25_results.iter().map(|(doc, _)| {
                let idx = *self.corpus_index.get(doc).unwrap();
                (idx, doc)
            }).collect()
        };

        let num_targets = target_docs.len();

        // ── Step 2: Fuzzy ranking over candidates ─────────────
        let q_bytes = query.as_bytes();
        let q_is_ascii = query.is_ascii();

        let mut fuzzy_ranked: Vec<(usize, f64)> = if num_targets > 2000 {
            // Parallel fuzzy for large candidate sets
            target_docs.par_iter().map(|(i, doc)| {
                let fuzzy = compute_fuzzy_score(query, q_bytes, q_is_ascii, doc);
                (*i, fuzzy)
            }).collect()
        } else {
            target_docs.iter().map(|(i, doc)| {
                let fuzzy = compute_fuzzy_score(query, q_bytes, q_is_ascii, doc);
                (*i, fuzzy)
            }).collect()
        };
        fuzzy_ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // ── Step 3: Dense ranking over candidates (if available) ──
        let dense_ranked: Option<Vec<(usize, f32)>> = match (&self.embeddings, &query_embedding) {
            (Some(emb), Some(q_emb)) => {
                if q_emb.len() != self.dim {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("query_embedding has {} dims, expected {}", q_emb.len(), self.dim)
                    ));
                }
                let q_norm = {
                    let norm = q_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 { norm } else { 1.0 }
                };

                let mut ranked: Vec<(usize, f32)> = if num_targets > 2000 {
                    target_docs.par_iter().map(|(i, _)| {
                        let doc_emb = &emb[*i];
                        let doc_norm = doc_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let dn = if doc_norm > 0.0 { doc_norm } else { 1.0 };
                        let cosine: f32 = q_emb.iter().zip(doc_emb.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f32>() / (q_norm * dn);
                        (*i, cosine)
                    }).collect()
                } else {
                    target_docs.iter().map(|(i, _)| {
                        let doc_emb = &emb[*i];
                        let doc_norm = doc_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let dn = if doc_norm > 0.0 { doc_norm } else { 1.0 };
                        let cosine: f32 = q_emb.iter().zip(doc_emb.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f32>() / (q_norm * dn);
                        (*i, cosine)
                    }).collect()
                };
                ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                Some(ranked)
            },
            _ => None,
        };

        // ── Step 4: Reciprocal Rank Fusion ────────────────────
        let mut rrf_scores: Vec<f64> = vec![0.0; self.corpus.len()];

        // BM25 ranks
        if !is_bm25_empty {
            for (rank, (doc, _)) in bm25_results.iter().enumerate() {
                let idx = *self.corpus_index.get(doc).unwrap();
                rrf_scores[idx] += 1.0 / (rrf_k + rank + 1) as f64;
            }
        }

        // Fuzzy ranks
        for (rank, (idx, _)) in fuzzy_ranked.iter().enumerate() {
            rrf_scores[*idx] += 1.0 / (rrf_k + rank + 1) as f64;
        }

        // Dense ranks (if available)
        if let Some(ref dr) = dense_ranked {
            for (rank, (idx, _)) in dr.iter().enumerate() {
                rrf_scores[*idx] += 1.0 / (rrf_k + rank + 1) as f64;
            }
        }

        // ── Step 5: Assemble and sort ────────────────────────
        let mut results: Vec<(String, f64)> = target_docs.into_iter()
            .map(|(idx, doc): (usize, &String)| (doc.clone(), rrf_scores[idx]))
            .collect();

        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(n);
        Ok(results)
    }

    /// Set per-document metadata as JSON strings for fast Rust-side filter evaluation.
    ///
    /// Parameters
    /// ----------
    /// json_strings : list[str]
    ///     JSON-serialized metadata dicts, one per document.
    pub fn set_metadata_json(&mut self, json_strings: Vec<String>) -> PyResult<()> {
        if json_strings.len() != self.corpus.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("metadata length ({}) must match corpus length ({})", json_strings.len(), self.corpus.len())
            ));
        }
        let parsed: Result<Vec<JsonValue>, _> = json_strings.iter()
            .map(|s| serde_json::from_str(s))
            .collect();
        match parsed {
            Ok(vals) => {
                self.metadata = Some(vals);
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Failed to parse metadata JSON: {}", e)
            )),
        }
    }

    /// Whether the index has Rust-side metadata for fast filtering.
    #[getter]
    pub fn has_metadata(&self) -> bool {
        self.metadata.is_some()
    }

    /// Evaluate a JSON filter expression against stored metadata and return a boolean mask.
    ///
    /// Parameters
    /// ----------
    /// filter_json : str
    ///     JSON-serialized filter expression (see Python-side serialization).
    ///
    /// Returns
    /// -------
    /// list[bool]
    ///     Boolean mask, True for documents that pass the filter.
    pub fn evaluate_filter_mask(&self, filter_json: &str) -> PyResult<Vec<bool>> {
        let meta = self.metadata.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No metadata set. Call set_metadata_json first.")
        })?;

        let filter_node: FilterAst = serde_json::from_str(filter_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("Failed to parse filter JSON: {}", e)
            ))?;

        let mask: Vec<bool> = if meta.len() >= 5000 {
            meta.par_iter().map(|m| evaluate_filter_ast(&filter_node, m)).collect()
        } else {
            meta.iter().map(|m| evaluate_filter_ast(&filter_node, m)).collect()
        };

        Ok(mask)
    }

    /// Combined filter + search + sort in a single Rust call.
    ///
    /// Parameters
    /// ----------
    /// query : str
    /// query_embedding : list[float] | None
    /// n : int
    /// rrf_k : int
    /// bm25_candidates : int
    /// filter_json : str | None
    ///     JSON-serialized filter expression.
    /// sort_keys : list[(str, bool)] | None
    ///     List of (attribute_path, reverse) pairs for sorting.
    #[pyo3(signature = (query, query_embedding=None, n=5, rrf_k=60, bm25_candidates=100, filter_json=None, sort_keys=None))]
    pub fn search_filtered_sorted(
        &self,
        query: &str,
        query_embedding: Option<Vec<f32>>,
        n: usize,
        rrf_k: usize,
        bm25_candidates: usize,
        filter_json: Option<String>,
        sort_keys: Option<Vec<(String, bool)>>,
    ) -> PyResult<Vec<(String, f64)>> {
        // Build filter mask if filter provided and metadata available
        let allowed = match (&filter_json, &self.metadata) {
            (Some(fj), Some(meta)) => {
                let filter_node: FilterAst = serde_json::from_str(fj)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                        format!("Failed to parse filter JSON: {}", e)
                    ))?;
                let mask: Vec<bool> = if meta.len() >= 5000 {
                    meta.par_iter().map(|m| evaluate_filter_ast(&filter_node, m)).collect()
                } else {
                    meta.iter().map(|m| evaluate_filter_ast(&filter_node, m)).collect()
                };
                Some(mask)
            }
            _ => None,
        };

        // Run search_filtered
        let mut results = self.search_filtered(query, query_embedding, n * 10, rrf_k, bm25_candidates, allowed)?;

        // Apply sort if sort_keys and metadata are available
        if let (Some(keys), Some(meta)) = (&sort_keys, &self.metadata) {
            if !keys.is_empty() {
                results.sort_by(|a, b| {
                    let idx_a = self.corpus_index.get(&a.0).copied();
                    let idx_b = self.corpus_index.get(&b.0).copied();
                    let meta_a = idx_a.and_then(|i| meta.get(i));
                    let meta_b = idx_b.and_then(|i| meta.get(i));
                    for (attr, reverse) in keys {
                        let va = meta_a.map(|m| resolve_json_attr(m, attr)).unwrap_or(None);
                        let vb = meta_b.map(|m| resolve_json_attr(m, attr)).unwrap_or(None);
                        let cmp = compare_json_values(&va, &vb, *reverse);
                        if cmp != std::cmp::Ordering::Equal {
                            return cmp;
                        }
                    }
                    std::cmp::Ordering::Equal
                });
            }
        }

        results.truncate(n);
        Ok(results)
    }

    /// Pickle support.
    fn __reduce__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
        let cls = PyType::new::<HybridSearchIndex>(py);

        let corpus_list: Vec<PyObject> = slf.corpus.iter()
            .map(|s| s.clone().into_pyobject(py).map(|v| v.into_any().unbind()))
            .collect::<Result<Vec<_>, _>>()?;
        let py_corpus = pyo3::types::PyList::new(py, corpus_list)?;

        let py_emb = match &slf.embeddings {
            Some(emb) => {
                let rows: Vec<PyObject> = emb.iter().map(|row| {
                    let pyrow = pyo3::types::PyList::new(py, row.to_vec()).unwrap();
                    pyrow.into_any().unbind()
                }).collect();
                pyo3::types::PyList::new(py, rows)?.into_any().unbind()
            },
            None => py.None(),
        };

        let py_delta = match slf.delta {
            Some(d) => d.into_pyobject(py)?.into_any().unbind(),
            None => py.None(),
        };

        let args = PyTuple::new(py, [
            py_corpus.into_any().unbind(),
            py_emb,
            slf.k1.into_pyobject(py)?.into_any().unbind(),
            slf.b.into_pyobject(py)?.into_any().unbind(),
            slf.algorithm.clone().into_pyobject(py)?.into_any().unbind(),
            py_delta,
        ])?;
        Ok(PyTuple::new(py, [cls.into_any().unbind(), args.into_any().unbind()])?.into_any().unbind())
    }
}

/// Compute fuzzy (indel ratio) score between a query and a document.
#[inline]
fn compute_fuzzy_score(query: &str, q_bytes: &[u8], q_is_ascii: bool, doc: &str) -> f64 {
    use crate::algorithms::indel_distance;

    if q_is_ascii && doc.is_ascii() {
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

// ===========================================================================
// Meilisearch-compatible filter AST — Rust-side evaluation
// ===========================================================================

/// JSON-serializable filter AST matching the Python filter.py output.
#[derive(serde::Deserialize, Debug, Clone)]
#[serde(tag = "type")]
enum FilterAst {
    #[serde(rename = "comparison")]
    Comparison { attribute: String, op: String, value: JsonValue },
    #[serde(rename = "range")]
    Range { attribute: String, low: f64, high: f64 },
    #[serde(rename = "exists")]
    Exists { attribute: String },
    #[serde(rename = "is_null")]
    IsNull { attribute: String },
    #[serde(rename = "is_empty")]
    IsEmpty { attribute: String },
    #[serde(rename = "in")]
    In { attribute: String, values: Vec<JsonValue> },
    #[serde(rename = "contains")]
    Contains { attribute: String, value: String },
    #[serde(rename = "starts_with")]
    StartsWith { attribute: String, value: String },
    #[serde(rename = "not")]
    Not { child: Box<FilterAst> },
    #[serde(rename = "and")]
    And { left: Box<FilterAst>, right: Box<FilterAst> },
    #[serde(rename = "or")]
    Or { left: Box<FilterAst>, right: Box<FilterAst> },
}

/// Resolve a dot-separated attribute path in a JSON value.
fn resolve_json_attr<'a>(val: &'a JsonValue, attr: &str) -> Option<&'a JsonValue> {
    let mut current = val;
    for part in attr.split('.') {
        match current {
            JsonValue::Object(map) => {
                current = map.get(part)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Compare a JSON value against a filter value using the given operator.
fn json_compare(a: &JsonValue, b: &JsonValue, op: &str) -> bool {
    match op {
        "=" => json_eq(a, b),
        "!=" => !json_eq(a, b),
        ">" | ">=" | "<" | "<=" => {
            if let (Some(fa), Some(fb)) = (json_as_f64(a), json_as_f64(b)) {
                match op {
                    ">" => fa > fb,
                    ">=" => fa >= fb,
                    "<" => fa < fb,
                    "<=" => fa <= fb,
                    _ => false,
                }
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Numeric extraction from JSON values.
#[inline]
fn json_as_f64(v: &JsonValue) -> Option<f64> {
    match v {
        JsonValue::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// Equality check handling booleans, numbers, strings, and null.
fn json_eq(a: &JsonValue, b: &JsonValue) -> bool {
    match (a, b) {
        (JsonValue::Null, JsonValue::Null) => true,
        (JsonValue::Bool(x), JsonValue::Bool(y)) => x == y,
        (JsonValue::Number(_), JsonValue::Number(_)) => {
            json_as_f64(a) == json_as_f64(b)
        }
        (JsonValue::String(x), JsonValue::String(y)) => x == y,
        _ => false,
    }
}

/// Evaluate filter AST against a single JSON metadata document.
fn evaluate_filter_ast(node: &FilterAst, meta: &JsonValue) -> bool {
    match node {
        FilterAst::Comparison { attribute, op, value } => {
            if let Some(actual) = resolve_json_attr(meta, attribute) {
                // Handle list membership for = and !=
                if let JsonValue::Array(arr) = actual {
                    if op == "=" {
                        return arr.iter().any(|v| json_eq(v, value));
                    }
                    if op == "!=" {
                        return !arr.iter().any(|v| json_eq(v, value));
                    }
                }
                json_compare(actual, value, op)
            } else {
                false
            }
        }
        FilterAst::Range { attribute, low, high } => {
            if let Some(actual) = resolve_json_attr(meta, attribute) {
                if let Some(v) = json_as_f64(actual) {
                    *low <= v && v <= *high
                } else {
                    false
                }
            } else {
                false
            }
        }
        FilterAst::Exists { attribute } => {
            resolve_json_attr(meta, attribute).is_some()
        }
        FilterAst::IsNull { attribute } => {
            matches!(resolve_json_attr(meta, attribute), Some(JsonValue::Null))
        }
        FilterAst::IsEmpty { attribute } => {
            match resolve_json_attr(meta, attribute) {
                Some(JsonValue::String(s)) => s.is_empty(),
                Some(JsonValue::Array(a)) => a.is_empty(),
                Some(JsonValue::Object(o)) => o.is_empty(),
                _ => false,
            }
        }
        FilterAst::In { attribute, values } => {
            if let Some(actual) = resolve_json_attr(meta, attribute) {
                // If metadata value is an array, check intersection
                if let JsonValue::Array(arr) = actual {
                    return arr.iter().any(|v| values.iter().any(|fv| json_eq(v, fv)));
                }
                values.iter().any(|v| json_eq(actual, v))
            } else {
                false
            }
        }
        FilterAst::Contains { attribute, value } => {
            if let Some(JsonValue::String(s)) = resolve_json_attr(meta, attribute) {
                s.contains(value.as_str())
            } else {
                false
            }
        }
        FilterAst::StartsWith { attribute, value } => {
            if let Some(JsonValue::String(s)) = resolve_json_attr(meta, attribute) {
                s.starts_with(value.as_str())
            } else {
                false
            }
        }
        FilterAst::Not { child } => {
            !evaluate_filter_ast(child, meta)
        }
        FilterAst::And { left, right } => {
            evaluate_filter_ast(left, meta) && evaluate_filter_ast(right, meta)
        }
        FilterAst::Or { left, right } => {
            evaluate_filter_ast(left, meta) || evaluate_filter_ast(right, meta)
        }
    }
}

/// Compare two optional JSON values for sort ordering.
/// Missing/null values sort to the end regardless of direction.
fn compare_json_values(a: &Option<&JsonValue>, b: &Option<&JsonValue>, reverse: bool) -> std::cmp::Ordering {
    match (a, b) {
        (None, None) | (Some(JsonValue::Null), Some(JsonValue::Null)) => std::cmp::Ordering::Equal,
        (None, _) | (Some(JsonValue::Null), _) => std::cmp::Ordering::Greater, // nulls last
        (_, None) | (_, Some(JsonValue::Null)) => std::cmp::Ordering::Less,
        (Some(va), Some(vb)) => {
            // Try numeric comparison first
            if let (Some(fa), Some(fb)) = (json_as_f64(va), json_as_f64(vb)) {
                let cmp = fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal);
                return if reverse { cmp.reverse() } else { cmp };
            }
            // String comparison
            if let (JsonValue::String(sa), JsonValue::String(sb)) = (va, vb) {
                let cmp = sa.to_lowercase().cmp(&sb.to_lowercase());
                return if reverse { cmp.reverse() } else { cmp };
            }
            // Bool comparison
            if let (JsonValue::Bool(ba), JsonValue::Bool(bb)) = (va, vb) {
                let cmp = ba.cmp(bb);
                return if reverse { cmp.reverse() } else { cmp };
            }
            std::cmp::Ordering::Equal
        }
    }
}
