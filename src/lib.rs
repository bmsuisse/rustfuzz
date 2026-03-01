use pyo3::prelude::*;

mod algorithms;
mod distance;
mod fuzz;
mod join;
mod process;
mod search;
mod types;
mod utils;

use distance::initialize;
use distance::metrics;

#[pymodule]
fn _rustfuzz(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<initialize::Editop>()?;
    m.add_class::<initialize::Editops>()?;
    m.add_class::<initialize::Opcode>()?;
    m.add_class::<initialize::Opcodes>()?;
    m.add_class::<initialize::MatchingBlock>()?;
    m.add_class::<initialize::ScoreAlignment>()?;

    m.add_function(wrap_pyfunction!(utils::default_process, m)?)?;

    m.add_function(wrap_pyfunction!(process::extract, m)?)?;
    m.add_function(wrap_pyfunction!(process::extract_one, m)?)?;
    m.add_function(wrap_pyfunction!(process::extract_iter, m)?)?;
    m.add_function(wrap_pyfunction!(process::cdist, m)?)?;

    // Search module
    m.add_class::<search::BM25Index>()?;
    m.add_class::<search::BM25L>()?;
    m.add_class::<search::BM25Plus>()?;
    m.add_class::<search::BM25T>()?;
    m.add_class::<search::HybridSearchIndex>()?;
    m.add_function(wrap_pyfunction!(search::cosine_similarity_matrix, m)?)?;

    // Multi-array join
    m.add_class::<join::MultiJoiner>()?;

    m.add_function(wrap_pyfunction!(fuzz::fuzz_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_partial_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_partial_ratio_alignment, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_token_sort_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_token_set_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_token_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_partial_token_sort_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_partial_token_set_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_partial_token_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_wratio, m)?)?;
    m.add_function(wrap_pyfunction!(fuzz::fuzz_qratio, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_editops, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_opcodes, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_editops, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_opcodes, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::indel_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_editops, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_opcodes, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::jaro_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_normalized_similarity, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::jaro_winkler_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_winkler_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_winkler_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_winkler_normalized_similarity, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::lcs_seq_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_editops, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_opcodes, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::osa_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::osa_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::osa_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::osa_normalized_similarity, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::damerau_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::damerau_levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::damerau_levenshtein_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::damerau_levenshtein_normalized_similarity, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::prefix_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::prefix_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::prefix_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::prefix_normalized_similarity, m)?)?;

    m.add_function(wrap_pyfunction!(metrics::postfix_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::postfix_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::postfix_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::postfix_normalized_similarity, m)?)?;

    distance::phonetic::register(m)?;
    distance::gotoh::register(m)?;
    distance::ngram::register(m)?;
    distance::bktree::register(m)?;

    Ok(())
}
