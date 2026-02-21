// SPDX-License-Identifier: MIT
// rustfuzz Rust extension module
use pyo3::prelude::*;

mod algorithms;
mod distance;
mod fuzz;
mod types;
mod utils;
mod process;

use distance::initialize;
use distance::metrics;

#[pymodule]
fn _rustfuzz(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- Data types (from distance._initialize_cpp) ---
    m.add_class::<initialize::Editop>()?;
    m.add_class::<initialize::Editops>()?;
    m.add_class::<initialize::Opcode>()?;
    m.add_class::<initialize::Opcodes>()?;
    m.add_class::<initialize::MatchingBlock>()?;
    m.add_class::<initialize::ScoreAlignment>()?;

    // --- utils ---
    m.add_function(wrap_pyfunction!(utils::default_process, m)?)?;

    // --- process batching ---
    m.add_function(wrap_pyfunction!(process::extract, m)?)?;
    m.add_function(wrap_pyfunction!(process::extract_one, m)?)?;
    m.add_function(wrap_pyfunction!(process::extract_iter, m)?)?;

    // --- fuzz scorers ---
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

    // --- distance metrics ---
    // Levenshtein
    m.add_function(wrap_pyfunction!(metrics::levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_editops, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::levenshtein_opcodes, m)?)?;
    // Hamming
    m.add_function(wrap_pyfunction!(metrics::hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_editops, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::hamming_opcodes, m)?)?;
    // Indel
    m.add_function(wrap_pyfunction!(metrics::indel_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_editops, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::indel_opcodes, m)?)?;
    // Jaro
    m.add_function(wrap_pyfunction!(metrics::jaro_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_normalized_similarity, m)?)?;
    // JaroWinkler
    m.add_function(wrap_pyfunction!(metrics::jaro_winkler_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_winkler_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_winkler_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::jaro_winkler_normalized_similarity, m)?)?;
    // LCSseq
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_editops, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::lcs_seq_opcodes, m)?)?;
    // OSA
    m.add_function(wrap_pyfunction!(metrics::osa_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::osa_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::osa_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::osa_normalized_similarity, m)?)?;
    // DamerauLevenshtein
    m.add_function(wrap_pyfunction!(metrics::damerau_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::damerau_levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::damerau_levenshtein_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::damerau_levenshtein_normalized_similarity, m)?)?;
    // Prefix
    m.add_function(wrap_pyfunction!(metrics::prefix_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::prefix_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::prefix_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::prefix_normalized_similarity, m)?)?;
    // Postfix
    m.add_function(wrap_pyfunction!(metrics::postfix_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::postfix_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::postfix_normalized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::postfix_normalized_similarity, m)?)?;

    // register submodule names in sys.modules so stubs can import them

    Ok(())
}
