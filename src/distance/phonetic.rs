// SPDX-License-Identifier: MIT
// Phonetic algorithms: Soundex and Metaphone

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Soundex
// ---------------------------------------------------------------------------

const SOUNDEX_MAPPING: [u8; 26] = [
    b'0', // A
    b'1', // B
    b'2', // C
    b'3', // D
    b'0', // E
    b'1', // F
    b'2', // G
    b'0', // H
    b'0', // I
    b'2', // J
    b'2', // K
    b'4', // L
    b'5', // M
    b'5', // N
    b'0', // O
    b'1', // P
    b'2', // Q
    b'6', // R
    b'2', // S
    b'3', // T
    b'0', // U
    b'1', // V
    b'0', // W
    b'2', // X
    b'0', // Y
    b'2', // Z
];

/// Computes the American Soundex code for a given string.
/// Always returns a 4-character string padded with zeros.
pub fn soundex(s: &str) -> String {
    let mut chars = s.chars().filter(|c| c.is_ascii_alphabetic());
    
    let first = match chars.next() {
        Some(c) => c.to_ascii_uppercase(),
        None => return "0000".to_string(),
    };

    let mut result = String::with_capacity(4);
    result.push(first);

    let first_idx = (first as u8 - b'A') as usize;
    let mut prev_code = SOUNDEX_MAPPING[first_idx];

    for c in chars {
        if result.len() == 4 {
            break;
        }
        let upper = c.to_ascii_uppercase();
        let idx = (upper as u8 - b'A') as usize;
        let code = SOUNDEX_MAPPING[idx];
        
        // H and W are ignored for adjacency rules, other vowels separate repeated consonants
        if code != b'0' && code != prev_code {
            result.push(code as char);
        }
        
        if code != b'0' || (upper != 'H' && upper != 'W') {
             prev_code = code;
        }
    }

    while result.len() < 4 {
        result.push('0');
    }

    result
}

// ---------------------------------------------------------------------------
// Distance & Similarity using Soundex
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (s))]
pub fn soundex_encode(s: &str) -> String {
    soundex(s)
}

#[pyfunction]
#[pyo3(signature = (s1, s2))]
pub fn soundex_distance(s1: &str, s2: &str) -> usize {
    let sx1 = soundex(s1);
    let sx2 = soundex(s2);
    crate::algorithms::hamming_distance(sx1.as_bytes(), sx2.as_bytes(), false)
}

#[pyfunction]
#[pyo3(signature = (s1, s2))]
pub fn soundex_similarity(s1: &str, s2: &str) -> usize {
    let sx1 = soundex(s1);
    let sx2 = soundex(s2);
    let b1 = sx1.as_bytes();
    let b2 = sx2.as_bytes();
    b1.iter().zip(b2.iter()).filter(|(a, b)| a == b).count()
}

#[pyfunction]
#[pyo3(signature = (s1, s2))]
pub fn soundex_normalized_similarity(s1: &str, s2: &str) -> f64 {
    soundex_similarity(s1, s2) as f64 / 4.0
}

#[pyfunction]
#[pyo3(signature = (s1, s2))]
pub fn soundex_normalized_distance(s1: &str, s2: &str) -> f64 {
    1.0 - soundex_normalized_similarity(s1, s2)
}

// ---------------------------------------------------------------------------
// Python Wrapper
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soundex_encode, m)?)?;
    m.add_function(wrap_pyfunction!(soundex_distance, m)?)?;
    m.add_function(wrap_pyfunction!(soundex_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(soundex_normalized_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(soundex_normalized_distance, m)?)?;
    Ok(())
}
