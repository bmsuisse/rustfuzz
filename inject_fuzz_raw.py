import re

# To execute natively without the GIL, we need raw versions of the fuzz_* functions
# that take Seq instead of py: Python and &Bound<PyAny>.

with open("src/fuzz.rs", "r") as f:
    text = f.read()

# E.g., pub fn fuzz_ratio(py: Python<'_>, s1: ..., s2:... ) -> PyResult<f64>
# We need to bypass `get_processed_args` and `extract_single` which require the GIL.

# Instead of rewriting all of fuzz.rs, let's write `execute_scorer_raw(stype: ScorerType, s1: &Seq<'_>, s2: &Seq<'_>, score_cutoff: Option<f64>) -> f64`
# In execute_scorer_raw, we call the normalized versions directly!

# e.g., ratio is indel_normalized_sim. 
# Look at fuzz_ratio:
#    let (a_obj, b_obj) = get_processed_args(py, s1, s2, &processor)?;
#    let a = extract_single(&a_obj)?;
#    let b = extract_single(&b_obj)?;
#    let sim = indel_normalized_sim(&a, &b, score_cutoff);
#    Ok(norm_sim_to_score(score_cutoff_check(sim, score_cutoff)))

raw_scorer = """
pub fn execute_scorer_raw(
    stype: crate::types::ScorerType,
    s1: &crate::types::Seq<'_>,
    s2: &crate::types::Seq<'_>,
    score_cutoff: Option<f64>,
) -> f64 {
    use crate::types::ScorerType::*;
    match stype {
        Ratio => norm_sim_to_score(score_cutoff_check(indel_normalized_sim(s1, s2, score_cutoff), score_cutoff)),
        _ => 0.0 // For now, only accelerate Ratio to test the Rayon concept!
    }
}
"""

with open("src/fuzz.rs", "a") as f:
    f.write(raw_scorer)

with open("src/process.rs", "r") as f:
    text = f.read()

impl = """
use rayon::prelude::*;
use crate::types::ScorerType;
use crate::fuzz::execute_scorer_raw;

// Reconfigure extract to use Rayon when possible
#[pyfunction]
#[pyo3(signature = (query, choices, scorer_name, scorer_obj, processor=None, limit=Some(5), score_cutoff=None))]
pub fn extract(
    py: Python<'_>,
    query: &Bound<'_, PyAny>,
    choices: &Bound<'_, PyAny>,
    scorer_name: &str,
    scorer_obj: Option<PyObject>,
    processor: Option<PyObject>,
    limit: Option<usize>,
    score_cutoff: Option<f64>,
) -> PyResult<Vec<(PyObject, f64, usize)>> {
    let stype = ScorerType::from_str(scorer_name);
    
    let processed_query = if let Some(ref proc) = processor {
        proc.call1(py, (query,))?.into_bound(py)
    } else {
        query.clone()
    };
    
    // Fast Path Check: If no custom processor, no custom scorer, AND it's Ratio (for now)
    if processor.is_none() && scorer_obj.is_none() && stype == ScorerType::Ratio {
        let q_seq = crate::types::extract_single(&processed_query)?;
        
        // Extract all choices holding GIL
        let mut raw_choices = Vec::new();
        for (idx, choice_res) in choices.try_iter()?.enumerate() {
            let choice = choice_res?;
            if !crate::types::is_none(&choice) {
                // Must extract it entirely to owned if we want to cross thread boundaries,
                // OR we can trust that `choices` keeps them alive but we can't express that safely in PyO3 without py.allow_threads
                // Actually, if we extract Seq::Ascii, it borrows from `choice` which dies at loop end!
                // PyO3's try_iter yields Bound<'py, PyAny>.
                // To safely multithread, we can just spawn Python threads with py.allow_threads? No, Python threads hit GIL.
                // We must extract strings to owned Vec<u32> or Rust String if we want safe Cross-thread iterators.
                // BUT extracting owned strings kills the Zero-Copy speed!
                
                // Let's just collect the Bound<'py, PyAny> into a Vec, and THEN run parallel with Python thread pool? No.
            }
        }
    }
"""

print("Written raw evaluator. PyO3 threading requires owned strings unless we use unsafe pointers.")
