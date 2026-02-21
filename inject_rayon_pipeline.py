import re

with open("src/process.rs", "r") as f:
    text = f.read()

# We need to insert our execution loop inside `extract`.
# First let's remove the placeholder native scorer, we already defined `execute_scorer_raw` in fuzz.rs!
# We just need to expose it.
# Actually, wait, `fuzz::execute_scorer_raw` was written in fuzz.rs.
# In `process.rs`, we just call it.
text = text.replace("""// Pure Rust dispatch for our algorithms without Python GIL
fn execute_native_scorer(
    stype: ScorerType,
    s1: &crate::types::Seq<'_>,
    s2: &crate::types::Seq<'_>,
    score_cutoff: Option<f64>,
) -> f64 {
    use ScorerType::*;
    match stype {
        Ratio => {
            // Because our current fuzz_* expects py: Python we need raw wrappers.
            // Let's just implement a quick dispatch directly to `algorithms` here.
            // Actually, we built `indel_normalized_sim` inside fuzz.rs but it's private.
            // Let's expose it or rewrite it.
            0.0 // Placeholder until we link the fn!
        }
        _ => 0.0
    }
}""", "")

# Now let's inject the parallel execution block into `extract`
# Where the loop currently says:
#     for (idx, choice_res) in iter.enumerate() {
#         let choice = choice_res?;

magic_block = """
    // Parallel Fast Path (No Python callbacks & Ratio)
    if processor.is_none() && scorer_obj.is_none() && stype == ScorerType::Ratio {
        // We must extract query once
        let q_seq = crate::types::extract_single(&processed_query)?;
        
        // Extract choices into owned Rust structures to escape GIL
        let mut raw_choices = Vec::new();
        for (idx, choice_res) in iter.enumerate() {
            let choice = choice_res?;
            if !crate::types::is_none(&choice) {
                let seq = crate::types::extract_single(&choice)?;
                // Clone to U32 arrays since we can't safely hold Py bytes across rayon threads
                // without 'py lifetime.
                // Wait! To beat C++, we cannot afford a `Vec<u32>` allocation per string!
                // PyO3 string buffers are IMMORTAL while the `choice` object is held in `iter`?
                // `choices` is a Python List. The items inside are alive as long as `choices` is alive!
                // We can grab the underlying `*const u8` of the PyString and reconstruct it!
                
                // For simplicity, let's just use `clone()` to prove functionality first before unsafe.
                let owned_seq = match seq {
                    crate::types::Seq::Ascii(slice) => crate::types::Seq::Chars(slice.iter().map(|&c| c as u32).collect()),
                    crate::types::Seq::U32(slice) => crate::types::Seq::Chars(slice.to_vec()),
                    crate::types::Seq::U64(slice) => crate::types::Seq::Chars(slice.iter().map(|&c| c as u32).collect()),
                    crate::types::Seq::Chars(vec) => crate::types::Seq::Chars(vec),
                };
                raw_choices.push((idx, choice.into_py(py), owned_seq));
            }
        }
        
        let mut par_results: Vec<(PyObject, f64, usize)> = raw_choices.into_par_iter().filter_map(|(idx, obj, seq)| {
            let score = crate::fuzz::execute_scorer_raw(stype, &q_seq, &seq, score_cutoff);
            if score_cutoff.is_none() || score >= score_cutoff.unwrap() {
                Some((obj, score, idx))
            } else {
                None
            }
        }).collect();
        
        par_results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(l) = limit {
            par_results.truncate(l);
        }
        return Ok(par_results);
    }

    for (idx, choice_res) in iter.enumerate() {"""

text = text.replace("    for (idx, choice_res) in iter.enumerate() {", magic_block)

with open("src/process.rs", "w") as f:
    f.write(text)

print("Injected parallel Rust engine")
