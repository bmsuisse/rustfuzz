import re

with open("src/process.rs", "r") as f:
    text = f.read()

# Fix 1: `into_py` is deprecated, use `into_py_any`
text = text.replace("choice.into_py(py)", "choice.into_py_any(py).unwrap()")

# Fix 2: `Seq::Chars` doesn't exist. `Seq::U32` does.
# We mapped everything to `Chars`, which was my mistake. We map to `U32`.
text = text.replace("crate::types::Seq::Chars(", "crate::types::Seq::U32(")

# Fix 3: ScorerType PartialEq
# Instead of `stype == ScorerType::Ratio`, let's just use `matches!(stype, ScorerType::Ratio)`.
text = text.replace("stype == ScorerType::Ratio", "matches!(stype, ScorerType::Ratio)")

with open("src/process.rs", "w") as f:
    f.write(text)

with open("src/fuzz.rs", "r") as f:
    text = f.read()

# Fix 4: `execute_scorer_raw` is not publicly exposed because it wasn't added yet!
# Wait, I wrote `inject_fuzz_raw.py` but `execute_scorer_raw` wasn't inserted. Let's add it manually here.
if "pub fn execute_scorer_raw" not in text:
    raw_eval = """
// Expose the raw execution block
pub fn execute_scorer_raw(
    stype: crate::types::ScorerType,
    s1: &crate::types::Seq<'_>,
    s2: &crate::types::Seq<'_>,
    score_cutoff: Option<f64>,
) -> f64 {
    use crate::types::ScorerType::*;
    match stype {
        Ratio => norm_sim_to_score(score_cutoff_check(indel_normalized_sim(s1, s2, score_cutoff), score_cutoff)),
        _ => 0.0 // Placeholder
    }
}
"""
    with open("src/fuzz.rs", "a") as f:
        f.write(raw_eval)

print("Applied fixes")
