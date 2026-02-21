import re

with open("src/distance/metrics.rs", "r") as f:
    text = f.read()

# Fix the internal macro to correctly pass variable arguments
orig_macro = """#[macro_export]
macro_rules! dispatch_metric {
    ($func:path, $s1:expr, $s2:expr $(, $args:expr)*) => {
        match ($s1, $s2) {
            (crate::types::Seq::Ascii(a), crate::types::Seq::Ascii(b)) => $func(*a, *b $(, $args)*),
            (crate::types::Seq::Ascii(a), crate::types::Seq::U32(b)) => $func(&a.iter().map(|&x| x as u32).collect::<Vec<_>>(), b $(, $args)*),
            (crate::types::Seq::Ascii(a), crate::types::Seq::U64(b)) => $func(&a.iter().map(|&x| x as u64).collect::<Vec<_>>(), b $(, $args)*),
            (crate::types::Seq::U32(a), crate::types::Seq::Ascii(b)) => $func(a, &b.iter().map(|&x| x as u32).collect::<Vec<_>>() $(, $args)*),
            (crate::types::Seq::U32(a), crate::types::Seq::U32(b)) => $func(a, b $(, $args)*),
            (crate::types::Seq::U32(a), crate::types::Seq::U64(b)) => $func(&a.iter().map(|&x| x as u64).collect::<Vec<_>>(), b $(, $args)*),
            (crate::types::Seq::U64(a), crate::types::Seq::Ascii(b)) => $func(a, &b.iter().map(|&x| x as u64).collect::<Vec<_>>() $(, $args)*),
            (crate::types::Seq::U64(a), crate::types::Seq::U32(b)) => $func(a, &b.iter().map(|&x| x as u64).collect::<Vec<_>>() $(, $args)*),
            (crate::types::Seq::U64(a), crate::types::Seq::U64(b)) => $func(a, b $(, $args)*),
        }
    };
}"""

# The macro is fine, we need to fix the call sites!
# In distance/metrics.rs some call `dispatch_metric!(alg::lcs_seq_distance, &av, &bv)` without the cutoff.

text = text.replace("dispatch_metric!(alg::indel_distance, &av, &bv)", 
                    "dispatch_metric!(alg::indel_distance, &av, &bv, score_cutoff)")

# Update LCS distance and similarity which had signature changes
text = text.replace("dispatch_metric!(alg::lcs_seq_similarity, &av, &bv)",
                    "dispatch_metric!(alg::lcs_seq_similarity, &av, &bv, score_cutoff)")

text = text.replace("dispatch_metric!(alg::lcs_seq_distance, &av, &bv)",
                    "dispatch_metric!(alg::lcs_seq_distance, &av, &bv, score_cutoff)")

with open("src/distance/metrics.rs", "w") as f:
    f.write(text)

with open("src/fuzz.rs", "r") as f:
    text = f.read()

# Fix the fuzz.rs partial_ratio inner loops which call indel_distance explicitly
# fn score_fn = |shorter: &[u64], window: &[u64]| -> f64 {
#     let dist = alg::indel_distance(shorter, window);
text = text.replace("alg::indel_distance(shorter, window)", "alg::indel_distance(shorter, window, None)")

with open("src/fuzz.rs", "w") as f:
    f.write(text)

print("Injected metrics.rs and fuzz.rs calls")
