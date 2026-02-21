import re

with open("src/fuzz.rs", "r") as f:
    text = f.read()

# Update indel_normalized_sim to take Option<f64> cutoff derived constraint
orig_indel = """fn indel_normalized_sim(av: &crate::types::Seq<'_>, bv: &crate::types::Seq<'_>) -> f64 {
    let lensum = av.len() + bv.len();
    if lensum == 0 {
        return 1.0;
    }
    let dist = dispatch_metric!(alg::indel_distance, av, bv);
    1.0 - (dist as f64 / lensum as f64)
}"""

new_indel = """fn indel_normalized_sim(av: &crate::types::Seq<'_>, bv: &crate::types::Seq<'_>, score_cutoff: Option<f64>) -> f64 {
    let lensum = av.len() + bv.len();
    if lensum == 0 {
        return 1.0;
    }
    
    let max_dist = score_cutoff.map(|mut c| {
        // e.g. cutoff = 90.0
        // max_dist = lensum - (cutoff / 100.0 * lensum)
        if c > 100.0 { c = 100.0; }
        let allowed = lensum as f64 * (1.0 - c / 100.0);
        // We can safely floor or ceil depending on bounds.
        // We floor it (or cast to usize directly which truncates).
        allowed.floor() as usize
    });
    
    // dispatch_metric passes exactly args. 
    // We updated `inject_bounds.py` to allow passing score_cutoff to `indel_distance`.
    let dist = dispatch_metric!(alg::indel_distance, av, bv, max_dist);
    
    if dist == usize::MAX {
        return 0.0; // Early exit cutoff fallback
    }
    
    1.0 - (dist as f64 / lensum as f64)
}"""

text = text.replace(orig_indel, new_indel)

# Update calls to indel_normalized_sim(..., score_cutoff)
text = text.replace("indel_normalized_sim(&crate::types::Seq::U64(av), &crate::types::Seq::U64(bv))",
                    "indel_normalized_sim(&crate::types::Seq::U64(av), &crate::types::Seq::U64(bv), None)")

text = text.replace("let score = norm_sim_to_score(indel_normalized_sim(&av, &bv));",
                    "let score = norm_sim_to_score(indel_normalized_sim(&av, &bv, score_cutoff));")

text = text.replace("let base = norm_sim_to_score(indel_normalized_sim(&av, &bv));",
                    "let base = norm_sim_to_score(indel_normalized_sim(&av, &bv, score_cutoff));")

with open("src/fuzz.rs", "w") as f:
    f.write(text)

print("Injected fuzz.rs bounds")
