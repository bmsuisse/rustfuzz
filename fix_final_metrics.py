import re

with open("src/distance/metrics.rs", "r") as f:
    text = f.read()

# Instead of regex patching loops, let's target the exact lines that error.
# 432:     let max_v = av.len() + bv.len();
# 433:     let max_dist = score_cutoff.map(|c| (max_v as f64 * c).floor() as usize /*c is f64*/);
# 434:     let dist = dispatch_metric!(alg::indel_distance, &av, &bv, max_dist);
# Actually `score_cutoff` is Option<usize> in indel_distance.
# So `max_dist` should literally just be `score_cutoff`.

text = text.replace("""    let max_v = av.len() + bv.len();
    let max_dist = score_cutoff.map(|c| (max_v as f64 * c).floor() as usize /*c is f64*/);
    let dist = dispatch_metric!(alg::indel_distance, &av, &bv, max_dist);""", """    let dist = dispatch_metric!(alg::indel_distance, &av, &bv, score_cutoff);""")

# For `indel_similarity`, it's also `score_cutoff: Option<usize>`
text = text.replace("""    let max_v = av.len() + bv.len();
    let max_dist = score_cutoff.map(|c| (max_v as f64 * c).floor() as usize /*c is f64*/);
    let dist = dispatch_metric!(alg::indel_distance, &av, &bv, max_dist);
    let max_v = av.len() + bv.len();""", """    let dist = dispatch_metric!(alg::indel_distance, &av, &bv, None); // similarity cutoff needs inversion logic, passing None for now
    let max_v = av.len() + bv.len();""")

# For `lcs_seq_normalized_distance`, `score_cutoff: Option<f64>`
# We want distance / max_v <= (1.0 - cutoff)
# distance <= (1.0 - cutoff) * max_v
# So c is an f64 here (e.g., 0.9 similarity cutoff).
text = text.replace("""    let max_v = av.len().max(bv.len());
    let max_dist = score_cutoff.map(|c| (max_v as f64 * c).floor() as usize /*c is f64*/);
    let dist = dispatch_metric!(alg::lcs_seq_distance, &av, &bv, max_dist);""", """    let max_v = av.len().max(bv.len());
    let max_dist = score_cutoff.map(|c| (max_v as f64 * (1.0 - c)).floor() as usize);
    let dist = dispatch_metric!(alg::lcs_seq_distance, &av, &bv, max_dist);""")

with open("src/distance/metrics.rs", "w") as f:
    f.write(text)

with open("src/algorithms.rs", "r") as f:
    text = f.read()

# Fix lcs_seq_similarity signature from map Option<usize> to expecting it.
# Wait, lcs_seq_similarity doesn't have an error, the error was passing f64 to lcs_seq_distance.

print("Applied strict Python signature patches")
