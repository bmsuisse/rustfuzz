import re

with open("src/distance/metrics.rs", "r") as f:
    text = f.read()

# For normalized metrics, the Python signature takes `score_cutoff: Option<f64>`.
# But `alg::indel_distance` or `alg::lcs_seq_distance` expects `Option<usize>`.
#
# We need to map the f64 similarity cutoff -> max permitted usize distance.
# 1. indel_normalized_distance: max_v = len1 + len2. max_dist = (max_v as f64 * cutoff).floor()
# Wait, if cutoff is for distance, then distance must be <= cutoff.
# Normalized distance = dist / max. 
# If dist / max <= cutoff_f64  => dist <= max * cutoff_f64

fix_indel_nd = """    let dist = dispatch_metric!(alg::indel_distance, &av, &bv, score_cutoff);"""
rep_indel_nd = """    let max_v = av.len() + bv.len();
    let max_dist = score_cutoff.map(|c| (max_v as f64 * c).floor() as usize);
    let dist = dispatch_metric!(alg::indel_distance, &av, &bv, max_dist);"""

text = text.replace(fix_indel_nd, rep_indel_nd)

fix_lcs_nd = """    let dist = dispatch_metric!(alg::lcs_seq_distance, &av, &bv, score_cutoff);"""
rep_lcs_nd = """    let max_v = av.len().max(bv.len());
    let max_dist = score_cutoff.map(|c| (max_v as f64 * c).floor() as usize);
    let dist = dispatch_metric!(alg::lcs_seq_distance, &av, &bv, max_dist);"""

# Replace both lcs_seq_normalized_distance and lcs_seq_normalized_similarity
text = text.replace(fix_lcs_nd, rep_lcs_nd)

with open("src/distance/metrics.rs", "w") as f:
    f.write(text)

print("Fixed metrics type bounds")
