import re

with open("src/distance/metrics.rs", "r") as f:
    text = f.read()

# Fix the casting error where we have:
# c | (max_v as f64 * c).floor() as usize
# `c` is already `usize` in indel_distance and indel_similarity because score_cutoff is Option<usize>!
# So we shouldn't be multiplying by a percentage anyway, the cutoff IS the max distance!

# Let's inspect the exact lines:
# 433: let max_dist = score_cutoff.map(|c| (max_v as f64 * c).floor() as usize);
# But wait, in indel_distance, `score_cutoff` is Option<usize>. 
# If it's `Option<usize>`, `c` is `usize`. We can just pass `score_cutoff` directly as `max_dist`!

# So:
fix1 = """    let max_v = av.len() + bv.len();
    let max_dist = score_cutoff.map(|c| (max_v as f64 * c).floor() as usize);
    let dist = dispatch_metric!(alg::indel_distance, &av, &bv, max_dist);"""
rep1 = """    let dist = dispatch_metric!(alg::indel_distance, &av, &bv, score_cutoff);"""
text = text.replace(fix1, rep1)

# Fix normalized functions which DO take Option<f64>
fix2 = """    let max_dist = score_cutoff.map(|c| (max_v as f64 * c).floor() as usize);"""
rep2 = """    let max_dist = score_cutoff.map(|c| (max_v as f64 * (1.0 - c)).floor() as usize);"""
# But earlier we replaced it blindly. Let's fix it properly using regex for `as f64 * c` to `as f64 * (1.0 - c)`
# Wait, for `c` being f64 similarity constraint: 
# normalized_sim >= c
# 1 - dist / max >= c
# dist / max <= 1 - c
# dist <= max * (1 - c)
text = text.replace("(max_v as f64 * c).floor() as usize", "(max_v as f64 * c).floor() as usize /*c is f64*/")
# Actually, the error `cannot multiply f64 by usize` means `c` was `usize`. 
# So `c` was the usize distance cutoff.

with open("src/distance/metrics.rs", "w") as f:
    f.write(text)

print("Fixed")
