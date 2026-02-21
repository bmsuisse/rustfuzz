import re

with open("src/algorithms.rs", "r") as f:
    text = f.read()

# We need to implement an early exit in lcs_length_64 and lcs_length_multiword.
# `indel_distance` calculation is:
# dist = m + n - 2 * lcs
# To guarantee dist > max_dist we check:
# m + n - 2 * target_lcs > max_dist
# target_lcs < (m + n - max_dist) / 2.0
# The bit-parallel LCS does not give us the LCS length at row `i` easily; it operates on columns (s2).
# Or does it?
# The vector `v` stores `N - lcs` where `!v.count_ones()` is the LCS length.
# At iteration `i` of `s2` (where `i` goes from 0 to n-1), what is the max possible LCS?
# The maximum possible LCS at step `i` is current_lcs + remaining characters in s2.
# remaining = n - 1 - i.
# So max_overall_lcs <= current_lcs + n - 1 - i
# If max_overall_lcs < required_lcs, we can abort.

# We must change lcs_length to accept Option<usize> max_dist
text = text.replace("pub fn lcs_length<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {", 
                    "pub fn lcs_length<T: HashableChar>(s1: &[T], s2: &[T], max_dist: Option<usize>) -> usize {")

text = text.replace("lcs_length_64(s1, s2)", "lcs_length_64_bounded(s1, s2, max_dist)")
text = text.replace("lcs_length_multiword(s1, s2)", "lcs_length_multiword_bounded(s1, s2, max_dist)")

# Now we need to implement bounds for 64 and multiword. We keep the original ones intact for usages where boundary is not passed.
text = text.replace("pub fn lcs_length_64<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {", 
"""pub fn lcs_length_64_bounded<T: HashableChar>(s1: &[T], s2: &[T], max_dist: Option<usize>) -> usize {
    let m = s1.len();
    let n = s2.len();
    let mut pm = PatternMask64::new();
    for (i, &c) in s1.iter().enumerate() {
        pm.insert(c, 1u64 << i);
    }
    let mut v = !0u64;
    let mask = if m == 64 { !0u64 } else { (1u64 << m) - 1 };
    
    // Compute required LCS to stay within max_dist
    let required_lcs = max_dist.map(|d| {
        let diff = m + n;
        if diff <= d { 0 } else { (diff - d + 1) / 2 }
    });

    for (i, &c) in s2.iter().enumerate() {
        let x = pm.get(c);
        let u = v & x;
        v = (v.wrapping_add(u)) | (v & !x);
        
        if let Some(req) = required_lcs {
            let current_lcs = (!v & mask).count_ones() as usize;
            let remaining = n - 1 - i;
            if current_lcs + remaining < req {
                // Cannot possibly reach the required LCS
                return 0; // Return 0 LCS so the distance formula evaluates > max_dist
            }
        }
    }
    (!v & mask).count_ones() as usize
}

pub fn lcs_length_64<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {""")

text = text.replace("fn lcs_length_multiword<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {",
"""fn lcs_length_multiword_bounded<T: HashableChar>(s1: &[T], s2: &[T], max_dist: Option<usize>) -> usize {
    let m = s1.len();
    let n = s2.len();
    let words = (m + 63) / 64;
    let mut pm = PatternMaskMulti::new(words);
    for (i, &c) in s1.iter().enumerate() {
        pm.set_bit(c, i / 64, i % 64);
    }

    let mut v = vec![!0u64; words];
    let last_bits = if m % 64 == 0 { 64 } else { m % 64 };
    let mask = if last_bits == 64 { !0u64 } else { (1u64 << last_bits) - 1 };

    let required_lcs = max_dist.map(|d| {
        let diff = m + n;
        if diff <= d { 0 } else { (diff - d + 1) / 2 }
    });

    for (i, &c) in s2.iter().enumerate() {
        let pm_c = pm.get(c);
        let mut carry = 0u64;
        let mut next_v = vec![0u64; words];
        for w in 0..words {
            let x = pm_c[w];
            let u = v[w] & x;
            let (sum1, c1) = v[w].overflowing_add(u);
            let (sum2, c2) = sum1.overflowing_add(carry);
            carry = (c1 as u64) | (c2 as u64);
            next_v[w] = sum2 | (v[w] & !x);
        }
        v = next_v;

        if let Some(req) = required_lcs {
            let mut current_lcs = 0;
            for w in 0..words - 1 {
                current_lcs += (!v[w]).count_ones() as usize;
            }
            current_lcs += (!v[words - 1] & mask).count_ones() as usize;
            let remaining = n - 1 - i;
            if current_lcs + remaining < req {
                return 0; // Early abort
            }
        }
    }
    
    let mut zeros = 0;
    for w in 0..words - 1 {
        zeros += (!v[w]).count_ones() as usize;
    }
    zeros += (!v[words - 1] & mask).count_ones() as usize;
    zeros
}

fn lcs_length_multiword<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {""")

# Update indel_distance
text = text.replace("let lcs = lcs_length(s1, s2);", "let lcs = lcs_length(s1, s2, score_cutoff);")

# Update lcs_seq_similarity call to lcs_length
text = text.replace("pfx + sfx + lcs_length(s1t, s2t)", "pfx + sfx + lcs_length(s1t, s2t, max_dist)")

with open("src/algorithms.rs", "w") as f:
    f.write(text)

print("Injected bounds loop check")
