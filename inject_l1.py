import re

# Since Rayon multi-threading would require copying memory strings out of the GIL (sacrificing the incredible 
# zero-copy speed we just gained), the only way to beat RapidFuzz on a single core is algorithmic pre-filtering.

# We will build a character frequency histogram $L_1$ filter in `indel_distance`.
# If `s1` has 5 'A's and 0 'B's, and `s2` has 0 'A's and 5 'B's, 
# The minimum possible edit distance is 10 (5 deletes + 5 inserts).
# Calculate histogram diff: sum(abs(count_s1[c] - count_s2[c])).
# If this diff > max_distance, we can return usize::MAX immediately!

# To do this efficiently, we can build a `[u16; 256]` or similar array for ASCII,
# and a `HashMap` or sorted vector for U32/unicode.

with open("src/algorithms.rs", "r") as f:
    text = f.read()

# E.g. inside `indel_distance` before even calling `common_prefix`
# We can do an ASCII L1 bound check.
text = text.replace("""    // If the minimum possible distance (length difference) is already greater than cutoff, abort.
    let len_diff = s1.len().abs_diff(s2.len());
    if let Some(cutoff) = score_cutoff {
        if len_diff > cutoff {
            return usize::MAX;
        }
    }""", """    // If the minimum possible distance (length difference) is already greater than cutoff, abort.
    let len_diff = s1.len().abs_diff(s2.len());
    if let Some(cutoff) = score_cutoff {
        if len_diff > cutoff {
            return usize::MAX;
        }
        
        // Fast Histogram L1 Pre-Filter
        // Minimum edits = sum(abs(count(s1, c) - count(s2, c)))
        // We only do this fast path if T happens to be u8 (ASCII) which we can cast via Any trick,
        // OR we can just use trait `Into<u32>` to index a 256-bin array safely.
        let mut hist1 = [0i32; 256];
        let mut hist2 = [0i32; 256];
        let mut can_use_hist = true;
        
        for &c in s1 {
            let u: u32 = c.into();
            if u < 256 {
                hist1[u as usize] += 1;
            } else {
                can_use_hist = false;
                break;
            }
        }
        if can_use_hist {
            for &c in s2 {
                let u: u32 = c.into();
                if u < 256 {
                    hist2[u as usize] += 1;
                } else {
                    can_use_hist = false;
                    break;
                }
            }
        }
        
        if can_use_hist {
            let mut l1_dist = 0;
            for i in 0..256 {
                l1_dist += (hist1[i] - hist2[i]).abs();
            }
            if (l1_dist as usize) > cutoff {
                return usize::MAX; // Statistically impossible to meet threshold
            }
        }
    }""")

with open("src/algorithms.rs", "w") as f:
    f.write(text)

print("Injected L1 bounds")
