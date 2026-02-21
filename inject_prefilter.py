import re

with open("src/process.rs", "r") as f:
    text = f.read()

# We want to remove the Rayon parallel execution block.
# We'll use a regex to match from `// Parallel Fast Path` to its closing brace `return Ok(par_results); }`

pattern = r"    // Parallel Fast Path.*?return Ok\(par_results\);\n    \}"
text = re.sub(pattern, "", text, flags=re.DOTALL)

# Now, we inject the L1 bounds filter into into the main iteration loop directly in `execute_scorer`.
# OR we can inject a pre-filter right in the `extract` function to skip calling `execute_scorer` entirely!
# If we precompute `q_hist` (query histogram) ONCE, we can do $O(1)$ rejection.

prefilter = """
    // Pre-calculate L1 Character bounds for early rejection natively
    let mut q_hist = [0i32; 256];
    let mut q_is_ascii = false;
    let mut q_len = 0;
    
    if let Ok(q_seq) = crate::types::extract_single(&processed_query) {
        if let crate::types::Seq::Ascii(slice) = q_seq {
            q_is_ascii = true;
            q_len = slice.len();
            for &c in slice {
                q_hist[c as usize] += 1;
            }
        }
    }
"""

text = text.replace("    for (idx, choice_res) in iter.enumerate() {", prefilter + "    for (idx, choice_res) in iter.enumerate() {")

inner_loop = """
        let choice = choice_res?;

        // Fast L1 Bounds Rejection (Only for ASCII matching)
        if q_is_ascii && score_cutoff.is_some() {
            let cutoff = score_cutoff.unwrap();
            if let Ok(crate::types::Seq::Ascii(slice)) = crate::types::extract_single(&choice) {
                // Determine `max_dist` dynamically from `cutoff` assuming Ratio
                // length sum
                let lensum = q_len + slice.len();
                if lensum > 0 {
                    let mut max_dist = lensum as f64 * (1.0 - cutoff / 100.0);
                    if max_dist < 0.0 { max_dist = 0.0; }
                    let allowed_edits = max_dist.floor() as usize;

                    // Compute L1 diff
                    let mut hist_diff = 0;
                    let mut c_hist = [0i32; 256];
                    for &c in slice {
                        c_hist[c as usize] += 1;
                    }
                    for i in 0..256 {
                        hist_diff += (q_hist[i] - c_hist[i]).abs();
                    }
                    
                    // If the character disparity is strictly larger than total errors allowed,
                    // we can mathematically skip this string completely!
                    if (hist_diff as usize) > allowed_edits {
                        continue;
                    }
                }
            }
        }
        
        let score = execute_scorer("""

text = text.replace("""        let choice = choice_res?;
        
        let score = execute_scorer(""", inner_loop)

# Let's write the same for `extract_iter` but wait, `extract_iter` calls `extract`, so it's handled automatically!

with open("src/process.rs", "w") as f:
    f.write(text)

print("Injected native L1 Prefilter in loop")
