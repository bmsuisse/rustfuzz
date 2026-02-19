// SPDX-License-Identifier: MIT
// Pure Rust algorithm implementations for all distance metrics.
// Performance: Myers' bit-parallel for Levenshtein/Indel (single + multi-word),
//              standard O(NM) for Jaro, OSA, DamerauLevenshtein.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: common prefix/suffix lengths
// ---------------------------------------------------------------------------
pub fn common_prefix(s1: &[i64], s2: &[i64]) -> usize {
    s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count()
}

pub fn common_suffix(s1: &[i64], s2: &[i64]) -> usize {
    s1.iter()
        .rev()
        .zip(s2.iter().rev())
        .take_while(|(a, b)| a == b)
        .count()
}

// ===========================================================================
// LEVENSHTEIN (uniform weights 1,1,1)
// Myers' bit-parallel algorithm, handles any length.
// ===========================================================================

/// Single-word Myers for len(s1) <= 64
fn myers_64(s1: &[i64], s2: &[i64]) -> usize {
    let len1 = s1.len();
    // Build pattern bitmask
    let mut pm: HashMap<i64, u64> = HashMap::new();
    for (i, &c) in s1.iter().enumerate() {
        *pm.entry(c).or_insert(0) |= 1u64 << i;
    }

    let mut vp: u64 = u64::MAX; // all ones
    let mut vn: u64 = 0;
    let mut dist = len1;
    let mask: u64 = 1u64 << (len1 - 1);

    for &c in s2 {
        let pm_j = pm.get(&c).copied().unwrap_or(0);
        let x = pm_j | vn;
        let d0 = (((pm_j & vp).wrapping_add(vp)) ^ vp) | x;
        let hp = vn | !(d0 | vp);
        let hn = d0 & vp;
        if hp & mask != 0 {
            dist += 1;
        }
        if hn & mask != 0 {
            dist -= 1;
        }
        let hp = (hp << 1) | 1;
        let hn = hn << 1;
        vp = hn | !(d0 | hp);
        vn = hp & d0;
    }
    dist
}

/// Multi-word Myers for len(s1) > 64, exactly mirrors myers_64's computation pattern.
fn myers_multiword(s1: &[i64], s2: &[i64]) -> usize {
    let len1 = s1.len();
    let words = (len1 + 63) / 64;

    // Build pattern bitmasks per word (keyed by character)
    let mut pm: HashMap<i64, Vec<u64>> = HashMap::new();
    for (i, &c) in s1.iter().enumerate() {
        let entry = pm.entry(c).or_insert_with(|| vec![0u64; words]);
        entry[i / 64] |= 1u64 << (i % 64);
    }

    let mut vp: Vec<u64> = vec![u64::MAX; words];
    let mut vn: Vec<u64> = vec![0u64; words];
    let mut dist = len1;

    // last_bits: number of valid bits in the last word
    let last_bits = if len1 % 64 == 0 { 64 } else { len1 % 64 };
    let last_valid_mask: u64 = if last_bits == 64 { u64::MAX } else { (1u64 << last_bits) - 1 };
    let last_mask: u64 = 1u64 << (last_bits - 1);
    // Mask last word of initial VP to valid bits only
    vp[words - 1] = last_valid_mask;

    for &c in s2 {
        let pm_c = pm.get(&c);
        // add_carry: carry for the (PM & VP) + VP addition, propagates low→high across words
        let mut add_carry: u64 = 0;
        // hp_carry: lowest bit shifted into HP (starts at 1 = X_{-1} in Myers' notation)
        let mut hp_carry: u64 = 1;
        // hn_carry: MSB of HN from previous word carried into next word's HN shift
        let mut hn_carry: u64 = 0;

        // Use separate arrays so reading vp[w]/vn[w] always gives old values
        let mut new_vp = vec![0u64; words];
        let mut new_vn = vec![0u64; words];

        for w in 0..words {
            let pm_j = pm_c.map(|p| p[w]).unwrap_or(0);
            let pv = vp[w]; // old VP for this word
            let nv = vn[w]; // old VN for this word

            // D0 = (((PM & VP) + VP) ^ VP) | (PM | VN)
            // Exactly mirrors myers_64: d0 = (((pm_j & vp) + vp) ^ vp) | (pm_j | vn)
            let pm_and_vp = pm_j & pv;
            let (t, c1) = pm_and_vp.overflowing_add(add_carry);
            let (sum, c2) = t.overflowing_add(pv);
            add_carry = (c1 as u64) | (c2 as u64);
            let d0 = (sum ^ pv) | (pm_j | nv);

            let hp = nv | !(d0 | pv);
            let hn = d0 & pv;

            // Check distance from HP/HN BEFORE shifting (like myers_64)
            if w == words - 1 {
                // Must mask hp/hn to valid bits first to avoid garbage from NOT
                let hp_valid = hp & last_valid_mask;
                let hn_valid = hn & last_valid_mask;
                if hp_valid & last_mask != 0 { dist += 1; }
                if hn_valid & last_mask != 0 { dist -= 1; }
            }

            // Shift HP/HN left, carrying MSB of each into the next word's bit 0
            let hp_shifted = (hp << 1) | hp_carry;
            let hn_shifted = (hn << 1) | hn_carry;
            new_vp[w] = hn_shifted | !(d0 | hp_shifted);
            new_vn[w] = hp_shifted & d0;
            hp_carry = hp >> 63;
            hn_carry = hn >> 63;
        }

        // Mask last word to valid bits (NOT produces garbage in upper bits)
        let lw = words - 1;
        new_vp[lw] &= last_valid_mask;
        new_vn[lw] &= last_valid_mask;

        vp = new_vp;
        vn = new_vn;
    }
    dist
}

pub fn levenshtein_uniform(s1: &[i64], s2: &[i64]) -> usize {
    if s1.is_empty() {
        return s2.len();
    }
    if s2.is_empty() {
        return s1.len();
    }
    // Ensure s1 is the shorter one for Myers
    let (s1, s2) = if s1.len() <= s2.len() {
        (s1, s2)
    } else {
        (s2, s1)
    };
    if s1.len() <= 64 {
        myers_64(s1, s2)
    } else {
        myers_multiword(s1, s2)
    }
}

/// Generic Levenshtein with custom weights (insert, delete, replace)
pub fn levenshtein_generic(
    s1: &[i64],
    s2: &[i64],
    ins_cost: usize,
    del_cost: usize,
    rep_cost: usize,
) -> usize {
    if s1.is_empty() {
        return s2.len() * ins_cost;
    }
    if s2.is_empty() {
        return s1.len() * del_cost;
    }
    let m = s1.len();
    let n = s2.len();
    let mut prev: Vec<usize> = (0..=n).map(|j| j * ins_cost).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for i in 0..m {
        curr[0] = (i + 1) * del_cost;
        for j in 0..n {
            let cost = if s1[i] == s2[j] { 0 } else { rep_cost };
            curr[j + 1] = (curr[j] + ins_cost)
                .min(prev[j + 1] + del_cost)
                .min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

pub fn levenshtein(s1: &[i64], s2: &[i64], weights: (usize, usize, usize)) -> usize {
    let (ins, del, rep) = weights;
    if ins == 1 && del == 1 && rep == 1 {
        // Strip common prefix/suffix first for speed
        let pfx = common_prefix(s1, s2);
        let s1 = &s1[pfx..];
        let s2 = &s2[pfx..];
        let sfx = common_suffix(s1, s2);
        let s1 = &s1[..s1.len() - sfx];
        let s2 = &s2[..s2.len() - sfx];
        levenshtein_uniform(s1, s2)
    } else {
        levenshtein_generic(s1, s2, ins, del, rep)
    }
}

/// Build editops using Python's Hyyrö VP/VN bit-parallel algorithm for exact tie-breaking match.
/// For m <= 127, uses u128 bit-parallel (matches Python's arbitrary-precision integers).
/// For m > 127, falls back to standard DP with delete-before-replace priority.
pub fn levenshtein_editops_trace(s1: &[i64], s2: &[i64]) -> Vec<(String, usize, usize)> {
    let full_m = s1.len();
    let full_n = s2.len();

    // Strip common prefix/suffix (like Python's common_affix)
    let pfx = s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count();
    let sfx = s1[pfx..].iter().rev().zip(s2[pfx..].iter().rev())
        .take_while(|(a, b)| a == b).count();
    let end1 = full_m - sfx;
    let end2 = full_n - sfx;

    let s1 = &s1[pfx..end1];
    let s2 = &s2[pfx..end2];
    let m = s1.len();
    let n = s2.len();

    if m == 0 {
        return (0..n).map(|j| ("insert".to_string(), pfx, pfx + j)).collect();
    }
    if n == 0 {
        return (0..m).map(|i| ("delete".to_string(), pfx + i, pfx)).collect();
    }

    if m <= 127 {
        levenshtein_editops_bitvec(s1, s2, pfx)
    } else {
        levenshtein_editops_multiword(s1, s2, pfx)
    }
}

/// Hyyrö bit-parallel VP/VN algorithm for m <= 127.
/// Exactly matches Python's _matrix + editops backtracking.
fn levenshtein_editops_bitvec(s1: &[i64], s2: &[i64], pfx: usize) -> Vec<(String, usize, usize)> {
    let m = s1.len();
    let n = s2.len();

    // Build PM (pattern match bitmask) for each unique char in s1
    // Using sorted Vec for cache-friendly lookup, much faster than HashMap for short strings
    let mut pm_pairs: Vec<(i64, u128)> = Vec::new();
    for (i, &ch) in s1.iter().enumerate() {
        match pm_pairs.binary_search_by_key(&ch, |&(c, _)| c) {
            Ok(pos) => pm_pairs[pos].1 |= 1u128 << i,
            Err(pos) => pm_pairs.insert(pos, (ch, 1u128 << i)),
        }
    }
    let pm_get = |ch: i64| -> u128 {
        match pm_pairs.binary_search_by_key(&ch, |&(c, _)| c) {
            Ok(pos) => pm_pairs[pos].1,
            Err(_) => 0,
        }
    };

    let mut vp: u128 = (1u128 << m) - 1; // all m bits set
    let mut vn: u128 = 0u128;
    let mask: u128 = 1u128 << (m - 1);

    let mut matrix_vp: Vec<u128> = Vec::with_capacity(n);
    let mut matrix_vn: Vec<u128> = Vec::with_capacity(n);

    for &ch2 in s2 {
        let pm_j = pm_get(ch2);
        let x = pm_j;
        let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x | vn;
        let hp = vn | !(d0 | vp);
        let hn = d0 & vp;
        let _ = (hp & mask, hn & mask); // distance tracking (unused here)
        let hp_s = (hp << 1) | 1;
        let hn_s = hn << 1;
        vp = hn_s | !(d0 | hp_s);
        vn = hp_s & d0;
        matrix_vp.push(vp);
        matrix_vn.push(vn);
    }

    // Backtrack exactly like Python's editops:
    // VP[row-1] & (1<<(col-1)) → DELETE
    // else: row--; VN[row-1] & (1<<(col-1)) → INSERT; else → MATCH/REPLACE (col--)
    let mut ops: Vec<(String, usize, usize)> = Vec::new();
    let mut col = m;
    let mut row = n;

    while row > 0 && col > 0 {
        if matrix_vp[row - 1] & (1u128 << (col - 1)) != 0 {
            ops.push(("delete".to_string(), pfx + col - 1, pfx + row));
            col -= 1;
        } else {
            row -= 1;
            if row > 0 && matrix_vn[row - 1] & (1u128 << (col - 1)) != 0 {
                ops.push(("insert".to_string(), pfx + col, pfx + row));
            } else {
                col -= 1;
                if s1[col] != s2[row] {
                    ops.push(("replace".to_string(), pfx + col, pfx + row));
                }
            }
        }
    }
    while col > 0 { col -= 1; ops.push(("delete".to_string(), pfx + col, pfx)); }
    while row > 0 { row -= 1; ops.push(("insert".to_string(), pfx, pfx + row)); }
    ops.reverse();
    ops
}

/// Multiword VP/VN bit-parallel editops for m > 127.
/// Stores VP and VN for every column j, then backtracks exactly like levenshtein_editops_bitvec.
/// This matches Python's arbitrary-precision integer approach for all string lengths.
fn levenshtein_editops_multiword(s1: &[i64], s2: &[i64], pfx: usize) -> Vec<(String, usize, usize)> {
    let m = s1.len();
    let n = s2.len();
    let words = (m + 63) / 64;
    let last_bits = if m % 64 == 0 { 64 } else { m % 64 };
    let last_valid_mask: u64 = if last_bits == 64 { u64::MAX } else { (1u64 << last_bits) - 1 };

    // Build pattern bitmasks
    let mut pm: HashMap<i64, Vec<u64>> = HashMap::new();
    for (i, &c) in s1.iter().enumerate() {
        let entry = pm.entry(c).or_insert_with(|| vec![0u64; words]);
        entry[i / 64] |= 1u64 << (i % 64);
    }

    let mut vp: Vec<u64> = vec![u64::MAX; words];
    let mut vn: Vec<u64> = vec![0u64; words];
    vp[words - 1] = last_valid_mask;

    // Store VP/VN for every column (each is a Vec<u64> of `words` words)
    let mut matrix_vp: Vec<Vec<u64>> = Vec::with_capacity(n);
    let mut matrix_vn: Vec<Vec<u64>> = Vec::with_capacity(n);

    for &c in s2 {
        let pm_c = pm.get(&c);
        let mut add_carry: u64 = 0;
        let mut hp_carry: u64 = 1;
        let mut hn_carry: u64 = 0;
        let mut new_vp = vec![0u64; words];
        let mut new_vn = vec![0u64; words];

        for w in 0..words {
            let pm_j = pm_c.map(|p| p[w]).unwrap_or(0);
            let pv = vp[w];
            let nv = vn[w];
            let pm_and_vp = pm_j & pv;
            let (t, c1) = pm_and_vp.overflowing_add(add_carry);
            let (sum, c2) = t.overflowing_add(pv);
            add_carry = (c1 as u64) | (c2 as u64);
            let d0 = (sum ^ pv) | (pm_j | nv);
            let hp = nv | !(d0 | pv);
            let hn = d0 & pv;
            let hp_shifted = (hp << 1) | hp_carry;
            let hn_shifted = (hn << 1) | hn_carry;
            new_vp[w] = hn_shifted | !(d0 | hp_shifted);
            new_vn[w] = hp_shifted & d0;
            hp_carry = hp >> 63;
            hn_carry = hn >> 63;
        }
        new_vp[words - 1] &= last_valid_mask;
        new_vn[words - 1] &= last_valid_mask;
        vp = new_vp.clone();
        vn = new_vn.clone();
        matrix_vp.push(new_vp);
        matrix_vn.push(new_vn);
    }

    // Backtrack exactly like levenshtein_editops_bitvec but with multiword VP/VN
    let mut ops: Vec<(String, usize, usize)> = Vec::new();
    let mut col = m; // index into s1 (1-based)
    let mut row = n; // index into s2 (1-based)

    while row > 0 && col > 0 {
        let word = (col - 1) / 64;
        let bit = (col - 1) % 64;
        if matrix_vp[row - 1][word] & (1u64 << bit) != 0 {
            ops.push(("delete".to_string(), pfx + col - 1, pfx + row));
            col -= 1;
        } else {
            row -= 1;
            if row > 0 && matrix_vn[row - 1][word] & (1u64 << bit) != 0 {
                ops.push(("insert".to_string(), pfx + col, pfx + row));
            } else {
                col -= 1;
                if s1[col] != s2[row] {
                    ops.push(("replace".to_string(), pfx + col, pfx + row));
                }
            }
        }
    }
    while col > 0 { col -= 1; ops.push(("delete".to_string(), pfx + col, pfx)); }
    while row > 0 { row -= 1; ops.push(("insert".to_string(), pfx, pfx + row)); }
    ops.reverse();
    ops
}

// ===========================================================================
// INDEL (insert/delete only, equivalent to LCS-based distance)
// ===========================================================================

/// LCS length via bit-parallel algorithm (Hyyro)
fn lcs_length_bitvector(s1: &[i64], s2: &[i64]) -> usize {
    if s1.is_empty() || s2.is_empty() {
        return 0;
    }
    let len1 = s1.len();
    let words = (len1 + 63) / 64;

    let mut pm: HashMap<i64, Vec<u64>> = HashMap::new();
    for (i, &c) in s1.iter().enumerate() {
        let entry = pm.entry(c).or_insert_with(|| vec![0u64; words]);
        entry[i / 64] |= 1u64 << (i % 64);
    }

    let mut m: Vec<u64> = vec![0u64; words];
    let mut lcs = 0usize;

    for &c in s2 {
        let pm_c = pm.get(&c);
        let mut carry = 0u64;
        for w in 0..words {
            let x = m[w] | pm_c.map(|p| p[w]).unwrap_or(0);
            let y = m[w] & pm_c.map(|p| p[w]).unwrap_or(0);
            let (t, c2) = y.overflowing_add(m[w]);
            let (t, c3) = t.overflowing_add(carry);
            carry = if c2 || c3 { 1 } else { 0 };
            m[w] = x & !(t ^ m[w]);
        }
        lcs += m.iter().map(|w| w.count_ones() as usize).sum::<usize>()
            - m[..words - 1]
                .iter()
                .map(|w| w.count_ones() as usize)
                .sum::<usize>()
            - m[words - 1].count_ones() as usize;
    }

    // Simpler: just count set bits in final m
    m.iter().map(|w| w.count_ones() as usize).sum()
}

/// LCS length using simple DP (fallback, always correct)
pub fn lcs_length(s1: &[i64], s2: &[i64]) -> usize {
    if s1.is_empty() || s2.is_empty() {
        return 0;
    }
    let m = s1.len();
    let n = s2.len();
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];
    for i in 0..m {
        for j in 0..n {
            curr[j + 1] = if s1[i] == s2[j] {
                prev[j] + 1
            } else {
                prev[j + 1].max(curr[j])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.iter_mut().for_each(|x| *x = 0);
    }
    prev[n]
}

pub fn indel_distance(s1: &[i64], s2: &[i64]) -> usize {
    let pfx = common_prefix(s1, s2);
    let s1 = &s1[pfx..];
    let s2 = &s2[pfx..];
    let sfx = common_suffix(s1, s2);
    let s1 = &s1[..s1.len() - sfx];
    let s2 = &s2[..s2.len() - sfx];
    let lcs = lcs_length(s1, s2);
    s1.len() + s2.len() - 2 * lcs
}

/// Build indel editops via LCS backtrack
pub fn indel_editops_trace(s1: &[i64], s2: &[i64]) -> Vec<(String, usize, usize)> {
    let full_m = s1.len();
    let full_n = s2.len();

    // Strip common prefix/suffix (like Python's common_affix) to match Python's tie-breaking
    let pfx = s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count();
    let sfx = s1[pfx..].iter().rev().zip(s2[pfx..].iter().rev()).take_while(|(a, b)| a == b).count();

    let s1 = &s1[pfx..full_m - sfx];
    let s2 = &s2[pfx..full_n - sfx];

    let m = s1.len();
    let n = s2.len();

    if m == 0 {
        return (0..n).map(|j| ("insert".to_string(), pfx, pfx + j)).collect();
    }
    if n == 0 {
        return (0..m).map(|i| ("delete".to_string(), pfx + i, pfx)).collect();
    }

    // Build LCS DP table
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if s1[i - 1] == s2[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    // Backtrack exactly matching Python's Hyyrö LCS bit-parallel backtracking.
    // Python uses `matrix` where bit k=0 (NOT set) means s1[k] IS in LCS.
    // Translated to standard LCS DP:
    //   `matrix[j] & (1<<i) != 0` ↔ dp[i+1][j+1] == dp[i][j+1]  (s1[i] NOT in LCS = bit set)
    //   `matrix[j] & (1<<i) == 0` ↔ dp[i+1][j+1] > dp[i][j+1]  (s1[i] IS in LCS = bit unset)
    let mut ops: Vec<(String, usize, usize)> = Vec::new();
    let mut i = m; // col in Python (s1 position)
    let mut j = n; // row in Python (s2 position)
    while i > 0 && j > 0 {
        // Python: `if matrix[row-1] & (1<<(col-1))` = bit set = dp[i][j] == dp[i-1][j] → delete
        if dp[i][j] == dp[i - 1][j] {
            ops.push(("delete".to_string(), pfx + i - 1, pfx + j));
            i -= 1;
        } else {
            j -= 1; // Python: row -= 1
            // Python: `if row and not (matrix[row-1] & bit)`
            // `not (matrix[row-1] & bit)` ↔ dp[i][j] > dp[i-1][j] (bit unset = s1[i-1] in LCS)
            // But wait: after decrement, is this INSERT or MATCH?
            // If bit NOT set (dp[i][j] > dp[i-1][j]): s1[i-1] IS in LCS at new j → INSERT (old j+1 was skipped)
            // If bit IS set or j==0 (dp[i][j] == dp[i-1][j]): → MATCH diagonal (col--, i--)
            if j > 0 && dp[i][j] > dp[i - 1][j] {
                // bit NOT set: s1[i-1] IS in LCS at j → INSERT s2[j] (0-indexed)
                ops.push(("insert".to_string(), pfx + i, pfx + j));
            } else {
                // bit IS set or j==0: MATCH, decrement i
                i -= 1;
            }
        }
    }
    // Handle remaining characters
    while i > 0 {
        ops.push(("delete".to_string(), pfx + i - 1, pfx));
        i -= 1;
    }
    while j > 0 {
        j -= 1;
        ops.push(("insert".to_string(), pfx, pfx + j));
    }
    ops.reverse();
    ops
}



// ===========================================================================
// HAMMING
// ===========================================================================

pub fn hamming_distance(s1: &[i64], s2: &[i64], pad: bool) -> usize {
    let max_len = s1.len().max(s2.len());
    let min_len = s1.len().min(s2.len());
    let mut dist = 0usize;
    for i in 0..min_len {
        if s1[i] != s2[i] {
            dist += 1;
        }
    }
    if pad {
        dist += max_len - min_len;
    } else if s1.len() != s2.len() {
        // Without pad, unequal lengths are an error in most implementations
        // return max to signal error; callers handle the TypeError
        return usize::MAX;
    }
    dist
}

pub fn hamming_editops_trace(s1: &[i64], s2: &[i64]) -> Vec<(String, usize, usize)> {
    let min_len = s1.len().min(s2.len());
    let mut ops = Vec::new();
    for i in 0..min_len {
        if s1[i] != s2[i] {
            ops.push(("replace".to_string(), i, i));
        }
    }
    // Extra characters from longer string
    if s1.len() > s2.len() {
        for i in min_len..s1.len() {
            ops.push(("delete".to_string(), i, s2.len()));
        }
    } else if s2.len() > s1.len() {
        for j in min_len..s2.len() {
            ops.push(("insert".to_string(), s1.len(), j));
        }
    }
    ops
}

// ===========================================================================
// JARO
// ===========================================================================

pub fn jaro(s1: &[i64], s2: &[i64]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let match_dist = (len1.max(len2) / 2).saturating_sub(1);

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];
    let mut matches = 0usize;
    let mut transpositions = 0usize;

    for i in 0..len1 {
        let start = i.saturating_sub(match_dist);
        let end = (i + match_dist + 1).min(len2);
        for j in start..end {
            if s2_matches[j] || s1[i] != s2[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while !s2_matches[k] {
            k += 1;
        }
        if s1[i] != s2[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    let t = (transpositions / 2) as f64;
    (m / len1 as f64 + m / len2 as f64 + (m - t) / m) / 3.0
}

// ===========================================================================
// JARO-WINKLER
// ===========================================================================

pub fn jaro_winkler(s1: &[i64], s2: &[i64], prefix_weight: f64) -> f64 {
    let jaro_score = jaro(s1, s2);
    if jaro_score < 0.7 {
        return jaro_score;
    }
    let prefix_len = s1
        .iter()
        .zip(s2.iter())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count() as f64;
    (jaro_score + prefix_len * prefix_weight * (1.0 - jaro_score)).min(1.0)
}

// ===========================================================================
// LCSseq
// ===========================================================================

pub fn lcs_seq_similarity(s1: &[i64], s2: &[i64]) -> usize {
    let pfx = common_prefix(s1, s2);
    let s1t = &s1[pfx..];
    let s2t = &s2[pfx..];
    let sfx = common_suffix(s1t, s2t);
    let s1t = &s1t[..s1t.len() - sfx];
    let s2t = &s2t[..s2t.len() - sfx];
    pfx + sfx + lcs_length(s1t, s2t)
}

pub fn lcs_seq_distance(s1: &[i64], s2: &[i64]) -> usize {
    let sim = lcs_seq_similarity(s1, s2);
    s1.len().max(s2.len()) - sim
}

pub fn lcs_seq_editops_trace(s1: &[i64], s2: &[i64]) -> Vec<(String, usize, usize)> {
    // Indel editops but without replace (only insert/delete)
    indel_editops_trace(s1, s2)
}

// ===========================================================================
// OSA (Optimal String Alignment)
// ===========================================================================

pub fn osa_distance(s1: &[i64], s2: &[i64]) -> usize {
    let m = s1.len();
    let n = s2.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut d = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        d[i][0] = i;
    }
    for j in 0..=n {
        d[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if s1[i - 1] == s2[j - 1] { 0 } else { 1 };
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && s1[i - 1] == s2[j - 2] && s1[i - 2] == s2[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + cost);
            }
        }
    }
    d[m][n]
}

// ===========================================================================
// DAMERAU-LEVENSHTEIN (true, with transpositions)
// ===========================================================================

pub fn damerau_levenshtein_distance(s1: &[i64], s2: &[i64]) -> usize {
    let m = s1.len();
    let n = s2.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let max_dist = m + n;
    let mut d: HashMap<(i64, i64), usize> = HashMap::new();

    // Last occurrence of each character in s1
    let mut da: HashMap<i64, usize> = HashMap::new();

    // d[-1..-1] border = max_dist
    let mut dp = vec![vec![0usize; n + 2]; m + 2];
    dp[0][0] = max_dist;
    for i in 0..=m {
        dp[i + 1][0] = max_dist;
        dp[i + 1][1] = i;
    }
    for j in 0..=n {
        dp[0][j + 1] = max_dist;
        dp[1][j + 1] = j;
    }

    for i in 1..=m {
        let mut db = 0usize; // last j where s1[i-1] == s2[j-1]
        for j in 1..=n {
            let i1 = da.get(&s2[j - 1]).copied().unwrap_or(0);
            let j1 = db;
            let cost = if s1[i - 1] == s2[j - 1] {
                db = j;
                0
            } else {
                1
            };
            dp[i + 1][j + 1] = dp[i][j] + cost;
            dp[i + 1][j + 1] = dp[i + 1][j + 1].min(dp[i + 1][j] + 1);
            dp[i + 1][j + 1] = dp[i + 1][j + 1].min(dp[i][j + 1] + 1);
            if i1 > 0 && j1 > 0 {
                dp[i + 1][j + 1] = dp[i + 1][j + 1]
                    .min(dp[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1));
            }
        }
        da.insert(s1[i - 1], i);
    }
    dp[m + 1][n + 1]
}

// ===========================================================================
// PREFIX / POSTFIX
// ===========================================================================

pub fn prefix_similarity(s1: &[i64], s2: &[i64]) -> usize {
    s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count()
}

pub fn prefix_distance(s1: &[i64], s2: &[i64]) -> usize {
    let sim = prefix_similarity(s1, s2);
    s1.len().max(s2.len()) - sim
}

pub fn postfix_similarity(s1: &[i64], s2: &[i64]) -> usize {
    s1.iter()
        .rev()
        .zip(s2.iter().rev())
        .take_while(|(a, b)| a == b)
        .count()
}

pub fn postfix_distance(s1: &[i64], s2: &[i64]) -> usize {
    let sim = postfix_similarity(s1, s2);
    s1.len().max(s2.len()) - sim
}

// ===========================================================================
// Normalized helpers (0.0..=1.0)
// ===========================================================================

pub fn normalized_distance(dist: usize, max_dist: usize) -> f64 {
    if max_dist == 0 {
        0.0
    } else {
        dist as f64 / max_dist as f64
    }
}

pub fn normalized_similarity(dist: usize, max_dist: usize) -> f64 {
    1.0 - normalized_distance(dist, max_dist)
}
