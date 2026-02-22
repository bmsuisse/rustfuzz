// SPDX-License-Identifier: MIT
// Pure Rust algorithm implementations for all distance metrics.
// Performance: Myers' bit-parallel for Levenshtein/Indel (single + multi-word),
//              standard O(NM) for Jaro, OSA, DamerauLevenshtein.

use rustc_hash::FxHashMap;

pub trait HashableChar: Copy + Eq + std::hash::Hash + Ord {
    fn as_usize(&self) -> Option<usize>;
}
impl HashableChar for u8 {
    #[inline] fn as_usize(&self) -> Option<usize> { Some(*self as usize) }
}
impl HashableChar for u32 {
    #[inline] fn as_usize(&self) -> Option<usize> { if *self < 256 { Some(*self as usize) } else { None } }
}
impl HashableChar for u64 {
    #[inline] fn as_usize(&self) -> Option<usize> { if *self < 256 { Some(*self as usize) } else { None } }
}



pub struct PatternMask64<T: HashableChar> {
    ascii: [u64; 256],
    fallback: FxHashMap<T, u64>,
}

impl<T: HashableChar> PatternMask64<T> {
    #[inline(always)]
    pub fn new() -> Self {
        PatternMask64 { ascii: [0; 256], fallback: FxHashMap::default() }
    }
    
    #[inline(always)]
    pub fn insert(&mut self, c: T, mask: u64) {
        if let Some(idx) = c.as_usize() {
            self.ascii[idx] |= mask;
        } else {
            *self.fallback.entry(c).or_insert(0) |= mask;
        }
    }
    
    #[inline(always)]
    pub fn get(&self, c: T) -> u64 {
        if let Some(idx) = c.as_usize() {
            self.ascii[idx]
        } else {
            self.fallback.get(&c).copied().unwrap_or(0)
        }
    }
}

pub struct PatternMaskMulti<T: HashableChar> {
    words: usize,
    ascii: Vec<u64>,
    fallback: FxHashMap<T, Vec<u64>>,
    zeros: Vec<u64>,
}

impl<T: HashableChar> PatternMaskMulti<T> {
    #[inline(always)]
    pub fn new(words: usize) -> Self {
        PatternMaskMulti { 
            words,
            ascii: vec![0u64; 256 * words],
            fallback: FxHashMap::default(),
            zeros: vec![0; words],
        }
    }
    
    #[inline(always)]
    pub fn set_bit(&mut self, c: T, word_idx: usize, bit_idx: usize) {
        if let Some(idx) = c.as_usize() {
            self.ascii[idx * self.words + word_idx] |= 1u64 << bit_idx;
        } else {
            let entry = self.fallback.entry(c).or_insert_with(|| vec![0u64; self.words]);
            entry[word_idx] |= 1u64 << bit_idx;
        }
    }
    
    #[inline(always)]
    pub fn get(&self, c: T) -> &[u64] {
        if let Some(idx) = c.as_usize() {
            &self.ascii[idx * self.words .. (idx + 1) * self.words]
        } else {
            self.fallback.get(&c).map(|v| v.as_slice()).unwrap_or(&self.zeros)
        }
    }
}

// Helper: common prefix/suffix lengths
// ---------------------------------------------------------------------------
#[inline]
pub fn common_prefix<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count()
}

#[inline]
pub fn common_suffix<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    s1.iter()
        .rev()
        .zip(s2.iter().rev())
        .take_while(|(a, b)| a == b)
        .count()
}

fn myers_64<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    let len1 = s1.len();
    // Build pattern bitmask
    let mut pm = PatternMask64::new();
    for (i, &c) in s1.iter().enumerate() {
        pm.insert(c, 1u64 << i);
    }

    let mut vp: u64 = u64::MAX; // all ones
    let mut vn: u64 = 0;
    let mut dist = len1;
    let mask: u64 = 1u64 << (len1 - 1);

    for &c in s2 {
        let pm_j = pm.get(c);
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

fn myers_multiword<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    let len1 = s1.len();
    let words = (len1 + 63) / 64;

    // Build pattern bitmasks per word (keyed by character)
    let mut pm = PatternMaskMulti::new(words);
    for (i, &c) in s1.iter().enumerate() {
        pm.set_bit(c, i / 64, i % 64);
    }

    let mut vp: Vec<u64> = vec![u64::MAX; words];
    let mut vn: Vec<u64> = vec![0u64; words];
    // Hoist scratch buffers outside the loop — allocating new_vp/new_vn per character
    // was O(len(s2)) heap allocations. Now we reuse and swap.
    let mut new_vp = vec![0u64; words];
    let mut new_vn = vec![0u64; words];
    let mut dist = len1;

    // last_bits: number of valid bits in the last word
    let last_bits = if len1 % 64 == 0 { 64 } else { len1 % 64 };
    let last_valid_mask: u64 = if last_bits == 64 { u64::MAX } else { (1u64 << last_bits) - 1 };
    let last_mask: u64 = 1u64 << (last_bits - 1);
    // Mask last word of initial VP to valid bits only
    vp[words - 1] = last_valid_mask;

    for &c in s2 {
        let pm_c = pm.get(c);
        // add_carry: carry for the (PM & VP) + VP addition, propagates low→high across words
        let mut add_carry: u64 = 0;
        // hp_carry: lowest bit shifted into HP (starts at 1 = X_{-1} in Myers' notation)
        let mut hp_carry: u64 = 1;
        // hn_carry: MSB of HN from previous word carried into next word's HN shift
        let mut hn_carry: u64 = 0;

        for w in 0..words {
            let pm_j = pm_c[w];
            let pv = vp[w]; // old VP for this word
            let nv = vn[w]; // old VN for this word

            // D0 = (((PM & VP) + VP) ^ VP) | (PM | VN)
            let pm_and_vp = pm_j & pv;
            let (t, c1) = pm_and_vp.overflowing_add(add_carry);
            let (sum, c2) = t.overflowing_add(pv);
            add_carry = (c1 as u64) | (c2 as u64);
            let d0 = (sum ^ pv) | (pm_j | nv);

            let hp = nv | !(d0 | pv);
            let hn = d0 & pv;

            // Check distance from HP/HN BEFORE shifting (like myers_64)
            if w == words - 1 {
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

        let lw = words - 1;
        new_vp[lw] &= last_valid_mask;
        new_vn[lw] &= last_valid_mask;

        std::mem::swap(&mut vp, &mut new_vp);
        std::mem::swap(&mut vn, &mut new_vn);
    }
    dist
}

pub fn levenshtein_uniform<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    if s1.is_empty() { return s2.len(); }
    if s2.is_empty() { return s1.len(); }
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

pub fn levenshtein_generic<T: HashableChar>(
    s1: &[T],
    s2: &[T],
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

pub fn levenshtein<T: HashableChar>(s1: &[T], s2: &[T], weights: (usize, usize, usize)) -> usize {
    let (ins, del, rep) = weights;
    if ins == 1 && del == 1 && rep == 1 {
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

pub fn levenshtein_editops_trace<T: HashableChar>(s1: &[T], s2: &[T]) -> Vec<(String, usize, usize)> {
    let full_m = s1.len();
    let full_n = s2.len();

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

fn levenshtein_editops_bitvec<T: HashableChar>(s1: &[T], s2: &[T], pfx: usize) -> Vec<(String, usize, usize)> {
    let m = s1.len();
    let n = s2.len();

    let mut pm_pairs: Vec<(T, u128)> = Vec::new();
    for (i, &ch) in s1.iter().enumerate() {
        match pm_pairs.binary_search_by_key(&ch, |&(c, _)| c) {
            Ok(pos) => pm_pairs[pos].1 |= 1u128 << i,
            Err(pos) => pm_pairs.insert(pos, (ch, 1u128 << i)),
        }
    }
    let pm_get = |ch: T| -> u128 {
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

fn levenshtein_editops_multiword<T: HashableChar>(s1: &[T], s2: &[T], pfx: usize) -> Vec<(String, usize, usize)> {
    let m = s1.len();
    let n = s2.len();
    let words = (m + 63) / 64;
    let last_bits = if m % 64 == 0 { 64 } else { m % 64 };
    let last_valid_mask: u64 = if last_bits == 64 { u64::MAX } else { (1u64 << last_bits) - 1 };

    // Build pattern bitmasks
    let mut pm = PatternMaskMulti::new(words);
    for (i, &c) in s1.iter().enumerate() {
        pm.set_bit(c, i / 64, i % 64);
    }

    let mut vp: Vec<u64> = vec![u64::MAX; words];
    let mut vn: Vec<u64> = vec![0u64; words];
    vp[words - 1] = last_valid_mask;

    let mut matrix_vp: Vec<Vec<u64>> = Vec::with_capacity(n);
    let mut matrix_vn: Vec<Vec<u64>> = Vec::with_capacity(n);

    // Hoist scratch buffers outside the loop — avoids O(len(s2)) Vec allocations.
    let mut new_vp = vec![0u64; words];
    let mut new_vn = vec![0u64; words];

    for &c in s2 {
        let pm_c = pm.get(c);
        let mut add_carry: u64 = 0;
        let mut hp_carry: u64 = 1;
        let mut hn_carry: u64 = 0;

        for w in 0..words {
            let pm_j = pm_c[w];
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
        std::mem::swap(&mut vp, &mut new_vp);
        std::mem::swap(&mut vn, &mut new_vn);
        matrix_vp.push(vp.clone());
        matrix_vn.push(vn.clone());
    }

    let mut ops: Vec<(String, usize, usize)> = Vec::new();
    let mut col = m;
    let mut row = n;

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

pub fn lcs_length_64_bounded<T: HashableChar>(s1: &[T], s2: &[T], max_dist: Option<usize>) -> usize {
    let m = s1.len();
    let n = s2.len();
    let mut pm = PatternMask64::new();
    for (i, &c) in s1.iter().enumerate() {
        pm.insert(c, 1u64 << i);
    }
    let mut v = !0u64;
    let mask = if m == 64 { !0u64 } else { (1u64 << m) - 1 };
    
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
                return 0;
            }
        }
    }
    (!v & mask).count_ones() as usize
}

/// LCS using a pre-built PatternMask64 for the query — avoids re-constructing
/// the PM for every choice in a batch. This is the rapidfuzz key trick:
/// build PM once per `process.extract` call, reuse for all N choices.
#[inline]
pub fn lcs_from_pm64(pm: &PatternMask64<u8>, q_len: usize, s2: &[u8], max_dist: Option<usize>) -> usize {
    let mask = if q_len == 64 { !0u64 } else { (1u64 << q_len) - 1 };
    let n = s2.len();

    let required_lcs = max_dist.map(|d| {
        let diff = q_len + n;
        if diff <= d { 0 } else { (diff - d + 1) / 2 }
    });

    let mut v = !0u64;
    for (i, &c) in s2.iter().enumerate() {
        let x = pm.get(c);
        let u = v & x;
        v = (v.wrapping_add(u)) | (v & !x);
        if let Some(req) = required_lcs {
            let current_lcs = (!v & mask).count_ones() as usize;
            let remaining = n - 1 - i;
            if current_lcs + remaining < req {
                return 0;
            }
        }
    }
    (!v & mask).count_ones() as usize
}

/// Fastest partial_ratio for ASCII strings where needle.len() <= 64.
/// Build needle's PM64 once, slide over haystack computing LCS for each
/// length-needle_len window in a single O(haystack_len) pass.
///
/// Key insight (rapidfuzz BitPal approach): the Myers DP for LCS runs
/// left-to-right; after processing needle_len characters from the haystack
/// we have LCS(needle, haystack[i..i+needle_len]). We reset state and repeat.
/// This is O(N) bit-ops total vs O(N × M/64) with per-window indel_distance.
///
/// Returns the best score in [0.0, 100.0].
pub fn partial_ratio_ascii_fast(needle: &[u8], haystack: &[u8]) -> f64 {
    let m = needle.len();
    let n = haystack.len();

    if m == 0 {
        return if n == 0 { 100.0 } else { 0.0 };
    }
    if n == 0 || n < m {
        // partial_ratio: shorter vs longer — if haystack < needle lengths are swapped outside
        return 0.0;
    }

    // Fast exact-substring check: if needle appears verbatim in haystack, score = 100.0
    // This is O(N) and very branch-predictor friendly — catches prefix/suffix/exact matches
    // before we build the PM or enter any LCS loop.
    if haystack.windows(m).any(|w| w == needle) {
        return 100.0;
    }

    // Build PatternMask for needle (query) — done ONCE
    let mut pm = PatternMask64::<u8>::new();
    for (i, &c) in needle.iter().enumerate() {
        pm.insert(c, 1u64 << i);
    }
    let mask = if m == 64 { !0u64 } else { (1u64 << m) - 1 };
    let lensum = 2 * m; // numerator for all windows: needle.len() + window.len() = 2m

    let mut best_score = 0.0f64;

    // Slide a window of size m across haystack
    // For each starting position `start`, process haystack[start..start+m]
    // using Myers LCS bit-parallel (no allocation).
    //
    // We vectorize by resetting V every m characters.
    let num_windows = n - m + 1;
    for start in 0..num_windows {
        let window = &haystack[start..start + m];
        let mut v = !0u64;
        for &c in window {
            let x = pm.get(c);
            let u = v & x;
            v = (v.wrapping_add(u)) | (v & !x);
        }
        let lcs = (!v & mask).count_ones() as usize;
        let dist = m + m - 2 * lcs; // indel = len(needle)+len(window) - 2*LCS
        let score = (1.0 - dist as f64 / lensum as f64) * 100.0;
        if score > best_score {
            best_score = score;
            if best_score == 100.0 { return 100.0; }
        }
    }

    // Also check suffix windows smaller than m (at the end of haystack)
    // These are windows where the window is shorter than needle (suffix overlap)
    for end in (1..m).rev() {
        if n < end { continue; }
        let window = &haystack[n - end..];
        if window.is_empty() { break; }
        let wlen = window.len();
        let mut v = !0u64;
        for &c in window {
            let x = pm.get(c);
            let u = v & x;
            v = v.wrapping_add(u) | (v ^ u);
        }
        let lcs = (!v & mask).count_ones() as usize;
        let ls = m + wlen;
        let dist = ls - 2 * lcs;
        let score = (1.0 - dist as f64 / ls as f64) * 100.0;
        if score > best_score {
            best_score = score;
            if best_score == 100.0 { return 100.0; }
        }
    }

    best_score
}


fn lcs_length_multiword_bounded<T: HashableChar>(s1: &[T], s2: &[T], max_dist: Option<usize>) -> usize {
    let m = s1.len();
    let n = s2.len();
    let words = (m + 63) / 64;
    let mut pm = PatternMaskMulti::new(words);
    for (i, &c) in s1.iter().enumerate() {
        pm.set_bit(c, i / 64, i % 64);
    }

    let mut v = vec![!0u64; words];
    let mut next_v = vec![0u64; words]; // hoisted scratch buffer
    let last_bits = if m % 64 == 0 { 64 } else { m % 64 };
    let mask = if last_bits == 64 { !0u64 } else { (1u64 << last_bits) - 1 };

    let required_lcs = max_dist.map(|d| {
        let diff = m + n;
        if diff <= d { 0 } else { (diff - d + 1) / 2 }
    });

    for (i, &c) in s2.iter().enumerate() {
        let pm_c = pm.get(c);
        let mut carry = 0u64;
        for w in 0..words {
            let x = pm_c[w];
            let u = v[w] & x;
            let sum = (v[w] as u128) + (u as u128) + (carry as u128);
            carry = (sum >> 64) as u64;
            next_v[w] = (sum as u64) | (v[w] ^ u);
        }
        std::mem::swap(&mut v, &mut next_v);

        if let Some(req) = required_lcs {
            let mut current_lcs = 0;
            for w in 0..words - 1 {
                current_lcs += (!v[w]).count_ones() as usize;
            }
            current_lcs += (!v[words - 1] & mask).count_ones() as usize;
            let remaining = n - 1 - i;
            if current_lcs + remaining < req {
                return 0;
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


pub fn lcs_length<T: HashableChar>(s1: &[T], s2: &[T], max_dist: Option<usize>) -> usize {
    if s1.is_empty() || s2.is_empty() {
        return 0;
    }
    if s1.len() <= 64 {
        lcs_length_64_bounded(s1, s2, max_dist)
    } else {
        lcs_length_multiword_bounded(s1, s2, max_dist)
    }
}

pub fn indel_distance<T: HashableChar>(s1: &[T], s2: &[T], score_cutoff: Option<usize>) -> usize {
    let len_diff = s1.len().abs_diff(s2.len());
    if let Some(cutoff) = score_cutoff {
        if len_diff > cutoff {
            return usize::MAX;
        }
        let mut hist1 = [0i32; 256];
        let mut hist2 = [0i32; 256];
        let mut can_use_hist = true;
        for &c in s1 {
            match c.as_usize() {
                Some(u) if u < 256 => hist1[u] += 1,
                _ => { can_use_hist = false; break; }
            }
        }
        if can_use_hist {
            for &c in s2 {
                match c.as_usize() {
                    Some(u) if u < 256 => hist2[u] += 1,
                    _ => { can_use_hist = false; break; }
                }
            }
        }
        if can_use_hist {
            let l1_dist: i32 = (0..256).map(|i| (hist1[i] - hist2[i]).abs()).sum();
            if l1_dist as usize > cutoff {
                return usize::MAX;
            }
        }
    }

    let pfx = common_prefix(s1, s2);
    let s1 = &s1[pfx..];
    let s2 = &s2[pfx..];
    let sfx = common_suffix(s1, s2);
    let s1 = &s1[..s1.len() - sfx];
    let s2 = &s2[..s2.len() - sfx];
    let lcs = lcs_length(s1, s2, score_cutoff);
    let dist = s1.len() + s2.len() - 2 * lcs;
    
    if score_cutoff.map_or(false, |c| dist > c) {
        return usize::MAX;
    }
    dist
}

pub fn indel_editops_trace<T: HashableChar>(s1: &[T], s2: &[T]) -> Vec<(String, usize, usize)> {
    let full_m = s1.len();
    let full_n = s2.len();

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

    let mut ops: Vec<(String, usize, usize)> = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 && j > 0 {
        if dp[i][j] == dp[i - 1][j] {
            ops.push(("delete".to_string(), pfx + i - 1, pfx + j));
            i -= 1;
        } else {
            j -= 1;
            if j > 0 && dp[i][j] > dp[i - 1][j] {
                ops.push(("insert".to_string(), pfx + i, pfx + j));
            } else {
                i -= 1;
            }
        }
    }
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

pub fn hamming_distance<T: HashableChar>(s1: &[T], s2: &[T], pad: bool) -> usize {
    let max_len = s1.len().max(s2.len());
    let min_len = s1.len().min(s2.len());
    let mut dist = 0usize;
    for i in 0..min_len {
        if s1[i] != s2[i] {
            dist += 1;
        }
    }
    match min_len {
        0 => if pad { max_len } else { 0 },
        _ => dist + if pad { max_len - min_len } else { 0 },
    }
}

pub fn hamming_editops_trace<T: HashableChar>(s1: &[T], s2: &[T]) -> Vec<(String, usize, usize)> {
    let min_len = s1.len().min(s2.len());
    let mut ops = Vec::new();
    for i in 0..min_len {
        if s1[i] != s2[i] {
            ops.push(("replace".to_string(), i, i));
        }
    }
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

pub fn jaro<T: HashableChar>(s1: &[T], s2: &[T]) -> f64 {
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

pub fn jaro_winkler<T: HashableChar>(s1: &[T], s2: &[T], prefix_weight: f64) -> f64 {
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

pub fn lcs_seq_similarity<T: HashableChar>(s1: &[T], s2: &[T], max_dist: Option<usize>) -> usize {
    let pfx = common_prefix(s1, s2);
    let s1t = &s1[pfx..];
    let s2t = &s2[pfx..];
    let sfx = common_suffix(s1t, s2t);
    let s1t = &s1t[..s1t.len() - sfx];
    let s2t = &s2t[..s2t.len() - sfx];
    pfx + sfx + lcs_length(s1t, s2t, max_dist)
}

pub fn lcs_seq_distance<T: HashableChar>(s1: &[T], s2: &[T], score_cutoff: Option<usize>) -> usize {
    let sim = lcs_seq_similarity(s1, s2, score_cutoff);
    s1.len().max(s2.len()) - sim
}

pub fn lcs_seq_editops_trace<T: HashableChar>(s1: &[T], s2: &[T]) -> Vec<(String, usize, usize)> {
    indel_editops_trace(s1, s2)
}

pub fn osa_distance<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    let m = s1.len();
    let n = s2.len();
    if m == 0 { return n; }
    if n == 0 { return m; }

    // 3-row rolling window: eliminates O(N×M) 2D allocation.
    // OSA needs current row, previous row, and two-rows-back (for transposition).
    let mut prev2 = vec![0usize; n + 1];
    let mut prev  = (0..=n).collect::<Vec<_>>();
    let mut curr  = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if s1[i - 1] == s2[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
            if i > 1 && j > 1 && s1[i - 1] == s2[j - 2] && s1[i - 2] == s2[j - 1] {
                curr[j] = curr[j].min(prev2[j - 2] + cost);
            }
        }
        std::mem::swap(&mut prev2, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ===========================================================================
// DAMERAU-LEVENSHTEIN (true, with transpositions)
// ===========================================================================

pub fn damerau_levenshtein_distance<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    let m = s1.len();
    let n = s2.len();
    if m == 0 { return n; }
    if n == 0 { return m; }

    let max_dist = m + n;
    // Last occurrence of each character in s1 (FxHashMap = fast integer-keyed hash)
    let mut da: FxHashMap<T, usize> = FxHashMap::default();

    // Flat single Vec instead of vec![vec![..]; m+2] — one contiguous allocation,
    // stride = n+2, so dp[i][j] = flat[i * (n+2) + j]. This eliminates m+2
    // sub-allocations and maximises cache locality for the inner j-loop.
    let stride = n + 2;
    let mut flat = vec![0usize; (m + 2) * stride];

    // Sentinel borders
    flat[0] = max_dist; // dp[0][0]
    for i in 0..=m {
        flat[(i + 1) * stride] = max_dist; // dp[i+1][0]
        flat[(i + 1) * stride + 1] = i;   // dp[i+1][1]
    }
    for j in 0..=n {
        flat[j + 1] = max_dist; // dp[0][j+1]
        flat[stride + j + 1] = j; // dp[1][j+1]
    }

    for i in 1..=m {
        let mut db = 0usize;
        for j in 1..=n {
            let i1 = da.get(&s2[j - 1]).copied().unwrap_or(0);
            let j1 = db;
            let cost = if s1[i - 1] == s2[j - 1] { db = j; 0 } else { 1 };

            let cur = flat[i * stride + j] + cost;         // dp[i][j] + cost
            let del = flat[(i + 1) * stride + j] + 1;     // dp[i+1][j] + 1
            let ins = flat[i * stride + j + 1] + 1;       // dp[i][j+1] + 1
            let mut best = cur.min(del).min(ins);

            if i1 > 0 && j1 > 0 {
                let trans = flat[i1 * stride + j1]
                    + (i - i1 - 1) + 1 + (j - j1 - 1);   // dp[i1][j1] + gaps
                best = best.min(trans);
            }
            flat[(i + 1) * stride + j + 1] = best;
        }
        da.insert(s1[i - 1], i);
    }
    flat[(m + 1) * stride + n + 1]
}

pub fn prefix_similarity<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count()
}

pub fn prefix_distance<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    let sim = prefix_similarity(s1, s2);
    s1.len().max(s2.len()) - sim
}

pub fn postfix_similarity<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    s1.iter()
        .rev()
        .zip(s2.iter().rev())
        .take_while(|(a, b)| a == b)
        .count()
}

pub fn postfix_distance<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    let sim = postfix_similarity(s1, s2);
    s1.len().max(s2.len()) - sim
}

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

// ---------------------------------------------------------------------------
// Gotoh Alignment (Affine Gap Penalty)
// ---------------------------------------------------------------------------

pub fn gotoh_distance<T: HashableChar>(
    s1: &[T],
    s2: &[T],
    open_penalty: usize,
    extend_penalty: usize,
) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 {
        return if n == 0 { 0 } else { open_penalty + n * extend_penalty };
    }
    if n == 0 {
        return open_penalty + m * extend_penalty;
    }

    let mut dp = vec![0usize; n + 1];
    let mut p_gap = vec![0usize; n + 1];
    
    dp[0] = 0;
    p_gap[0] = 0;
    for j in 1..=n {
        dp[j] = open_penalty + j * extend_penalty;
        p_gap[j] = dp[j] + open_penalty; 
    }
    
    for i in 1..=m {
        let mut prev_dp = dp[0];
        dp[0] = open_penalty + i * extend_penalty;
        let mut row_gap = dp[0] + open_penalty;
        
        for j in 1..=n {
            let match_cost = if s1[i - 1] == s2[j - 1] { 0 } else { 100 };
            
            p_gap[j] = (dp[j] + open_penalty + extend_penalty)
                        .min(p_gap[j] + extend_penalty);
                        
            row_gap = (dp[j - 1] + open_penalty + extend_penalty)
                       .min(row_gap + extend_penalty);
            
            let current_dp = (prev_dp + match_cost)
                             .min(p_gap[j])
                             .min(row_gap);
            
            prev_dp = dp[j];
            dp[j] = current_dp;
        }
    }
    
    dp[n]
}
