import re

with open("src/algorithms.rs", "r") as f:
    text = f.read()

# Update signature of lcs_seq_similarity/distance
text = text.replace("pub fn lcs_seq_similarity<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {", 
                   "pub fn lcs_seq_similarity<T: HashableChar>(s1: &[T], s2: &[T], max_dist: Option<usize>) -> usize {")

text = text.replace("pub fn lcs_seq_distance<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {",
                   "pub fn lcs_seq_distance<T: HashableChar>(s1: &[T], s2: &[T], score_cutoff: Option<usize>) -> usize {")

# Update lcs_length_64
text = text.replace("pub fn lcs_length_64<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {",
                   "pub fn lcs_length_64<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {")

# Update lcs_length
text = text.replace("pub fn lcs_length<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {",
                   "pub fn lcs_length<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {")

# Update indel_distance
orig_indel = """pub fn indel_distance<T: HashableChar>(s1: &[T], s2: &[T]) -> usize {
    let pfx = common_prefix(s1, s2);
    let s1 = &s1[pfx..];
    let s2 = &s2[pfx..];
    let sfx = common_suffix(s1, s2);
    let s1 = &s1[..s1.len() - sfx];
    let s2 = &s2[..s2.len() - sfx];
    let lcs = lcs_length(s1, s2);
    s1.len() + s2.len() - 2 * lcs
}"""

new_indel = """pub fn indel_distance<T: HashableChar>(s1: &[T], s2: &[T], score_cutoff: Option<usize>) -> usize {
    // If the minimum possible distance (length difference) is already greater than cutoff, abort.
    let len_diff = s1.len().abs_diff(s2.len());
    if let Some(cutoff) = score_cutoff {
        if len_diff > cutoff {
            return usize::MAX;
        }
    }

    let pfx = common_prefix(s1, s2);
    let s1 = &s1[pfx..];
    let s2 = &s2[pfx..];
    let sfx = common_suffix(s1, s2);
    let s1 = &s1[..s1.len() - sfx];
    let s2 = &s2[..s2.len() - sfx];
    let lcs = lcs_length(s1, s2);
    let dist = s1.len() + s2.len() - 2 * lcs;
    
    // Validate final distance against the cutoff
    if let Some(cutoff) = score_cutoff {
        if dist > cutoff {
            return usize::MAX;
        }
    }
    dist
}"""

text = text.replace(orig_indel, new_indel)

with open("src/algorithms.rs", "w") as f:
    f.write(text)

print("Injected algorithms.rs")
