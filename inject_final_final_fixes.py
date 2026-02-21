import re

with open("src/process.rs", "r") as f:
    text = f.read()

# Fix 1: Conflicting implementation of Clone.
# There is already `#[derive(Clone, Copy)]` on `ScorerType`.
# Remove `#[derive(PartialEq, Clone)]`
text = text.replace("#[derive(PartialEq, Clone)]\npub enum ScorerType", "pub enum ScorerType")

# Fix 2: match arms have incompatible types
# I replaced `Seq::Chars` with `Seq::U32(vec) => crate::types::Seq::U32(vec)` but now it returns Vec<u32>
text = text.replace("crate::types::Seq::U32(vec) => crate::types::Seq::U32(vec),", "crate::types::Seq::U32(vec) => vec,")

# We had two U32 rules actually:
# 1. crate::types::Seq::U32(slice) => slice.to_vec(),
# 2. crate::types::Seq::U32(vec) => vec, 
# The script replaced `Chars` but there is no `Chars`. Let's just fix it completely.

# Wait, `Seq` only has 3 variants: Ascii, U32, U64.
# Let's replace the whole match block.

new_match = """let owned_seq: Vec<u32> = match seq {
                    crate::types::Seq::Ascii(slice) => slice.iter().map(|&c| c as u32).collect(),
                    crate::types::Seq::U32(slice) => slice.to_vec(),
                    crate::types::Seq::U64(slice) => slice.iter().map(|&c| c as u32).collect(),
                };"""

# We just replace any lingering 4-arm matches with 3-arm matches.
# I'll just regex replace the assignment block
pattern = r"let owned_seq: Vec<u32> = match seq \{.*?\};"
text = re.sub(pattern, new_match, text, flags=re.DOTALL)

with open("src/process.rs", "w") as f:
    f.write(text)

print("Applied Type resolution fixes")
