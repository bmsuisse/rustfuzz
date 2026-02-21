import re

# `raw_choices.into_par_iter()` fails because `Seq<'_>` contains lifetime `'_` which cannot be Sent across threads using normal `into_par_iter()`.
# To use Rayon safely, we need fully owned data structures without lifetimes for the thread dispatch.

with open("src/process.rs", "r") as f:
    text = f.read()

text = text.replace("let owned_seq = match seq {", "let owned_seq: Vec<u32> = match seq {")
text = text.replace("crate::types::Seq::Ascii(slice) => crate::types::Seq::U32(slice.iter().map(|&c| c as u32).collect()),", "crate::types::Seq::Ascii(slice) => slice.iter().map(|&c| c as u32).collect(),")
text = text.replace("crate::types::Seq::U32(slice) => crate::types::Seq::U32(slice.to_vec()),", "crate::types::Seq::U32(slice) => slice.to_vec(),")
text = text.replace("crate::types::Seq::U64(slice) => crate::types::Seq::U32(slice.iter().map(|&c| c as u32).collect()),", "crate::types::Seq::U64(slice) => slice.iter().map(|&c| c as u32).collect(),")

# Replace the inner map using fully owned U32 struct
text = text.replace("let score = crate::fuzz::execute_scorer_raw(stype.clone(), &q_seq, &seq, score_cutoff);", "let seq_wrap = crate::types::Seq::U32(seq.clone());\n            let score = crate::fuzz::execute_scorer_raw(stype.clone(), &q_seq, &seq_wrap, score_cutoff);")

# We still need `q_seq` to be owned `Vec<u32>` to pass to threads if it captures it
text = text.replace("let q_seq = crate::types::extract_single(&processed_query)?;", """let q_seq_temp = crate::types::extract_single(&processed_query)?;
        let q_owned: Vec<u32> = match q_seq_temp {
            crate::types::Seq::Ascii(slice) => slice.iter().map(|&c| c as u32).collect(),
            crate::types::Seq::U32(slice) => slice.to_vec(),
            crate::types::Seq::U64(slice) => slice.iter().map(|&c| c as u32).collect(),
        };""")
        
text = text.replace("let score = crate::fuzz::execute_scorer_raw(stype, &q_seq, &seq, score_cutoff);", """let seq_wrap = crate::types::Seq::U32(seq.clone());
            let q_wrap = crate::types::Seq::U32(q_owned.clone());
            let score = crate::fuzz::execute_scorer_raw(stype, &q_wrap, &seq_wrap, score_cutoff);""")

# Drop the `clone()` logic from previous manual updates if stype clone throws error. It's an Enum, let's derive Clone.
if "#[derive(PartialEq, Clone)]" not in text:
    text = text.replace("#[derive(PartialEq)]\npub enum ScorerType", "#[derive(PartialEq, Clone)]\npub enum ScorerType")

with open("src/process.rs", "w") as f:
    f.write(text)
    
print("Fixed thread lifetimes")
