import re

with open("src/fuzz.rs", "r") as f:
    text = f.read()

text = text.replace("crate::types::ScorerType", "crate::process::ScorerType")
with open("src/fuzz.rs", "w") as f:
    f.write(text)

with open("src/process.rs", "r") as f:
    text = f.read()

# Add IntoPyObjectExt trait
if "use pyo3::IntoPyObjectExt;" not in text:
    text = text.replace("use pyo3::prelude::*;", "use pyo3::prelude::*;\nuse pyo3::IntoPyObjectExt;")

# Check ScorerType deriving PartialEq
if "#[derive(PartialEq)]\npub enum ScorerType" not in text:
    text = text.replace("pub enum ScorerType", "#[derive(PartialEq)]\npub enum ScorerType")

with open("src/process.rs", "w") as f:
    f.write(text)
print("Applied remaining trait fixes")
