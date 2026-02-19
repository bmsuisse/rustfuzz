# RapidFuzz Agent Guidelines

## Project Overview

RapidFuzz is a 1:1 Rust port of the original C++/Cython RapidFuzz for performance. It uses:
- **Rust** (via PyO3 + maturin) for all core fuzzy-matching algorithms
- **Python** (3.10+) for the public API surface — thin wrappers that re-export Rust symbols
- **uv** for Python package management
- **rapidfuzz-rs** / `strsim` crates for low-level string algorithms where applicable

## Architecture

```
src/rapidfuzz/          ← thin Python wrappers (re-export Rust symbols)
  __init__.py           ← exposes: distance, fuzz, process, utils
  fuzz.py               ← re-exports from _rapidfuzz (Rust ext module)
  process.py            ← re-exports from _rapidfuzz
  utils.py              ← re-exports from _rapidfuzz
  distance/
    __init__.py
    _initialize.py      ← Editop, Editops, Opcode, Opcodes, MatchingBlock, ScoreAlignment
    Levenshtein.py      ← re-exports from _rapidfuzz
    Hamming.py          ← re-exports from _rapidfuzz
    Indel.py            ← re-exports from _rapidfuzz
    Jaro.py             ← re-exports from _rapidfuzz
    JaroWinkler.py      ← re-exports from _rapidfuzz
    LCSseq.py           ← re-exports from _rapidfuzz
    OSA.py              ← re-exports from _rapidfuzz
    DamerauLevenshtein.py
    Prefix.py / Postfix.py
src/              ← Rust source (Cargo workspace)
  lib.rs          ← PyO3 module root, registers all submodules
  fuzz.rs         ← ratio, partial_ratio, token_*, WRatio, QRatio
  process.rs      ← extract, extractOne, extract_iter, cdist, cpdist
  utils.rs        ← default_process
  distance/
    mod.rs
    initialize.rs ← Editop, Editops, Opcode, Opcodes, MatchingBlock, ScoreAlignment
    levenshtein.rs
    hamming.rs
    indel.rs
    jaro.rs
    jaro_winkler.rs
    lcs_seq.rs
    osa.rs
    damerau_levenshtein.rs
    prefix.rs / postfix.rs
    metrics.rs     ← common metrics helpers
Cargo.toml
pyproject.toml    ← maturin build backend
```

## Public API Surface

### `rapidfuzz.fuzz`
`ratio`, `partial_ratio`, `partial_ratio_alignment`, `token_sort_ratio`, `token_set_ratio`,
`token_ratio`, `partial_token_sort_ratio`, `partial_token_set_ratio`, `partial_token_ratio`,
`WRatio`, `QRatio`

### `rapidfuzz.process`
`extract`, `extractOne`, `extract_iter`, `cdist`, `cpdist`

### `rapidfuzz.utils`
`default_process`

### `rapidfuzz.distance`
**Data types:** `Editop`, `Editops`, `Opcode`, `Opcodes`, `MatchingBlock`, `ScoreAlignment`

**Per-metric (all modules):** `distance`, `similarity`, `normalized_distance`,
`normalized_similarity`, `editops` (where applicable), `opcodes` (where applicable)

**Modules:** `Levenshtein`, `Hamming`, `Indel`, `Jaro`, `JaroWinkler`, `LCSseq`,
`OSA`, `DamerauLevenshtein`, `Prefix`, `Postfix`

## Critical Rules

### Always Use `uv`
- **All Python commands MUST use `uv run`** — never use `.venv/bin/python` or bare `python`
- Tests: `uv run pytest tests/ -x -q`
- Type checking: `uv run pyright`
- Smoke test: `uv run python -c "import rapidfuzz; print(rapidfuzz.__version__)"`

### Build Workflow
1. `cargo check` — fast compilation check
2. `maturin develop --release` — build optimised `.so`
3. Copy `.so` to in-tree location if needed (stale `.so` issue):
   ```bash
   cp .venv/lib/python3.*/site-packages/rapidfuzz/_rapidfuzz*.so \
      src/rapidfuzz/_rapidfuzz*.so
   ```

### Pre-Commit Checklist
1. `cargo check` — no Rust errors
2. `maturin develop --release` + copy `.so` if needed
3. `uv run pytest tests/ -x -q` — **all original tests must pass**
4. `uv run pyright` — type checking passes

### File Size Limit
- No file should exceed 1000 lines of code

### Testing
- Always run the **full** original test suite before committing
- Run e2e tests after implementing new features
- Create a branch for each feature/algorithm group

### Implementation Strategy
- Use `RAPIDFUZZ_IMPLEMENTATION=python` env var to run against pure-Python fallback as a
  reference during development
- Each metric module (`Levenshtein`, `Hamming`, etc.) should have parity with the
  `*_py.py` pure-Python implementation — use those as the correctness reference
- `scorer_flags` / `get_scorer_flags` attributes must be preserved for `process.cdist`
  compatibility

## Releasing a New Version

The CI automatically builds wheels for all platforms, generates a changelog, and publishes
to PyPI when a **git tag** is pushed.

### Steps
1. **Bump version** in `Cargo.toml` (the `version` field under `[package]`)
2. **Commit** the version bump: `git commit -am "release: v0.X.Y"`
3. **Tag** the commit: `git tag v0.X.Y`
4. **Push** the tag: `git push origin main && git push origin v0.X.Y`
5. CI will:
   - Run tests on all Python versions
   - Build wheels (linux, musllinux, macos, windows)
   - Generate changelog from conventional commits (via `git-cliff`)
   - Publish to PyPI
   - Create GitHub Release with changelog and wheel assets

### Commit Convention
Use [Conventional Commits](https://www.conventionalcommits.org/) for automatic changelog
categorization:

| Prefix | Category | Example |
|--------|----------|---------| 
| `feat:` | 🚀 Features | `feat: implement Jaro-Winkler in Rust` |
| `fix:` | 🐛 Bug Fixes | `fix: handle empty string in partial_ratio` |
| `perf:` | ⚡ Performance | `perf: use SIMD in Levenshtein` |
| `refactor:` | 🔧 Refactoring | `refactor: split distance module` |
| `docs:` | 📖 Documentation | `docs: update README` |
| `ci:` | 🔄 CI/CD | `ci: add Python 3.13 to matrix` |
| `chore:` | 📦 Miscellaneous | `chore: update deps` |
| `release:` | _(skipped)_ | `release: v3.15.0` |

### Verify Before Releasing
```bash
cargo check
uv run maturin develop --release
uv run pytest tests/ -x -q
uv run pyright
```
