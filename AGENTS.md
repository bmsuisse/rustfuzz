# rustfuzz Agent Guidelines

## Project Overview

rustfuzz is a high-performance fuzzy string matching library implemented entirely in Rust, published as `rustfuzz` on PyPI. It uses:
- **Rust** (via PyO3 + maturin) for all core fuzzy-matching algorithms
- **Python** (3.10+) for the public API surface — thin wrappers that re-export Rust symbols
- **uv** for Python package management

## Architecture

```
src/                    ← Rust source (compiled as rustfuzz._rustfuzz)
  lib.rs                ← PyO3 module root — fn _rustfuzz
  algorithms.rs         ← Core algorithm implementations (Myers, LCS, Jaro, etc.)
  fuzz.rs               ← ratio, partial_ratio, token_*, WRatio, QRatio
  utils.rs              ← default_process
  types.rs              ← Seq type + Python object extraction helpers
  distance/
    mod.rs
    initialize.rs       ← Editop, Editops, Opcode, Opcodes, MatchingBlock, ScoreAlignment
    metrics.rs          ← All distance/similarity pyfunction wrappers
rustfuzz/               ← Python package (thin wrappers, imports from ._rustfuzz)
Cargo.toml              ← lib name = "_rustfuzz"
pyproject.toml          ← module-name = "rustfuzz._rustfuzz"
```

## Public API Surface

### `rustfuzz.fuzz`
`ratio`, `partial_ratio`, `partial_ratio_alignment`, `token_sort_ratio`, `token_set_ratio`,
`token_ratio`, `partial_token_sort_ratio`, `partial_token_set_ratio`, `partial_token_ratio`,
`WRatio`, `QRatio`

### `rustfuzz.process`
`extract`, `extractOne`, `extract_iter`, `cdist`

### `rustfuzz.utils`
`default_process`

### `rustfuzz.distance`
**Data types:** `Editop`, `Editops`, `Opcode`, `Opcodes`, `MatchingBlock`, `ScoreAlignment`

**Per-metric (all modules):** `distance`, `similarity`, `normalized_distance`,
`normalized_similarity`, `editops` (where applicable), `opcodes` (where applicable)

**Modules:** `Levenshtein`, `Hamming`, `Indel`, `Jaro`, `JaroWinkler`, `LCSseq`,
`OSA`, `DamerauLevenshtein`, `Prefix`, `Postfix`

## Critical Rules

### Always Use `uv`
- **All Python commands MUST use `uv run`** — never use `.venv/bin/python` or bare `python`
- Tests: `uv run pytest tests/ -x -q`
- Benchmarks: `uv run pytest tests/test_benchmarks.py --benchmark-save=baseline`
- Benchmark regression: `uv run pytest tests/test_benchmarks.py --benchmark-compare=baseline --benchmark-compare-fail=mean:10%`
- Type checking: `uv run pyright`
- Smoke test: `uv run python -c "import rustfuzz; print(rustfuzz.__version__)"`

### Build Workflow
1. `cargo check` — fast compilation check
2. `uv run maturin develop --release` — build optimised `.so`

### Pre-Commit Checklist
1. `cargo check` — no Rust errors
2. `uv run maturin develop --release`
3. `uv run pytest tests/ -x -q` — all tests must pass
4. `uv run pytest tests/test_benchmarks.py --benchmark-compare=baseline --benchmark-compare-fail=mean:10%` — no regressions
5. `uv run pyright` — type checking passes

### File Size Limit
- No file should exceed 1000 lines of code

### Testing
- Always run the **full** original test suite before committing
- Run e2e tests after implementing new features
- Create a branch for each feature/algorithm group

### Implementation Strategy
- Each metric module (`Levenshtein`, `Hamming`, etc.) must agree with the reference algorithms
- `process.cdist` consumes any scorer callable accepting `(str, str, **kwargs) -> float`
- Benchmarks baseline is saved in `.benchmarks/` — commit it so CI can compare

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
uv run pytest tests/test_benchmarks.py --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
uv run pyright
```
