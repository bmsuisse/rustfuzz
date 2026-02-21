---
description: Research → Build → Test → Benchmark → Repeat — the core optimisation loop for beating RapidFuzz
---

# Research → Build → Test → Benchmark → Repeat

This is the core AI-driven optimisation loop for rustfuzz.
Each iteration identifies a bottleneck, implements a Rust fix, validates correctness, and measures the gain against RapidFuzz.

> **Rule:** never skip the Test step. A faster but broken library is worthless.

---

## 🔍 Step 1 — Research

Profile the current hottest path and identify what to improve next.

Run a quick benchmark first to get a baseline and see where time is spent:
```
uv run pytest tests/benchmarks/ -v --benchmark-only --benchmark-sort=mean 2>&1 | head -60
```

If you need function-level profiling, use `py-spy` on a benchmark call:
```
uv run py-spy record -o profile.svg -- python -c "
import rustfuzz.process as p
import rapidfuzz.process as rf
choices = ['New York','Newark','New Orleans','Los Angeles'] * 5000
p.extract('new york', choices, limit=5)
"
```

Identify the specific Rust function or Python↔Rust boundary causing overhead.
Document the hypothesis: _"the bottleneck is X because Y"_.

---

## 🦀 Step 2 — Build

Implement the fix in Rust and rebuild the extension module.

Make your Rust changes in `src/`, then build in dev mode:
// turbo
```
uv run maturin develop --release
```

Check that the Rust code compiles cleanly with no warnings:
// turbo
```
cargo check 2>&1
```

---

## ✅ Step 3 — Test

**All tests must pass before proceeding.** No exceptions.

// turbo
```
uv run pytest tests/ -x -q
```

If any test fails, go back to Step 2 and fix the regression before moving on.

Also run pyright to catch any type errors introduced in Python stubs or wrappers:
// turbo
```
uv run pyright
```

---

## 📊 Step 4 — Benchmark

Run the full benchmark suite and compare against RapidFuzz.

```
uv run pytest tests/benchmarks/ -v --benchmark-only --benchmark-compare --benchmark-sort=mean
```

Key metrics to record for each function (`ratio`, `partial_ratio`, `token_sort_ratio`, `process.extract`, `process.extractOne`):

| Metric | Target |
|--------|--------|
| Mean time | ≤ RapidFuzz mean |
| Min time | competitive |
| Throughput | ≥ RapidFuzz |

Save the results for comparison in the next iteration:
```
uv run pytest tests/benchmarks/ -v --benchmark-only --benchmark-save=iteration_N
```

Interpret the results:
- **Better** → document the gain, pick the next bottleneck → Step 5.
- **Worse / no change** → revisit the approach in Step 1.

---

## 🔁 Step 5 — Repeat

Pick the next biggest gap from the benchmark output and go back to Step 1.

Useful questions to guide the next Research phase:
- Which function is still slowest relative to RapidFuzz?
- Is the bottleneck in the algorithm itself, the Python↔Rust boundary, or string allocation?
- Can batch operations (`cdist`, `extract`) be parallelised with Rayon?
- Are we paying for UTF-8 re-encoding on every call?

---

## Quick-reference commands

| Command | Purpose |
|---------|---------|
| `uv run maturin develop --release` | Rebuild Rust extension (optimised) |
| `cargo check` | Fast compile check without linking |
| `uv run pytest tests/ -x -q` | Full test suite, stop on first failure |
| `uv run pyright` | Type-check Python stubs |
| `uv run pytest tests/benchmarks/ -v --benchmark-only` | Full benchmark run |
| `uv run pytest tests/benchmarks/ -k ratio --benchmark-only` | Benchmark a single function |
| `uv run pytest tests/benchmarks/ --benchmark-compare` | Compare vs saved baseline |
