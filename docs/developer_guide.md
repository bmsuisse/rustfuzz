# Developer & Contributor Guide

Welcome to the `rustfuzz` project! This guide explains how to build, test, and contribute to the library. The project follows strict guidelines to ensure maximum performance, safety, and maintainability.

## 1. Toolchain & Setup

We use modern, fast tools built in Rust across the entire stack:
- **Rust / Cargo:** For the underlying implementations.
- **Maturin:** For building Python wheels and managing the PyO3 bindings.
- **uv:** For incredibly fast Python environment and dependency management.
- **Ruff:** For Python linting and formatting.
- **Pyright:** For strict static type checking in Python.

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/bmsuisse/rustfuzz.git
cd rustfuzz

# Create a virtual environment and sync dependencies using uv
uv sync --all-groups
```

To build the development version of the Rust extension:

```bash
uv run maturin develop
# Use --release for benchmarking to disable debug assertions and enable optimizations
uv run maturin develop --release
```

## 2. The Core Optimization Loop

Our development philosophy is summarized in the following loop, designed to aggressively optimize performance vs. benchmark targets (like RapidFuzz):

**Research → Build → Test → Benchmark → Repeat**

1. **Research:** Run profilers to identify bottlenecks or missing algorithm fast-paths.
2. **Build:** Implement the logic in Rust and expose it via PyO3.
3. **Test:** Validate against existing tests to ensure 100% equivalence and correctness (Memory safety).
4. **Benchmark:** Run the `pytest-benchmark` suite to measure the impact.
5. **Repeat:** Repeat the cycle.

*Note: You can trigger the automated benchmark workflow via the `/research-build-test-benchmark` command if using the dedicated AI agent flow.*

## 3. Coding Standards & Rules

To maintain high code quality, please adhere strictly to the following rules:

- **1000 Lines Max:** No file may exceed 1000 lines of code. If a module grows too large, refactor it into smaller, manageable pieces to ensure cognitive load remains low.
- **Branching:** Create a dedicated branch for every new feature or bugfix. Do not push directly to `main`.
- **Typing First:** Python features must be fully typed. Use `pyright` to catch typing errors before committing. 
  ```bash
  uv run pyright
  ```
- **Linting:** We enforce `ruff` for linting. 
  ```bash
  uv run ruff check .
  ```

## 4. Testing & QA

- **Ensure tests pass before committing:**
  ```bash
  uv run pytest
  ```
- **End-to-End (E2E) Verification:** Whenever implementing a major new feature (like a new core algorithm or architectural integration), write comprehensive unit tests or E2E tests validating the full pipeline from Python through FFI into Rust and back out.

## 5. Web/Browser Verification (If Applicable)

`rustfuzz` is strictly a backend/library tool. However, if visual output or browser-based dashboards are added in the future (e.g., for reporting), **never** attempt manual browser testing. **Always use Playwright** for automated, reproducible E2E browser tests. 

## 6. Documenting Functions

- Python wrappers must have standard docstrings.
- Ensure any new Python stubs (`.pyi`) are kept perfectly in sync with the `src/*.rs` Rust implementation signatures.
