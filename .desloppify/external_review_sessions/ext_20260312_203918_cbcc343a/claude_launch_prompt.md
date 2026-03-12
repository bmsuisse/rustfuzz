# Claude Blind Reviewer Launch Prompt

You are an isolated blind reviewer. Do not use prior chat context, prior score history, or target-score anchoring.

Session id: ext_20260312_203918_cbcc343a
Session token: 306dfa449ce832d20f627f89463bb2e3
Blind packet: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/review_packet_blind.json
Template JSON: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/external_review_sessions/ext_20260312_203918_cbcc343a/review_result.template.json
Output JSON path: /Users/dominikpeter/DevOps/RapidFuzz/.desloppify/external_review_sessions/ext_20260312_203918_cbcc343a/review_result.json

--- Batch 1: cross_module_architecture ---
Rationale: seed files for cross_module_architecture review
DIMENSION TO EVALUATE:

## cross_module_architecture
Dependency direction, cycles, hub modules, and boundary integrity
Look for:
- Layer/dependency direction violations repeated across multiple modules
- Cycles or hub modules that create large blast radius for common changes
- Documented architecture contracts drifting from runtime (e.g. dynamic import boundaries)
- Cross-module coordination through shared mutable state or import-time side effects
- Compatibility shim paths that persist without active external need and blur boundaries
- Cross-package duplication that indicates a missing shared boundary
- Subsystem or package consuming a disproportionate share of the codebase — see package_size_census evidence
Skip:
- Intentional facades/re-exports with clear API purpose
- Framework-required patterns (Django settings, plugin registries)
- Package naming/placement tidy-ups without boundary harm (belongs to package_organization)
- Local readability/craft issues (belongs to low_level_elegance)

Seed files (start here):
- rustfuzz/search.py
- rustfuzz/__init__.py
- rustfuzz/distance/__init__.py
- rustfuzz/fuzz.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/distance/JaroWinkler.py

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show cycles --no-budget      # 1 findings
  desloppify show private_imports --no-budget      # 3 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

--- Batch 2: high_level_elegance ---
Rationale: seed files for high_level_elegance review
DIMENSION TO EVALUATE:

## high_level_elegance
Clear decomposition, coherent ownership, domain-aligned structure
Look for:
- Top-level packages/files map to domain capabilities rather than historical accidents
- Ownership and change boundaries are predictable — a new engineer can explain why this exists
- Public surface (exports/entry points) is small and consistent with stated responsibility
- Project contracts and reference docs match runtime reality (README/structure/philosophy are trustworthy)
- Subsystem decomposition localizes change without surprising ripple edits
- A small set of architectural patterns is used consistently across major areas
Skip:
- When dependency direction/cycle/hub failures are the PRIMARY issue, report under cross_module_architecture (still include here if they materially blur ownership/decomposition)
- When handoff mechanics are the PRIMARY issue, report under mid_level_elegance (still include here if they materially affect top-level role clarity)
- When function/class internals are the PRIMARY issue, report under low_level_elegance or logic_clarity
- Pure naming/style nits with no impact on role clarity

Seed files (start here):
- rustfuzz/__init__.py
- rustfuzz/_types.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/Hamming.py

--- Batch 3: convention_outlier ---
Rationale: seed files for convention_outlier review
DIMENSION TO EVALUATE:

## convention_outlier
Naming convention drift, inconsistent file organization, style islands
Look for:
- Naming convention drift: snake_case functions in a camelCase codebase or vice versa
- Inconsistent file organization that impedes navigation (not mere structural variation between dirs)
- Mixed export patterns across sibling modules (named vs default, class vs function)
- Style islands: one directory uses a completely different pattern than the rest
- Sibling modules following different behavioral protocols (e.g. most call a shared function, one doesn't)
- Inconsistent plugin organization: sibling plugins structured differently
- Large __init__.py re-export surfaces that obscure internal module structure
- Mixed type strategies for domain objects (TypedDict for some, dataclass for others, NamedTuple for yet others) without documented rationale — see type_strategy_census evidence
Skip:
- Intentional variation for different module types (config vs logic)
- Third-party code or generated files following their own conventions
- Do NOT recommend adding index/barrel files, re-export facades, or directory wrappers to 'standardize' — prefer the simpler existing pattern over consistency-for-its-own-sake
- When sibling modules use different structures, report the inconsistency but do NOT suggest adding abstraction layers to unify them

Seed files (start here):
- rustfuzz/fuzz.py
- rustfuzz/utils.py
- rustfuzz/__init__.py
- rustfuzz/_types.py
- rustfuzz/compat.py
- rustfuzz/search.py
- rustfuzz/filter.py

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show dupes --no-budget      # 11 findings
  desloppify show signature --no-budget      # 3 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

--- Batch 4: error_consistency ---
Rationale: seed files for error_consistency review
DIMENSION TO EVALUATE:

## error_consistency
Consistent error strategies, preserved context, predictable failure modes
Look for:
- Mixed error strategies: some functions throw, others return null, others use Result types
- Error context lost at boundaries: catch-and-rethrow without wrapping original
- Inconsistent error types: custom error classes in some modules, bare strings in others
- Silent error swallowing: catches that log but don't propagate or recover
- Missing error handling on I/O boundaries (file, network, parse operations)
Skip:
- Intentional error boundaries at top-level handlers
- Different strategies for different layers (e.g. Result in core, throw in CLI)

Seed files (start here):
- rustfuzz/fuzz.py
- rustfuzz/utils.py
- rustfuzz/__init__.py
- rustfuzz/_types.py
- rustfuzz/compat.py
- rustfuzz/search.py
- rustfuzz/filter.py

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show smells --no-budget      # 27 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

--- Batch 5: naming_quality ---
Rationale: seed files for naming_quality review
DIMENSION TO EVALUATE:

## naming_quality
Function/variable/file names that communicate intent
Look for:
- Generic verbs that reveal nothing: process, handle, do, run, manage
- Name/behavior mismatch: getX() that mutates state, isX() returning non-boolean
- Vocabulary divergence from codebase norms (context provides the norms)
- Abbreviations inconsistent with codebase conventions
Skip:
- Standard framework names (render, mount, useEffect)
- Short-lived loop variables (i, j, k)
- Well-known abbreviations matching codebase convention (ctx, req, res)
- Short names that are established project conventions used consistently — a name used 50+ times is a convention, not an outlier

Seed files (start here):
- rustfuzz/fuzz.py
- rustfuzz/utils.py
- rustfuzz/__init__.py
- rustfuzz/_types.py
- rustfuzz/compat.py
- rustfuzz/search.py
- rustfuzz/filter.py

--- Batch 6: abstraction_fitness ---
Rationale: seed files for abstraction_fitness review
DIMENSION TO EVALUATE:

## abstraction_fitness
Python abstraction fitness: favor direct modules, explicit domain APIs, and bounded packages over indirection and generic helper surfaces.
Look for:
- Functions that only forward args/kwargs to another function without policy or translation
- Protocol/base-class abstractions with one concrete implementation and no extension pressure
- Cross-module wrapper chains where calls hop through helper layers before reaching real logic
- Project-wide reliance on generic helper modules instead of bounded domain packages
- Over-broad dict/config/context parameters used as implicit parameter bags
Skip:
- Django/FastAPI/SQLAlchemy framework boundaries that require adapters or dependency hooks
- Wrappers that add retries, metrics, auth checks, caching, or tracing
- Intentional package facades used to stabilize public import paths
- Migration shims with active callers and clear sunset plan

Seed files (start here):
- rustfuzz/utils.py
- rustfuzz/search.py
- rustfuzz/engine.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/spark.py
- rustfuzz/query.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/NGram.py
- rustfuzz/__init__.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Hamming.py
- rustfuzz/distance/Indel.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/JaroWinkler.py
- rustfuzz/distance/LCSseq.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/distance/OSA.py
- rustfuzz/distance/Postfix.py
- rustfuzz/distance/Prefix.py
- rustfuzz/distance/__init__.py
- rustfuzz/distance/_initialize.py
- rustfuzz/langchain.py
- rustfuzz/fuzz.py
- rustfuzz/filter.py

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show facade --no-budget      # 14 findings
  desloppify show props --no-budget      # 1 findings
  desloppify show responsibility_cohesion --no-budget      # 2 findings
  desloppify show structural --no-budget      # 5 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

--- Batch 7: dependency_health ---
Rationale: seed files for dependency_health review
DIMENSION TO EVALUATE:

## dependency_health
Unused deps, version conflicts, multiple libs for same purpose, heavy deps
Look for:
- Multiple libraries for the same purpose (e.g. moment + dayjs, axios + fetch wrapper)
- Heavy dependencies pulled in for light use (e.g. lodash for one function)
- Circular dependency cycles visible in the import graph
- Unused dependencies in package.json/requirements.txt
- Version conflicts or pinning issues visible in lock files
Skip:
- Dev dependencies (test, build, lint tools)
- Peer dependencies required by frameworks

Seed files (start here):
- rustfuzz/utils.py
- rustfuzz/search.py
- rustfuzz/engine.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/spark.py
- rustfuzz/query.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/NGram.py
- rustfuzz/__init__.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Hamming.py
- rustfuzz/distance/Indel.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/JaroWinkler.py
- rustfuzz/distance/LCSseq.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/distance/OSA.py
- rustfuzz/distance/Postfix.py
- rustfuzz/distance/Prefix.py
- rustfuzz/distance/__init__.py
- rustfuzz/distance/_initialize.py
- rustfuzz/langchain.py
- rustfuzz/fuzz.py
- rustfuzz/filter.py

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show cycles --no-budget      # 1 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

--- Batch 8: low_level_elegance ---
Rationale: seed files for low_level_elegance review
DIMENSION TO EVALUATE:

## low_level_elegance
Direct, precise function and class internals
Look for:
- Control flow is direct and intention-revealing; branches are necessary and distinct
- State mutation and side effects are explicit, local, and bounded
- Edge-case handling is precise without defensive sprawl
- Extraction level is balanced: avoids both monoliths and micro-fragmentation
- Helper extraction style is consistent across related modules
Skip:
- When file responsibility/package role is the PRIMARY issue, report under high_level_elegance
- When inter-module seam choreography is the PRIMARY issue, report under mid_level_elegance
- When dependency topology is the PRIMARY issue, report under cross_module_architecture
- Provable logic/type/error defects already captured by logic_clarity, type_safety, or error_consistency

Seed files (start here):
- rustfuzz/utils.py
- rustfuzz/search.py
- rustfuzz/engine.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/spark.py
- rustfuzz/query.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/NGram.py
- rustfuzz/__init__.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Hamming.py
- rustfuzz/distance/Indel.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/JaroWinkler.py
- rustfuzz/distance/LCSseq.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/distance/OSA.py
- rustfuzz/distance/Postfix.py
- rustfuzz/distance/Prefix.py
- rustfuzz/distance/__init__.py
- rustfuzz/distance/_initialize.py
- rustfuzz/langchain.py
- rustfuzz/fuzz.py
- rustfuzz/filter.py

--- Batch 9: mid_level_elegance ---
Rationale: seed files for mid_level_elegance review
DIMENSION TO EVALUATE:

## mid_level_elegance
Quality of handoffs and integration seams across modules and layers
Look for:
- Inputs/outputs across boundaries are explicit, minimal, and unsurprising
- Data translation at boundaries happens in one obvious place
- Error and lifecycle propagation across boundaries follows predictable patterns
- Orchestration reads as composition of collaborators, not tangled back-and-forth calls
- Integration seams avoid glue-code entropy (ad-hoc mappers and boundary conditionals)
Skip:
- When top-level decomposition/package shape is the PRIMARY issue, report under high_level_elegance
- When implementation craft inside one function/class is the PRIMARY issue, report under low_level_elegance
- Pure API/type contract defects with no seam design impact (belongs to contract_coherence)
- Standalone naming/style preferences that do not affect handoffs

Seed files (start here):
- rustfuzz/__init__.py
- rustfuzz/_types.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/Hamming.py

--- Batch 10: test_strategy ---
Rationale: seed files for test_strategy review
DIMENSION TO EVALUATE:

## test_strategy
Untested critical paths, coupling, snapshot overuse, fragility patterns
Look for:
- Critical paths with zero test coverage (high-importer files, core business logic)
- Test-production coupling: tests that break when implementation details change
- Snapshot test overuse: >50% of tests are snapshot-based
- Missing integration tests: unit tests exist but no cross-module verification
- Test fragility: tests that depend on timing, ordering, or external state
Skip:
- Low-value files intentionally untested (types, constants, index files)
- Generated code that shouldn't have custom tests

Seed files (start here):
- rustfuzz/_types.py

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show test_coverage --no-budget      # 1 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

--- Batch 11: api_surface_coherence ---
Rationale: seed files for api_surface_coherence review
DIMENSION TO EVALUATE:

## api_surface_coherence
Inconsistent API shapes, mixed sync/async, overloaded interfaces
Look for:
- Inconsistent API shapes: similar functions with different parameter ordering or naming
- Mixed sync/async in the same module's public API
- Overloaded interfaces: one function doing too many things based on argument types
- Missing error contracts: no documentation or types indicating what can fail
- Public functions with >5 parameters (API boundary may be wrong)
Skip:
- Internal/private APIs where flexibility is acceptable
- Framework-imposed patterns (React hooks must follow rules of hooks)

Seed files (start here):
- rustfuzz/_types.py

--- Batch 12: package_organization ---
Rationale: seed files for package_organization review
DIMENSION TO EVALUATE:

## package_organization
Directory layout quality and navigability: whether placement matches ownership and change boundaries
Look for:
- Use holistic_context.structure as objective evidence: root_files (fan_in/fan_out + role), directory_profiles (file_count/avg fan-in/out), and coupling_matrix (cross-directory edges)
- Straggler roots: root-level files with low fan-in (<5 importers) that share concern/theme with other files should move under a focused package
- Import-affinity mismatch: file imports/references are mostly from one sibling domain (>60%), but file lives outside that domain
- Coupling-direction failures: reciprocal/bidirectional directory edges or obvious downstream→upstream imports indicate boundary placement problems
- Flat directory overload: >10 files with mixed concerns and low cohesion should be split into purpose-driven subfolders
- Ambiguous folder naming: directory names do not reflect contained responsibilities
Skip:
- Root-level files that ARE genuinely core — high fan-in (≥5 importers), imported across multiple subdirectories (cli.py, state.py, utils.py, config.py)
- Small projects (<20 files) where flat structure is appropriate
- Framework-imposed directory layouts (src/, lib/, dist/, __pycache__/)
- Test directories mirroring production structure
- Aesthetic preferences without measurable navigation, ownership, or coupling impact

Seed files (start here):
- rustfuzz/__init__.py
- rustfuzz/_types.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/Hamming.py

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show flat_dirs --no-budget      # 1 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

--- Batch 13: initialization_coupling ---
Rationale: seed files for initialization_coupling review
DIMENSION TO EVALUATE:

## initialization_coupling
Boot-order dependencies, import-time side effects, global singletons
Look for:
- Module-level code that depends on another module having been imported first
- Import-time side effects: DB connections, file I/O, network calls at module scope
- Global singletons where creation order matters across modules
- Environment variable reads at import time (fragile in testing)
- Circular init dependencies hidden behind conditional or lazy imports
- Module-level constants computed at import time alongside a dynamic getter function — consumers referencing the stale snapshot instead of calling the getter
Skip:
- Standard library initialization (logging.basicConfig)
- Framework bootstrap (app.configure, server.listen)

Seed files (start here):
- rustfuzz/filter.py
- rustfuzz/search.py
- rustfuzz/engine.py
- rustfuzz/langchain.py
- rustfuzz/join.py
- rustfuzz/spark.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/_types.py
- rustfuzz/compat.py

--- Batch 14: design_coherence ---
Rationale: seed files for design_coherence review
DIMENSION TO EVALUATE:

## design_coherence
Are structural design decisions sound — functions focused, abstractions earned, patterns consistent?
Look for:
- Functions doing too many things — multiple distinct responsibilities in one body
- Parameter lists that should be config/context objects — many related params passed together
- Files accumulating issues across many dimensions — likely mixing unrelated concerns
- Deep nesting that could be flattened with early returns or extraction
- Repeated structural patterns that should be data-driven
Skip:
- Functions that are long but have a single coherent responsibility
- Parameter lists where grouping would obscure meaning — do NOT recommend config/context objects or dependency injection wrappers just to reduce parameter count; only group when the grouping has independent semantic meaning
- Files that are large because their domain is genuinely complex, not because they mix concerns
- Nesting that is inherent to the problem (e.g., recursive tree processing)
- Do NOT recommend extracting callable parameters or injecting dependencies for 'testability' — direct function calls are simpler and preferred unless there is a concrete decoupling need

Seed files (start here):
- rustfuzz/filter.py
- rustfuzz/search.py
- rustfuzz/engine.py
- rustfuzz/langchain.py
- rustfuzz/join.py
- rustfuzz/spark.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/_types.py
- rustfuzz/compat.py

Mechanical concern signals — investigate and adjudicate:
Overview (9 signals):
  design_concern: 4 — rustfuzz/distance/Gotoh.py, rustfuzz/join.py, ...
  mixed_responsibilities: 3 — rustfuzz/engine.py, rustfuzz/filter.py, rustfuzz/search.py
  systemic_smell: 2 — rustfuzz/_types.py, rustfuzz/compat.py

For each concern, read the source code and report your verdict in issues[]:
  - Confirm → full issue object with concern_verdict: "confirmed"
  - Dismiss → minimal object: {concern_verdict: "dismissed", concern_fingerprint: "<hash>"}
    (only these 2 fields required — add optional reasoning/concern_type/concern_file)
  - Unsure → skip it (will be re-evaluated next review)

  - [design_concern] rustfuzz/distance/Gotoh.py
    summary: Design signals from private_imports, smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: private_imports, smells
    evidence: [smells] 4x Too many optional params — consider a config object
    fingerprint: 0b8110e93e0f542b
  - [design_concern] rustfuzz/join.py
    summary: Design signals from props, smells
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: props, smells
    evidence: [props] Passthrough: fuzzy_join (11/14 forwarded, 79%)
    fingerprint: e4c590a39c145854
  - [design_concern] rustfuzz/langchain.py
    summary: Design signals from orphaned, smells
    question: Is this file truly dead, or is it used via a non-import mechanism (dynamic import, CLI entry point, plugin)?
    evidence: Flagged by: orphaned, smells
    evidence: [orphaned] Orphaned file (160 LOC): zero importers, not an entry point
    fingerprint: a9a81f6b72b0c6f9
  - [design_concern] rustfuzz/spark.py
    summary: Design signals from smells, uncalled_functions
    question: Review the flagged patterns — are they design problems that need addressing, or acceptable given the file's role?
    evidence: Flagged by: smells, uncalled_functions
    evidence: [uncalled_functions] Uncalled private function: _make_udf() — 26 LOC, zero references
    fingerprint: b8250131e6256e5b
  - [mixed_responsibilities] rustfuzz/engine.py
    summary: Issues from 3 detectors — may have too many responsibilities
    question: This file has issues across 3 dimensions (signature, smells, structural). Is it trying to do too many things, or is this complexity inherent to its domain?
    evidence: Flagged by: signature, smells, structural
    evidence: File size: 808 lines
    fingerprint: ad681fff83f61084
  - [mixed_responsibilities] rustfuzz/filter.py
    summary: Issues from 3 detectors — may have too many responsibilities
    question: This file has issues across 3 dimensions (dupes, smells, structural). Is it trying to do too many things, or is this complexity inherent to its domain? Can the nesting be reduced with early returns, guard clauses, or extraction into helper functions? Is the duplication worth extracting into a shared utility, or is it intentional variation?
    evidence: Flagged by: dupes, smells, structural
    evidence: File size: 740 lines
    fingerprint: 7b8f6bf03ead840c
  - [mixed_responsibilities] rustfuzz/search.py
    summary: Issues from 3 detectors — may have too many responsibilities
    question: This file has issues across 3 dimensions (dupes, smells, structural). Is it trying to do too many things, or is this complexity inherent to its domain? Is the duplication worth extracting into a shared utility, or is it intentional variation?
    evidence: Flagged by: dupes, smells, structural
    evidence: File size: 1035 lines
    fingerprint: 31a98d04727a67bc
  - [systemic_smell] rustfuzz/_types.py
    summary: 'deferred_import' appears in 9 files — likely a systemic pattern
    question: The smell 'deferred_import' appears across 9 files. Is this a codebase-wide convention that should be addressed systemically (lint rule, shared utility, architecture change), or are these independent occurrences?
    evidence: Smell: deferred_import
    evidence: Affected files (9): rustfuzz/_types.py, rustfuzz/compat.py, rustfuzz/engine.py, rustfuzz/filter.py, rustfuzz/langchain.py, rustfuzz/process.py, rustfuzz/query.py, rustfuzz/search.py, rustfuzz/spark.py
    fingerprint: 565bab4c90e82257
  - [systemic_smell] rustfuzz/compat.py
    summary: 'high_cyclomatic_complexity' appears in 7 files — likely a systemic pattern
    question: The smell 'high_cyclomatic_complexity' appears across 7 files. Is this a codebase-wide convention that should be addressed systemically (lint rule, shared utility, architecture change), or are these independent occurrences?
    evidence: Smell: high_cyclomatic_complexity
    evidence: Affected files (7): rustfuzz/compat.py, rustfuzz/engine.py, rustfuzz/filter.py, rustfuzz/join.py, rustfuzz/query.py, rustfuzz/search.py, rustfuzz/sort.py
    fingerprint: 975a48fc7b86f11b

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show dupes --no-budget      # 11 findings
  desloppify show orphaned --no-budget      # 1 findings
  desloppify show private_imports --no-budget      # 1 findings
  desloppify show props --no-budget      # 1 findings
  desloppify show signature --no-budget      # 3 findings
  desloppify show smells --no-budget      # 25 findings
  desloppify show structural --no-budget      # 3 findings
  desloppify show uncalled_functions --no-budget      # 1 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

--- Batch 15: contract_coherence ---
Rationale: seed files for contract_coherence review
DIMENSION TO EVALUATE:

## contract_coherence
Functions and modules that honor their stated contracts
Look for:
- Return type annotation lies: declared type doesn't match all return paths
- Docstring/signature divergence: params described in docs but not in function signature
- Functions named getX that mutate state (side effect hidden behind getter name)
- Module-level API inconsistency: some exports follow a pattern, one doesn't
- Error contracts: function says it throws but silently returns None, or vice versa
Skip:
- Protocol/interface stubs (abstract methods with placeholder returns)
- Test helpers where loose typing is intentional
- Overloaded functions with multiple valid return types

Seed files (start here):
- rustfuzz/utils.py
- rustfuzz/search.py
- rustfuzz/engine.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/spark.py
- rustfuzz/query.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/NGram.py
- rustfuzz/__init__.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Hamming.py
- rustfuzz/distance/Indel.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/JaroWinkler.py
- rustfuzz/distance/LCSseq.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/distance/OSA.py
- rustfuzz/distance/Postfix.py
- rustfuzz/distance/Prefix.py
- rustfuzz/distance/__init__.py
- rustfuzz/distance/_initialize.py
- rustfuzz/langchain.py
- rustfuzz/fuzz.py
- rustfuzz/filter.py

--- Batch 16: logic_clarity ---
Rationale: seed files for logic_clarity review
DIMENSION TO EVALUATE:

## logic_clarity
Control flow and logic that provably does what it claims
Look for:
- Identical if/else or ternary branches (same code on both sides)
- Dead code paths: code after unconditional return/raise/throw/break
- Always-true or always-false conditions (e.g. checking a constant)
- Redundant null/undefined checks on values that cannot be null
- Async functions that never await (synchronous wrapped in async)
- Boolean expressions that simplify: `if x: return True else: return False`
Skip:
- Deliberate no-op branches with explanatory comments
- Framework lifecycle methods that must be async by contract
- Guard clauses that are defensive by design

Seed files (start here):
- rustfuzz/utils.py
- rustfuzz/search.py
- rustfuzz/engine.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/spark.py
- rustfuzz/query.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/NGram.py
- rustfuzz/__init__.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Hamming.py
- rustfuzz/distance/Indel.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/JaroWinkler.py
- rustfuzz/distance/LCSseq.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/distance/OSA.py
- rustfuzz/distance/Postfix.py
- rustfuzz/distance/Prefix.py
- rustfuzz/distance/__init__.py
- rustfuzz/distance/_initialize.py
- rustfuzz/langchain.py
- rustfuzz/fuzz.py
- rustfuzz/filter.py

RELEVANT FINDINGS — explore with CLI:
These detectors found patterns related to this dimension. Explore the findings,
then read the actual source code.

  desloppify show smells --no-budget      # 27 findings
  desloppify show structural --no-budget      # 5 findings

Report actionable issues in issues[]. Use concern_verdict and concern_fingerprint
for findings you want to confirm or dismiss.

--- Batch 17: type_safety ---
Rationale: seed files for type_safety review
DIMENSION TO EVALUATE:

## type_safety
Type annotations that match runtime behavior
Look for:
- Return type annotations that don't cover all code paths (e.g., -> str but can return None)
- Parameters typed as X but called with Y (e.g., str param receiving None)
- Union types that could be narrowed (Optional used where None is never valid)
- Missing annotations on public API functions
- Type: ignore comments without explanation
- TypedDict fields marked Required but accessed via .get() with defaults — the type promises a shape the code doesn't trust
- Parameters typed as dict[str, Any] where a specific TypedDict or dataclass exists
- Enum types defined in the codebase but bypassed with raw string or int literal comparisons — see enum_bypass_patterns evidence
- Parallel type definitions: a Literal alias that duplicates an existing enum's values
Skip:
- Untyped private helpers in well-typed modules
- Dynamic framework code where typing is impractical
- Test code with loose typing

Seed files (start here):
- rustfuzz/utils.py
- rustfuzz/search.py
- rustfuzz/engine.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/spark.py
- rustfuzz/query.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/NGram.py
- rustfuzz/__init__.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Hamming.py
- rustfuzz/distance/Indel.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/JaroWinkler.py
- rustfuzz/distance/LCSseq.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/distance/OSA.py
- rustfuzz/distance/Postfix.py
- rustfuzz/distance/Prefix.py
- rustfuzz/distance/__init__.py
- rustfuzz/distance/_initialize.py
- rustfuzz/langchain.py
- rustfuzz/fuzz.py
- rustfuzz/filter.py

--- Batch 18: ai_generated_debt ---
Rationale: no direct batch mapping for ai_generated_debt; using representative files
DIMENSION TO EVALUATE:

## ai_generated_debt
LLM-hallmark patterns: restating comments, defensive overengineering, boilerplate
Look for:
- Restating comments that echo the code without adding insight (// increment counter above i++)
- Nosy debug logging: entry/exit logs on every function, full object dumps to console
- Defensive overengineering: null checks on non-nullable typed values, try-catch around pure expressions
- Docstring bloat: multi-line docstrings on trivial 2-line functions
- Pass-through wrapper functions with no added logic (just forward args to another function)
- Generic names in domain code: handleData, processItem, doOperation where domain terms exist
- Identical boilerplate error handling copied verbatim across multiple files
Skip:
- Comments explaining WHY (business rules, non-obvious constraints, external dependencies)
- Defensive checks at genuine API boundaries (user input, network, file I/O)
- Generated code (protobuf, GraphQL codegen, ORM migrations)
- Wrapper functions that add auth, logging, metrics, or caching

Seed files (start here):
- rustfuzz/search.py
- rustfuzz/__init__.py
- rustfuzz/distance/__init__.py
- rustfuzz/fuzz.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/distance/JaroWinkler.py
- rustfuzz/_types.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/Hamming.py
- rustfuzz/utils.py
- rustfuzz/filter.py
- rustfuzz/engine.py
- rustfuzz/spark.py
- rustfuzz/query.py
- rustfuzz/distance/NGram.py
- rustfuzz/distance/Indel.py
- rustfuzz/distance/LCSseq.py
- rustfuzz/distance/OSA.py
- rustfuzz/distance/Postfix.py
- rustfuzz/distance/Prefix.py
- rustfuzz/distance/_initialize.py
- rustfuzz/langchain.py

--- Batch 19: authorization_consistency ---
Rationale: no direct batch mapping for authorization_consistency; using representative files
DIMENSION TO EVALUATE:

## authorization_consistency
Auth/permission patterns consistently applied across the codebase
Look for:
- Route handlers with auth decorators/middleware on some siblings but not others
- RLS enabled on some tables but not siblings in the same domain
- Permission strings as magic literals instead of shared constants
- Mixed trust boundaries: some endpoints validate user input, siblings don't
- Service role / admin bypass without audit logging or access control
Skip:
- Public routes explicitly documented as unauthenticated (health checks, login, webhooks)
- Internal service-to-service calls behind network-level auth
- Dev/test endpoints behind feature flags or environment checks

Seed files (start here):
- rustfuzz/search.py
- rustfuzz/__init__.py
- rustfuzz/distance/__init__.py
- rustfuzz/fuzz.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/distance/JaroWinkler.py
- rustfuzz/_types.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/Hamming.py
- rustfuzz/utils.py
- rustfuzz/filter.py
- rustfuzz/engine.py
- rustfuzz/spark.py
- rustfuzz/query.py
- rustfuzz/distance/NGram.py
- rustfuzz/distance/Indel.py
- rustfuzz/distance/LCSseq.py
- rustfuzz/distance/OSA.py
- rustfuzz/distance/Postfix.py
- rustfuzz/distance/Prefix.py
- rustfuzz/distance/_initialize.py
- rustfuzz/langchain.py

--- Batch 20: incomplete_migration ---
Rationale: no direct batch mapping for incomplete_migration; using representative files
DIMENSION TO EVALUATE:

## incomplete_migration
Old+new API coexistence, deprecated-but-called symbols, stale migration shims
Look for:
- Old and new API patterns coexisting: class+functional components, axios+fetch, moment+dayjs
- Deprecated symbols still called by active code (@deprecated, DEPRECATED markers)
- Compatibility shims that no caller actually needs anymore
- Mixed JS/TS files for the same module (incomplete TypeScript migration)
- Stale migration TODOs: TODO/FIXME referencing 'migrate', 'legacy', 'old api', 'remove after'
Skip:
- Active, intentional migrations with tracked progress
- Backward-compatibility for external consumers (published APIs, libraries)
- Gradual rollouts behind feature flags with clear ownership

Seed files (start here):
- rustfuzz/search.py
- rustfuzz/__init__.py
- rustfuzz/distance/__init__.py
- rustfuzz/fuzz.py
- rustfuzz/distance/Jaro.py
- rustfuzz/distance/Levenshtein.py
- rustfuzz/join.py
- rustfuzz/process.py
- rustfuzz/distance/JaroWinkler.py
- rustfuzz/_types.py
- rustfuzz/compat.py
- rustfuzz/distance/DamerauLevenshtein.py
- rustfuzz/distance/Gotoh.py
- rustfuzz/distance/Hamming.py
- rustfuzz/utils.py
- rustfuzz/filter.py
- rustfuzz/engine.py
- rustfuzz/spark.py
- rustfuzz/query.py
- rustfuzz/distance/NGram.py
- rustfuzz/distance/Indel.py
- rustfuzz/distance/LCSseq.py
- rustfuzz/distance/OSA.py
- rustfuzz/distance/Postfix.py
- rustfuzz/distance/Prefix.py
- rustfuzz/distance/_initialize.py
- rustfuzz/langchain.py

YOUR TASK: Read the code for this batch's dimension. Judge how well the codebase serves a developer from that perspective. The dimension rubric above defines what good looks like. Cite specific observations that explain your judgment.

Mechanical scan evidence — navigation aid, not scoring evidence:
The blind packet contains `holistic_context.scan_evidence` with aggregated signals from all mechanical detectors — including complexity hotspots, error hotspots, signal density index, boundary violations, and systemic patterns. Use these as starting points for where to look beyond the seed files.

Task requirements:
1. Read the blind packet's `system_prompt` — it contains scoring rules and calibration.
2. Start from the seed files, then freely explore the repository to build your understanding.
3. Keep issues and scoring scoped to this batch's dimension.
4. Respect scope controls: do not include files/directories marked by `exclude`, `suppress`, or non-production zone overrides.
5. Return 0-200 issues for this batch (empty array allowed).
6. For package_organization, ground scoring in objective structure signals from `holistic_context.structure` (root_files fan_in/fan_out roles, directory_profiles, coupling_matrix). Prefer thresholded evidence (for example: fan_in < 5 for root stragglers, import-affinity > 60%, directories > 10 files with mixed concerns).
7. Suggestions must include a staged reorg plan (target folders, move order, and import-update/validation commands).
8. Also consult `holistic_context.structure.flat_dir_issues` for directories flagged as overloaded, fragmented, or thin-wrapper patterns.
9. For abstraction_fitness, use evidence from `holistic_context.abstractions`:
10. - `delegation_heavy_classes`: classes where most methods forward to an inner object — entries include class_name, delegate_target, sample_methods, and line number.
11. - `facade_modules`: re-export-only modules with high re_export_ratio — entries include samples (re-exported names) and loc.
12. - `typed_dict_violations`: TypedDict fields accessed via .get()/.setdefault()/.pop() — entries include typed_dict_name, violation_type, field, and line number.
13. - `complexity_hotspots`: files where mechanical analysis found extreme parameter counts, deep nesting, or disconnected responsibility clusters.
14. Include `delegation_density`, `definition_directness`, and `type_discipline` alongside existing sub-axes in dimension_notes when evidence supports it.
15. For initialization_coupling, use evidence from `holistic_context.scan_evidence.mutable_globals` and `holistic_context.errors.mutable_globals`. Investigate initialization ordering dependencies, coupling through shared mutable state, and whether state should be encapsulated behind a proper registry/context manager.
16. For design_coherence, use evidence from `holistic_context.scan_evidence.signal_density` — files where multiple mechanical detectors fired. Investigate what design change would address multiple signals simultaneously. Check `scan_evidence.complexity_hotspots` for files with high responsibility cluster counts.
17. For error_consistency, use evidence from `holistic_context.errors.exception_hotspots` — files with concentrated exception handling issues. Investigate whether error handling is designed or accidental. Check for broad catches masking specific failure modes.
18. For cross_module_architecture, also consult `holistic_context.coupling.boundary_violations` for import paths that cross architectural boundaries, and `holistic_context.dependencies.deferred_import_density` for files with many function-level imports (proxy for cycle pressure).
19. For convention_outlier, also consult `holistic_context.conventions.duplicate_clusters` for cross-file function duplication and `conventions.naming_drift` for directory-level naming inconsistency.
20. Workflow integrity checks: when reviewing orchestration/queue/review flows,
21. xplicitly look for loop-prone patterns and blind spots:
22. - repeated stale/reopen churn without clear exit criteria or gating,
23. - packet/batch data being generated but dropped before prompt execution,
24. - ranking/triage logic that can starve target-improving work,
25. - reruns happening before existing open review work is drained.
26. If found, propose concrete guardrails and where to implement them.
27. Complete `dimension_judgment` for your dimension — all three fields (strengths, issue_character, score_rationale) are required. Write the judgment BEFORE setting the score.
28. Do not edit repository files.
29. Return ONLY valid JSON, no markdown fences.

Scope enums:
- impact_scope: "local" | "module" | "subsystem" | "codebase"
- fix_scope: "single_edit" | "multi_file_refactor" | "architectural_change"

Output schema:
{
  "session": {"id": "<preserve from template>", "token": "<preserve from template>"},
  "assessments": {"<dimension>": <0-100 with one decimal place>},
  "dimension_notes": {
    "<dimension>": {
      "evidence": ["specific code observations"],
      "impact_scope": "local|module|subsystem|codebase",
      "fix_scope": "single_edit|multi_file_refactor|architectural_change",
      "confidence": "high|medium|low"
    }
  },
  "issues": [{
    "dimension": "<dimension>",
    "identifier": "short_id",
    "summary": "one-line defect summary",
    "related_files": ["relative/path.py"],
    "evidence": ["specific code observation"],
    "suggestion": "concrete fix recommendation",
    "confidence": "high|medium|low",
    "impact_scope": "local|module|subsystem|codebase",
    "fix_scope": "single_edit|multi_file_refactor|architectural_change",
    "root_cause_cluster": "optional_cluster_name",
    "concern_verdict": "confirmed|dismissed  // for concern signals only",
    "concern_fingerprint": "abc123  // required when dismissed; copy from signal fingerprint",
    "reasoning": "why dismissed  // optional, for dismissed only"
  }]
}

Session requirements:
1. Keep `session.id` exactly `ext_20260312_203918_cbcc343a`.
2. Keep `session.token` exactly `306dfa449ce832d20f627f89463bb2e3`.
3. Do not include provenance metadata (CLI injects canonical provenance).

