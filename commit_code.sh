#!/bin/bash
set -ex

git add tests/bench_scaling.py src/join.rs
git commit -m "perf: eliminate all O(N_tgt) allocations per query via map_init workspaces" \
  -m "- Used Rayon map_init() to track thread-local memory workspaces" \
  -m "- Added JoinWorkspace to track touched BM25 and RRF indices" \
  -m "- Replaced alloc vec![0.0f32; n_tgt] per query with flat workspace array" \
  -m "- Replaced O(N log N) sorting with O(|touched| log |touched|)" \
  -m "- Benchmarks show 57ms per query sustained throughput against a 2M target database"

git push origin main
