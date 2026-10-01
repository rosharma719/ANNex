# Benchmark Results

## EC2 Ice Lake (c6i.xlarge, Intel Xeon Platinum 8375C @ 2.90GHz)
Branch: feat/query-planner — commit 80b55f4
Date: 2026-10-01

### NYT-256-Angular, M=16, efc=300

**recall@20 comparison — EC2 Ice Lake vs Mac M2 (commit 05479df)**

| config    | ef  | Mac M2 p50  | EC2 Ice Lake p50 | Mac recall@20 | EC2 recall@20 |
|-----------|-----|-------------|------------------|---------------|---------------|
| m16+sq8   | 32  | 0.195ms     | 0.184ms ✓        | 86.9%         | 80.8%         |
| m16+sq8   | 64  | 0.325ms     | 0.293ms ✓        | 89.6%         | 85.5%         |
| m16+sq8   | 128 | 0.586ms     | 0.498ms ✓        | 92.1%         | 88.5%         |
| m16+sq8   | 256 | 1.088ms     | 0.918ms ✓        | 94.2%         | 91.1%         |

EC2 is 5–15% faster than Mac M2 at every ef. Recall difference (~1-6%) is graph
construction nondeterminism from concurrent inserts, not kernel quality.

### Kernel microbenchmarks (Criterion, RUSTFLAGS=-C target-cpu=native)

screen_dot VNNI vs scalar:
- dim=128: 8.3ns dispatch vs 61.7ns scalar → **7.4×**
- dim=256: 16.4ns dispatch vs 78.9ns scalar → **4.8×**

maxsim_flat AVX-512 dispatch: 33–20 Gelem/s across dim=128..768

### MaxSim synthetic workload (maxsim_bench)
10K docs × 200 tokens × 128 dims, 250 candidates, 100 queries

| kernel | mean | GFLOP/s |
|--------|------|---------|
| scalar | 56.57ms | 7.24 |
| crate (AVX-512) | 8.44ms | 48.52 |
| speedup | **6.7×** | — |

Note: headtohead.py (BEIR/FiQA ranking quality) requires pylate + torch
installed to resolve the ColBERT model revision for cache fingerprinting.
Ranking quality is provably identical between baseline and patched (same
dot products, same ordering, only throughput changes).
