# Multi-Surface Kernel Optimization

**Date:** 2026-10-01 (revised ×2 after review)
**Branch:** feat/query-planner
**Author:** rosharma719

---

## Goal

Vectorize all hot kernels for every real deployment surface, eliminate in-loop dispatch overhead, and produce benchmarks that prove gains — including real-dataset MaxSim validation and a structured competitor comparison.

Three sequential stages:
1. **Correct kernels + per-backend microbenchmarks** (this spec)
2. **EC2 end-to-end validation** — HNSW and MaxSim against ANNex baseline at matched recall
3. **Competitor comparison** — separate project; methodology defined at end of this doc

---

## Current State

### `annex-core` — HNSW search kernels (`core.rs`)

| Kernel | ARM64 | AVX2 | AVX2+FMA | AVX-512 |
|---|---|---|---|---|
| `dot_product` | NEON ✓ | AVX2 ✓ | AVX2+FMA ✓ | — |
| `l2_squared` | NEON ✓ | AVX2 ✓ | AVX2+FMA ✓ | — |
| `screen_dot` | NEON sdot ✓ | AVX2 ✓ (widening) | — | — |

All three dispatch via `is_x86_feature_detected!()` inside the function body. `fast_score()` (which calls `dot_product`/`l2_squared`) is called at 14 sites in `search.rs` — inside BFS candidate loops, filtered-search loops, exact-scan loops, insertion, and graph-maintenance.

**`screen_dot_avx2` status:** the existing widening-to-i16 path is the correctness baseline and must not be replaced with `_mm256_maddubs_epi16` without proving adjacent-pair sums cannot exceed `i16::MAX` (32767). `maddubs` saturates: `255×127 + 255×127 = 64770` overflows. This rewrite is deferred out of scope.

### `annex-multivector` — MaxSim/FDE kernels (`fde.rs`)

| Kernel | ARM64 | AVX2+FMA | AVX-512 |
|---|---|---|---|
| `dot` | NEON FMA ✓ | — scalar only | — |
| `maxsim_flat` | NEON ✓ + dim=128 | — scalar only | — |

### Benchmarks

`benches/kernels.rs` (Criterion) covers `maxsim_flat` and `maxsim_naive`. `screen_dot` has no Criterion benchmark.

---

## Dispatch Architecture

### What Rust already does

`std_detect` caches CPU feature flags in an atomic initialized on first use. Each `is_x86_feature_detected!("avx2")` call is an atomic load + bit test — not a CPUID re-execution. Adding a `OnceLock<CpuLevel>` on top reduces nothing.

**The real overhead** is that dispatch branches live inside functions called per candidate or per token. The goal is to select the kernel function *once per search* (outside the candidate loop) rather than once per candidate.

### `CpuLevel` enum

A `CpuLevel` value is computed by the caller outside the hot loop to select a function pointer; it is not stored globally.

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CpuLevel {
    NeonDotprod,  // aarch64 + dotprod
    Neon,         // aarch64 baseline
    Avx512Vnni,   // avx512f + avx512vnni
    Avx512F,      // avx512f, no vnni
    Avx2Fma,      // avx2 + fma
    Avx2,         // avx2 only
    Scalar,
}
pub fn cpu_level() -> CpuLevel { /* detects once per call; std_detect already caches */ }
```

### Per-kernel dispatch table

Each kernel selects its best implementation independently. The complete dispatch for all new and existing kernels:

**`dot_product` / `l2_squared`:**
- `Avx512Vnni | Avx512F` → new `dot_avx512` / `l2_avx512`
- `Avx2Fma` → existing `dot_avx2_fma` / `l2_avx2_fma`
- `Avx2` → existing `dot_avx2` / `l2_avx2` ← **must not regress**
- `NeonDotprod | Neon` → existing `dot_neon` / `l2_neon`
- `Scalar` → scalar

**`screen_dot`:**
- `Avx512Vnni` → new `screen_dot_avx512_vnni`
- `Avx2Fma | Avx2` → existing `screen_dot_avx2` (widening, unchanged)
- `NeonDotprod` → existing `screen_dot_neon_sdot`
- `Neon | Scalar` → scalar

**`fde::dot` / `maxsim_flat`:**
- `Avx512Vnni | Avx512F` → new `dot_avx512` / `maxsim_flat_avx512`
- `Avx2Fma` → new `dot_avx2_fma` / `maxsim_flat_avx2_fma`
- `Avx2` → new `dot_avx2` / `maxsim_flat_avx2` (2-accumulator variant, no FMA)
- `NeonDotprod | Neon` → existing NEON paths
- `Scalar` → scalar

### Hoisting `fast_score` dispatch in `HNSWIndex`

`fast_score` is a method on `HNSWIndex` calling `dot_product`/`l2_squared` at 14 call sites across search, insertion, and graph-maintenance paths. Hoisting dispatch at each call site individually is fragile and incomplete.

**Design:** store a scorer on the `HNSWIndex` struct, selected once at construction/open time:

```rust
struct HNSWIndex {
    ...
    dot_fn: fn(&[f32], &[f32]) -> f32,
    l2_fn:  fn(&[f32], &[f32]) -> f32,
}
```

`fast_score` calls `self.dot_fn` or `self.l2_fn` directly. All 14 call sites are covered with no signature changes to search paths.

**Indirect-call caveat.** An indirect function-pointer call may cost as much as or more than the current predictable branch on a well-trained predictor. **Measure before claiming a win.** The acceptance criterion is:

> Keep the struct-stored scorer only if it produces no statistically meaningful regression vs. the current dispatch at representative dimensions. If indirect-call overhead is measurable, retain cached branch dispatch (`is_x86_feature_detected!`) in `dot_product`/`l2_squared` and specialize the enclosing loop instead.

The benchmark section below specifies how to measure this.

---

## Kernel Additions

All throughput figures are theoretical maximums based on register widths. Treat as hypotheses; validate with generated assembly and Criterion numbers before asserting improvement.

### `annex-core/src/vector/hnsw/core.rs`

#### `dot_avx512` / `l2_avx512`

`#[target_feature(enable = "avx512f")]`. Use `_mm512_fmadd_ps` with multiple `__m512` accumulators across the loop. Reduction: accumulate at loop end; `_mm512_reduce_add_ps` is a multi-instruction convenience — inspect generated assembly and replace with an explicit extract-and-add tree if the compiler emits suboptimal code. Scalar tail for `len % 16`.

Number of accumulators: start with 8 (matching NEON's register count), adjust after benchmarking.

#### `screen_dot_avx512_vnni`

`#[target_feature(enable = "avx512f,avx512vnni")]`. Uses `_mm512_dpbusd_epi32(acc, u8_stored, i8_query)` which computes `acc += sum(u8[i] * i8[i])` over 64 elements per call.

**Centering correction.** The kernel contract is `sum(query_i8[i] × (stored_u8[i] − 128))`. Expanding:

```
sum(query_i8[i] × stored_u8[i]) − 128 × sum(query_i8[i])
```

The VNNI accumulator computes the first term. The second term — `128 × sum(query_i8)` — is a scalar precomputed **once per query** (outside the document loop) and subtracted from the VNNI result after reduction.

**Length contract.** Precomputing the correction once per query is only valid when every stored vector has the same dimension as the query. In the HNSW index this is an invariant (all vectors share `self.dim`). The public `screen_dot` function must document: when called via the HNSW path, `query_i8.len() == stored.len()` is guaranteed. When lengths differ (the existing `min` contract), fall through to scalar.

**Overflow safety.** Correction value = `128 × sum(query_i8)`. Worst case: `dim=4096`, all values at `+127`. `4096 × 127 × 128 = 66,584,576 < i32::MAX (2,147,483,647)`. Safe. VNNI accumulator worst case: `4096 × 255 × 127 = 132,464,640 < i32::MAX`. Document both bounds in the implementation.

4 accumulators covering 64 elements each = 256 elements/iteration before tail.

### `annex-multivector/src/fde.rs`

#### `dot_avx2_fma` / `dot_avx512`

Mirror `dot_neon_multiple_of_16`. Multiple accumulators over the multiple-of-16 prefix; scalar tail. Reduction without `_mm256_hadd_ps`: use `_mm256_extractf128_ps` + `_mm_add_ps` + `_mm_movehl_ps` + `_mm_add_ss`.

`dot_avx2` (no-FMA variant): 2 accumulators using `_mm256_add_ps(_mm256_mul_ps(...))`.

#### `maxsim_flat_avx2_fma` / `maxsim_flat_avx512`

Mirror `maxsim_flat_neon`. General path calls per-doc-token dot; dim=128 specialization uses a const-length variant allowing full loop unrolling. Dispatch: call `cpu_level()` once before the query-token outer loop; use a function pointer inside the inner loop.

---

## Correctness

### Required: direct per-backend tests

Tests must call each kernel function directly (not through public dispatch), gated by `#[cfg(target_arch = "x86_64")]` with a runtime feature check. Testing `dot_product(...)` on AVX-512 hardware does not test the AVX2 or scalar paths.

**For `screen_dot` integer kernels — required cases per backend:**
- Empty input (`len = 0`)
- Below SIMD width: lengths 1, 4, 15, 16, 31, 32
- SIMD boundaries: 32, 64, 128, 256
- Integer extremes: `query_i8 = [127; N]` with `stored_u8 = [255; N]` and `[0; N]`
- Adversarial pairs: `q=[127,127,…]`, `s=[255,255,…]` (validates saturation behavior of any future rewrite)
- Correctness: exact integer equality with `screen_dot_scalar`

**For f32 kernels — required cases per backend:**
- Below SIMD width: 1, 4, 7, 8, 15, 16, 31, 32
- SIMD boundaries: 16, 32, 64, 128, 256, 512
- Floating-point cancellation: alternating `+1.0` and `-1.0`
- Tolerance formula: `abs(actual − reference) ≤ 1e-3 × max(1.0, abs(reference))`

### Benchmark access for `screen_dot`

`screen_dot` is `pub(crate)` in `annex-core`. Add a thin re-export in `annex-core` behind a Cargo feature:

```toml
# annex-core/Cargo.toml
[features]
bench-internals = []
```

```rust
// annex-core/src/lib.rs or vector/mod.rs
#[cfg(feature = "bench-internals")]
pub use crate::vector::hnsw::core::{screen_dot, screen_dot_scalar};
```

`annex-multivector/Cargo.toml` enables this feature for the bench binary only:

```toml
[[bench]]
name = "kernels"
required-features = []  # bench-internals enabled via dev-dependency feature activation

[dev-dependencies]
annex = { path = "../annex-core", version = "0.2.0", features = ["bench-internals"] }
```

---

## Benchmark Additions

### Criterion (`benches/kernels.rs`)

The benchmark must measure dispatch overhead directly, not just kernel throughput. For each of `screen_dot` and `maxsim_flat`, benchmark four variants at representative dimensions:

1. **Existing public dispatch** — current behavior baseline
2. **Hoisted function pointer** — selected before the loop, called unconditionally inside
3. **Direct backend invocation** — call the SIMD function directly (requires `bench-internals`)
4. **Scalar reference**

This produces a 4-row table per dimension that shows whether the hoist helps, hurts, or is within noise vs. the current dispatch. Use this data to make the hoist/no-hoist decision.

**New group `bench_screen_dot`:** dim=128, dim=256, dim=768, four variants each.

**`bench_maxsim_flat` additions:** dim=512, dim=768 added to existing dim=128, dim=384 cases; four variants each where practical.

**`maxsim-bench` binary** (Stage 1): run the existing `maxsim_bench` binary on EC2 to compare scalar, AVX2, AVX-512 paths at the ColBERT workload shape (10K docs × 200 tokens × 128 dims, 250 candidates). This isolates raw kernel throughput before the full pipeline.

---

## Stage 2 — EC2 End-to-End Validation

Run after all Stage 1 kernels pass backend tests and the Criterion bench shows no regression.

### HNSW kernel validation

ANNex `master` vs. patched branch on NYT-256-angular. Same M=16 + SQ8 screening + RCM config. Warm up with at least 3 full query sweeps before recording. Fix CPU affinity and thread count. Randomize baseline/patched run order. Report:

- Recall@10 at matched ef_search settings
- Median query latency with confidence interval over paired query deltas
- No improvement is claimed if the CI overlaps zero

### MaxSim kernel validation

Use the real BEIR/ColBERT harness — `maxsim-bench` is synthetic and belongs in Stage 1:

```bash
benchmark/headtohead.py \
  --dataset beir/fiqa/test \
  --engines annex_exact,annex_hnsw \
  --annex-candidates 250
```

Run with identical cached ColBERT embeddings, query partition, candidate count, and settings for both baseline and patched builds. Record:

- MaxSim-stage time via `MULTIVECTOR_TIMING` instrumentation
- Total query p50/p95
- Ranking agreement (NDCG@10, MRR) between baseline and patched
- Exact baseline and patched commit SHAs in the report

Claim no improvement unless the MaxSim-stage CI does not overlap zero.

---

## Stage 3 — Competitor Comparison (Separate Project)

Comparing ANNex against Qdrant, Weaviate, and pgvector requires its own spec covering:
- Controlled dataset version pinning (NYT-256-angular, BEIR)
- Equal `M`, `ef_construction`, thread count, and RAM limits per system
- Network/client overhead isolated (loopback or UNIX socket where available)
- Warmup protocol and measurement window
- Published configurations commited to the repo

This is a non-trivial methodology effort. Open a new brainstorm when Stage 2 is complete.

---

## Files Changed

| File | Change |
|---|---|
| `crates/annex-core/src/vector/simd.rs` | **New** — `CpuLevel` enum + `cpu_level()` |
| `crates/annex-core/src/vector/mod.rs` | Re-export `simd::cpu_level` |
| `crates/annex-core/src/vector/hnsw/core.rs` | Add `dot_avx512`, `l2_avx512`, `screen_dot_avx512_vnni`; store scorer on `HNSWIndex`; per-backend correctness tests; `bench-internals` re-export |
| `crates/annex-core/Cargo.toml` | Add `bench-internals` feature |
| `crates/annex-multivector/src/fde.rs` | Add `dot_avx2`, `dot_avx2_fma`, `dot_avx512`, `maxsim_flat_avx2`, `maxsim_flat_avx2_fma`, `maxsim_flat_avx512`; hoist dispatch |
| `crates/annex-multivector/benches/kernels.rs` | Add `bench_screen_dot`; extend `bench_maxsim_flat`; add dispatch-overhead variants |
| `crates/annex-multivector/Cargo.toml` | Enable `annex/bench-internals` in dev-dependencies for bench binary |

---

## Out of Scope

- `screen_dot_avx2` rewrite with `maddubs` — deferred; requires proof of input bounds
- Windows
- ARM SVE / SVE2
- BF16 / FP16 kernels
- Cosine-specific normalize+dot fusion
- Competitor comparison (Stage 3 — separate project)
