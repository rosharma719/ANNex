# Multi-Surface Kernel Optimization

**Date:** 2026-10-01 (revised after review)
**Branch:** feat/query-planner
**Author:** rosharma719

---

## Goal

Vectorize all hot kernels for every real deployment surface, eliminate in-loop dispatch overhead, and produce benchmarks that prove gains — including a structured comparison against competing retrieval stacks.

Three sequential stages:
1. **Correct kernels + per-backend microbenchmarks** — the bulk of this spec
2. **EC2 end-to-end validation** — HNSW and MaxSim against ANNex baseline at matched recall
3. **Competitor comparison** — separate project; methodology defined at end of this doc

---

## Current State

### `annex-core` — HNSW search kernels (`core.rs`)

| Kernel | ARM64 | AVX2+FMA | AVX-512 |
|---|---|---|---|
| `dot_product` | NEON ✓ | AVX2+FMA ✓ | — |
| `l2_squared` | NEON ✓ | AVX2+FMA ✓ | — |
| `screen_dot` | NEON sdot ✓ | AVX2 ✓ (widening, keep as-is) | — |

All three dispatch via `is_x86_feature_detected!()` inside the function body, which is called per candidate in the HNSW inner loop.

Note on `screen_dot_avx2`: the existing widening-to-i16 path (`cvtepu8_epi16`, `cvtepi8_epi16`, `sub_epi16`, `mullo_epi16`, `madd_epi16`) is the correctness baseline and must not be replaced with `_mm256_maddubs_epi16` without proving that adjacent-pair sums cannot exceed `i16::MAX` (32767). `maddubs` saturates: `255×127 + 255×127 = 64770` overflows. This rewrite is deferred.

### `annex-multivector` — MaxSim/FDE kernels (`fde.rs`)

| Kernel | ARM64 | x86-64 |
|---|---|---|
| `dot` | NEON FMA ✓ | scalar only — no x86 path |
| `maxsim_flat` | NEON ✓ + dim=128 specialization | scalar only — no x86 path |

### Benchmarks

`benches/kernels.rs` (Criterion) covers `maxsim_flat` and `maxsim_naive`. `screen_dot` has no Criterion benchmark.

---

## Dispatch Architecture

### What changes — and why

Rust's `std_detect` already caches CPU feature flags in an atomic initialized on first use. Each `is_x86_feature_detected!("avx2")` call is an atomic load + bit test — not a CPUID. A `OnceLock<CpuLevel>` replacing it would add a second atomic load on top of the first, with no net benefit.

**The real overhead** is that the dispatch branch lives inside functions called per candidate (HNSW) or per token (MaxSim). Even with perfect branch prediction, this is a branch in the hot inner loop.

**Fix:** select a function pointer *before* the candidate/token loop; call it unconditionally inside the loop. The selection itself uses the existing `is_x86_feature_detected!` machinery (already cached).

### `CpuLevel` enum — for selection, not detection

A `CpuLevel` enum provides a clean way to select function pointers at the call site. It is computed once by the caller outside the loop, not stored globally.

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

pub fn cpu_level() -> CpuLevel { /* one call, result used by caller */ }
```

**Per-kernel selection, not a global ranking.** A single `CpuLevel` value does not mean "use AVX-512 for everything." Each kernel selects its best implementation independently based on the available feature set:

- `screen_dot`: `Avx512Vnni` → VNNI path; `Avx2Fma | Avx2` → existing widening path; else scalar.
- `dot_product` / `l2_squared`: `Avx512F | Avx512Vnni` → AVX-512 path; `Avx2Fma` → existing `dot_avx2_fma`; else scalar.
- `maxsim_flat`: same as `dot_product`.

**Measure dispatch overhead before and after.** The criterion bench must show whether hoisting outside the loop produces a measurable improvement. If the difference is within noise, the hoist is still correct but the spec should not claim a specific speedup.

### Location

New file `crates/annex-core/src/vector/simd.rs`. `annex-multivector` imports `cpu_level` from `annex` (the existing dep on `annex-core`).

---

## Kernel Additions

All throughput figures below are theoretical maximums based on register widths. Actual performance depends on memory bandwidth, loop structure, and unrolling — **treat as hypotheses to validate with generated assembly and measurement, not assertions.**

### `annex-core/src/vector/hnsw/core.rs`

#### `dot_avx512` / `l2_avx512`

`#[target_feature(enable = "avx512f")]` kernels using `_mm512_fmadd_ps` (16 f32/register).

Reduction: accumulate into multiple `__m512` registers across the loop; reduce to scalar at the end. `_mm512_reduce_add_ps` is a multi-instruction reduction convenience — if the compiler emits suboptimal code, replace with explicit extract-and-add tree. Determine final form after inspecting generated assembly.

Tail: scalar loop for `len % 16` elements.

Required features: `avx512f`. No VNNI dependency.

#### `screen_dot_avx512_vnni`

`#[target_feature(enable = "avx512f,avx512vnni")]` kernel using `_mm512_dpbusd_epi32`.

`_mm512_dpbusd_epi32(acc, a, b)` computes `acc += sum(a[i] * b[i])` where `a` is **u8** and `b` is **i8**, 64 elements per call. This matches the operand types (stored = u8, query = i8) but not the centering.

**Centering correction.** The existing contract computes `sum(query_i8[i] × (stored_u8[i] − 128))`. Expanding:

```
sum(query_i8[i] × stored_u8[i]) − 128 × sum(query_i8[i])
```

The VNNI accumulator computes the first term directly. The correction `128 × sum(query_i8[i])` is a scalar computed once per query vector (outside the document loop) and subtracted from the VNNI result after reduction. This must handle:

- The existing slice-length contract: `n = query_i8.len().min(stored.len())` — correction sum must cover exactly the same `n` elements.
- Overflow: `sum(query_i8)` for a 768-dim vector with all values at +127 = 97536, times 128 = 12M — fits in i32 (max ~2.1B). Document this bound.

4 accumulators, each covering 64 elements per iteration = 256 elements/iteration before tail.

Required features: `avx512f` + `avx512vnni`.

### `annex-multivector/src/fde.rs`

#### `dot_avx2_fma` / `dot_avx512`

Mirror the NEON `dot_neon_multiple_of_16` structure. Multiple accumulators to cover the multiple-of-16 prefix; scalar tail for remainder. Reduction without `_mm256_hadd_ps`: use `_mm256_extractf128_ps` + `_mm_add_ps` + `_mm_shuffle_ps` or `_mm_movehl_ps`.

The number of accumulators and unroll depth is a tuning decision — start with 4 (matching NEON), adjust after benchmarking.

#### `maxsim_flat_avx2_fma` / `maxsim_flat_avx512`

Mirror `maxsim_flat_neon`:
- General path: call per-doc-token dot on each doc chunk.
- dim=128 specialization: const-length variant allowing full unrolling.

Dispatch: call `cpu_level()` once before the query-token outer loop; pass a function pointer into the inner loop.

---

## Correctness

### Problem with dispatch-path testing

Testing `dot_product(...)` on Ice Lake with `RUSTFLAGS=-C target-cpu=native` exercises the AVX-512 path. It does not test AVX2, AVX2+FMA, or scalar. The same is true for `screen_dot` and `maxsim_flat`.

### Required: direct per-backend tests

Each kernel function (`dot_avx2`, `dot_avx2_fma`, `dot_avx512`, `l2_avx2`, `l2_avx2_fma`, `l2_avx512`, `screen_dot_avx2`, `screen_dot_avx512_vnni`, `maxsim_flat_avx2_fma`, `maxsim_flat_avx512`) needs its own test calling the function directly, gated by `#[cfg(target_arch = "x86_64")]` and a runtime feature check.

**For screen_dot (integer kernels) — required cases:**
- Empty input (len = 0)
- Lengths below SIMD width: 1, 4, 15, 16, 31, 32
- Lengths at SIMD boundaries: 32, 64, 128, 256
- Integer extremes: `query_i8 = [127; N]`, `stored_u8 = [255; N]` and `[0; N]`
- Adversarial adjacent pairs designed to expose saturation bugs in any future `maddubs` attempt: `q=[127,127,...]`, `s=[255,255,...]`
- Correctness: assert exact integer equality with `screen_dot_scalar` for all cases

**For f32 kernels — required cases:**
- Lengths below SIMD width: 1, 4, 7, 8, 15, 16, 31, 32
- Lengths at SIMD boundaries: 16, 32, 64, 128, 256, 512
- Floating-point cancellation: alternating +1.0 and -1.0
- Tolerance: ≤ 1e-3 absolute difference vs scalar, with dimension-appropriate scaling

### Benchmark access for `screen_dot`

`screen_dot` is currently `pub(crate)` in `annex-core`. To benchmark it in the Criterion bench (which lives in `annex-multivector`), add a thin `pub` re-export in `annex-core` behind a `#[cfg(any(test, feature = "bench-internals"))]` gate. The `bench-internals` Cargo feature is activated only in the benchmark binary. Do not expose it in the public library API.

---

## Benchmark Additions

### Criterion (`benches/kernels.rs`)

New group `bench_screen_dot`:
- Call `screen_dot` dispatch (via `bench-internals` feature) for dim=128, dim=256, dim=768
- This gives the first Criterion coverage of the SQ8 screening kernel

`bench_maxsim_flat` gains dim=512 and dim=768.

**Expected outputs:** absolute timing with Criterion confidence intervals. Claim no speedup until numbers are in hand.

---

## Stages 2 and 3

### Stage 2 — EC2 end-to-end validation

Run after stage 1 kernels pass all backend tests.

**HNSW kernel validation:** ANNex baseline (current `master`) vs patched (this branch) on NYT-256-angular. Same M=16 + SQ8 screening + RCM config. Metric: recall@10 vs median query latency at matched recall. Expected: no regression in recall; latency improvement measurable at ≥2× criterion sample count for statistical significance.

**MaxSim kernel validation:** Use the existing `maxsim_bench` binary workload (10K docs × 200 tokens × 128 dims, 250 candidates). Compare scalar vs AVX2 vs AVX-512 paths directly. NYT-256 is single-vector retrieval and does **not** exercise `maxsim_flat`.

### Stage 3 — Competitor comparison (separate project)

Comparing ANNex against Qdrant, Weaviate, and pgvector requires:

- Controlled dataset (NYT-256-angular, same version)
- Each system configured at comparable `M` and `ef_construction`
- Warmup rounds documented
- Network/client overhead separated from pure index latency (test with local UNIX socket or loopback where possible)
- Resource limits (RAM, CPU threads) equalized or reported
- Published configurations (not default-tuned for one system)

This is a non-trivial methodology effort. It belongs in a separate spec, not as a tail of the kernel work. When the kernel stages are complete and validated, open a new brainstorm for the competitor comparison.

---

## Files Changed

| File | Change |
|---|---|
| `crates/annex-core/src/vector/simd.rs` | **New** — `CpuLevel` enum + `cpu_level()` detection function |
| `crates/annex-core/src/vector/mod.rs` | Re-export `simd::cpu_level` |
| `crates/annex-core/src/vector/hnsw/core.rs` | Add `dot_avx512`, `l2_avx512`, `screen_dot_avx512_vnni`; hoist dispatch outside candidate loop; per-backend correctness tests |
| `crates/annex-core/Cargo.toml` | Add `bench-internals` feature flag |
| `crates/annex-multivector/src/fde.rs` | Add `dot_avx2_fma`, `dot_avx512`, `maxsim_flat_avx2_fma`, `maxsim_flat_avx512`; hoist dispatch outside token loop |
| `crates/annex-multivector/benches/kernels.rs` | Add `bench_screen_dot` group; extend `bench_maxsim_flat` dims |

---

## Out of Scope

- `screen_dot_avx2` rewrite with `maddubs` — deferred; requires proof of input bounds
- Windows
- ARM SVE / SVE2
- BF16 / FP16 kernels
- Cosine-specific normalize+dot fusion
- Competitor comparison (Stage 3 is a separate project)
