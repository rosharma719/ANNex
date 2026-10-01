# Multi-Surface Kernel Optimization

**Date:** 2026-10-01  
**Branch:** feat/query-planner  
**Author:** rosharma719

---

## Goal

Vectorize all hot kernels for every real deployment surface, eliminate per-call SIMD detection overhead, and produce benchmarks that prove the gains — including a comparison against competing RAG retrieval stacks.

Success criteria:
- `maxsim_flat` and `screen_dot` have explicit SIMD implementations on x86-64 (AVX2+FMA and AVX-512 VNNI), not just on aarch64.
- All dispatch decisions are made once at process startup, not per call.
- The criterion bench covers `screen_dot` for the first time.
- An end-to-end RAG comparison (ANNex vs Qdrant, Weaviate, pgvector) runs on the EC2 instance with published recall/latency numbers.

---

## Current State

### `annex-core` — HNSW search kernels (`core.rs`)

| Kernel | ARM64 | AVX2+FMA | AVX-512 | Notes |
|---|---|---|---|---|
| `dot_product` | NEON ✓ | AVX2+FMA ✓ | — | |
| `l2_squared` | NEON ✓ | AVX2+FMA ✓ | — | |
| `screen_dot` | NEON sdot ✓ | AVX2 ✓ (suboptimal) | — | AVX2 path widens i8/u8 → i16 (6 ops/32B); should use `maddubs` (2 ops/32B) |

All three dispatch via `is_x86_feature_detected!()` on every call.

### `annex-multivector` — MaxSim/FDE kernels (`fde.rs`)

| Kernel | ARM64 | x86-64 | Notes |
|---|---|---|---|
| `dot` | NEON FMA ✓ | scalar only | No x86 path at all |
| `maxsim_flat` | NEON ✓ + dim=128 specialization | scalar only | No x86 path at all |

### Benchmarks

`benches/kernels.rs` (Criterion) covers `maxsim_flat` and `maxsim_naive`. **`screen_dot` has no Criterion benchmark.**

---

## Dispatch Architecture

### New module: `crates/annex-core/src/vector/simd.rs`

Single source of truth for platform capability, used by both crates.

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CpuLevel {
    NeonDotprod,   // aarch64 + dotprod (M-series, Graviton3+)
    Neon,          // aarch64 baseline
    Avx512Vnni,    // avx512f + avx512vnni (Ice Lake, Zen4, Sapphire Rapids)
    Avx512F,       // avx512f only (Skylake-X, Cascade Lake pre-VNNI)
    Avx2Fma,       // avx2 + fma, no avx512 (most current cloud: c5, m5, c6a)
    Avx2,          // avx2 only
    Scalar,
}

static CPU_LEVEL: OnceLock<CpuLevel> = OnceLock::new();

pub fn cpu_level() -> CpuLevel { *CPU_LEVEL.get_or_init(detect) }
```

Detection order (first match wins):
1. `cfg(target_arch = "aarch64")` → check `is_aarch64_feature_detected!("dotprod")` → `NeonDotprod` else `Neon`
2. `cfg(target_arch = "x86_64")` → check `avx512vnni` → `Avx512Vnni`; `avx512f` → `Avx512F`; `avx2+fma` → `Avx2Fma`; `avx2` → `Avx2`
3. Else `Scalar`

`CpuLevel` is `Copy`. Call sites hold no state — `match simd::cpu_level()` is a single load + compare, predicted perfectly after the first two calls.

`fde.rs` imports `cpu_level` via a re-export from `annex-core`. Since the two crates share a process the `OnceLock` is initialized once and the same `CpuLevel` value is returned everywhere.

---

## Kernel Additions

### `annex-core/src/vector/hnsw/core.rs`

#### `dot_avx512` / `l2_avx512`

8 accumulators × 16 f32/register = 128 floats/iteration.

```
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(q: &[f32], v: &[f32]) -> f32
```

Reduction: `_mm512_reduce_add_ps` (single instruction on SKX+).  
Tail: scalar loop for `len % 16` remainder.

Dispatch: `Avx512Vnni | Avx512F` → `dot_avx512`; `Avx2Fma` → `dot_avx2_fma`; `Avx2` → `dot_avx2`; else scalar.

#### `screen_dot_avx512_vnni`

```
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn screen_dot_avx512_vnni(query_i8: &[i8], stored: &[u8]) -> i32
```

`_mm512_dpbusd_epi32(acc, u8_stored, i8_query_adjusted)`: 64 bytes/instruction, one cycle throughput on Ice Lake. Handles u8-128 centering by pre-adjusting the i8 query: `q_adj[d] = q[d] - 128_bias_correction` (factored out of the inner loop). 4 accumulators = 256 bytes/iteration before tail.

Dispatch: `Avx512Vnni` → `screen_dot_avx512_vnni`; `Avx2Fma | Avx2` → `screen_dot_avx2`; else scalar.

#### `screen_dot_avx2` rewrite

Replace the current widen-to-i16 path (extract 128-bit halves, `cvtepu8_epi16`, `cvtepi8_epi16`, subtract bias, `mullo_epi16`, `madd_epi16` — 6 ops per 32 elements) with:

```
maddubs(u8_stored_minus_128_as_u8, i8_query)  // saturating u8×i8 → i16 pairs
madd(result, ones)                             // horizontal add pairs → i32
add_epi32(acc, ...)
```

2 ops per 32 elements. The centering `stored[i] - 128` becomes a reinterpretation after subtracting the scalar bias from the u8 values before loading — precomputed as `vsubq_u8(s, vdupq_n_u8(128))` in the NEON version; on AVX2: `_mm256_sub_epi8` after treating the subtracted value as a signed interpretation (needs care: use `_mm256_xor_si256` with `0x80` mask to flip sign bit, making u8-128 behave as signed i8).

---

### `annex-multivector/src/fde.rs`

#### `dot_avx2_fma`

Mirrors `dot_neon_multiple_of_16`: 4 accumulators × 8 f32 = 32 floats/iteration.

```
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: *const f32, b: *const f32, len: usize) -> f32
```

Reduction: `_mm256_extractf128_ps` + `_mm_add_ps` + two `_mm_shuffle_ps` (no `hadd`).

#### `maxsim_flat_avx2_fma`

Mirrors `maxsim_flat_neon`:
- General path: calls `dot_avx2_fma` per doc token
- dim=128 specialization: unrolled const-length version `dot_avx2_fma_128`

Dispatched from `maxsim_flat` when `Avx2Fma | Avx2` (AVX2 only uses 2 accumulators × 8 = 16f/iter variant).

#### `maxsim_flat_avx512`

8 accumulators × 16 f32 = 128 floats/iteration. dim=128 specialization fully unrolled.

Dispatched from `maxsim_flat` when `Avx512Vnni | Avx512F`.

---

## Correctness

All new kernels added to the existing tolerance-check loops:

- `core.rs::simd_kernels_match_scalar_within_tolerance` — extended to call AVX-512 variants when `cpu_level()` reports them; tolerance ≤ 1e-3 (matches existing threshold).
- `fde.rs::maxsim_flat_scalar_and_dispatch_agree_on_dim{128,384}` — dispatch now routes to AVX2/AVX-512 on x86; no test changes needed, coverage is automatic.
- New: `screen_dot_all_paths_agree` — calls scalar, AVX2, AVX-512 VNNI on the same input; asserts exact equality (integer kernel, no FP error).

---

## Benchmark Additions

### Criterion (`benches/kernels.rs`)

New group `bench_screen_dot`:
- dim=128, dim=256 (NYT-256 production shape), dim=768
- Calls the public `screen_dot` dispatch, so all paths are exercised automatically depending on build machine

`bench_maxsim_flat` group gains dim=512 and dim=768.

### RAG comparison (end-to-end, on EC2)

After kernel optimization is complete, run a cross-system comparison on the EC2 instance using `scripts/evaluate_ann_benchmark.py` extended to cover competing stacks:

**Competitors:** Qdrant (Docker, HNSW), Weaviate (Docker, HNSW), pgvector (PostgreSQL, ivfflat + hnsw).

**Dataset:** NYT-256-angular (same as the existing Mac benchmark — 290K vectors, 256 dims, cosine).

**Metrics:**
- Recall@10 vs median query latency (Pareto curve)
- Build time
- Index size on disk

**Deliverable:** A table comparable to the existing Mac benchmark wording in `project_strategy.md`, reportable as: "On NYT-256-Angular, x86-64 (Intel Ice Lake), ANNex reaches [recall]% Recall@10 at [X] ms using M=16 + SQ8 screening + RCM. Qdrant HNSW M=16: [Y] ms. pgvector HNSW: [Z] ms."

The `scripts/bench_competitors.py` script already has scaffolding for Qdrant and pgvector — extend it for Weaviate and wire it to the EC2 benchmark runner.

---

## Files Changed

| File | Change |
|---|---|
| `crates/annex-core/src/vector/simd.rs` | **New** — `CpuLevel` enum + `OnceLock` detection |
| `crates/annex-core/src/vector/mod.rs` | Re-export `simd::cpu_level` |
| `crates/annex-core/src/vector/hnsw/core.rs` | Add AVX-512 kernels; rewrite `screen_dot_avx2`; replace runtime detection with `cpu_level()` |
| `crates/annex-multivector/src/fde.rs` | Add `dot_avx2_fma`, `maxsim_flat_avx2_fma`, `maxsim_flat_avx512`; wire dispatch |
| `crates/annex-multivector/benches/kernels.rs` | Add `bench_screen_dot` group; extend `bench_maxsim_flat` dims |
| `scripts/bench_competitors.py` | Add Weaviate; wire to EC2 runner |

---

## Out of Scope

- Windows (separate effort)
- ARM SVE / SVE2 (no current test hardware)
- BF16 / FP16 kernels
- Cosine-specific kernel fusion (normalize-then-dot in one pass)
