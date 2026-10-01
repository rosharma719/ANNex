# BF16 Transparent Compute Auto-Selection

**Date:** 2026-10-01
**Branch:** feat/query-planner
**Author:** rosharma719

---

## Goal

Automatically use BF16 fused dot-product instructions on hardware that supports `avx512bf16` (Intel Sapphire Rapids, AMD Zen4), while leaving all storage and APIs unchanged. Users on capable hardware get a compute throughput increase with zero configuration; users on older hardware see no change at all.

This is a **compute-only** optimization. Vectors remain stored as f32. The BF16 conversion is ephemeral — it happens in registers during the scoring hot path and is not persisted anywhere.

---

## Scope

**In scope:**
- New `CpuLevel::Avx512Bf16` variant detected above `Avx512Vnni`
- New `dot_avx512_bf16` kernel in `crates/annex-core/src/vector/hnsw/core.rs`
- New `dot_avx512_bf16_len` kernel in `crates/annex-multivector/src/fde.rs`
- Dispatch wiring in `select_dot_fn()`, `select_l2_fn()`, `select_dot_many_fn()`, `fde::dot`, and `fde::maxsim_flat`
- Correctness tests for the new backend
- Criterion benchmark additions

**Out of scope:**
- BF16 storage (halves memory footprint; separate project with snapshot migration requirements)
- `l2_avx512_bf16` — L2 distance does not benefit from BF16 fused accumulation the same way dot does; defer
- ARM SVE BF16 (`bf16` extension)
- Windows
- AMX (tile-based BF16 on Sapphire Rapids; requires OS support and a fully different programming model)

---

## Why BF16 Compute (Not Storage)

BF16 storage would halve memory footprint per vector (2 bytes vs. 4 per f32) and therefore halve memory bandwidth per query. That is a larger potential win than compute-only BF16.

However, BF16 storage requires:
- A snapshot format migration (breaking existing indexes)
- Precision loss in stored data (BF16 has ~3 decimal digits of mantissa vs. ~7 for f32)
- A user-visible API change or silent degradation on incompatible indexes

BF16 compute is the opposite: invisible, zero compatibility risk, no stored-data precision change. The f32 values loaded from memory are converted to BF16 **on the fly in registers** immediately before the fused multiply-accumulate. The bottleneck on x86 is memory bandwidth, so this optimization helps most when the working set fits in L2/L3 cache; it does not reduce memory traffic. The claim is not "faster than AVX-512 f32 on memory-bound workloads" but "higher instruction throughput on in-cache workloads and lower latency per operation when computing fused dot products."

BF16 storage is the right next step after this, once a snapshot migration plan exists. This spec explicitly defers it.

---

## Hardware Availability

| Feature | Intel | AMD | Availability |
|---|---|---|---|
| `avx512bf16` | Sapphire Rapids (Xeon 4th gen, 2023+), Alder Lake-P client (2021+) | Zen4 (EPYC Genoa, 2022+) | AWS c7i, GCP c3, Azure Dv5 |
| `avx512vnni` | Ice Lake (Xeon 3rd gen, 2021+), Sapphire Rapids | Zen4 | AWS c6i/m6i/r6i, c7i |
| `avx512f` | Skylake-X (2017+) | Zen4 | Most AVX-512 cloud instances |

The current dev/CI machine (Intel Ice Lake EC2) has `avx512f` + `avx512vnni` but **not** `avx512bf16`. The new kernel will be compiled but not reachable at runtime on Ice Lake; it will activate on Sapphire Rapids (c7i) instances. Detection is safe — `is_x86_feature_detected!("avx512bf16")` returns false gracefully on Ice Lake and older hardware.

---

## Current State

### `annex-core` — HNSW scoring kernels

| Level | `dot_product` | `l2_squared` | `dot_many` |
|---|---|---|---|
| `NeonDotprod` / `Neon` | NEON ✓ | NEON ✓ | NEON ✓ |
| `Avx512Vnni` / `Avx512F` | AVX-512 FMA ✓ | AVX-512 FMA ✓ | AVX-512 ✓ |
| `Avx2Fma` | AVX2+FMA ✓ | AVX2+FMA ✓ | AVX2+FMA ✓ |
| `Avx2` | AVX2 ✓ | AVX2 ✓ | AVX2 ✓ |
| `Scalar` | scalar ✓ | scalar ✓ | scalar ✓ |
| **`Avx512Bf16`** | **missing** | **missing** | **missing** |

### `annex-multivector` — FDE / MaxSim kernels

| Level | `fde::dot` | `maxsim_flat` |
|---|---|---|
| `NeonDotprod` / `Neon` | NEON ✓ | NEON ✓ |
| `Avx512Vnni` / `Avx512F` | AVX-512 ✓ | AVX-512 ✓ |
| `Avx2Fma` | AVX2+FMA ✓ | AVX2+FMA ✓ |
| `Avx2` | AVX2 ✓ | AVX2 ✓ |
| `Scalar` | scalar ✓ | scalar ✓ |
| **`Avx512Bf16`** | **missing** | **missing** |

---

## Design

### 1. `CpuLevel` — new variant

Add `Avx512Bf16` to `crates/annex-core/src/vector/simd.rs`, ranked **above** `Avx512Vnni` so hardware with both features gets the BF16 path:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuLevel {
    NeonDotprod,   // aarch64 + dotprod
    Neon,          // aarch64 baseline
    Avx512Bf16,    // x86: avx512f + avx512bf16 (Sapphire Rapids, Zen4)  ← NEW
    Avx512Vnni,    // x86: avx512f + avx512vnni (Ice Lake, Zen4, Sapphire Rapids)
    Avx512F,       // x86: avx512f only
    Avx2Fma,       // x86: avx2 + fma
    Avx2,          // x86: avx2 only
    Scalar,
}
```

Detection order in `cpu_level()` — insert before the `avx512vnni` check:

```rust
if is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512f") {
    return CpuLevel::Avx512Bf16;
}
```

`avx512bf16` implies `avx512f` in practice, but check both explicitly to match the pattern of every other variant and satisfy the `#[target_feature]` requirement.

Note: on real Sapphire Rapids hardware, `avx512vnni` is also present. The detection order ensures `Avx512Bf16` is returned first — it subsumes `Avx512Vnni` for the dot kernels. `screen_dot` (integer VNNI) is unaffected because it has its own dispatch and does not route through `CpuLevel::Avx512Bf16` for now.

### 2. BF16 dot kernel — `annex-core`

Add to `crates/annex-core/src/vector/hnsw/core.rs`:

```rust
/// BF16 fused dot product. Converts f32 inputs to BF16 on the fly and uses
/// dpbf16_ps to accumulate into an f32 register. Handles 32 BF16 elements
/// (= 16 f32 pairs) per iteration. Scalar tail for len % 32.
///
/// # Safety
/// Requires avx512f and avx512bf16. `a` and `b` must be valid for `len` reads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn dot_avx512_bf16(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut i = 0usize;
    // Process 128 f32 values per iteration (4 × 32-element BF16 groups).
    while i + 128 <= len {
        let a0 = _mm512_loadu_ps(a.add(i));
        let a1 = _mm512_loadu_ps(a.add(i + 16));
        let b0 = _mm512_loadu_ps(b.add(i));
        let b1 = _mm512_loadu_ps(b.add(i + 16));
        let abh0 = _mm512_cvtne2ps_pbh(a1, a0); // pack a[i..i+32] into BF16
        let bbh0 = _mm512_cvtne2ps_pbh(b1, b0); // pack b[i..i+32] into BF16
        acc0 = _mm512_dpbf16_ps(acc0, abh0, bbh0);

        let a2 = _mm512_loadu_ps(a.add(i + 32));
        let a3 = _mm512_loadu_ps(a.add(i + 48));
        let b2 = _mm512_loadu_ps(b.add(i + 32));
        let b3 = _mm512_loadu_ps(b.add(i + 48));
        let abh1 = _mm512_cvtne2ps_pbh(a3, a2);
        let bbh1 = _mm512_cvtne2ps_pbh(b3, b2);
        acc1 = _mm512_dpbf16_ps(acc1, abh1, bbh1);

        let a4 = _mm512_loadu_ps(a.add(i + 64));
        let a5 = _mm512_loadu_ps(a.add(i + 80));
        let b4 = _mm512_loadu_ps(b.add(i + 64));
        let b5 = _mm512_loadu_ps(b.add(i + 80));
        let abh2 = _mm512_cvtne2ps_pbh(a5, a4);
        let bbh2 = _mm512_cvtne2ps_pbh(b5, b4);
        acc2 = _mm512_dpbf16_ps(acc2, abh2, bbh2);

        let a6 = _mm512_loadu_ps(a.add(i + 96));
        let a7 = _mm512_loadu_ps(a.add(i + 112));
        let b6 = _mm512_loadu_ps(b.add(i + 96));
        let b7 = _mm512_loadu_ps(b.add(i + 112));
        let abh3 = _mm512_cvtne2ps_pbh(a7, a6);
        let bbh3 = _mm512_cvtne2ps_pbh(b7, b6);
        acc3 = _mm512_dpbf16_ps(acc3, abh3, bbh3);

        i += 128;
    }
    // Handle remaining 32-element groups.
    while i + 32 <= len {
        let a0 = _mm512_loadu_ps(a.add(i));
        let a1 = _mm512_loadu_ps(a.add(i + 16));
        let b0 = _mm512_loadu_ps(b.add(i));
        let b1 = _mm512_loadu_ps(b.add(i + 16));
        let abh = _mm512_cvtne2ps_pbh(a1, a0);
        let bbh = _mm512_cvtne2ps_pbh(b1, b0);
        acc0 = _mm512_dpbf16_ps(acc0, abh, bbh);
        i += 32;
    }
    // Reduce accumulators.
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    let mut result = _mm512_reduce_add_ps(acc0);
    // Scalar tail for len % 32 — uses f32 multiply to avoid double-rounding.
    while i < len {
        result += *a.add(i) * *b.add(i);
        i += 1;
    }
    result
}
```

**Intrinsic semantics:**
- `_mm512_cvtne2ps_pbh(hi: __m512, lo: __m512) -> __m512bh` — converts 32 f32 values from two `__m512` registers into 32 BF16 values (nearest-even rounding) packed into a single `__m512bh`.
- `_mm512_dpbf16_ps(acc: __m512, a: __m512bh, b: __m512bh) -> __m512` — fused BF16 dot product: for each of 16 pairs of BF16 pairs, computes `a[2i]*b[2i] + a[2i+1]*b[2i+1]` and accumulates into f32. Effectively 32 BF16 multiplications + 16 f32 additions per call.

**Why 4 accumulators:** `dpbf16_ps` has ~4-cycle latency and 0.5-cycle throughput on Sapphire Rapids; 4 independent accumulators hide latency and fill the execution ports.

**Tail handling:** The tail (remaining `len % 32` elements) uses scalar f32 multiplication, not BF16 conversion. This avoids a partial BF16 pack and keeps the scalar tail simple. For the common ColBERT dimensions (128, 384, 512, 768) that are multiples of 32, the tail never executes.

### 3. Dispatch wiring — `annex-core`

In `select_dot_fn()` and `select_dot_many_fn()`, add `Avx512Bf16` arms above `Avx512Vnni`. `select_l2_fn()` is unchanged (L2 distance deferred):

```rust
// select_dot_fn:
CpuLevel::Avx512Bf16 => {
    |q: &[f32], v: &[f32]| unsafe { dot_avx512_bf16(q.as_ptr(), v.as_ptr(), q.len()) }
}
CpuLevel::Avx512Vnni | CpuLevel::Avx512F => { ... }  // unchanged

// select_l2_fn: Avx512Bf16 falls through to Avx512F path (no BF16 L2 yet)
CpuLevel::Avx512Bf16 | CpuLevel::Avx512Vnni | CpuLevel::Avx512F => {
    |q: &[f32], v: &[f32]| unsafe { l2_avx512(q, v) }
}

// select_dot_many_fn:
CpuLevel::Avx512Bf16 => dot_many_avx512_bf16,
CpuLevel::Avx512Vnni | CpuLevel::Avx512F => dot_many_avx512,  // unchanged
```

`dot_many_avx512_bf16` mirrors `dot_many_avx512` but calls `dot_avx512_bf16` per candidate. The signature is unchanged: `unsafe fn(&[f32], &[&[f32]], &mut [f32])`.

### 4. `fde.rs` — new kernel and dispatch

Add `dot_avx512_bf16_len` to `crates/annex-multivector/src/fde.rs`:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn dot_avx512_bf16_len(a: *const f32, b: *const f32, len: usize) -> f32 {
    // Same body as dot_avx512_bf16 in annex-core.
    // Kept separate to avoid a cross-crate dependency on a private function.
}
```

In `fde::dot`, insert a new arm above the existing `Avx512Vnni | Avx512F` arm:

```rust
CpuLevel::Avx512Bf16 => {
    return unsafe { dot_avx512_bf16_len(left.as_ptr(), right.as_ptr(), n) };
}
```

In `maxsim_flat`, add a new branch above `Avx512Vnni | Avx512F`:

```rust
CpuLevel::Avx512Bf16 => {
    return maxsim_flat_avx512_bf16(query, document, dimension);
}
```

`maxsim_flat_avx512_bf16` mirrors `maxsim_flat_avx512` but sets `dot_fn = dot_avx512_bf16_len`.

**Full dispatch table after this change:**

| `CpuLevel` | `core::dot_fn` | `core::l2_fn` | `core::dot_many_fn` | `fde::dot` | `fde::maxsim_flat` |
|---|---|---|---|---|---|
| `Avx512Bf16` | `dot_avx512_bf16` | `l2_avx512` (fallback) | `dot_many_avx512_bf16` | `dot_avx512_bf16_len` | `maxsim_flat_avx512_bf16` |
| `Avx512Vnni` | `dot_avx512` | `l2_avx512` | `dot_many_avx512` | `dot_avx512_len` | `maxsim_flat_avx512` |
| `Avx512F` | `dot_avx512` | `l2_avx512` | `dot_many_avx512` | `dot_avx512_len` | `maxsim_flat_avx512` |
| `Avx2Fma` | `dot_avx2_fma` | `l2_avx2_fma` | `dot_many_avx2_fma` | `dot_avx2_fma_len` | `maxsim_flat_avx2_fma` |
| `Avx2` | `dot_avx2` | `l2_avx2` | `dot_many_avx2_fma` | `dot_avx2_len` | `maxsim_flat_avx2` |
| `NeonDotprod`/`Neon` | `dot_neon` | `l2_neon` | `dot_many_neon` | NEON path | `maxsim_flat_neon` |
| `Scalar` | `dot_scalar` | `l2_scalar` | `dot_many_scalar` | `dot_scalar` | `maxsim_flat_scalar` |

---

## Correctness

### Precision model

BF16 has 7 mantissa bits (vs. 23 for f32), giving ~3 decimal digits of precision. Rounding error per product is bounded by `|a_i × b_i| × 2^-7`. For unit-normalized vectors (`|a| = |b| = 1`), per-element products are at most 1.0, and the accumulated rounding error over `N` elements is bounded by approximately `N × 2^-7`. For `N = 768` (the largest standard dimension): `768 / 128 = 6`. The existing tolerance formula `|actual − reference| ≤ 1e-3 × max(1.0, |reference|)` covers this comfortably.

For larger dimensions (1024, 2048, 4096) the accumulated error is proportionally larger but still within 1e-3 × reference given unit-normalized inputs. This is acceptable for approximate nearest-neighbor ranking, where score order matters more than absolute value. If sub-1e-3 error is required in future exact modes, the BF16 path will be skipped there.

### Required test: `x86_simd_kernels_match_scalar_across_shapes` — BF16 arm

Add to the existing shape-coverage test in `core.rs` (gated on `#[cfg(target_arch = "x86_64")]`):

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn dot_avx512_bf16_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx512bf16") {
        return; // skip on hardware without avx512bf16
    }
    for &len in &[0usize, 1, 4, 16, 31, 32, 33, 64, 128, 256, 384, 512, 768] {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0).recip()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 + 2.0).recip()).collect();
        let ref_val = dot_scalar(&a, &b);
        let bf16_val = unsafe { dot_avx512_bf16(a.as_ptr(), b.as_ptr(), len) };
        let tol = 1e-3_f32 * ref_val.abs().max(1.0);
        assert!(
            (bf16_val - ref_val).abs() <= tol,
            "len={len}: ref={ref_val} bf16={bf16_val} diff={diff}",
            diff = (bf16_val - ref_val).abs()
        );
    }
}
```

Required length cases:
- `len = 0` — empty slice, must return 0.0
- `len < 32` (1, 4, 16, 31) — tail-only path
- `len = 32` — exactly one `dpbf16_ps` group, no tail
- `len = 33` — one group + one scalar tail element
- `len = 64, 128` — aligned multi-group
- `len = 384, 512, 768` — standard ColBERT dimensions, all multiples of 32

**Adversarial case:** alternating `+1.0 / -1.0` at len=512 (cancellation test — the sum should be 0, but BF16 rounding may produce a small nonzero; confirm it stays within tolerance).

Add the same coverage for `fde::dot_avx512_bf16_len` in `fde.rs` tests.

Also add a `maxsim_flat` agreement test for `Avx512Bf16` in the `fde.rs` test suite, mirroring the existing `maxsim_flat_scalar_and_dispatch_agree_on_dim128/384` tests.

---

## Benchmarks

### Criterion additions (`benches/kernels.rs`)

Add a new group `maxsim_flat_bf16` that runs alongside the existing `maxsim_flat` group. On hardware without `avx512bf16`, skip with a Criterion `println!` to signal the hardware gap rather than silently omitting the benchmark. On Sapphire Rapids, run all variants.

```rust
fn bench_maxsim_flat_bf16(c: &mut Criterion) {
    // On non-avx512bf16 hardware: emit a note and return immediately.
    if !std::arch::is_x86_feature_detected!("avx512bf16") {
        println!("[bench_maxsim_flat_bf16] avx512bf16 not available — skipping");
        return;
    }
    let mut group = c.benchmark_group("maxsim_flat_bf16");
    for &(dim, doc_tokens, query_tokens) in &[
        (128usize, 200usize, 32usize),
        (128, 100, 32),
        (384, 200, 32),
        (512, 200, 32),
        (768, 200, 32),
    ] {
        // ... same setup as bench_maxsim_flat, calls maxsim_flat which now
        // routes to the BF16 kernel on this hardware.
    }
    group.finish();
}
```

Run baseline (AVX-512 f32) vs. BF16 on the same machine by temporarily checking `Avx512Vnni` dispatch to get comparable numbers.

### Expected throughput

On Sapphire Rapids, `dpbf16_ps` has 0.5-cycle throughput (2 ops/cycle per port) vs. 0.5-cycle for `fmadd_ps`. Both are throughput-equivalent per cycle, but `dpbf16_ps` processes 32 BF16 elements (16 f32 pairs) per call vs. 16 f32 elements for `fmadd_ps`. This gives a theoretical **2× compute density** — the kernel needs half as many loop iterations to cover the same number of element pairs.

In practice, the win is bounded by memory bandwidth when the data is not cached. Expected outcomes:

| Workload | Expected gain vs. AVX-512 f32 |
|---|---|
| Small dimensions (dim=128), data in L1/L2 | ~1.5–2.0× throughput |
| Large dimensions (dim=768), data in L3 | ~1.2–1.5× throughput |
| Memory-bandwidth-bound (cold HNSW scan, large index) | ~1.0–1.1× (memory dominates) |

Do not claim a gain until Criterion numbers on Sapphire Rapids confirm it. The benchmark section produces the evidence.

---

## Files Changed

| File | Change |
|---|---|
| `crates/annex-core/src/vector/simd.rs` | Add `Avx512Bf16` variant to `CpuLevel`; add detection arm in `cpu_level()` |
| `crates/annex-core/src/vector/hnsw/core.rs` | Add `dot_avx512_bf16`; add `Avx512Bf16` arm in `select_dot_fn()`, `select_l2_fn()`, `select_dot_many_fn()`; add `dot_many_avx512_bf16`; add correctness tests |
| `crates/annex-multivector/src/fde.rs` | Add `dot_avx512_bf16_len`; add `Avx512Bf16` arm in `fde::dot` and `fde::maxsim_flat`; add `maxsim_flat_avx512_bf16`; add correctness tests |
| `crates/annex-multivector/benches/kernels.rs` | Add `bench_maxsim_flat_bf16` group |

---

## Out of Scope

- **BF16 storage** — halves memory footprint per vector; requires snapshot format migration, user-visible precision change in stored data, and a separate design spec. This is the natural follow-on project once the compute path proves stable.
- **L2 distance BF16** — `dpbf16_ps` is a dot-product accumulator; adapting it for L2 requires `(a-b)^2 = a^2 - 2ab + b^2` decomposition with precomputed norms, which changes the kernel contract. Defer to the BF16 storage project.
- **`screen_dot` BF16** — `screen_dot` uses integer quantized inputs (u8/i8), not f32; BF16 does not apply.
- **AMX** — tile-based BF16 on Sapphire Rapids; requires OS kernel support (`ARCH_REQ_XCOMP_PERM`), a fundamentally different programming model, and much larger tile buffers. Separate project if ever pursued.
- **ARM BF16** — Graviton3 and Apple M-series support `bf16` extension; a future `NeonBf16` level would follow the same pattern but is not part of this change.
- **Windows**
