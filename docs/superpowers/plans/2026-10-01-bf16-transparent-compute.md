# BF16 Transparent Compute Auto-Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `CpuLevel::Avx512Bf16` detection tier and BF16 dot-product kernels that activate automatically on Sapphire Rapids / Zen4 hardware, with no API or storage changes.

**Architecture:** `simd.rs` gains `Avx512Bf16` ranked above `Avx512Vnni`. `core.rs` gets `dot_avx512_bf16` (uses `_mm512_cvtne2ps_pbh` + `_mm512_dpbf16_ps`) wired into `select_dot_fn` and `select_dot_many_fn`. `fde.rs` gets a matching `dot_avx512_bf16_len` and `maxsim_flat_avx512_bf16`. `select_l2_fn` routes `Avx512Bf16` to the existing `l2_avx512` — L2 BF16 is deferred.

**Tech Stack:** Rust 1.99, `std::arch::x86_64` (`avx512f,avx512bf16` target features), Criterion 0.5. No new crate dependencies.

**Spec:** `docs/superpowers/specs/2026-10-01-bf16-transparent-compute-design.md`

## Global Constraints

- `avx512bf16` kernel gated by `#[cfg(target_arch = "x86_64")]` and `#[target_feature(enable = "avx512f,avx512bf16")]`; never compiled on aarch64 or scalar-only targets
- Correctness tolerance: `|actual − reference| ≤ 1e-3 × max(1.0, |reference|)` — spec proves BF16 rounding stays within this for unit-normalized vectors up to dim=768
- Tests that call `dot_avx512_bf16` directly must `return` early (not `skip`) when `!is_x86_feature_detected!("avx512bf16")` — the function is compiled but unreachable on non-BF16 hardware
- `select_l2_fn` for `Avx512Bf16` falls through to `l2_avx512` — no BF16 L2 kernel
- Benchmark in `kernels.rs` silently no-ops on hardware without `avx512bf16`; it must not fail the bench compile
- Branch: `feat/query-planner`; commit after each task

## Review Focus

1. **`cpu_level()` detection order** — `Avx512Bf16` must fire before `Avx512Vnni` since Sapphire Rapids has both features. If the order is wrong, BF16 hardware never uses the new kernel. Test: `cpu_level_does_not_panic` must include `Avx512Bf16` in the valid-on-avx2 set so the match doesn't panic on Sapphire Rapids. Added to Task 1.
2. **Empty-slice zero return** — `dot_avx512_bf16(ptr, ptr, 0)` must return `0.0` without reading any memory. The loop body never executes, and `_mm512_reduce_add_ps(setzero) = 0.0`. Covered by `len=0` in the correctness test in Task 2.
3. **Tail for non-multiple-of-32 lengths** — dims 1, 31, 33, 769 etc. exercise the scalar tail. These are in the correctness test's required length list in Task 2.
4. **`fde::dot` called from `normalize()`** — `normalize()` calls `dot(v, v)` which on BF16 hardware routes to `dot_avx512_bf16_len`. Self-dot on a unit-normalized vector should return 1.0 ± 1e-3. Covered implicitly by the `maxsim_flat` agreement test in Task 3 (which calls `normalize` internally), but add an explicit `dot_self_is_near_one_on_bf16` test to Task 3.
5. **`dot_many_avx512_bf16` partial-group fallback** — when `vecs.len() % 4 != 0` or some vector is shorter than `query.len()`, the function falls back to per-vector `dot_avx512_bf16`. The `fast_score_many_matches_fast_score_individually` test in Task 2 uses exactly 3 vectors (not a multiple of 4), exercising this path.

---

### Task 1: Add `Avx512Bf16` to `CpuLevel` enum and detection

**Files:**
- Modify: `crates/annex-core/src/vector/simd.rs`

**Interfaces:**
- Produces: `CpuLevel::Avx512Bf16` variant; `cpu_level()` returns it on hardware with `avx512bf16 + avx512f`

- [ ] **Step 1: Write the failing test — detection test must include `Avx512Bf16`**

In `simd.rs` tests, the existing `cpu_level_does_not_panic` test will need updating after adding the variant (it won't fail to compile before the enum change, but the `matches!` assert won't cover the new variant and will be wrong on BF16 hardware). Add a second test now that will only compile once `Avx512Bf16` exists:

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn cpu_level_bf16_detection_is_consistent() {
    // If avx512bf16 is present, cpu_level() must return Avx512Bf16.
    // If absent, it must NOT return Avx512Bf16.
    let avx512bf16 = std::arch::is_x86_feature_detected!("avx512bf16");
    let avx512f    = std::arch::is_x86_feature_detected!("avx512f");
    let level = cpu_level();
    if avx512bf16 && avx512f {
        assert_eq!(level, CpuLevel::Avx512Bf16, "avx512bf16 present but not selected");
    } else {
        assert_ne!(level, CpuLevel::Avx512Bf16, "avx512bf16 absent but incorrectly selected");
    }
}
```

- [ ] **Step 2: Run to confirm compile error**

```bash
cargo test -p annex --lib vector::simd 2>&1 | tail -5
```
Expected: compile error (`CpuLevel::Avx512Bf16` not found).

- [ ] **Step 3: Add `Avx512Bf16` variant to `CpuLevel` enum**

In `crates/annex-core/src/vector/simd.rs`, replace the enum definition:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuLevel {
    NeonDotprod, // aarch64 + dotprod extension (M-series, Graviton3+)
    Neon,        // aarch64 baseline
    Avx512Bf16,  // x86: avx512f + avx512bf16 (Sapphire Rapids, Zen4)
    Avx512Vnni,  // x86: avx512f + avx512vnni (Ice Lake, Zen4, Sapphire Rapids)
    Avx512F,     // x86: avx512f only (Skylake-X, Cascade Lake pre-VNNI)
    Avx2Fma,     // x86: avx2 + fma (most current cloud: c5, m5, c6a)
    Avx2,        // x86: avx2 only
    Scalar,
}
```

Add the detection arm in `cpu_level()` **before** the `avx512vnni` check (lines 24–28):

```rust
// Insert before the avx512vnni check:
if std::arch::is_x86_feature_detected!("avx512bf16")
    && std::arch::is_x86_feature_detected!("avx512f")
{
    return CpuLevel::Avx512Bf16;
}
```

Update the `cpu_level_does_not_panic` test's `matches!` to include `Avx512Bf16`:

```rust
assert!(matches!(
    level,
    CpuLevel::Avx2
        | CpuLevel::Avx2Fma
        | CpuLevel::Avx512F
        | CpuLevel::Avx512Vnni
        | CpuLevel::Avx512Bf16   // ← add this
));
```

- [ ] **Step 4: Confirm all simd tests pass**

```bash
cargo test -p annex --lib vector::simd -- --nocapture 2>&1 | tail -8
```
Expected: 2 passed (both simd tests). On Ice Lake EC2: `cpu_level_bf16_detection_is_consistent` passes (avx512bf16 absent → `Avx512Bf16` not selected).

- [ ] **Step 5: Run full workspace to confirm no regressions**

```bash
cargo test --workspace 2>&1 | grep -E "^test result|FAILED" | head -10
```
Expected: all pass. The `select_dot_fn` and `select_l2_fn` match arms now have a non-exhaustive pattern warning — that's expected and will be fixed in Task 2.

- [ ] **Step 6: Commit**

```bash
git add crates/annex-core/src/vector/simd.rs
git commit -m "feat(simd): add Avx512Bf16 CpuLevel variant and detection"
```

---

### Task 2: BF16 dot kernels in `core.rs` + dispatch wiring

**Files:**
- Modify: `crates/annex-core/src/vector/hnsw/core.rs`

**Interfaces:**
- Consumes: `CpuLevel::Avx512Bf16` from Task 1
- Produces: `unsafe fn dot_avx512_bf16(query: &[f32], vec: &[f32]) -> f32`, `unsafe fn dot_many_avx512_bf16(query: &[f32], vecs: &[&[f32]], out: &mut [f32])`, updated `select_dot_fn`/`select_l2_fn`/`select_dot_many_fn`

- [ ] **Step 1: Write the failing correctness test**

Add to the `tests` mod in `core.rs`:

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn dot_avx512_bf16_matches_scalar_across_shapes() {
    if !std::arch::is_x86_feature_detected!("avx512bf16") {
        return; // kernel compiled but skip on non-BF16 hardware
    }
    let close = |actual: f32, expected: f32| {
        let tol = 1e-3_f32 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tol,
            "bf16 mismatch: expected={expected} actual={actual} tol={tol}"
        );
    };
    // Standard ColBERT / embedding dimensions, plus boundary lengths.
    for &len in &[0usize, 1, 4, 16, 31, 32, 33, 64, 128, 256, 384, 512, 768] {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0).recip()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 + 2.0).recip()).collect();
        let ref_val = dot_scalar(&a, &b);
        let bf16_val = unsafe { dot_avx512_bf16(a.as_ptr(), b.as_ptr(), len) };
        close(bf16_val, ref_val);
    }
    // Adversarial: alternating +1/-1, cancellation sum ≈ 0.
    let len = 512usize;
    let a: Vec<f32> = (0..len).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let b = a.clone();
    let ref_val = dot_scalar(&a, &b);      // = 512.0
    let bf16_val = unsafe { dot_avx512_bf16(a.as_ptr(), b.as_ptr(), len) };
    close(bf16_val, ref_val);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn fast_score_many_uses_bf16_on_bf16_hardware() {
    if !std::arch::is_x86_feature_detected!("avx512bf16") {
        return;
    }
    // fast_score_many routes to dot_many_avx512_bf16 on BF16 hardware;
    // verify results match fast_score on 3 vectors (exercises partial-group fallback).
    let idx = HNSWIndex::new(crate::utils::types::DistanceMetric::Cosine, 4, 8, 4, 4);
    let q = vec![1.0f32, 0.0, 0.0, 0.0];
    let vs: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.707, 0.707, 0.0, 0.0],
    ];
    let vref: Vec<&[f32]> = vs.iter().map(|v| v.as_slice()).collect();
    let mut out = [0.0f32; 4];
    idx.fast_score_many(&q, &vref, &mut out);
    for (i, v) in vs.iter().enumerate() {
        let single = idx.fast_score(&q, v);
        assert!((out[i] - single).abs() < 1e-3, "many[{i}]={} single={}", out[i], single);
    }
}
```

- [ ] **Step 2: Run to confirm compile error (function not yet defined)**

```bash
cargo test -p annex --lib "dot_avx512_bf16_matches_scalar|fast_score_many_uses_bf16" 2>&1 | tail -5
```
Expected: compile error (`dot_avx512_bf16` not found). On non-x86 (Mac/aarch64) the cfg block compiles away — the test body is skipped entirely, not a compile error. That's correct.

- [ ] **Step 3: Implement `dot_avx512_bf16`**

Add after `dot_avx512` (around line 1070) in `core.rs`:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx512_bf16(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut i = 0usize;
    // Main loop: 4 × 32-element BF16 groups = 128 f32 values per iteration.
    while i + 128 <= len {
        let a0 = _mm512_loadu_ps(a.add(i));
        let a1 = _mm512_loadu_ps(a.add(i + 16));
        let b0 = _mm512_loadu_ps(b.add(i));
        let b1 = _mm512_loadu_ps(b.add(i + 16));
        acc0 = _mm512_dpbf16_ps(acc0, _mm512_cvtne2ps_pbh(a1, a0), _mm512_cvtne2ps_pbh(b1, b0));

        let a2 = _mm512_loadu_ps(a.add(i + 32));
        let a3 = _mm512_loadu_ps(a.add(i + 48));
        let b2 = _mm512_loadu_ps(b.add(i + 32));
        let b3 = _mm512_loadu_ps(b.add(i + 48));
        acc1 = _mm512_dpbf16_ps(acc1, _mm512_cvtne2ps_pbh(a3, a2), _mm512_cvtne2ps_pbh(b3, b2));

        let a4 = _mm512_loadu_ps(a.add(i + 64));
        let a5 = _mm512_loadu_ps(a.add(i + 80));
        let b4 = _mm512_loadu_ps(b.add(i + 64));
        let b5 = _mm512_loadu_ps(b.add(i + 80));
        acc2 = _mm512_dpbf16_ps(acc2, _mm512_cvtne2ps_pbh(a5, a4), _mm512_cvtne2ps_pbh(b5, b4));

        let a6 = _mm512_loadu_ps(a.add(i + 96));
        let a7 = _mm512_loadu_ps(a.add(i + 112));
        let b6 = _mm512_loadu_ps(b.add(i + 96));
        let b7 = _mm512_loadu_ps(b.add(i + 112));
        acc3 = _mm512_dpbf16_ps(acc3, _mm512_cvtne2ps_pbh(a7, a6), _mm512_cvtne2ps_pbh(b7, b6));
        i += 128;
    }
    // Remaining 32-element groups.
    while i + 32 <= len {
        let a0 = _mm512_loadu_ps(a.add(i));
        let a1 = _mm512_loadu_ps(a.add(i + 16));
        let b0 = _mm512_loadu_ps(b.add(i));
        let b1 = _mm512_loadu_ps(b.add(i + 16));
        acc0 = _mm512_dpbf16_ps(acc0, _mm512_cvtne2ps_pbh(a1, a0), _mm512_cvtne2ps_pbh(b1, b0));
        i += 32;
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    let mut result = _mm512_reduce_add_ps(acc0);
    // Scalar tail for len % 32 — avoids a partial BF16 pack.
    while i < len {
        result += *a.add(i) * *b.add(i);
        i += 1;
    }
    result
}
```

- [ ] **Step 4: Implement `dot_many_avx512_bf16`**

Add after `dot_avx512_bf16`:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_many_avx512_bf16(query: &[f32], vecs: &[&[f32]], out: &mut [f32]) {
    // Delegate to per-vector dot_avx512_bf16; 4-vector grouping doesn't add
    // benefit over the per-vector BF16 path (register width already saturated).
    let n = query.len();
    let q = query.as_ptr();
    for (o, v) in out.iter_mut().zip(vecs) {
        *o = dot_avx512_bf16(q, v.as_ptr(), n.min(v.len()));
    }
}
```

- [ ] **Step 5: Wire `Avx512Bf16` into the three selectors**

In `select_dot_fn()` (around line 167), add above the `Avx512Vnni | Avx512F` arm:

```rust
#[cfg(target_arch = "x86_64")]
CpuLevel::Avx512Bf16 => {
    |q: &[f32], v: &[f32]| unsafe { dot_avx512_bf16(q.as_ptr(), v.as_ptr(), q.len()) }
}
```

In `select_l2_fn()` (around line 186), merge `Avx512Bf16` into the existing AVX-512 arm:

```rust
#[cfg(target_arch = "x86_64")]
CpuLevel::Avx512Bf16 | CpuLevel::Avx512Vnni | CpuLevel::Avx512F => {
    |q: &[f32], v: &[f32]| unsafe { l2_avx512(q, v) }
}
```

In `select_dot_many_fn()` (around line 205), add above the `Avx512Vnni | Avx512F` arm:

```rust
#[cfg(target_arch = "x86_64")]
CpuLevel::Avx512Bf16 => dot_many_avx512_bf16,
```

- [ ] **Step 6: Run correctness tests locally**

```bash
cargo test -p annex --lib 2>&1 | tail -10
```
Expected: all pass (14+ tests). On Mac/aarch64 the BF16 tests compile but return early (no-op). On Ice Lake EC2 they return early too.

- [ ] **Step 7: Run on EC2 to confirm no regression (BF16 path unreachable on Ice Lake — should still pass)**

```bash
rsync -av --exclude='target/' -e "ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no" \
  /Users/rohansharma/Desktop/Code/vectordb/ ec2-user@3.137.213.55:~/vectordb/

ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 \
  "cd ~/vectordb && ANNEX_REQUIRE_X86_FEATURES=avx2,fma,avx512f,avx512vnni \
   cargo test -p annex --lib -- --nocapture 2>&1 | tail -10" 2>/dev/null
```
Expected: all pass. No BF16 path activated (Ice Lake lacks `avx512bf16`).

- [ ] **Step 8: Commit**

```bash
git add crates/annex-core/src/vector/hnsw/core.rs
git commit -m "feat(simd): add dot_avx512_bf16 and dispatch wiring in core.rs"
```

---

### Task 3: BF16 kernels in `fde.rs` + dispatch wiring

**Files:**
- Modify: `crates/annex-multivector/src/fde.rs`

**Interfaces:**
- Consumes: `CpuLevel::Avx512Bf16` from Task 1
- Produces: `unsafe fn dot_avx512_bf16_len(a: *const f32, b: *const f32, len: usize) -> f32`, `fn maxsim_flat_avx512_bf16(query: &[Vector], document: &[f32], dimension: usize) -> f32`, updated `dot` and `maxsim_flat` dispatch

- [ ] **Step 1: Write failing tests**

Add to the `tests` mod in `fde.rs`:

```rust
#[test]
fn maxsim_flat_bf16_agrees_with_scalar_on_dim128() {
    if cfg!(not(target_arch = "x86_64")) { return; }
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx512bf16") { return; }

    let mut rng = 0xcafe_babe_u64;
    let mut next = || -> f32 {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
    };
    let query: Vec<_> = (0..8).map(|_| (0..128).map(|_| next()).collect::<Vec<f32>>()).collect();
    let doc_tokens: Vec<_> = (0..50).map(|_| (0..128).map(|_| next()).collect::<Vec<f32>>()).collect();
    let (flat_doc, dim) = {
        let d = doc_tokens.iter().flat_map(|v| v.iter().copied()).collect::<Vec<f32>>();
        (d, 128usize)
    };
    let scalar = maxsim_flat_scalar(&query, &flat_doc, dim);
    let dispatch = maxsim_flat(&query, &flat_doc, dim);
    let tol = 1e-3_f32 * scalar.abs().max(1.0);
    assert!((dispatch - scalar).abs() <= tol, "bf16 maxsim mismatch: scalar={scalar} dispatch={dispatch}");
}

#[test]
fn dot_self_is_near_one_after_normalize_on_bf16() {
    if cfg!(not(target_arch = "x86_64")) { return; }
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx512bf16") { return; }
    // normalize() calls dot(v, v) which routes to BF16 on BF16 hardware.
    for dim in [64usize, 128, 384, 768] {
        let raw: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).recip()).collect();
        let normed = normalize(&raw);
        let self_dot = dot(&normed, &normed);
        assert!((self_dot - 1.0).abs() < 1e-3, "dim={dim}: self_dot={self_dot}");
    }
}
```

- [ ] **Step 2: Run to confirm they compile but pass immediately on non-BF16 hardware**

```bash
cargo test -p annex-multivector --lib "maxsim_flat_bf16|dot_self_is_near_one" -- --nocapture 2>&1 | tail -6
```
Expected: PASS (both return early via `if !avx512bf16 { return; }`).

- [ ] **Step 3: Implement `dot_avx512_bf16_len`**

Add after `dot_avx512_len` in `fde.rs` (around line 219):

```rust
/// BF16 fused dot product for fde.rs — same algorithm as dot_avx512_bf16 in
/// annex-core, duplicated to avoid a cross-crate dependency on a private fn.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx512_bf16_len(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 128 <= len {
        let a0 = _mm512_loadu_ps(a.add(i));       let a1 = _mm512_loadu_ps(a.add(i + 16));
        let b0 = _mm512_loadu_ps(b.add(i));       let b1 = _mm512_loadu_ps(b.add(i + 16));
        acc0 = _mm512_dpbf16_ps(acc0, _mm512_cvtne2ps_pbh(a1, a0), _mm512_cvtne2ps_pbh(b1, b0));
        let a2 = _mm512_loadu_ps(a.add(i + 32));  let a3 = _mm512_loadu_ps(a.add(i + 48));
        let b2 = _mm512_loadu_ps(b.add(i + 32));  let b3 = _mm512_loadu_ps(b.add(i + 48));
        acc1 = _mm512_dpbf16_ps(acc1, _mm512_cvtne2ps_pbh(a3, a2), _mm512_cvtne2ps_pbh(b3, b2));
        let a4 = _mm512_loadu_ps(a.add(i + 64));  let a5 = _mm512_loadu_ps(a.add(i + 80));
        let b4 = _mm512_loadu_ps(b.add(i + 64));  let b5 = _mm512_loadu_ps(b.add(i + 80));
        acc2 = _mm512_dpbf16_ps(acc2, _mm512_cvtne2ps_pbh(a5, a4), _mm512_cvtne2ps_pbh(b5, b4));
        let a6 = _mm512_loadu_ps(a.add(i + 96));  let a7 = _mm512_loadu_ps(a.add(i + 112));
        let b6 = _mm512_loadu_ps(b.add(i + 96));  let b7 = _mm512_loadu_ps(b.add(i + 112));
        acc3 = _mm512_dpbf16_ps(acc3, _mm512_cvtne2ps_pbh(a7, a6), _mm512_cvtne2ps_pbh(b7, b6));
        i += 128;
    }
    while i + 32 <= len {
        let a0 = _mm512_loadu_ps(a.add(i));
        let a1 = _mm512_loadu_ps(a.add(i + 16));
        let b0 = _mm512_loadu_ps(b.add(i));
        let b1 = _mm512_loadu_ps(b.add(i + 16));
        acc0 = _mm512_dpbf16_ps(acc0, _mm512_cvtne2ps_pbh(a1, a0), _mm512_cvtne2ps_pbh(b1, b0));
        i += 32;
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    let mut result = _mm512_reduce_add_ps(acc0);
    while i < len {
        result += *a.add(i) * *b.add(i);
        i += 1;
    }
    result
}
```

- [ ] **Step 4: Implement `maxsim_flat_avx512_bf16`**

Add after `maxsim_flat_avx512` (around line 397):

```rust
#[cfg(target_arch = "x86_64")]
fn maxsim_flat_avx512_bf16(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    let dot_fn: unsafe fn(*const f32, *const f32, usize) -> f32 = dot_avx512_bf16_len;
    query
        .iter()
        .map(|q| {
            debug_assert_eq!(q.len(), dimension);
            let qp = q.as_ptr();
            let mut best = f32::NEG_INFINITY;
            for doc in document.chunks_exact(dimension) {
                let s = unsafe { dot_fn(qp, doc.as_ptr(), dimension) };
                if s > best {
                    best = s;
                }
            }
            best
        })
        .sum()
}
```

- [ ] **Step 5: Wire `Avx512Bf16` into `fde::dot` dispatch**

In `fde::dot`, add above the `Avx512Vnni | Avx512F` arm (around line 59):

```rust
CpuLevel::Avx512Bf16 => {
    return unsafe { dot_avx512_bf16_len(left.as_ptr(), right.as_ptr(), n) };
}
```

- [ ] **Step 6: Wire `Avx512Bf16` into `maxsim_flat` dispatch**

In `maxsim_flat`, add above the `Avx512Vnni | Avx512F` arm (around line 281):

```rust
CpuLevel::Avx512Bf16 => {
    return maxsim_flat_avx512_bf16(query, document, dimension);
}
```

- [ ] **Step 7: Run full test suite**

```bash
cargo test --workspace 2>&1 | grep -E "^test result|FAILED" | head -10
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add crates/annex-multivector/src/fde.rs
git commit -m "feat(simd): add dot_avx512_bf16_len and maxsim_flat_avx512_bf16 in fde.rs"
```

---

### Task 4: Criterion benchmark additions

**Files:**
- Modify: `crates/annex-multivector/benches/kernels.rs`

**Interfaces:**
- Consumes: nothing new — `maxsim_flat` now routes to BF16 on BF16 hardware, so the existing `bench_maxsim_flat` automatically exercises the new path on Sapphire Rapids

- [ ] **Step 1: Add `bench_maxsim_flat_bf16` group**

The existing `bench_maxsim_flat` already calls `maxsim_flat()` which routes to BF16 on BF16 hardware. Add a separate group to make the BF16 path explicitly visible in benchmark reports, and to allow side-by-side runs by temporarily overriding dispatch on the same machine:

```rust
fn bench_maxsim_flat_bf16(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx512bf16") {
        // Emit a note so it's clear the bench was considered, just not runnable.
        let _ = c; // suppress unused warning
        println!("[bench_maxsim_flat_bf16] avx512bf16 not available on this host — skipping");
        return;
    }
    // On avx512bf16 hardware, maxsim_flat() routes here automatically.
    // Benchmarks identical to bench_maxsim_flat so results can be diffed.
    let mut group = c.benchmark_group("maxsim_flat_bf16");
    for &(dim, doc_tokens, query_tokens) in &[
        (128usize, 200usize, 32usize),
        (128, 100, 32),
        (384, 200, 32),
        (512, 200, 32),
        (768, 200, 32),
    ] {
        let mut rng_state = 0xdead_beef_cafe_babe_u64;
        let doc_matrix = gen_normalized(&mut rng_state, dim, doc_tokens);
        let doc = flat(&doc_matrix);
        let query = gen_normalized(&mut rng_state, dim, query_tokens);
        let label = format!("dim={dim}/doc_tokens={doc_tokens}/query_tokens={query_tokens}");
        let ops = (dim as u64) * (doc_tokens as u64) * (query_tokens as u64) * 2;
        group.throughput(Throughput::Elements(ops));
        group.bench_function(&label, |b| {
            b.iter(|| maxsim_flat(black_box(&query), black_box(&doc), black_box(dim)))
        });
    }
    group.finish();
}
```

Update `criterion_group!` and `criterion_main!` at the bottom of the file:

```rust
criterion_group!(kernels, bench_maxsim_flat, bench_dot, bench_screen_dot, bench_maxsim_flat_bf16);
criterion_main!(kernels);
```

- [ ] **Step 2: Compile-check bench binary**

```bash
cargo bench --no-run -p annex-multivector 2>&1 | grep -E "^error|Finished" | head -5
```
Expected: `Finished` with no errors.

- [ ] **Step 3: Run bench locally to verify no-op message (Mac/Ice Lake — no BF16)**

```bash
cargo bench --bench kernels -- bench_maxsim_flat_bf16 --noplot 2>&1 | grep -E "avx512bf16|Benchmarking|time:" | head -5
```
Expected on non-BF16 hardware: prints `avx512bf16 not available on this host — skipping` and exits immediately.

- [ ] **Step 4: Commit**

```bash
git add crates/annex-multivector/benches/kernels.rs
git commit -m "bench: add bench_maxsim_flat_bf16 group for Sapphire Rapids validation"
```

---

## Notes for Sapphire Rapids validation

Once a c7i instance (AWS) is available, run:

```bash
# On the Sapphire Rapids instance:
RUSTFLAGS='-C target-cpu=native' cargo bench --bench kernels -- --noplot 2>&1 | \
  grep -E "maxsim_flat/|maxsim_flat_bf16/" | head -20
```

Expected: `maxsim_flat_bf16` ~1.5–2× faster than `maxsim_flat` on cached dims (128, 384). The `maxsim_flat` group will also show BF16 numbers since dispatch routes there automatically.

To compare BF16 vs AVX-512 f32 on the same hardware, temporarily change `select_dot_fn` to skip `Avx512Bf16` → get baseline, then restore → get BF16 numbers.
