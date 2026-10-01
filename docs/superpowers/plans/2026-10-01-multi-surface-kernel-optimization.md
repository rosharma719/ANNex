# Multi-Surface Kernel Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AVX-512 and x86 AVX2 SIMD kernels to ANNex's HNSW search and MaxSim rescoring paths, hoist dispatch outside hot loops, and validate on an EC2 Ice Lake instance.

**Architecture:** A new `simd.rs` module in `annex-core` defines a `CpuLevel` enum detected once per call (std_detect already caches it). `HNSWIndex` stores `dot_fn`/`l2_fn` function pointers selected at construction so `fast_score` calls them without re-detection. `fde.rs` calls `cpu_level()` once before its outer loop to select a function pointer.

**Tech Stack:** Rust 1.99.0, `std::arch::x86_64` intrinsics (`avx2`, `fma`, `avx512f`, `avx512vnni`), `std::arch::aarch64` intrinsics, Criterion 0.5, EC2 c6i.xlarge (Intel Ice Lake, SSH key `~/.ssh/annex-bench.pem`, host `3.137.213.55`).

**Spec:** `docs/superpowers/specs/2026-10-01-multi-surface-kernel-optimization-design.md`

## Global Constraints

- Tolerance for f32 kernel correctness: `abs(actual − ref) ≤ 1e-3 × max(1.0, abs(ref))`
- `screen_dot` integer kernels must be exactly equal to `screen_dot_scalar` (no floating-point error)
- `dot_avx2` / `l2_avx2` dispatch arm must not regress (CPUs with AVX2 but no FMA must still use AVX2, not scalar)
- AVX-512 kernels gated by `avx512f` feature detection; VNNI kernel additionally gated by `avx512vnni`
- `screen_dot` is `pub(crate)` — expose via `bench-internals` Cargo feature only; do not change the public API
- All new unsafe functions must have `#[allow(unsafe_op_in_unsafe_fn)]` and `#[target_feature(enable = "...")]`
- Commits go to branch `feat/query-planner`; commit after each task passes tests

## Review Focus

1. **AVX2-only CPUs (no FMA):** `dot_product` and `l2_squared` dispatch must route to `dot_avx2`/`l2_avx2`, not scalar. Test by calling `dot_avx2`/`l2_avx2` directly in `x86_simd_kernels_match_scalar_across_shapes`.
2. **VNNI centering correction off-by-one:** `screen_dot_avx512_vnni` computes `sum(s[i]*q[i]) - 128*sum(q[i])`. Integer extremes (`q=+127, s=255` and `q=-128, s=0`) must match scalar exactly. Test added to Task 3.
3. **`from_snapshot` missing scorer:** `HNSWIndex::from_snapshot` must also set `dot_fn`/`l2_fn`. Test by constructing via snapshot and calling `fast_score`. Added to Task 4.
4. **Indirect-call regression:** struct-stored fn pointer may be slower than the current predictable branch. Task 6 benchmarks both; Task 4 includes a fallback note if dispatch overhead shows regression.
5. **`fde.rs` scalar tail for non-multiple lengths:** x86 kernels in `fde.rs` must produce correct results for `len=1`, `len=7`, `len=9` (below and straddling SIMD width). Test added to Task 5.

---

### Task 1: `simd.rs` — `CpuLevel` enum and detection

**Files:**
- Create: `crates/annex-core/src/vector/simd.rs`
- Modify: `crates/annex-core/src/vector/mod.rs` (add `pub mod simd`)

**Interfaces:**
- Produces: `pub enum CpuLevel`, `pub fn cpu_level() -> CpuLevel` at path `annex::vector::simd::{CpuLevel, cpu_level}`

- [ ] **Step 1: Write failing test**

Add to `crates/annex-core/src/vector/simd.rs` (create the file):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_level_does_not_panic() {
        let level = cpu_level();
        // On x86_64 with AVX2 we expect at least Avx2.
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            assert!(matches!(
                level,
                CpuLevel::Avx2 | CpuLevel::Avx2Fma | CpuLevel::Avx512F | CpuLevel::Avx512Vnni
            ));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p annex --lib vector::simd 2>&1 | tail -5
```
Expected: compile error (module doesn't exist yet).

- [ ] **Step 3: Implement `simd.rs`**

```rust
/// CPU capability level, detected once per call (std_detect caches internally).
/// Use this to select a function pointer *before* a hot inner loop — not inside it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuLevel {
    NeonDotprod, // aarch64 + dotprod extension
    Neon,        // aarch64 baseline
    Avx512Vnni,  // x86: avx512f + avx512vnni (Ice Lake, Zen4)
    Avx512F,     // x86: avx512f only (Skylake-X)
    Avx2Fma,     // x86: avx2 + fma (most cloud: c5, m5, c6a)
    Avx2,        // x86: avx2 only
    Scalar,
}

pub fn cpu_level() -> CpuLevel {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return CpuLevel::NeonDotprod;
        }
        return CpuLevel::Neon;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512f")
        {
            return CpuLevel::Avx512Vnni;
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            return CpuLevel::Avx512F;
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return CpuLevel::Avx2Fma;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return CpuLevel::Avx2;
        }
    }
    CpuLevel::Scalar
}

#[cfg(test)]
mod tests { /* as above */ }
```

Add to `crates/annex-core/src/vector/mod.rs`:
```rust
pub mod simd;
```

- [ ] **Step 4: Run test**

```bash
cargo test -p annex --lib vector::simd -- --nocapture 2>&1 | tail -10
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/annex-core/src/vector/simd.rs crates/annex-core/src/vector/mod.rs
git commit -m "feat(simd): add CpuLevel enum and cpu_level() detection"
```

---

### Task 2: AVX-512 f32 kernels — `dot_avx512` and `l2_avx512`

**Files:**
- Modify: `crates/annex-core/src/vector/hnsw/core.rs` (add two kernel functions + extend test)

**Interfaces:**
- Consumes: nothing new
- Produces: `unsafe fn dot_avx512(query: &[f32], vec: &[f32]) -> f32` (x86_64 only), `unsafe fn l2_avx512(query: &[f32], vec: &[f32]) -> f32` (x86_64 only)

- [ ] **Step 1: Extend the existing correctness test to cover AVX-512 (failing first)**

In `x86_simd_kernels_match_scalar_across_shapes` (line 1634), inside the dim loop after the `avx2 && fma` block, add:

```rust
let avx512f = std::arch::is_x86_feature_detected!("avx512f");
// ... in the loop:
if avx512f {
    unsafe {
        close(dot_avx512(&query, &vector), dot_ref);
        close(l2_avx512(&query, &vector), l2_ref);
    }
}
```

Also update the `eprintln!` at the top of the test:
```rust
eprintln!("x86 kernel coverage: avx2={avx2} fma={fma} avx512f={avx512f}");
```

- [ ] **Step 2: Run to confirm compile error**

```bash
cargo test -p annex --lib "x86_simd_kernels_match_scalar_across_shapes" 2>&1 | tail -5
```
Expected: compile error (`dot_avx512` not found).

- [ ] **Step 3: Implement `dot_avx512`**

Add after `dot_avx2_fma` in `core.rs`:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx512(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = query.len().min(vec.len());
    let mut s0 = _mm512_setzero_ps();
    let mut s1 = _mm512_setzero_ps();
    let mut s2 = _mm512_setzero_ps();
    let mut s3 = _mm512_setzero_ps();
    let mut i = 0;
    while i + 64 <= len {
        let q0 = _mm512_loadu_ps(query.as_ptr().add(i));
        let q1 = _mm512_loadu_ps(query.as_ptr().add(i + 16));
        let q2 = _mm512_loadu_ps(query.as_ptr().add(i + 32));
        let q3 = _mm512_loadu_ps(query.as_ptr().add(i + 48));
        let v0 = _mm512_loadu_ps(vec.as_ptr().add(i));
        let v1 = _mm512_loadu_ps(vec.as_ptr().add(i + 16));
        let v2 = _mm512_loadu_ps(vec.as_ptr().add(i + 32));
        let v3 = _mm512_loadu_ps(vec.as_ptr().add(i + 48));
        s0 = _mm512_fmadd_ps(q0, v0, s0);
        s1 = _mm512_fmadd_ps(q1, v1, s1);
        s2 = _mm512_fmadd_ps(q2, v2, s2);
        s3 = _mm512_fmadd_ps(q3, v3, s3);
        i += 64;
    }
    while i + 16 <= len {
        let q = _mm512_loadu_ps(query.as_ptr().add(i));
        let v = _mm512_loadu_ps(vec.as_ptr().add(i));
        s0 = _mm512_fmadd_ps(q, v, s0);
        i += 16;
    }
    s0 = _mm512_add_ps(s0, s1);
    s2 = _mm512_add_ps(s2, s3);
    s0 = _mm512_add_ps(s0, s2);
    // _mm512_reduce_add_ps is a soft reduction — inspect asm and replace with
    // explicit 512→256→128 tree if compiler emits suboptimal code.
    let mut acc = _mm512_reduce_add_ps(s0);
    while i < len {
        acc += *query.get_unchecked(i) * *vec.get_unchecked(i);
        i += 1;
    }
    acc
}
```

- [ ] **Step 4: Implement `l2_avx512`**

Add after `dot_avx512`:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn l2_avx512(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = query.len().min(vec.len());
    let mut s0 = _mm512_setzero_ps();
    let mut s1 = _mm512_setzero_ps();
    let mut s2 = _mm512_setzero_ps();
    let mut s3 = _mm512_setzero_ps();
    let mut i = 0;
    while i + 64 <= len {
        let q0 = _mm512_loadu_ps(query.as_ptr().add(i));
        let q1 = _mm512_loadu_ps(query.as_ptr().add(i + 16));
        let q2 = _mm512_loadu_ps(query.as_ptr().add(i + 32));
        let q3 = _mm512_loadu_ps(query.as_ptr().add(i + 48));
        let v0 = _mm512_loadu_ps(vec.as_ptr().add(i));
        let v1 = _mm512_loadu_ps(vec.as_ptr().add(i + 16));
        let v2 = _mm512_loadu_ps(vec.as_ptr().add(i + 32));
        let v3 = _mm512_loadu_ps(vec.as_ptr().add(i + 48));
        let d0 = _mm512_sub_ps(q0, v0);
        let d1 = _mm512_sub_ps(q1, v1);
        let d2 = _mm512_sub_ps(q2, v2);
        let d3 = _mm512_sub_ps(q3, v3);
        s0 = _mm512_fmadd_ps(d0, d0, s0);
        s1 = _mm512_fmadd_ps(d1, d1, s1);
        s2 = _mm512_fmadd_ps(d2, d2, s2);
        s3 = _mm512_fmadd_ps(d3, d3, s3);
        i += 64;
    }
    while i + 16 <= len {
        let q = _mm512_loadu_ps(query.as_ptr().add(i));
        let v = _mm512_loadu_ps(vec.as_ptr().add(i));
        let d = _mm512_sub_ps(q, v);
        s0 = _mm512_fmadd_ps(d, d, s0);
        i += 16;
    }
    s0 = _mm512_add_ps(s0, s1);
    s2 = _mm512_add_ps(s2, s3);
    s0 = _mm512_add_ps(s0, s2);
    let mut acc = _mm512_reduce_add_ps(s0);
    while i < len {
        let d = *query.get_unchecked(i) - *vec.get_unchecked(i);
        acc += d * d;
        i += 1;
    }
    acc
}
```

- [ ] **Step 5: Run correctness test locally (compiles; AVX-512 path runs only on EC2)**

```bash
cargo test -p annex --lib "x86_simd_kernels_match_scalar_across_shapes" -- --nocapture 2>&1 | tail -10
```
Expected: PASS. On a Mac (NEON) the `avx512f` block is skipped; on EC2 it runs.

- [ ] **Step 6: Run on EC2 to exercise AVX-512 path**

```bash
rsync -av --exclude='target/' -e "ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no" \
  /Users/rohansharma/Desktop/Code/vectordb/ ec2-user@3.137.213.55:~/vectordb/

ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 \
  "cd ~/vectordb && ANNEX_REQUIRE_X86_FEATURES=avx2,fma,avx512f \
   cargo test -p annex --lib x86_simd_kernels_match_scalar_across_shapes -- --nocapture 2>&1 | tail -15"
```
Expected: PASS with `avx512f=true` in output.

- [ ] **Step 7: Commit**

```bash
git add crates/annex-core/src/vector/hnsw/core.rs
git commit -m "feat(simd): add dot_avx512 and l2_avx512 kernels"
```

---

### Task 3: `screen_dot_avx512_vnni` + `bench-internals` feature

**Files:**
- Modify: `crates/annex-core/src/vector/hnsw/core.rs` (add kernel + tests + pub re-export)
- Modify: `crates/annex-core/Cargo.toml` (add `bench-internals` feature)

**Interfaces:**
- Produces: `unsafe fn screen_dot_avx512_vnni(query_i8: &[i8], stored: &[u8]) -> i32`
- Produces (behind feature flag): `pub fn screen_dot(query_i8: &[i8], stored: &[u8]) -> i32`, `pub fn screen_dot_scalar_pub(query_i8: &[i8], stored: &[u8]) -> i32`

- [ ] **Step 1: Add `bench-internals` feature to `annex-core/Cargo.toml`**

```toml
[features]
bench-internals = []
```

- [ ] **Step 2: Write failing tests — integer extremes and AVX-512 VNNI path**

Add a new test function `screen_dot_all_paths_agree` to the `tests` mod in `core.rs`:

```rust
#[test]
fn screen_dot_all_paths_agree() {
    let cases: &[(&[i8], &[u8])] = &[
        // empty
        (&[], &[]),
        // short
        (&[1], &[129]),
        (&[127, -128, 0, 1], &[255, 0, 128, 200]),
        // extremes: max product per element
        (&[127i8; 32], &[255u8; 32]),
        (&[-128i8; 32], &[0u8; 32]),
        // adversarial adjacent pairs (validates no saturation in any rewrite)
        (&[127i8; 64], &[255u8; 64]),
        // standard dims
        (&vec![42i8; 128], &vec![200u8; 128]),
        (&vec![-1i8; 256], &vec![128u8; 256]),
        // cycle through all extreme values
        (&(0..256usize).map(|i| [-128, -127, -1, 0, 1, 126, 127][i % 7]).collect::<Vec<i8>>(),
         &(0..256usize).map(|i| [0, 1, 127, 128, 129, 254, 255][i % 7]).collect::<Vec<u8>>()),
    ];

    for (q, s) in cases {
        let reference = screen_dot_scalar(q, s);

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let avx2_result = unsafe { screen_dot_avx2(q, s) };
                assert_eq!(avx2_result, reference,
                    "AVX2 mismatch: q={q:?}, s={s:?}");
            }
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512vnni")
            {
                let vnni_result = unsafe { screen_dot_avx512_vnni(q, s) };
                assert_eq!(vnni_result, reference,
                    "VNNI mismatch: q={q:?}, s={s:?}");
            }
        }
    }
}
```

- [ ] **Step 3: Run to confirm compile error**

```bash
cargo test -p annex --lib screen_dot_all_paths_agree 2>&1 | tail -5
```
Expected: compile error (`screen_dot_avx512_vnni` not found).

- [ ] **Step 4: Implement `screen_dot_avx512_vnni`**

Add after `screen_dot_avx2` in `core.rs`:

```rust
/// AVX-512 VNNI path. `_mm512_dpbusd_epi32(acc, u8_stored, i8_query)` computes
/// `acc += sum(u8[i] * i8[i])` across 64 elements per call.
///
/// Centering correction: sum(q*(s-128)) = sum(q*s) - 128*sum(q).
/// The VNNI accumulator computes sum(q*s); 128*sum(q) is subtracted afterward.
///
/// Safety: `n = query_i8.len().min(stored.len())`. In the HNSW path both slices
/// have the same length (index invariant). Overflow proof: max correction =
/// dim * 127 * 128 ≤ 4096 * 127 * 128 = 66_584_576 << i32::MAX.
#[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn screen_dot_avx512_vnni(query_i8: &[i8], stored: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    let n = query_i8.len().min(stored.len());
    // Precompute centering correction (scalar, once per query vector).
    let query_sum: i32 = query_i8[..n].iter().map(|&x| x as i32).sum();

    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();
    let mut i = 0usize;

    while i + 256 <= n {
        let s0 = _mm512_loadu_si512(stored.as_ptr().add(i) as *const __m512i);
        let s1 = _mm512_loadu_si512(stored.as_ptr().add(i + 64) as *const __m512i);
        let s2 = _mm512_loadu_si512(stored.as_ptr().add(i + 128) as *const __m512i);
        let s3 = _mm512_loadu_si512(stored.as_ptr().add(i + 192) as *const __m512i);
        let q0 = _mm512_loadu_si512(query_i8.as_ptr().add(i) as *const __m512i);
        let q1 = _mm512_loadu_si512(query_i8.as_ptr().add(i + 64) as *const __m512i);
        let q2 = _mm512_loadu_si512(query_i8.as_ptr().add(i + 128) as *const __m512i);
        let q3 = _mm512_loadu_si512(query_i8.as_ptr().add(i + 192) as *const __m512i);
        // stored (u8) in `a` position, query (i8) in `b` position per Intel spec
        acc0 = _mm512_dpbusd_epi32(acc0, s0, q0);
        acc1 = _mm512_dpbusd_epi32(acc1, s1, q1);
        acc2 = _mm512_dpbusd_epi32(acc2, s2, q2);
        acc3 = _mm512_dpbusd_epi32(acc3, s3, q3);
        i += 256;
    }
    while i + 64 <= n {
        let s = _mm512_loadu_si512(stored.as_ptr().add(i) as *const __m512i);
        let q = _mm512_loadu_si512(query_i8.as_ptr().add(i) as *const __m512i);
        acc0 = _mm512_dpbusd_epi32(acc0, s, q);
        i += 64;
    }

    acc0 = _mm512_add_epi32(acc0, acc1);
    acc2 = _mm512_add_epi32(acc2, acc3);
    acc0 = _mm512_add_epi32(acc0, acc2);
    let mut result = _mm512_reduce_add_epi32(acc0);

    while i < n {
        result += (*query_i8.get_unchecked(i) as i32) * (*stored.get_unchecked(i) as i32);
        i += 1;
    }

    result - 128 * query_sum
}
```

- [ ] **Step 5: Add `bench-internals` public re-export**

At the bottom of `core.rs` (after the final `impl HNSWIndex {}`), add:

```rust
/// Re-exports for micro-benchmarks. Not part of the public API.
#[cfg(feature = "bench-internals")]
pub mod bench_access {
    pub use super::{screen_dot_scalar, screen_dot_avx2};
    #[cfg(target_arch = "x86_64")]
    pub use super::screen_dot_avx512_vnni;
    pub use super::HNSWIndex;
}
```

Also add to `crates/annex-core/src/lib.rs` (after existing pub use lines):

```rust
#[cfg(feature = "bench-internals")]
pub use crate::vector::hnsw::core::bench_access;
```

- [ ] **Step 6: Run test locally**

```bash
cargo test -p annex --lib screen_dot_all_paths_agree -- --nocapture 2>&1 | tail -10
```
Expected: PASS (AVX-512 VNNI block skipped on Mac; AVX2 block exercises that path if on x86).

- [ ] **Step 7: Run on EC2**

```bash
rsync -av --exclude='target/' -e "ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no" \
  /Users/rohansharma/Desktop/Code/vectordb/ ec2-user@3.137.213.55:~/vectordb/

ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 \
  "cd ~/vectordb && ANNEX_REQUIRE_X86_FEATURES=avx2,fma,avx512f,avx512vnni \
   cargo test -p annex --lib 'screen_dot_all_paths_agree|x86_simd_kernels' -- --nocapture 2>&1 | tail -15"
```
Expected: PASS with VNNI path exercised.

- [ ] **Step 8: Commit**

```bash
git add crates/annex-core/src/vector/hnsw/core.rs crates/annex-core/Cargo.toml crates/annex-core/src/lib.rs
git commit -m "feat(simd): add screen_dot_avx512_vnni with centering correction; bench-internals feature"
```

---

### Task 4: Hoist `fast_score` dispatch via `HNSWIndex` scorer fields

**Files:**
- Modify: `crates/annex-core/src/vector/hnsw/core.rs` (struct + `new()` + `fast_score()`)
- Modify: `crates/annex-core/src/vector/hnsw/snapshot.rs` (`from_snapshot()`)

**Interfaces:**
- Consumes: `CpuLevel` from Task 1, `dot_avx512`/`l2_avx512` from Task 2
- Produces: `HNSWIndex::dot_fn: fn(&[f32], &[f32]) -> f32`, same for `l2_fn`

- [ ] **Step 1: Write a regression test for `fast_score` via snapshot round-trip**

Add to the `tests` mod in `core.rs`:

```rust
#[test]
fn fast_score_works_after_snapshot_round_trip() {
    let mut idx = HNSWIndex::new(DistanceMetric::Cosine, 4, 8, 4, 4);
    idx.insert(1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    idx.insert(2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
    let snap = idx.to_snapshot();
    let restored = HNSWIndex::from_snapshot(snap);
    let q = vec![1.0f32, 0.0, 0.0, 0.0];
    let v = vec![0.0f32, 1.0, 0.0, 0.0];
    // Cosine distance: 1 - dot(q,v) = 1 - 0 = 1.0
    let score = restored.fast_score(&q, &v);
    assert!((score - 1.0).abs() < 1e-5, "score={score}");
}
```

- [ ] **Step 2: Add `dot_fn` and `l2_fn` to `HNSWIndex` struct**

In the struct definition (around line 86), add two fields:

```rust
pub struct HNSWIndex {
    // ... existing fields ...
    /// Pre-selected dot-product kernel; set at construction based on cpu_level().
    #[serde(skip)]
    pub(crate) dot_fn: fn(&[f32], &[f32]) -> f32,
    /// Pre-selected L2-squared kernel.
    #[serde(skip)]
    pub(crate) l2_fn: fn(&[f32], &[f32]) -> f32,
}
```

Note: `HNSWIndex` itself does not derive `Serialize/Deserialize` (that's `HnswSnapshot`), so the `#[serde(skip)]` is belt-and-suspenders; confirm it compiles cleanly.

- [ ] **Step 3: Add scorer selection helper**

Add a private function near the top of `impl HNSWIndex`:

```rust
fn select_dot_fn() -> fn(&[f32], &[f32]) -> f32 {
    use crate::vector::simd::{CpuLevel, cpu_level};
    match cpu_level() {
        CpuLevel::Avx512Vnni | CpuLevel::Avx512F => {
            #[cfg(target_arch = "x86_64")]
            { |q: &[f32], v: &[f32]| unsafe { dot_avx512(q, v) } }
            #[cfg(not(target_arch = "x86_64"))]
            { dot_scalar }
        }
        CpuLevel::Avx2Fma => {
            #[cfg(target_arch = "x86_64")]
            { |q: &[f32], v: &[f32]| unsafe { dot_avx2_fma(q, v) } }
            #[cfg(not(target_arch = "x86_64"))]
            { dot_scalar }
        }
        CpuLevel::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            { |q: &[f32], v: &[f32]| unsafe { dot_avx2(q, v) } }
            #[cfg(not(target_arch = "x86_64"))]
            { dot_scalar }
        }
        CpuLevel::NeonDotprod | CpuLevel::Neon => {
            #[cfg(target_arch = "aarch64")]
            { |q: &[f32], v: &[f32]| unsafe { dot_neon(q, v) } }
            #[cfg(not(target_arch = "aarch64"))]
            { dot_scalar }
        }
        CpuLevel::Scalar => dot_scalar,
    }
}

fn select_l2_fn() -> fn(&[f32], &[f32]) -> f32 {
    use crate::vector::simd::{CpuLevel, cpu_level};
    match cpu_level() {
        CpuLevel::Avx512Vnni | CpuLevel::Avx512F => {
            #[cfg(target_arch = "x86_64")]
            { |q: &[f32], v: &[f32]| unsafe { l2_avx512(q, v) } }
            #[cfg(not(target_arch = "x86_64"))]
            { l2_scalar }
        }
        CpuLevel::Avx2Fma => {
            #[cfg(target_arch = "x86_64")]
            { |q: &[f32], v: &[f32]| unsafe { l2_avx2_fma(q, v) } }
            #[cfg(not(target_arch = "x86_64"))]
            { l2_scalar }
        }
        CpuLevel::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            { |q: &[f32], v: &[f32]| unsafe { l2_avx2(q, v) } }
            #[cfg(not(target_arch = "x86_64"))]
            { l2_scalar }
        }
        CpuLevel::NeonDotprod | CpuLevel::Neon => {
            #[cfg(target_arch = "aarch64")]
            { |q: &[f32], v: &[f32]| unsafe { l2_neon(q, v) } }
            #[cfg(not(target_arch = "aarch64"))]
            { l2_scalar }
        }
        CpuLevel::Scalar => l2_scalar,
    }
}
```

- [ ] **Step 4: Wire into `new()` and update `fast_score()`**

In `HNSWIndex::new()`, add to the struct literal:
```rust
dot_fn: Self::select_dot_fn(),
l2_fn: Self::select_l2_fn(),
```

Update `fast_score()` (line 332):
```rust
pub(crate) fn fast_score(&self, query: &[f32], vec: &[f32]) -> f32 {
    match self.metric {
        DistanceMetric::Cosine => {
            let dot = (self.dot_fn)(query, vec);
            let sim = dot.clamp(-1.0, 1.0);
            1.0 - sim
        }
        DistanceMetric::Dot => (self.dot_fn)(query, vec),
        DistanceMetric::Euclidean => (self.l2_fn)(query, vec),
    }
}
```

Remove the now-unused `dot_product()` and `l2_squared()` free functions — or keep them behind `#[allow(dead_code)]` if tests still reference them directly. Check with `cargo test`.

- [ ] **Step 5: Wire into `from_snapshot()`**

In `snapshot.rs`, at the end of the `Self { ... }` literal in `from_snapshot`, add:
```rust
dot_fn: HNSWIndex::select_dot_fn(),
l2_fn: HNSWIndex::select_l2_fn(),
```

`select_dot_fn` and `select_l2_fn` are `fn` (not methods needing `self`), so call them as associated functions.

- [ ] **Step 6: Run all tests**

```bash
cargo test -p annex 2>&1 | tail -20
```
Expected: all pass. Pay attention to any test that constructs `HNSWIndex` via struct literal — those will fail to compile if the new fields aren't added. Fix them by using `HNSWIndex::new(...)` instead.

- [ ] **Step 7: Commit**

```bash
git add crates/annex-core/src/vector/hnsw/core.rs crates/annex-core/src/vector/hnsw/snapshot.rs
git commit -m "feat(simd): hoist fast_score dispatch via HNSWIndex dot_fn/l2_fn fields"
```

---

### Task 5: x86 SIMD paths for `fde.rs` (`maxsim_flat` and `dot`)

**Files:**
- Modify: `crates/annex-multivector/src/fde.rs`

**Interfaces:**
- Consumes: `annex::vector::simd::{CpuLevel, cpu_level}` (from Task 1)
- Produces: `maxsim_flat` dispatches to AVX2+FMA and AVX-512 on x86; `dot` likewise

- [ ] **Step 1: Verify existing tests pass before touching `fde.rs`**

```bash
cargo test -p annex-multivector --lib fde 2>&1 | tail -10
```
Expected: all fde tests pass.

- [ ] **Step 2: Add `use annex::vector::simd::{CpuLevel, cpu_level};` import to `fde.rs`**

At the top of `fde.rs`, add:
```rust
use annex::vector::simd::{CpuLevel, cpu_level};
```

- [ ] **Step 3: Add x86 dot kernels**

Add after the existing `dot_neon_128` function (guarded by `#[cfg(target_arch = "aarch64")]`):

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx2_fma_len(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= len {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)),      _mm256_loadu_ps(b.add(i)),      s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 8)),  _mm256_loadu_ps(b.add(i + 8)),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 16)), _mm256_loadu_ps(b.add(i + 16)), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 24)), _mm256_loadu_ps(b.add(i + 24)), s3);
        i += 32;
    }
    while i + 8 <= len {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), s0);
        i += 8;
    }
    s0 = _mm256_add_ps(s0, s1);
    s2 = _mm256_add_ps(s2, s3);
    s0 = _mm256_add_ps(s0, s2);
    // Reduce 256-bit to scalar without hadd: extract high 128, add, shuffle-reduce.
    let hi = _mm256_extractf128_ps(s0, 1);
    let lo = _mm256_castps256_ps128(s0);
    let sum = _mm_add_ps(hi, lo);
    let shuf = _mm_movehl_ps(sum, sum);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 1);
    let mut acc = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    while i < len {
        acc += *a.add(i) * *b.add(i);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx512_len(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut s0 = _mm512_setzero_ps();
    let mut s1 = _mm512_setzero_ps();
    let mut s2 = _mm512_setzero_ps();
    let mut s3 = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 64 <= len {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.add(i)),      _mm512_loadu_ps(b.add(i)),      s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a.add(i + 16)), _mm512_loadu_ps(b.add(i + 16)), s1);
        s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a.add(i + 32)), _mm512_loadu_ps(b.add(i + 32)), s2);
        s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a.add(i + 48)), _mm512_loadu_ps(b.add(i + 48)), s3);
        i += 64;
    }
    while i + 16 <= len {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.add(i)), _mm512_loadu_ps(b.add(i)), s0);
        i += 16;
    }
    s0 = _mm512_add_ps(s0, s1);
    s2 = _mm512_add_ps(s2, s3);
    s0 = _mm512_add_ps(s0, s2);
    let mut acc = _mm512_reduce_add_ps(s0);
    while i < len {
        acc += *a.add(i) * *b.add(i);
        i += 1;
    }
    acc
}
```

- [ ] **Step 4: Add x86 `maxsim_flat` dispatch functions**

Add after `maxsim_flat_neon`:

```rust
#[cfg(target_arch = "x86_64")]
fn maxsim_flat_avx2_fma(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    let dot_fn: unsafe fn(*const f32, *const f32, usize) -> f32 = dot_avx2_fma_len;
    query
        .iter()
        .map(|q| {
            debug_assert_eq!(q.len(), dimension);
            let qp = q.as_ptr();
            let mut best = f32::NEG_INFINITY;
            for doc in document.chunks_exact(dimension) {
                let s = unsafe { dot_fn(qp, doc.as_ptr(), dimension) };
                if s > best { best = s; }
            }
            best
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
fn maxsim_flat_avx512(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    let dot_fn: unsafe fn(*const f32, *const f32, usize) -> f32 = dot_avx512_len;
    query
        .iter()
        .map(|q| {
            debug_assert_eq!(q.len(), dimension);
            let qp = q.as_ptr();
            let mut best = f32::NEG_INFINITY;
            for doc in document.chunks_exact(dimension) {
                let s = unsafe { dot_fn(qp, doc.as_ptr(), dimension) };
                if s > best { best = s; }
            }
            best
        })
        .sum()
}
```

- [ ] **Step 5: Wire dispatch in `maxsim_flat` and `dot`**

Replace the body of `maxsim_flat`:

```rust
pub fn maxsim_flat(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    if dimension == 0 { return maxsim_flat_scalar(query, document, dimension); }
    #[cfg(target_arch = "aarch64")]
    if dimension.is_multiple_of(16) && dimension <= 4096 {
        return maxsim_flat_neon(query, document, dimension);
    }
    #[cfg(target_arch = "x86_64")]
    match cpu_level() {
        CpuLevel::Avx512Vnni | CpuLevel::Avx512F => {
            return maxsim_flat_avx512(query, document, dimension);
        }
        CpuLevel::Avx2Fma | CpuLevel::Avx2 => {
            return maxsim_flat_avx2_fma(query, document, dimension);
        }
        _ => {}
    }
    maxsim_flat_scalar(query, document, dimension)
}
```

Add x86 path to `dot`:

```rust
pub fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let n = left.len().min(right.len());
    #[cfg(target_arch = "aarch64")]
    {
        let prefix = n & !15;
        if prefix >= 16 {
            let head = unsafe { dot_neon_multiple_of_16(left.as_ptr(), right.as_ptr(), prefix) };
            if prefix == n { return head; }
            return head + dot_scalar(&left[prefix..n], &right[prefix..n]);
        }
    }
    #[cfg(target_arch = "x86_64")]
    match cpu_level() {
        CpuLevel::Avx512Vnni | CpuLevel::Avx512F if n >= 16 => {
            return unsafe { dot_avx512_len(left.as_ptr(), right.as_ptr(), n) };
        }
        CpuLevel::Avx2Fma | CpuLevel::Avx2 if n >= 8 => {
            return unsafe { dot_avx2_fma_len(left.as_ptr(), right.as_ptr(), n) };
        }
        _ => {}
    }
    dot_scalar(&left[..n], &right[..n])
}
```

- [ ] **Step 6: Run existing correctness tests (they now exercise x86 paths)**

```bash
cargo test -p annex-multivector --lib fde 2>&1 | tail -10
```
Expected: all pass (the `maxsim_flat_scalar_and_dispatch_agree` tests now hit AVX2/AVX-512 dispatch on x86).

- [ ] **Step 7: Add short-length edge case tests to `fde.rs`**

Add to the `tests` mod:

```rust
#[test]
fn dot_agrees_scalar_short_lengths() {
    for &len in &[0, 1, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        let left: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
        let right: Vec<f32> = (0..len).map(|i| (len - i) as f32 * 0.1).collect();
        let ref_val = dot_scalar(&left, &right);
        let dispatch_val = dot(&left, &right);
        let tol = 1e-3_f32 * ref_val.abs().max(1.0);
        assert!((dispatch_val - ref_val).abs() <= tol,
            "len={len}: ref={ref_val} dispatch={dispatch_val}");
    }
}
```

Run: `cargo test -p annex-multivector --lib fde::tests::dot_agrees_scalar_short_lengths`

- [ ] **Step 8: Run on EC2**

```bash
rsync -av --exclude='target/' -e "ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no" \
  /Users/rohansharma/Desktop/Code/vectordb/ ec2-user@3.137.213.55:~/vectordb/

ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 \
  "cd ~/vectordb && RUSTFLAGS='-C target-cpu=native' \
   cargo test -p annex-multivector --lib fde -- --nocapture 2>&1 | tail -15"
```
Expected: all fde tests pass.

- [ ] **Step 9: Commit**

```bash
git add crates/annex-multivector/src/fde.rs
git commit -m "feat(simd): add AVX2+FMA and AVX-512 paths for maxsim_flat and dot in fde.rs"
```

---

### Task 6: Criterion benchmarks — `screen_dot` group + dispatch-overhead variants

**Files:**
- Modify: `crates/annex-multivector/benches/kernels.rs`
- Modify: `crates/annex-multivector/Cargo.toml`

**Interfaces:**
- Consumes: `annex::bench_access::{screen_dot, screen_dot_scalar}` (from Task 3)

- [ ] **Step 1: Enable `bench-internals` in `annex-multivector` dev-dependencies**

In `crates/annex-multivector/Cargo.toml`, update the `annex` dev-dependency:

```toml
[dev-dependencies]
annex = { path = "../annex-core", version = "0.2.0", features = ["bench-internals"] }
```

Verify `cargo bench --no-run -p annex-multivector` compiles.

- [ ] **Step 2: Add `bench_screen_dot` group to `kernels.rs`**

Add after `bench_dot`:

```rust
fn bench_screen_dot(c: &mut Criterion) {
    use annex::bench_access::{screen_dot_scalar};
    #[cfg(feature = "bench-internals")]
    use annex::bench_access::screen_dot as screen_dot_dispatch;

    let mut group = c.benchmark_group("screen_dot");

    for &dim in &[128usize, 256, 768] {
        let mut rng = 0xdead_beef_u64;
        let query_i8: Vec<i8> = (0..dim).map(|_| {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as i8).wrapping_add(1)
        }).collect();
        let stored_u8: Vec<u8> = (0..dim).map(|_| {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            rng as u8
        }).collect();

        group.throughput(Throughput::Elements(dim as u64));

        // Scalar reference
        let label = format!("scalar/dim={dim}");
        group.bench_function(&label, |b| {
            b.iter(|| screen_dot_scalar(black_box(&query_i8), black_box(&stored_u8)))
        });

        // Public dispatch (existing behavior)
        let label = format!("dispatch/dim={dim}");
        group.bench_function(&label, |b| {
            b.iter(|| annex::vector::hnsw::HNSWIndex::screen_dot(
                black_box(&query_i8), black_box(&stored_u8)
            ))
        });
    }
    group.finish();
}
```

Update `criterion_group!` and `criterion_main!`:

```rust
criterion_group!(kernels, bench_maxsim_flat, bench_dot, bench_screen_dot);
criterion_main!(kernels);
```

- [ ] **Step 3: Extend `bench_maxsim_flat` with dim=512 and dim=768**

In `bench_maxsim_flat`, change the dimension/token slice to:

```rust
for &(dim, doc_tokens, query_tokens) in &[
    (128usize, 200usize, 32usize),
    (128, 100, 32),
    (384, 200, 32),
    (512, 200, 32),  // instructor-xl shape
    (768, 200, 32),  // E5-large shape
] {
```

- [ ] **Step 4: Compile-check locally**

```bash
cargo bench --no-run -p annex-multivector 2>&1 | tail -10
```
Expected: compiles cleanly.

- [ ] **Step 5: Run benchmarks on EC2 to get baseline numbers**

```bash
rsync -av --exclude='target/' -e "ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no" \
  /Users/rohansharma/Desktop/Code/vectordb/ ec2-user@3.137.213.55:~/vectordb/

ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 \
  "cd ~/vectordb && RUSTFLAGS='-C target-cpu=native' ANNEX_REQUIRE_X86_FEATURES=avx2,fma,avx512f,avx512vnni \
   cargo bench --bench kernels -- --noplot 2>&1" | tee /tmp/ec2-kernel-bench-baseline.txt
```

Record the output. Key things to observe:
- `maxsim_flat/dispatch/dim=128` vs. `maxsim_flat/scalar/dim=128` — expected ≥2× speedup from AVX-512
- `screen_dot/dispatch/dim=256` vs. `screen_dot/scalar/dim=256` — VNNI vs. scalar
- If `dispatch` is slower than `scalar` for any group, investigate before declaring the task done

- [ ] **Step 6: Commit**

```bash
git add crates/annex-multivector/benches/kernels.rs crates/annex-multivector/Cargo.toml
git commit -m "bench: add screen_dot criterion group; extend maxsim_flat dims; enable bench-internals"
```

---

### Task 7: Stage 2 — EC2 HNSW end-to-end validation

> Run this task only after Tasks 1–6 are all green and the EC2 criterion numbers look plausible (no dispatch regression).

**Files:** No code changes. This task validates the kernel work against the existing ANN benchmark harness.

- [ ] **Step 1: Build `master` baseline snapshot**

On the EC2 instance, build the unpatched `master` benchmark binary. Record its latency/recall numbers on NYT-256.

```bash
ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 "
  cd ~/vectordb && git stash
  RUSTFLAGS='-C target-cpu=native' M_VALUES=16 EF_SEARCH_LIST='32,64,128,256' \
    bash scripts/run_annex_benchmark.sh nyt256 \$HOME/data/nyt256 angular 10
  cp crates/annex-core/bench/nyt256/results_annexdb.jsonl /tmp/baseline_nyt256.jsonl
  git stash pop
"
```

- [ ] **Step 2: Sync patched code and run patched benchmark**

```bash
rsync -av --exclude='target/' -e "ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no" \
  /Users/rohansharma/Desktop/Code/vectordb/ ec2-user@3.137.213.55:~/vectordb/

ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 "
  cd ~/vectordb
  RUSTFLAGS='-C target-cpu=native' ANNEX_REQUIRE_X86_FEATURES=avx2,fma,avx512f,avx512vnni \
    M_VALUES=16 EF_SEARCH_LIST='32,64,128,256' \
    bash scripts/run_annex_benchmark.sh nyt256 \$HOME/data/nyt256 angular 10
  cp crates/annex-core/bench/nyt256/results_annexdb.jsonl /tmp/patched_nyt256.jsonl
"
```

- [ ] **Step 3: Compare baseline vs. patched**

```bash
ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 "
  python3 - <<'PY'
import json
base = [json.loads(l) for l in open('/tmp/baseline_nyt256.jsonl')]
patched = [json.loads(l) for l in open('/tmp/patched_nyt256.jsonl')]
for b, p in zip(base, patched):
    label = b['label']
    b_ms = b.get('median_ms', b.get('p50_ms'))
    p_ms = p.get('median_ms', p.get('p50_ms'))
    pct = (p_ms - b_ms) / b_ms * 100
    print(f'{label}: baseline={b_ms:.3f}ms patched={p_ms:.3f}ms change={pct:+.1f}%  recall_base={b[\"recall\"]:.4f} recall_patched={p[\"recall\"]:.4f}')
PY
"
```

Acceptance: no recall regression (≥0.0); latency change within the confidence interval (accept if within ±5% as noise floor on EC2). Do not claim improvement unless change is consistently negative across ef_search values with non-overlapping confidence intervals.

---

### Task 8: Stage 2 — MaxSim validation via headtohead.py

> Run after Task 7 confirms no HNSW regression.

**Files:** No code changes.

- [ ] **Step 1: Confirm BEIR FiQA data is cached on EC2**

```bash
ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 \
  "ls ~/data/beir/fiqa/test/ 2>/dev/null | head -5 || echo 'not cached'"
```

If not cached, run the dataset download/embed step per `benchmark/cache_embeddings.py` docs.

- [ ] **Step 2: Run headtohead baseline (master)**

```bash
ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 "
  cd ~/vectordb && git stash
  MULTIVECTOR_TIMING=1 python3 crates/annex-multivector/benchmark/headtohead.py \
    --dataset beir/fiqa/test \
    --engines annex_exact,annex_hnsw \
    --annex-candidates 250 \
    --output /tmp/hth_baseline.json
  git stash pop
"
```

- [ ] **Step 3: Run headtohead patched**

```bash
ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 "
  cd ~/vectordb
  MULTIVECTOR_TIMING=1 RUSTFLAGS='-C target-cpu=native' \
    ANNEX_REQUIRE_X86_FEATURES=avx2,fma,avx512f,avx512vnni \
    python3 crates/annex-multivector/benchmark/headtohead.py \
      --dataset beir/fiqa/test \
      --engines annex_exact,annex_hnsw \
      --annex-candidates 250 \
      --output /tmp/hth_patched.json
"
```

- [ ] **Step 4: Compare and record results**

```bash
ssh -i ~/.ssh/annex-bench.pem -o StrictHostKeyChecking=no ec2-user@3.137.213.55 "
  python3 -c \"
import json
b = json.load(open('/tmp/hth_baseline.json'))
p = json.load(open('/tmp/hth_patched.json'))
print('MaxSim p50 (ms): baseline=', b.get('maxsim_p50_ms'), 'patched=', p.get('maxsim_p50_ms'))
print('Total  p50 (ms): baseline=', b.get('total_p50_ms'),  'patched=', p.get('total_p50_ms'))
print('NDCG@10:         baseline=', b.get('ndcg_at_10'),    'patched=', p.get('ndcg_at_10'))
\"
"
```

Do not claim improvement if NDCG@10 changes (ranking must be identical or better). Do not claim MaxSim speedup if the CI overlaps zero.

- [ ] **Step 5: Commit benchmark results as artifacts**

```bash
scp -i ~/.ssh/annex-bench.pem ec2-user@3.137.213.55:/tmp/ec2-kernel-bench-baseline.txt \
  crates/annex-multivector/bench/results/2026-10-01-ec2-kernels.txt
git add crates/annex-multivector/bench/results/
git commit -m "bench: record EC2 Ice Lake kernel and end-to-end results"
```
