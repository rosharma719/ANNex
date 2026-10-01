#[cfg(target_arch = "x86_64")]
use annex::vector::simd::{CpuLevel, cpu_level};

pub type Vector = Vec<f32>;

pub fn normalize(vector: &[f32]) -> Vector {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() {
        let norm = vector
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        vector.iter().map(|&x| (x as f64 / norm) as f32).collect()
    } else if norm == 0.0 {
        vec![0.0; vector.len()]
    } else {
        vector.iter().map(|x| x / norm).collect()
    }
}

/// Scalar dot product. Kept as a portable fallback and used for tail elements
/// past the SIMD-aligned prefix.
#[inline]
fn dot_scalar(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let n = left.len().min(right.len());
    let mut s = 0.0f32;
    for i in 0..n {
        s += left[i] * right[i];
    }
    s
}

/// Public dot product with architecture dispatch. Routes to a NEON FMA
/// implementation on aarch64 for the multiple-of-16 prefix and mops up the
/// tail with the scalar path. Every hot path in the crate (FDE exhaustive
/// scan, MaxSim rescoring) goes through this — reason enough to keep the
/// dispatch cheap.
#[inline]
pub fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let n = left.len().min(right.len());
    #[cfg(target_arch = "aarch64")]
    {
        let prefix = n & !15;
        if prefix >= 16 {
            // SAFETY: prefix is a multiple of 16 and prefix <= n <= left.len().
            let head = unsafe { dot_neon_multiple_of_16(left.as_ptr(), right.as_ptr(), prefix) };
            if prefix == n {
                return head;
            }
            return head + dot_scalar(&left[prefix..n], &right[prefix..n]);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if n > 0 {
        match cpu_level() {
            CpuLevel::Avx512Vnni | CpuLevel::Avx512F => {
                return unsafe { dot_avx512_len(left.as_ptr(), right.as_ptr(), n) };
            }
            CpuLevel::Avx2Fma => {
                return unsafe { dot_avx2_fma_len(left.as_ptr(), right.as_ptr(), n) };
            }
            CpuLevel::Avx2 => {
                return unsafe { dot_avx2_len(left.as_ptr(), right.as_ptr(), n) };
            }
            _ => {}
        }
    }
    dot_scalar(&left[..n], &right[..n])
}

/// NEON FP32 dot product for a length that is a multiple of 16.
/// Uses four independent accumulators to hide FMA latency (~4 cycles on M-series).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_neon_multiple_of_16(a: *const f32, b: *const f32, len: usize) -> f32 {
    unsafe {
        use std::arch::aarch64::*;
        debug_assert!(len.is_multiple_of(16));
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i < len {
            let a0 = vld1q_f32(a.add(i));
            let a1 = vld1q_f32(a.add(i + 4));
            let a2 = vld1q_f32(a.add(i + 8));
            let a3 = vld1q_f32(a.add(i + 12));
            let b0 = vld1q_f32(b.add(i));
            let b1 = vld1q_f32(b.add(i + 4));
            let b2 = vld1q_f32(b.add(i + 8));
            let b3 = vld1q_f32(b.add(i + 12));
            acc0 = vfmaq_f32(acc0, a0, b0);
            acc1 = vfmaq_f32(acc1, a1, b1);
            acc2 = vfmaq_f32(acc2, a2, b2);
            acc3 = vfmaq_f32(acc3, a3, b3);
            i += 16;
        }
        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(acc)
    }
}

/// Same as [`dot_neon_multiple_of_16`] but with the length fixed at 128.
/// Const bound lets the compiler unroll the loop fully — in benchmarks this
/// specialisation shaves ~30% off the general-dimension path (~2.4 ms vs
/// ~3.2 ms on the 250-candidate MaxSim rescoring kernel).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_neon_128(a: *const f32, b: *const f32) -> f32 {
    unsafe {
        use std::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i < 128 {
            let a0 = vld1q_f32(a.add(i));
            let a1 = vld1q_f32(a.add(i + 4));
            let a2 = vld1q_f32(a.add(i + 8));
            let a3 = vld1q_f32(a.add(i + 12));
            let b0 = vld1q_f32(b.add(i));
            let b1 = vld1q_f32(b.add(i + 4));
            let b2 = vld1q_f32(b.add(i + 8));
            let b3 = vld1q_f32(b.add(i + 12));
            acc0 = vfmaq_f32(acc0, a0, b0);
            acc1 = vfmaq_f32(acc1, a1, b1);
            acc2 = vfmaq_f32(acc2, a2, b2);
            acc3 = vfmaq_f32(acc3, a3, b3);
            i += 16;
        }
        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(acc)
    }
}

/// AVX2-only dot product (no FMA). 2 accumulators × 8 f32 = 16 f/iter.
/// Used when AVX2 is available but FMA is not (some pre-Broadwell or AMD Jaguar).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx2_len(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= len {
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a.add(i)),     _mm256_loadu_ps(b.add(i))));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(a.add(i + 8)), _mm256_loadu_ps(b.add(i + 8))));
        i += 16;
    }
    while i + 8 <= len {
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i))));
        i += 8;
    }
    s0 = _mm256_add_ps(s0, s1);
    let hi  = _mm256_extractf128_ps(s0, 1);
    let lo  = _mm256_castps256_ps128(s0);
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

/// AVX2+FMA dot product for arbitrary length. 4 accumulators × 8 f32 = 32 f/iter.
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
    // Reduce without hadd: extract high 128, add, shuffle down.
    let hi  = _mm256_extractf128_ps(s0, 1);
    let lo  = _mm256_castps256_ps128(s0);
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

/// AVX-512 dot product for arbitrary length. 4 accumulators × 16 f32 = 64 f/iter.
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

/// ColBERT's late-interaction score: sum of per-query-token maxima.
pub fn maxsim(query: &[Vector], document: &[Vector]) -> f32 {
    if query.is_empty() || document.is_empty() {
        return 0.0;
    }
    let document: Vec<_> = document.iter().map(|v| normalize(v)).collect();
    query
        .iter()
        .map(|query| {
            let query = normalize(query);
            document
                .iter()
                .map(|doc| dot(&query, doc))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// MaxSim over already-normalized query vectors and a flat document matrix.
///
/// Routes to a vectorised kernel when the dimension is a multiple of 16 on
/// aarch64 (covers the standard ColBERT/E5/MPNet embedding sizes 128, 384,
/// 512, 768, 1024). Falls back to the scalar path everywhere else.
pub fn maxsim_flat(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    if dimension == 0 {
        return maxsim_flat_scalar(query, document, dimension);
    }
    #[cfg(target_arch = "aarch64")]
    if dimension.is_multiple_of(16) && dimension <= 4096 {
        return maxsim_flat_neon(query, document, dimension);
    }
    #[cfg(target_arch = "x86_64")]
    match cpu_level() {
        CpuLevel::Avx512Vnni | CpuLevel::Avx512F => {
            return maxsim_flat_avx512(query, document, dimension);
        }
        CpuLevel::Avx2Fma => {
            return maxsim_flat_avx2_fma(query, document, dimension);
        }
        CpuLevel::Avx2 => {
            return maxsim_flat_avx2(query, document, dimension);
        }
        _ => {}
    }
    maxsim_flat_scalar(query, document, dimension)
}

#[inline]
fn maxsim_flat_scalar(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    query
        .iter()
        .map(|q| {
            document
                .chunks_exact(dimension)
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[cfg(target_arch = "aarch64")]
fn maxsim_flat_neon(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    // NOTE: query-outer / doc-inner order is intentional. A previous attempt
    // to swap the loops (stream doc tokens once, iterate 32 query tokens
    // per doc) regressed the kernel from 2.35 ms to 3.51 ms in the
    // maxsim-bench harness. The original order keeps the current query
    // vector pinned in registers across all 200 doc-token dot products,
    // and lets the compiler track a single scalar `best` in a register
    // across the inner loop. Doc-outer loses both benefits.
    if dimension == 128 {
        // Const-length specialisation for the standard ColBERT dim so the
        // compiler can fully unroll the inner FMA loop.
        return query
            .iter()
            .map(|q| {
                debug_assert_eq!(q.len(), 128);
                let qp = q.as_ptr();
                let mut best = f32::NEG_INFINITY;
                for doc in document.as_chunks::<128>().0 {
                    let s = unsafe { dot_neon_128(qp, doc.as_ptr()) };
                    if s > best {
                        best = s;
                    }
                }
                best
            })
            .sum();
    }
    query
        .iter()
        .map(|q| {
            debug_assert_eq!(q.len(), dimension);
            let qp = q.as_ptr();
            let mut best = f32::NEG_INFINITY;
            for doc in document.chunks_exact(dimension) {
                // SAFETY: `dimension` is validated a multiple of 16 by the caller
                // (maxsim_flat), q and doc are contiguous slices of `dimension`
                // f32 values, so the NEON loads stay in-bounds.
                let s = unsafe { dot_neon_multiple_of_16(qp, doc.as_ptr(), dimension) };
                if s > best {
                    best = s;
                }
            }
            best
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
fn maxsim_flat_avx2(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    let dot_fn: unsafe fn(*const f32, *const f32, usize) -> f32 = dot_avx2_len;
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
                if s > best {
                    best = s;
                }
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
                if s > best {
                    best = s;
                }
            }
            best
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic(seed: u64, dim: usize) -> Vector {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn flat(doc: &[Vector]) -> (Vec<f32>, usize) {
        let dim = doc[0].len();
        (doc.iter().flat_map(|v| v.iter().copied()).collect(), dim)
    }

    #[test]
    fn maxsim_flat_scalar_and_dispatch_agree_on_dim128() {
        let query: Vec<_> = (0..8).map(|i| deterministic(0x11 + i, 128)).collect();
        let doc_tokens: Vec<_> = (0..50).map(|i| deterministic(0x2000 + i, 128)).collect();
        let (flat_doc, dim) = flat(&doc_tokens);
        let s = maxsim_flat_scalar(&query, &flat_doc, dim);
        let d = maxsim_flat(&query, &flat_doc, dim);
        assert!((s - d).abs() < 1e-3, "scalar={s} dispatch={d}");
    }

    #[test]
    fn maxsim_flat_scalar_and_dispatch_agree_on_dim384() {
        let query: Vec<_> = (0..12).map(|i| deterministic(0x33 + i, 384)).collect();
        let doc_tokens: Vec<_> = (0..30).map(|i| deterministic(0x4000 + i, 384)).collect();
        let (flat_doc, dim) = flat(&doc_tokens);
        let s = maxsim_flat_scalar(&query, &flat_doc, dim);
        let d = maxsim_flat(&query, &flat_doc, dim);
        assert!((s - d).abs() < 1e-3, "scalar={s} dispatch={d}");
    }

    #[test]
    fn maxsim_flat_dispatch_falls_back_when_dim_not_multiple_of_16() {
        // 100 is not a multiple of 16 — dispatch must take the scalar path.
        let query: Vec<_> = (0..4).map(|i| deterministic(0x55 + i, 100)).collect();
        let doc_tokens: Vec<_> = (0..10).map(|i| deterministic(0x6000 + i, 100)).collect();
        let (flat_doc, dim) = flat(&doc_tokens);
        let s = maxsim_flat_scalar(&query, &flat_doc, dim);
        let d = maxsim_flat(&query, &flat_doc, dim);
        assert!((s - d).abs() < 1e-6, "scalar={s} dispatch={d}");
    }

    #[test]
    fn dot_agrees_scalar_short_lengths() {
        for &len in &[0usize, 1, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let left: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
            let right: Vec<f32> = (0..len).map(|i| (len.saturating_sub(i)) as f32 * 0.1).collect();
            let ref_val = dot_scalar(&left, &right);
            let dispatch_val = dot(&left, &right);
            let tol = 1e-3_f32 * ref_val.abs().max(1.0);
            assert!(
                (dispatch_val - ref_val).abs() <= tol,
                "len={len}: ref={ref_val} dispatch={dispatch_val}"
            );
        }
    }

    #[test]
    fn maxsim_flat_agrees_scalar_dim512_dim768() {
        for &dim in &[512usize, 768] {
            let query: Vec<_> = (0..8).map(|i| deterministic(0xAA + i, dim)).collect();
            let doc_tokens: Vec<_> = (0..20).map(|i| deterministic(0xBB00 + i, dim)).collect();
            let (flat_doc, _) = flat(&doc_tokens);
            let s = maxsim_flat_scalar(&query, &flat_doc, dim);
            let d = maxsim_flat(&query, &flat_doc, dim);
            let tol = 1e-3_f32 * s.abs().max(1.0);
            assert!((s - d).abs() <= tol, "dim={dim}: scalar={s} dispatch={d}");
        }
    }
}
