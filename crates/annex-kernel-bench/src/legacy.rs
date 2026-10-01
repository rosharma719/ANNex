//! Verbatim copy of the x86_64 kernels as of commit 66231da (before the
//! shared `annex::vector::kernels` module), kept as the "legacy" baseline so
//! before/after numbers come from one harness. Only the visibility and cfg
//! attributes were changed.
#![allow(unsafe_op_in_unsafe_fn, dead_code, clippy::all)]

/// Legacy SQ8 dispatch (`HNSWIndex::screen_dot` on x86_64).
#[inline]
pub fn screen_dot(query_i8: &[i8], stored: &[u8]) -> i32 {
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { screen_dot_avx2(query_i8, stored) };
    }
    screen_dot_scalar(query_i8, stored)
}

/// Legacy multivector dot (scalar on x86_64) and MaxSim.
#[inline]
pub fn fde_dot(left: &[f32], right: &[f32]) -> f32 {
    let n = left.len().min(right.len());
    let mut s = 0.0f32;
    for i in 0..n {
        s += left[i] * right[i];
    }
    s
}

pub fn maxsim_flat(query: &[Vec<f32>], document: &[f32], dimension: usize) -> f32 {
    query
        .iter()
        .map(|q| {
            document
                .chunks_exact(dimension)
                .map(|d| fde_dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[inline]
pub fn dot_product(query: &[f32], vec: &[f32]) -> f32 {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe { dot_avx2_fma(query, vec) }
    } else if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { dot_avx2(query, vec) }
    } else {
        dot_scalar(query, vec)
    }
}

#[inline]
pub fn l2_squared(query: &[f32], vec: &[f32]) -> f32 {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe { l2_avx2_fma(query, vec) }
    } else if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { l2_avx2(query, vec) }
    } else {
        l2_scalar(query, vec)
    }
}

#[allow(dead_code)]
#[inline]
fn dot_scalar(query: &[f32], vec: &[f32]) -> f32 {
    query.iter().zip(vec.iter()).map(|(x, y)| x * y).sum()
}

#[allow(dead_code)]
#[inline]
fn l2_scalar(query: &[f32], vec: &[f32]) -> f32 {
    query
        .iter()
        .zip(vec.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
}

#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn dot_avx2(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(q, v));
        i += 8;
    }
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        acc += query.get_unchecked(i) * vec.get_unchecked(i);
        i += 1;
    }
    acc
}

#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx2_fma(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // 4 accumulators × 8 floats/register = 32 floats/iteration.
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 32 <= len {
        let q0 = _mm256_loadu_ps(query.as_ptr().add(i));
        let q1 = _mm256_loadu_ps(query.as_ptr().add(i + 8));
        let q2 = _mm256_loadu_ps(query.as_ptr().add(i + 16));
        let q3 = _mm256_loadu_ps(query.as_ptr().add(i + 24));
        let v0 = _mm256_loadu_ps(vec.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(vec.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(vec.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(vec.as_ptr().add(i + 24));
        s0 = _mm256_fmadd_ps(q0, v0, s0);
        s1 = _mm256_fmadd_ps(q1, v1, s1);
        s2 = _mm256_fmadd_ps(q2, v2, s2);
        s3 = _mm256_fmadd_ps(q3, v3, s3);
        i += 32;
    }
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        s0 = _mm256_fmadd_ps(q, v, s0);
        i += 8;
    }
    s0 = _mm256_add_ps(s0, s1);
    s2 = _mm256_add_ps(s2, s3);
    s0 = _mm256_add_ps(s0, s2);
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        acc += query.get_unchecked(i) * vec.get_unchecked(i);
        i += 1;
    }
    acc
}

#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn l2_avx2(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        let diff = _mm256_sub_ps(q, v);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        i += 8;
    }
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        let diff = query.get_unchecked(i) - vec.get_unchecked(i);
        acc += diff * diff;
        i += 1;
    }
    acc
}

#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn l2_avx2_fma(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 32 <= len {
        let q0 = _mm256_loadu_ps(query.as_ptr().add(i));
        let q1 = _mm256_loadu_ps(query.as_ptr().add(i + 8));
        let q2 = _mm256_loadu_ps(query.as_ptr().add(i + 16));
        let q3 = _mm256_loadu_ps(query.as_ptr().add(i + 24));
        let v0 = _mm256_loadu_ps(vec.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(vec.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(vec.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(vec.as_ptr().add(i + 24));
        let d0 = _mm256_sub_ps(q0, v0);
        let d1 = _mm256_sub_ps(q1, v1);
        let d2 = _mm256_sub_ps(q2, v2);
        let d3 = _mm256_sub_ps(q3, v3);
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
        i += 32;
    }
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        let d = _mm256_sub_ps(q, v);
        s0 = _mm256_fmadd_ps(d, d, s0);
        i += 8;
    }
    s0 = _mm256_add_ps(s0, s1);
    s2 = _mm256_add_ps(s2, s3);
    s0 = _mm256_add_ps(s0, s2);
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        let diff = query.get_unchecked(i) - vec.get_unchecked(i);
        acc += diff * diff;
        i += 1;
    }
    acc
}

pub fn screen_dot_scalar(query_i8: &[i8], stored: &[u8]) -> i32 {
    let n = query_i8.len().min(stored.len());
    let mut a = [0i32; 4];
    let mut i = 0;
    while i + 4 <= n {
        a[0] += (query_i8[i] as i32) * (stored[i] as i32 - 128);
        a[1] += (query_i8[i + 1] as i32) * (stored[i + 1] as i32 - 128);
        a[2] += (query_i8[i + 2] as i32) * (stored[i + 2] as i32 - 128);
        a[3] += (query_i8[i + 3] as i32) * (stored[i + 3] as i32 - 128);
        i += 4;
    }
    let mut acc = a[0] + a[1] + a[2] + a[3];
    while i < n {
        acc += (query_i8[i] as i32) * (stored[i] as i32 - 128);
        i += 1;
    }
    acc
}

/// NEON sdot: vdotq_s32 processes 4 groups of 4 i8 products per instruction.
/// For 256-dim: 256/16 = 16 sdot calls with 4 accumulators = 4 iterations of 64 values.

#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn screen_dot_avx2(query_i8: &[i8], stored: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    let n = query_i8.len().min(stored.len());
    let ones = _mm256_set1_epi16(1);
    let bias = _mm256_set1_epi16(128);
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 32 <= n {
        let s = _mm256_loadu_si256(stored.as_ptr().add(i) as *const __m256i);
        let q_raw = _mm256_loadu_si256(query_i8.as_ptr().add(i) as *const __m256i);
        let s_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s));
        let s_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s, 1));
        let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q_raw));
        let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q_raw, 1));
        let s_lo_c = _mm256_sub_epi16(s_lo, bias);
        let s_hi_c = _mm256_sub_epi16(s_hi, bias);
        let prod_lo = _mm256_madd_epi16(_mm256_mullo_epi16(s_lo_c, q_lo), ones);
        let prod_hi = _mm256_madd_epi16(_mm256_mullo_epi16(s_hi_c, q_hi), ones);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(prod_lo, prod_hi));
        i += 32;
    }
    // Reduce acc (8 × i32) to scalar
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256(acc, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b_01_00_11_10));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 1));
    let mut result = _mm_cvtsi128_si32(sum32);
    while i < n {
        result += (*query_i8.get_unchecked(i) as i32) * (*stored.get_unchecked(i) as i32 - 128);
        i += 1;
    }
    result
}
