//! Distance kernels shared by the HNSW index and the multivector crate.
//!
//! Every public function has a scalar reference (`*_scalar`) and one or more
//! architecture-specific implementations:
//!
//! | ISA | f32 dot / L2 | SQ8 `i8 x (u8 - 128)` |
//! | --- | --- | --- |
//! | AVX-512 (F+BW) | 16 lanes, 4 accumulators, masked tail | `vpmaddwd` on widened bytes |
//! | AVX-512 VNNI | as above | `vpdpbusd` |
//! | AVX2 + FMA | 8 lanes, 4 accumulators, masked tail | `vpmaddwd` on widened bytes |
//! | NEON | 4 lanes, 4 accumulators | `sdot` when `dotprod` is present |
//!
//! On x86_64 the best supported ISA is chosen once, on first use, and cached in
//! per-function pointers, so each call costs one relaxed load plus an indirect
//! call. When the crate is compiled with the ISA already enabled (for example
//! `-C target-cpu=native` on an AVX-512 host), calls go straight to the kernel
//! and can be inlined. All kernels operate on `min(a.len(), b.len())` elements.
#![allow(unsafe_op_in_unsafe_fn)]

use std::fmt;

/// Instruction set a [`Kernels`] table dispatches to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Isa {
    Scalar,
    /// AVX2 with FMA.
    Avx2,
    /// AVX-512 F and BW.
    Avx512,
    /// AVX-512 F, BW and VNNI.
    Avx512Vnni,
    Neon,
}

impl Isa {
    pub const ALL: [Isa; 5] = [
        Isa::Scalar,
        Isa::Avx2,
        Isa::Avx512,
        Isa::Avx512Vnni,
        Isa::Neon,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2+fma",
            Isa::Avx512 => "avx512",
            Isa::Avx512Vnni => "avx512+vnni",
            Isa::Neon => "neon",
        }
    }

    /// Whether the running CPU can execute kernels for this ISA.
    pub fn is_supported(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => {
                Isa::Avx512.is_supported() && std::arch::is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

type F32Fn = unsafe fn(&[f32], &[f32]) -> f32;
type Sq8Fn = unsafe fn(&[i8], &[u8]) -> i32;

/// A table of kernels for one ISA. Obtain one with [`Kernels::detect`] or
/// [`Kernels::for_isa`]; construction checks CPU support, so calls are safe.
#[derive(Clone, Copy)]
pub struct Kernels {
    isa: Isa,
    dot: F32Fn,
    l2_squared: F32Fn,
    dot_i8_u8_centered: Sq8Fn,
}

impl fmt::Debug for Kernels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernels").field("isa", &self.isa).finish()
    }
}

impl Kernels {
    /// The fastest kernels the running CPU supports.
    pub fn detect() -> Self {
        [Isa::Avx512Vnni, Isa::Avx512, Isa::Avx2, Isa::Neon]
            .into_iter()
            .find_map(Kernels::for_isa)
            .unwrap_or_else(|| Kernels::for_isa(Isa::Scalar).unwrap())
    }

    /// Kernels for `isa`, or `None` when the CPU (or target) lacks it.
    pub fn for_isa(isa: Isa) -> Option<Self> {
        if !isa.is_supported() {
            return None;
        }
        let scalar = Kernels {
            isa: Isa::Scalar,
            dot: dot_scalar_unsafe,
            l2_squared: l2_squared_scalar_unsafe,
            dot_i8_u8_centered: dot_i8_u8_centered_scalar_unsafe,
        };
        Some(match isa {
            Isa::Scalar => scalar,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => Kernels {
                isa,
                dot: x86::dot_avx2,
                l2_squared: x86::l2_avx2,
                dot_i8_u8_centered: x86::sq8_avx2,
            },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => Kernels {
                isa,
                dot: x86::dot_avx512,
                l2_squared: x86::l2_avx512,
                dot_i8_u8_centered: x86::sq8_avx512bw,
            },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => Kernels {
                isa,
                dot: x86::dot_avx512,
                l2_squared: x86::l2_avx512,
                dot_i8_u8_centered: x86::sq8_avx512vnni,
            },
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => Kernels {
                isa,
                dot: neon::dot,
                l2_squared: neon::l2,
                dot_i8_u8_centered: neon::sq8,
            },
            #[allow(unreachable_patterns)]
            _ => return None,
        })
    }

    /// Every kernel table the running CPU supports, scalar first.
    pub fn all_supported() -> Vec<Self> {
        Isa::ALL.into_iter().filter_map(Kernels::for_isa).collect()
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    #[inline]
    pub fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: construction verified CPU support for `self.isa`.
        unsafe { (self.dot)(a, b) }
    }

    #[inline]
    pub fn l2_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: as above.
        unsafe { (self.l2_squared)(a, b) }
    }

    #[inline]
    pub fn dot_i8_u8_centered(&self, query: &[i8], stored: &[u8]) -> i32 {
        // SAFETY: as above.
        unsafe { (self.dot_i8_u8_centered)(query, stored) }
    }
}

/// The ISA used by [`dot`], [`l2_squared`] and [`dot_i8_u8_centered`].
pub fn selected_isa() -> Isa {
    Kernels::detect().isa
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Inner product of `a` and `b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // SAFETY: the crate was compiled for a CPU with AVX-512F.
        unsafe { x86::dot_avx512(a, b) }
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    {
        x86::dispatch::dot(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64.
        unsafe { neon::dot(a, b) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_scalar(a, b)
    }
}

/// Squared Euclidean distance between `a` and `b`.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // SAFETY: the crate was compiled for a CPU with AVX-512F.
        unsafe { x86::l2_avx512(a, b) }
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    {
        x86::dispatch::l2_squared(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64.
        unsafe { neon::l2(a, b) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        l2_squared_scalar(a, b)
    }
}

/// SQ8 screening product `sum(query[i] * (stored[i] - 128))`, exact in i32.
#[inline]
pub fn dot_i8_u8_centered(query: &[i8], stored: &[u8]) -> i32 {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    ))]
    {
        // SAFETY: the crate was compiled for a CPU with AVX-512BW and VNNI.
        unsafe { x86::sq8_avx512vnni(query, stored) }
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx512bw", target_feature = "avx512vnni"))
    ))]
    {
        x86::dispatch::dot_i8_u8_centered(query, stored)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `neon::sq8` checks for `dotprod` before using it.
        unsafe { neon::sq8(query, stored) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_i8_u8_centered_scalar(query, stored)
    }
}

// ---------------------------------------------------------------------------
// Scalar references
// ---------------------------------------------------------------------------

#[inline]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

pub fn dot_i8_u8_centered_scalar(query: &[i8], stored: &[u8]) -> i32 {
    let n = query.len().min(stored.len());
    let mut a = [0i32; 4];
    let mut i = 0;
    while i + 4 <= n {
        for (lane, acc) in a.iter_mut().enumerate() {
            *acc += (query[i + lane] as i32) * (stored[i + lane] as i32 - 128);
        }
        i += 4;
    }
    let mut acc = a[0] + a[1] + a[2] + a[3];
    while i < n {
        acc += (query[i] as i32) * (stored[i] as i32 - 128);
        i += 1;
    }
    acc
}

unsafe fn dot_scalar_unsafe(a: &[f32], b: &[f32]) -> f32 {
    dot_scalar(a, b)
}

unsafe fn l2_squared_scalar_unsafe(a: &[f32], b: &[f32]) -> f32 {
    l2_squared_scalar(a, b)
}

unsafe fn dot_i8_u8_centered_scalar_unsafe(query: &[i8], stored: &[u8]) -> i32 {
    dot_i8_u8_centered_scalar(query, stored)
}

// ---------------------------------------------------------------------------
// x86_64
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Lazily resolved function pointers for builds without AVX-512 enabled
    /// at compile time. The first call of each function detects the CPU and
    /// replaces its pointer.
    #[cfg_attr(target_feature = "avx512f", allow(dead_code))]
    pub(super) mod dispatch {
        use super::super::{F32Fn, Kernels, Sq8Fn};
        use std::sync::atomic::{AtomicPtr, Ordering};

        static DOT: AtomicPtr<()> = AtomicPtr::new(resolve_dot as *mut ());
        static L2: AtomicPtr<()> = AtomicPtr::new(resolve_l2 as *mut ());
        static SQ8: AtomicPtr<()> = AtomicPtr::new(resolve_sq8 as *mut ());

        unsafe fn resolve_dot(a: &[f32], b: &[f32]) -> f32 {
            let f = Kernels::detect().dot;
            DOT.store(f as *mut (), Ordering::Relaxed);
            f(a, b)
        }

        unsafe fn resolve_l2(a: &[f32], b: &[f32]) -> f32 {
            let f = Kernels::detect().l2_squared;
            L2.store(f as *mut (), Ordering::Relaxed);
            f(a, b)
        }

        unsafe fn resolve_sq8(q: &[i8], s: &[u8]) -> i32 {
            let f = Kernels::detect().dot_i8_u8_centered;
            SQ8.store(f as *mut (), Ordering::Relaxed);
            f(q, s)
        }

        #[inline]
        pub fn dot(a: &[f32], b: &[f32]) -> f32 {
            // SAFETY: DOT only ever holds an `F32Fn` valid on this CPU.
            unsafe {
                let f: F32Fn = std::mem::transmute(DOT.load(Ordering::Relaxed));
                f(a, b)
            }
        }

        #[inline]
        pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
            // SAFETY: as above.
            unsafe {
                let f: F32Fn = std::mem::transmute(L2.load(Ordering::Relaxed));
                f(a, b)
            }
        }

        #[inline]
        pub fn dot_i8_u8_centered(q: &[i8], s: &[u8]) -> i32 {
            // SAFETY: as above.
            unsafe {
                let f: Sq8Fn = std::mem::transmute(SQ8.load(Ordering::Relaxed));
                f(q, s)
            }
        }
    }

    // ----- AVX-512 ---------------------------------------------------------

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn tail_mask16(rem: usize) -> __mmask16 {
        ((1u32 << rem) - 1) as __mmask16
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut s0 = _mm512_setzero_ps();
        let mut s1 = _mm512_setzero_ps();
        let mut s2 = _mm512_setzero_ps();
        let mut s3 = _mm512_setzero_ps();
        let mut i = 0;
        while i + 64 <= n {
            s0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), s0);
            s1 = _mm512_fmadd_ps(
                _mm512_loadu_ps(pa.add(i + 16)),
                _mm512_loadu_ps(pb.add(i + 16)),
                s1,
            );
            s2 = _mm512_fmadd_ps(
                _mm512_loadu_ps(pa.add(i + 32)),
                _mm512_loadu_ps(pb.add(i + 32)),
                s2,
            );
            s3 = _mm512_fmadd_ps(
                _mm512_loadu_ps(pa.add(i + 48)),
                _mm512_loadu_ps(pb.add(i + 48)),
                s3,
            );
            i += 64;
        }
        while i + 16 <= n {
            s0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), s0);
            i += 16;
        }
        if i < n {
            // Masked-off lanes are neither read nor faulted on.
            let m = tail_mask16(n - i);
            s1 = _mm512_fmadd_ps(
                _mm512_maskz_loadu_ps(m, pa.add(i)),
                _mm512_maskz_loadu_ps(m, pb.add(i)),
                s1,
            );
        }
        _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn l2_avx512(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut s0 = _mm512_setzero_ps();
        let mut s1 = _mm512_setzero_ps();
        let mut s2 = _mm512_setzero_ps();
        let mut s3 = _mm512_setzero_ps();
        let mut i = 0;
        while i + 64 <= n {
            let d0 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
            let d1 = _mm512_sub_ps(
                _mm512_loadu_ps(pa.add(i + 16)),
                _mm512_loadu_ps(pb.add(i + 16)),
            );
            let d2 = _mm512_sub_ps(
                _mm512_loadu_ps(pa.add(i + 32)),
                _mm512_loadu_ps(pb.add(i + 32)),
            );
            let d3 = _mm512_sub_ps(
                _mm512_loadu_ps(pa.add(i + 48)),
                _mm512_loadu_ps(pb.add(i + 48)),
            );
            s0 = _mm512_fmadd_ps(d0, d0, s0);
            s1 = _mm512_fmadd_ps(d1, d1, s1);
            s2 = _mm512_fmadd_ps(d2, d2, s2);
            s3 = _mm512_fmadd_ps(d3, d3, s3);
            i += 64;
        }
        while i + 16 <= n {
            let d = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
            s0 = _mm512_fmadd_ps(d, d, s0);
            i += 16;
        }
        if i < n {
            let m = tail_mask16(n - i);
            let d = _mm512_sub_ps(
                _mm512_maskz_loadu_ps(m, pa.add(i)),
                _mm512_maskz_loadu_ps(m, pb.add(i)),
            );
            s1 = _mm512_fmadd_ps(d, d, s1);
        }
        _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)))
    }

    /// `sum(q * (s - 128))` via `vpdpbusd` (u8 x i8): computes `sum(s * q)`
    /// and `sum(128 * q)` in parallel and subtracts. Exact in i32.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(super) unsafe fn sq8_avx512vnni(query: &[i8], stored: &[u8]) -> i32 {
        let n = query.len().min(stored.len());
        let (pq, ps) = (query.as_ptr(), stored.as_ptr());
        let bias = _mm512_set1_epi8(-128); // 0x80 == 128 as u8
        let mut dot0 = _mm512_setzero_si512();
        let mut dot1 = _mm512_setzero_si512();
        let mut off0 = _mm512_setzero_si512();
        let mut off1 = _mm512_setzero_si512();
        let mut i = 0;
        while i + 128 <= n {
            let q0 = _mm512_loadu_si512(pq.add(i) as *const _);
            let q1 = _mm512_loadu_si512(pq.add(i + 64) as *const _);
            let s0 = _mm512_loadu_si512(ps.add(i) as *const _);
            let s1 = _mm512_loadu_si512(ps.add(i + 64) as *const _);
            dot0 = _mm512_dpbusd_epi32(dot0, s0, q0);
            dot1 = _mm512_dpbusd_epi32(dot1, s1, q1);
            off0 = _mm512_dpbusd_epi32(off0, bias, q0);
            off1 = _mm512_dpbusd_epi32(off1, bias, q1);
            i += 128;
        }
        while i < n {
            let rem = n - i;
            let m: __mmask64 = if rem >= 64 {
                u64::MAX
            } else {
                (1u64 << rem) - 1
            };
            let q = _mm512_maskz_loadu_epi8(m, pq.add(i));
            let s = _mm512_maskz_loadu_epi8(m, ps.add(i) as *const i8);
            dot0 = _mm512_dpbusd_epi32(dot0, s, q);
            off0 = _mm512_dpbusd_epi32(off0, bias, q);
            i += 64;
        }
        let dot = _mm512_add_epi32(dot0, dot1);
        let off = _mm512_add_epi32(off0, off1);
        _mm512_reduce_add_epi32(_mm512_sub_epi32(dot, off))
    }

    /// AVX-512BW without VNNI: widen to i16 and use `vpmaddwd`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn sq8_avx512bw(query: &[i8], stored: &[u8]) -> i32 {
        let n = query.len().min(stored.len());
        let (pq, ps) = (query.as_ptr(), stored.as_ptr());
        let bias = _mm512_set1_epi16(128);
        let mut acc0 = _mm512_setzero_si512();
        let mut acc1 = _mm512_setzero_si512();
        let mut i = 0;
        while i < n {
            let rem = n - i;
            let m: __mmask64 = if rem >= 64 {
                u64::MAX
            } else {
                (1u64 << rem) - 1
            };
            let q = _mm512_maskz_loadu_epi8(m, pq.add(i));
            // Masked-off stored bytes load as 0 and become -128 after
            // centering, but their query bytes are 0, so they add nothing.
            let s = _mm512_maskz_loadu_epi8(m, ps.add(i) as *const i8);
            let q_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(q));
            let q_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(q, 1));
            let s_lo = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(s)), bias);
            let s_hi =
                _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(s, 1)), bias);
            acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(q_lo, s_lo));
            acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(q_hi, s_hi));
            i += 64;
        }
        _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1))
    }

    // ----- AVX2 + FMA ------------------------------------------------------

    /// Eight -1 lanes followed by eight 0 lanes; loading 8 lanes starting at
    /// `8 - rem` yields a mask enabling the first `rem` lanes.
    static TAIL_MASK: [i32; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn tail_mask8(rem: usize) -> __m256i {
        _mm256_loadu_si256(TAIL_MASK.as_ptr().add(8 - rem) as *const __m256i)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                s1,
            );
            s2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
                s2,
            );
            s3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
                s3,
            );
            i += 32;
        }
        while i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            i += 8;
        }
        if i < n {
            let m = tail_mask8(n - i);
            s1 = _mm256_fmadd_ps(
                _mm256_maskload_ps(pa.add(i), m),
                _mm256_maskload_ps(pb.add(i), m),
                s1,
            );
        }
        hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn l2_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
            );
            let d2 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
            );
            let d3 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
            );
            s0 = _mm256_fmadd_ps(d0, d0, s0);
            s1 = _mm256_fmadd_ps(d1, d1, s1);
            s2 = _mm256_fmadd_ps(d2, d2, s2);
            s3 = _mm256_fmadd_ps(d3, d3, s3);
            i += 32;
        }
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            s0 = _mm256_fmadd_ps(d, d, s0);
            i += 8;
        }
        if i < n {
            let m = tail_mask8(n - i);
            let d = _mm256_sub_ps(
                _mm256_maskload_ps(pa.add(i), m),
                _mm256_maskload_ps(pb.add(i), m),
            );
            s1 = _mm256_fmadd_ps(d, d, s1);
        }
        hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)))
    }

    /// Widen 16 bytes of each input to i16, center the stored codes, and
    /// accumulate with `vpmaddwd` (pairwise i16 products summed into i32).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sq8_avx2(query: &[i8], stored: &[u8]) -> i32 {
        let n = query.len().min(stored.len());
        let (pq, ps) = (query.as_ptr(), stored.as_ptr());
        let bias = _mm256_set1_epi16(128);
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let q0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pq.add(i) as *const __m128i));
            let q1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pq.add(i + 16) as *const __m128i));
            let s0 = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128(ps.add(i) as *const __m128i)),
                bias,
            );
            let s1 = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128(ps.add(i + 16) as *const __m128i)),
                bias,
            );
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q0, s0));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(q1, s1));
            i += 32;
        }
        if i + 16 <= n {
            let q = _mm256_cvtepi8_epi16(_mm_loadu_si128(pq.add(i) as *const __m128i));
            let s = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128(ps.add(i) as *const __m128i)),
                bias,
            );
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q, s));
            i += 16;
        }
        let acc = _mm256_add_epi32(acc0, acc1);
        let s = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256(acc, 1),
        );
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        let mut sum = _mm_cvtsi128_si32(s);
        while i < n {
            sum += (*pq.add(i) as i32) * (*ps.add(i) as i32 - 128);
            i += 1;
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// aarch64
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[inline]
    pub(super) unsafe fn dot(query: &[f32], vec: &[f32]) -> f32 {
        // 4 independent accumulators break the FMA latency chain.
        // At 4 floats/register × 4 accumulators = 16 floats/iteration.
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let mut i = 0;
        let len = query.len().min(vec.len());
        while i + 16 <= len {
            let q0 = vld1q_f32(query.as_ptr().add(i));
            let q1 = vld1q_f32(query.as_ptr().add(i + 4));
            let q2 = vld1q_f32(query.as_ptr().add(i + 8));
            let q3 = vld1q_f32(query.as_ptr().add(i + 12));
            let v0 = vld1q_f32(vec.as_ptr().add(i));
            let v1 = vld1q_f32(vec.as_ptr().add(i + 4));
            let v2 = vld1q_f32(vec.as_ptr().add(i + 8));
            let v3 = vld1q_f32(vec.as_ptr().add(i + 12));
            s0 = vmlaq_f32(s0, q0, v0);
            s1 = vmlaq_f32(s1, q1, v1);
            s2 = vmlaq_f32(s2, q2, v2);
            s3 = vmlaq_f32(s3, q3, v3);
            i += 16;
        }
        // Drain remaining full NEON registers.
        while i + 4 <= len {
            let q = vld1q_f32(query.as_ptr().add(i));
            let v = vld1q_f32(vec.as_ptr().add(i));
            s0 = vmlaq_f32(s0, q, v);
            i += 4;
        }
        // Reduce four accumulators to one.
        s0 = vaddq_f32(s0, s1);
        s2 = vaddq_f32(s2, s3);
        s0 = vaddq_f32(s0, s2);
        let mut acc = vaddvq_f32(s0);
        while i < len {
            acc += query.get_unchecked(i) * vec.get_unchecked(i);
            i += 1;
        }
        acc
    }

    #[inline]
    pub(super) unsafe fn l2(query: &[f32], vec: &[f32]) -> f32 {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let mut i = 0;
        let len = query.len().min(vec.len());
        while i + 16 <= len {
            let q0 = vld1q_f32(query.as_ptr().add(i));
            let q1 = vld1q_f32(query.as_ptr().add(i + 4));
            let q2 = vld1q_f32(query.as_ptr().add(i + 8));
            let q3 = vld1q_f32(query.as_ptr().add(i + 12));
            let v0 = vld1q_f32(vec.as_ptr().add(i));
            let v1 = vld1q_f32(vec.as_ptr().add(i + 4));
            let v2 = vld1q_f32(vec.as_ptr().add(i + 8));
            let v3 = vld1q_f32(vec.as_ptr().add(i + 12));
            let d0 = vsubq_f32(q0, v0);
            let d1 = vsubq_f32(q1, v1);
            let d2 = vsubq_f32(q2, v2);
            let d3 = vsubq_f32(q3, v3);
            s0 = vmlaq_f32(s0, d0, d0);
            s1 = vmlaq_f32(s1, d1, d1);
            s2 = vmlaq_f32(s2, d2, d2);
            s3 = vmlaq_f32(s3, d3, d3);
            i += 16;
        }
        while i + 4 <= len {
            let q = vld1q_f32(query.as_ptr().add(i));
            let v = vld1q_f32(vec.as_ptr().add(i));
            let d = vsubq_f32(q, v);
            s0 = vmlaq_f32(s0, d, d);
            i += 4;
        }
        s0 = vaddq_f32(s0, s1);
        s2 = vaddq_f32(s2, s3);
        s0 = vaddq_f32(s0, s2);
        let mut acc = vaddvq_f32(s0);
        while i < len {
            let diff = query.get_unchecked(i) - vec.get_unchecked(i);
            acc += diff * diff;
            i += 1;
        }
        acc
    }

    #[inline]
    pub(super) unsafe fn sq8(query_i8: &[i8], stored: &[u8]) -> i32 {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            sq8_sdot(query_i8, stored)
        } else {
            super::dot_i8_u8_centered_scalar(query_i8, stored)
        }
    }

    /// NEON sdot: vdotq_s32 processes 4 groups of 4 i8 products per instruction.
    /// For 256-dim: 256/16 = 16 sdot calls with 4 accumulators = 4 iterations of 64 values.
    /// Memory: 256 B/vector (4 cache lines) vs 1024 B for f32 (16 cache lines).
    #[target_feature(enable = "dotprod")]
    unsafe fn sq8_sdot(query_i8: &[i8], stored: &[u8]) -> i32 {
        let n = query_i8.len().min(stored.len());
        let sub128 = vdupq_n_u8(128);
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);
        let mut i = 0;
        while i + 64 <= n {
            let q0 = vld1q_s8(query_i8.as_ptr().add(i));
            let q1 = vld1q_s8(query_i8.as_ptr().add(i + 16));
            let q2 = vld1q_s8(query_i8.as_ptr().add(i + 32));
            let q3 = vld1q_s8(query_i8.as_ptr().add(i + 48));
            // Subtract 128 from u8: u8-128 wraps to the correct signed i8 bit pattern.
            let s0 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i)), sub128));
            let s1 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i + 16)), sub128));
            let s2 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i + 32)), sub128));
            let s3 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i + 48)), sub128));
            acc0 = vdotq_s32(acc0, q0, s0);
            acc1 = vdotq_s32(acc1, q1, s1);
            acc2 = vdotq_s32(acc2, q2, s2);
            acc3 = vdotq_s32(acc3, q3, s3);
            i += 64;
        }
        while i + 16 <= n {
            let q = vld1q_s8(query_i8.as_ptr().add(i));
            let s = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i)), sub128));
            acc0 = vdotq_s32(acc0, q, s);
            i += 16;
        }
        acc0 = vaddq_s32(acc0, acc1);
        acc2 = vaddq_s32(acc2, acc3);
        acc0 = vaddq_s32(acc0, acc2);
        let mut sum = vaddvq_s32(acc0);
        while i < n {
            sum += (*query_i8.get_unchecked(i) as i32) * (*stored.get_unchecked(i) as i32 - 128);
            i += 1;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_f32(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    const SHAPES: &[usize] = &[
        0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257,
        384, 784, 960, 1536, 1537,
    ];

    #[test]
    fn every_supported_isa_matches_scalar() {
        let tables = Kernels::all_supported();
        eprintln!(
            "kernel ISAs under test: {:?}",
            tables.iter().map(|k| k.isa()).collect::<Vec<_>>()
        );
        for k in &tables {
            for &dim in SHAPES {
                let a = gen_f32(dim as u64 + 1, dim);
                // Unequal lengths: kernels must use the shorter one.
                let b = gen_f32(dim as u64 + 7919, dim + usize::from(dim % 2 == 0));
                let close = |got: f32, want: f32, what: &str| {
                    let tol = 1e-4 + 1e-5 * (dim as f32) + 2e-5 * want.abs();
                    assert!(
                        (got - want).abs() <= tol,
                        "{} {what} dim={dim}: got {got}, want {want}",
                        k.isa()
                    );
                };
                close(k.dot(&a, &b), dot_scalar(&a, &b), "dot");
                close(k.l2_squared(&a, &b), l2_squared_scalar(&a, &b), "l2");

                let q: Vec<i8> = (0..dim)
                    .map(|i| [-128, -127, -1, 0, 1, 126, 127][i % 7])
                    .collect();
                let s: Vec<u8> = (0..dim + usize::from(dim % 2 == 1))
                    .map(|i| [0, 1, 127, 128, 129, 254, 255][(i * 3) % 7])
                    .collect();
                assert_eq!(
                    k.dot_i8_u8_centered(&q, &s),
                    dot_i8_u8_centered_scalar(&q, &s),
                    "{} sq8 dim={dim}",
                    k.isa()
                );
            }
        }
    }

    #[test]
    fn public_entry_points_match_detected_table() {
        let k = Kernels::detect();
        let a = gen_f32(3, 257);
        let b = gen_f32(4, 257);
        assert_eq!(dot(&a, &b).to_bits(), k.dot(&a, &b).to_bits());
        assert_eq!(l2_squared(&a, &b).to_bits(), k.l2_squared(&a, &b).to_bits());
        let q: Vec<i8> = (0..257i32).map(|i| (i - 128) as i8).collect();
        let s: Vec<u8> = (0..257).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(dot_i8_u8_centered(&q, &s), k.dot_i8_u8_centered(&q, &s));
    }

    #[test]
    fn required_isas_are_present() {
        // CI hosts set ANNEX_REQUIRE_ISA=avx2,avx512 to fail loudly when a
        // runner silently lacks the instruction set it is meant to cover.
        let required = std::env::var("ANNEX_REQUIRE_ISA").unwrap_or_default();
        for name in required.split(',').map(str::trim).filter(|v| !v.is_empty()) {
            let isa = Isa::ALL
                .into_iter()
                .find(|isa| isa.name() == name || format!("{isa:?}").eq_ignore_ascii_case(name))
                .unwrap_or_else(|| panic!("unknown ISA {name}"));
            assert!(isa.is_supported(), "required ISA {name} is unavailable");
        }
    }
}
