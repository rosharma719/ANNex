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
#[cfg(target_arch = "aarch64")]
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
/// tail with the scalar path; x86_64 uses `annex::vector::kernels::dot`. Every hot path in the crate (FDE exhaustive
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
        dot_scalar(&left[..n], &right[..n])
    }
    #[cfg(target_arch = "x86_64")]
    if n > 0 && matches!(cpu_level(), CpuLevel::Avx512Bf16) {
        return unsafe { dot_avx512_bf16_len(left.as_ptr(), right.as_ptr(), n) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // AVX-512 / AVX2+FMA with runtime dispatch; scalar elsewhere.
        annex::vector::kernels::dot(&left[..n], &right[..n])
    }
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
/// 512, 768, 1024), and to the packed AVX-512/AVX2 kernel for any dimension
/// on x86_64. Falls back to the scalar path everywhere else.
///
/// When one query is scored against many documents, build a [`MaxSimQuery`]
/// once instead: on x86_64 this call re-packs the query every time.
pub fn maxsim_flat(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if dimension > 0 && dimension.is_multiple_of(16) && dimension <= 4096 {
            return maxsim_flat_neon(query, document, dimension);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if dimension > 0 && matches!(cpu_level(), CpuLevel::Avx512Bf16) {
        return maxsim_flat_avx512_bf16(query, document, dimension);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(kernel) = x86::PackedKernel::detect()
            && x86::applicable(query, document, dimension)
        {
            thread_local! {
                static PACKED: std::cell::RefCell<x86::Panel> =
                    const { std::cell::RefCell::new(x86::Panel::new()) };
            }
            return PACKED.with(|cell| {
                let mut packed = cell.borrow_mut();
                x86::pack(query, dimension, kernel.lanes(), &mut packed);
                // SAFETY: `detect` verified CPU support; `pack` sized the panel.
                unsafe { kernel.score(&packed, query.len(), dimension, document) }
            });
        }
    }
    maxsim_flat_scalar(query, document, dimension)
}

/// A normalized query prepared for repeated MaxSim scoring.
///
/// On x86_64 with AVX2 or AVX-512 the query tokens are packed once into a
/// dimension-major panel, so each document is scored with a register-tiled
/// kernel that needs no horizontal reductions. Elsewhere it forwards to
/// [`maxsim_flat`].
pub struct MaxSimQuery<'a> {
    tokens: &'a [Vector],
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    dimension: usize,
    #[cfg(target_arch = "x86_64")]
    packed: Option<(x86::PackedKernel, x86::Panel)>,
}

impl<'a> MaxSimQuery<'a> {
    pub fn new(tokens: &'a [Vector], dimension: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        let packed = x86::PackedKernel::detect()
            .filter(|_| dimension > 0 && !tokens.is_empty())
            .map(|kernel| {
                let mut panel = x86::Panel::new();
                x86::pack(tokens, dimension, kernel.lanes(), &mut panel);
                (kernel, panel)
            });
        MaxSimQuery {
            tokens,
            dimension,
            #[cfg(target_arch = "x86_64")]
            packed,
        }
    }

    /// Equivalent to `maxsim_flat(tokens, document, dimension)`.
    #[inline]
    pub fn score(&self, document: &[f32], dimension: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if let Some((kernel, panel)) = &self.packed
            && dimension == self.dimension
            && x86::applicable(self.tokens, document, dimension)
        {
            // SAFETY: `detect` verified CPU support; `pack` sized the panel.
            return unsafe { kernel.score(panel, self.tokens.len(), dimension, document) };
        }
        maxsim_flat(self.tokens, document, dimension)
    }
}

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
        let a0 = _mm512_loadu_ps(a.add(i));
        let a1 = _mm512_loadu_ps(a.add(i + 16));
        let b0 = _mm512_loadu_ps(b.add(i));
        let b1 = _mm512_loadu_ps(b.add(i + 16));
        acc0 = _mm512_dpbf16_ps(
            acc0,
            _mm512_cvtne2ps_pbh(a1, a0),
            _mm512_cvtne2ps_pbh(b1, b0),
        );
        let a2 = _mm512_loadu_ps(a.add(i + 32));
        let a3 = _mm512_loadu_ps(a.add(i + 48));
        let b2 = _mm512_loadu_ps(b.add(i + 32));
        let b3 = _mm512_loadu_ps(b.add(i + 48));
        acc1 = _mm512_dpbf16_ps(
            acc1,
            _mm512_cvtne2ps_pbh(a3, a2),
            _mm512_cvtne2ps_pbh(b3, b2),
        );
        let a4 = _mm512_loadu_ps(a.add(i + 64));
        let a5 = _mm512_loadu_ps(a.add(i + 80));
        let b4 = _mm512_loadu_ps(b.add(i + 64));
        let b5 = _mm512_loadu_ps(b.add(i + 80));
        acc2 = _mm512_dpbf16_ps(
            acc2,
            _mm512_cvtne2ps_pbh(a5, a4),
            _mm512_cvtne2ps_pbh(b5, b4),
        );
        let a6 = _mm512_loadu_ps(a.add(i + 96));
        let a7 = _mm512_loadu_ps(a.add(i + 112));
        let b6 = _mm512_loadu_ps(b.add(i + 96));
        let b7 = _mm512_loadu_ps(b.add(i + 112));
        acc3 = _mm512_dpbf16_ps(
            acc3,
            _mm512_cvtne2ps_pbh(a7, a6),
            _mm512_cvtne2ps_pbh(b7, b6),
        );
        i += 128;
    }
    while i + 32 <= len {
        let a0 = _mm512_loadu_ps(a.add(i));
        let a1 = _mm512_loadu_ps(a.add(i + 16));
        let b0 = _mm512_loadu_ps(b.add(i));
        let b1 = _mm512_loadu_ps(b.add(i + 16));
        acc0 = _mm512_dpbf16_ps(
            acc0,
            _mm512_cvtne2ps_pbh(a1, a0),
            _mm512_cvtne2ps_pbh(b1, b0),
        );
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

/// Packed MaxSim for x86_64.
///
/// MaxSim is a small GEMM (`query [nq x dim] * doc^T [dim x nd]`) followed by a
/// row max. Instead of `nq * nd` independent dot products, each needing a
/// horizontal reduction, the query is packed dimension-major into blocks of
/// one vector register (16 lanes on AVX-512, 8 on AVX2), so lane `t` of a
/// block holds query token `t`. A tile of `QB` query blocks x `DB` document
/// tokens then accumulates in `QB * DB` registers: for each dimension `k` it
/// loads `QB` query rows, broadcasts `DB` document scalars and issues
/// `QB * DB` FMAs. The row max is a vertical `max` against a running best,
/// so no shuffles are needed until the final per-block lane sum.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Vector;
    use annex::vector::kernels::Isa;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum PackedKernel {
        Avx2,
        Avx512,
    }

    impl PackedKernel {
        pub(super) fn detect() -> Option<Self> {
            [PackedKernel::Avx512, PackedKernel::Avx2]
                .into_iter()
                .find(|k| k.is_supported())
        }

        pub(super) fn is_supported(self) -> bool {
            match self {
                PackedKernel::Avx2 => Isa::Avx2.is_supported(),
                PackedKernel::Avx512 => std::arch::is_x86_feature_detected!("avx512f"),
            }
        }

        pub(super) fn lanes(self) -> usize {
            match self {
                PackedKernel::Avx2 => 8,
                PackedKernel::Avx512 => 16,
            }
        }

        /// # Safety
        /// The CPU must support `self`, and `panel` must come from [`pack`]
        /// with the same `nq`, `dim` and `self.lanes()`.
        pub(super) unsafe fn score(self, panel: &Panel, nq: usize, dim: usize, doc: &[f32]) -> f32 {
            debug_assert!(panel.len() >= nq.div_ceil(self.lanes()) * dim * self.lanes());
            let nd = doc.len() / dim;
            unsafe {
                match self {
                    PackedKernel::Avx2 => avx2::maxsim(panel.as_ptr(), nq, dim, doc.as_ptr(), nd),
                    PackedKernel::Avx512 => {
                        avx512::maxsim(panel.as_ptr(), nq, dim, doc.as_ptr(), nd)
                    }
                }
            }
        }
    }

    /// The packed kernels need a query token and a whole document token;
    /// everything else keeps the scalar path's semantics.
    pub(super) fn applicable(query: &[Vector], document: &[f32], dimension: usize) -> bool {
        dimension > 0 && !query.is_empty() && document.len() >= dimension
    }

    #[derive(Clone, Copy)]
    #[repr(C, align(64))]
    struct Line([f32; 16]);

    /// Cache-line-aligned f32 buffer, so no packed query row splits a line.
    #[derive(Default)]
    pub(super) struct Panel(Vec<Line>);

    impl Panel {
        pub(super) const fn new() -> Self {
            Panel(Vec::new())
        }

        fn len(&self) -> usize {
            self.0.len() * 16
        }

        fn as_ptr(&self) -> *const f32 {
            self.0.as_ptr().cast()
        }

        fn reset(&mut self, len: usize) -> &mut [f32] {
            self.0.clear();
            self.0.resize(len.div_ceil(16), Line([0.0; 16]));
            // SAFETY: `Line` is 16 contiguous f32s with no padding.
            unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), self.len()) }
        }
    }

    /// Pack `query` into blocks of `lanes` tokens, dimension-major: token
    /// `t`, dimension `k` lands at `(t / lanes) * dim * lanes + k * lanes +
    /// t % lanes`. Unused lanes and dimensions past a short token stay zero,
    /// matching `dot`'s shorter-length semantics.
    pub(super) fn pack(query: &[Vector], dim: usize, lanes: usize, panel: &mut Panel) {
        let out = panel.reset(query.len().div_ceil(lanes) * dim * lanes);
        for (block, tokens) in out.chunks_exact_mut(dim * lanes).zip(query.chunks(lanes)) {
            if tokens.len() == lanes && tokens.iter().all(|t| t.len() >= dim) {
                // Full block: write each output row sequentially.
                for (k, row) in block.chunks_exact_mut(lanes).enumerate() {
                    for (slot, token) in row.iter_mut().zip(tokens) {
                        // SAFETY: every token has at least `dim` values.
                        *slot = unsafe { *token.get_unchecked(k) };
                    }
                }
            } else {
                for (lane, token) in tokens.iter().enumerate() {
                    for (k, &v) in token.iter().take(dim).enumerate() {
                        block[k * lanes + lane] = v;
                    }
                }
            }
        }
    }

    macro_rules! packed_maxsim {
        (
            $name:ident, $feature:literal, $reg:ty, $lanes:literal,
            $db_big:literal, $db_mid:literal, $qb_max:literal,
            zero: $zero:expr, splat: $splat:path, load: $load:path,
            fma: $fma:path, max: $max:path, sum: $sum:ident
        ) => {
            mod $name {
                use std::arch::x86_64::*;
                const L: usize = $lanes;

                #[target_feature(enable = $feature)]
                pub(super) unsafe fn maxsim(
                    panel: *const f32,
                    nq: usize,
                    dim: usize,
                    doc: *const f32,
                    nd: usize,
                ) -> f32 {
                    let blocks = nq.div_ceil(L);
                    let mut total = 0.0f32;
                    let mut b = 0;
                    while b < blocks {
                        let q = unsafe { panel.add(b * dim * L) };
                        if b + $qb_max <= blocks {
                            let best = unsafe { sweep::<$qb_max>(q, dim, doc, nd) };
                            for (i, v) in best.into_iter().enumerate() {
                                total += unsafe { $sum(v, nq - (b + i) * L) };
                            }
                            b += $qb_max;
                        } else {
                            let best = unsafe { sweep::<1>(q, dim, doc, nd) };
                            total += unsafe { $sum(best[0], nq - b * L) };
                            b += 1;
                        }
                    }
                    total
                }

                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn sweep<const QB: usize>(
                    q: *const f32,
                    dim: usize,
                    doc: *const f32,
                    nd: usize,
                ) -> [$reg; QB] {
                    let mut best = [$splat(f32::NEG_INFINITY); QB];
                    let mut d = 0;
                    unsafe {
                        while d + $db_big <= nd {
                            tile::<QB, $db_big>(q, dim, doc.add(d * dim), &mut best);
                            d += $db_big;
                        }
                        if d + $db_mid <= nd {
                            tile::<QB, $db_mid>(q, dim, doc.add(d * dim), &mut best);
                            d += $db_mid;
                        }
                        while d < nd {
                            tile::<QB, 1>(q, dim, doc.add(d * dim), &mut best);
                            d += 1;
                        }
                    }
                    best
                }

                /// `QB` query blocks x `DB` document tokens, all in registers.
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn tile<const QB: usize, const DB: usize>(
                    q: *const f32,
                    dim: usize,
                    docs: *const f32,
                    best: &mut [$reg; QB],
                ) {
                    let mut acc = [[$zero; QB]; DB];
                    for k in 0..dim {
                        let mut qv = [$zero; QB];
                        for (b, v) in qv.iter_mut().enumerate() {
                            *v = unsafe { $load(q.add(b * dim * L + k * L)) };
                        }
                        for (j, row) in acc.iter_mut().enumerate() {
                            let x = $splat(unsafe { *docs.add(j * dim + k) });
                            for (a, &v) in row.iter_mut().zip(qv.iter()) {
                                *a = $fma(v, x, *a);
                            }
                        }
                    }
                    for row in &acc {
                        for (m, &a) in best.iter_mut().zip(row) {
                            *m = $max(*m, a);
                        }
                    }
                }

                #[allow(dead_code)]
                #[inline]
                #[target_feature(enable = "avx512f")]
                unsafe fn sum512(v: __m512, valid: usize) -> f32 {
                    let m = if valid >= 16 {
                        u16::MAX
                    } else {
                        ((1u32 << valid) - 1) as u16
                    };
                    _mm512_mask_reduce_add_ps(m, v)
                }

                #[allow(dead_code)]
                #[inline]
                #[target_feature(enable = "avx")]
                unsafe fn sum256(v: __m256, valid: usize) -> f32 {
                    let mut lanes = [0.0f32; 8];
                    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
                    lanes[..valid.min(8)].iter().sum()
                }
            }
        };
    }

    // AVX-512: 2 query blocks (32 tokens) x 8 document tokens = 16 zmm
    // accumulators + 2 query rows + 1 broadcast, of 32 registers.
    packed_maxsim!(
        avx512, "avx512f", __m512, 16, 8, 4, 2,
        zero: _mm512_setzero_ps(), splat: _mm512_set1_ps, load: _mm512_load_ps,
        fma: _mm512_fmadd_ps, max: _mm512_max_ps, sum: sum512
    );

    // AVX2: 2 query blocks (16 tokens) x 6 document tokens = 12 ymm
    // accumulators + 2 query rows + 1 broadcast, of 16 registers.
    packed_maxsim!(
        avx2, "avx2,fma", __m256, 8, 6, 3, 2,
        zero: _mm256_setzero_ps(), splat: _mm256_set1_ps, load: _mm256_load_ps,
        fma: _mm256_fmadd_ps, max: _mm256_max_ps, sum: sum256
    );
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
        // Exact on aarch64 (scalar path); x86_64 vectorises every dimension.
        let tol = if cfg!(target_arch = "aarch64") {
            1e-6
        } else {
            1e-4
        };
        assert!((s - d).abs() < tol, "scalar={s} dispatch={d}");
    }

    #[test]
    fn maxsim_kernels_match_scalar_across_shapes() {
        for &dim in &[1usize, 7, 16, 33, 96, 100, 128, 129, 384, 768] {
            for &nq in &[1usize, 3, 8, 15, 16, 17, 31, 32, 33, 48] {
                for &nd in &[1usize, 2, 5, 6, 7, 8, 9, 13, 64, 201] {
                    let query: Vec<_> = (0..nq)
                        .map(|i| normalize(&deterministic(0x77 + i as u64, dim)))
                        .collect();
                    let doc: Vec<_> = (0..nd)
                        .map(|i| normalize(&deterministic(0x9000 + (i * 31 + dim) as u64, dim)))
                        .collect();
                    let (flat_doc, _) = flat(&doc);
                    let want = maxsim_flat_scalar(&query, &flat_doc, dim);
                    // FMA single-rounding vs scalar two-step rounding diverges
                    // at non-power-of-2 dims; 5e-4 base accommodates AVX FMA.
                    let tol = 1e-4 * nq as f32 + 5e-4;
                    let got = maxsim_flat(&query, &flat_doc, dim);
                    assert!(
                        (got - want).abs() <= tol,
                        "flat dim={dim} nq={nq} nd={nd}: {got} vs {want}"
                    );
                    let prepared = MaxSimQuery::new(&query, dim).score(&flat_doc, dim);
                    assert!(
                        (prepared - want).abs() <= tol,
                        "prepared dim={dim} nq={nq} nd={nd}: {prepared} vs {want}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_x86_maxsim_kernel_matches_scalar() {
        for kernel in [x86::PackedKernel::Avx2, x86::PackedKernel::Avx512] {
            if !kernel.is_supported() {
                continue;
            }
            for &(dim, nq, nd) in &[(128, 32, 200), (100, 17, 13), (384, 5, 7), (1, 1, 1)] {
                let query: Vec<_> = (0..nq)
                    .map(|i| normalize(&deterministic(0x5 + i as u64, dim)))
                    .collect();
                let doc: Vec<_> = (0..nd)
                    .map(|i| normalize(&deterministic(0x700 + i as u64, dim)))
                    .collect();
                let (flat_doc, _) = flat(&doc);
                let mut panel = x86::Panel::new();
                x86::pack(&query, dim, kernel.lanes(), &mut panel);
                let got = unsafe { kernel.score(&panel, nq, dim, &flat_doc) };
                let want = maxsim_flat_scalar(&query, &flat_doc, dim);
                assert!(
                    (got - want).abs() < 1e-3,
                    "{kernel:?} dim={dim}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn maxsim_edge_cases_match_scalar() {
        let q = vec![normalize(&deterministic(1, 16))];
        assert_eq!(maxsim_flat(&[], &[1.0; 16], 16), 0.0);
        assert_eq!(maxsim_flat(&q, &[], 16), f32::NEG_INFINITY);
        assert_eq!(MaxSimQuery::new(&q, 16).score(&[], 16), f32::NEG_INFINITY);
        // A partial trailing token is ignored, as with `chunks_exact`.
        let doc = deterministic(2, 20);
        let want = maxsim_flat_scalar(&q, &doc, 16);
        assert!((maxsim_flat(&q, &doc, 16) - want).abs() < 1e-5);
    }

    #[test]
    fn maxsim_flat_bf16_agrees_with_scalar_on_dim128() {
        if cfg!(not(target_arch = "x86_64")) {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx512bf16") {
            return;
        }

        let mut rng = 0xcafe_babe_u64;
        let mut next = || -> f32 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let query: Vec<_> = (0..8)
            .map(|_| (0..128).map(|_| next()).collect::<Vec<f32>>())
            .collect();
        let doc_tokens: Vec<_> = (0..50)
            .map(|_| (0..128).map(|_| next()).collect::<Vec<f32>>())
            .collect();
        let flat_doc: Vec<f32> = doc_tokens.iter().flat_map(|v| v.iter().copied()).collect();
        let scalar = maxsim_flat_scalar(&query, &flat_doc, 128);
        let dispatch = maxsim_flat(&query, &flat_doc, 128);
        let tol = 1e-3_f32 * scalar.abs().max(1.0);
        assert!(
            (dispatch - scalar).abs() <= tol,
            "bf16 maxsim mismatch: scalar={scalar} dispatch={dispatch}"
        );
    }

    #[test]
    fn dot_self_is_near_one_after_normalize_on_bf16() {
        if cfg!(not(target_arch = "x86_64")) {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx512bf16") {
            return;
        }
        for dim in [64usize, 128, 384, 768] {
            let raw: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).recip()).collect();
            let normed = normalize(&raw);
            let self_dot = dot(&normed, &normed);
            // BF16 has 7-bit mantissa (~0.8% relative error); accumulated
            // over `dim` terms a self-dot can deviate up to ~5e-3 from 1.0.
            assert!(
                (self_dot - 1.0).abs() < 5e-3,
                "dim={dim}: self_dot={self_dot}"
            );
        }
    }
}
