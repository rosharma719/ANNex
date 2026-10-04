//! 4-bit scalar quantization (SQ4) for compressed graph traversal.
//!
//! SQ4 exists for one reason: **cache traffic**, not disk space. At 256
//! dimensions a 32-neighbour expansion touches roughly
//!
//! | representation | bytes per 32-neighbour expansion |
//! | --- | --- |
//! | f32 | 32 KiB |
//! | SQ8 | 8 KiB |
//! | SQ4 | 4 KiB |
//!
//! Halving the bytes read per expansion is the motivation for a future
//! neighbour-local graph layout. This module provides the codec and scalar
//! scoring baseline; it does not yet integrate SQ4 into graph traversal.
//!
//! # Scheme
//!
//! Per-dimension affine quantization over 16 levels:
//!
//! ```text
//! code_d(v)   = clamp(round((v_d - min_d) / scale_d), 0, 15)
//! reconstruct = min_d + code_d * scale_d
//! ```
//!
//! Two codes are packed per byte (low nibble = even dimension, high nibble =
//! odd dimension), so a `dim`-length vector occupies `ceil(dim / 2)` bytes.
//!
//! # Scoring
//!
//! Any dot-product-shaped metric reduces to a *linear* form in the codes:
//!
//! ```text
//! dot(q, v) = sum_d q_d * (min_d + code_d * scale_d)
//!           = C(q)          + sum_d (q_d * scale_d) * code_d
//!             ^^^^^^ constant across every stored vector
//! ```
//!
//! Dropping `C(q)` preserves ranking exactly, so traversal only needs
//! `sum_d w_d * code_d` with `w_d = q_d * scale_d`. The weights are quantized
//! to `i8` once per query ([`Sq4Query`]), which turns scoring into an
//! `i8 x u4` integer dot product — the same instruction shape as SQ8 screening
//! (`sdot` on NEON, `vpmaddubsw`/`vpdpbusd` on x86).
//!
//! Euclidean does **not** reduce to a linear form (`||q - v||^2` contains a
//! `code^2` term), so SQ4 scores are an approximate proxy there. That is
//! only useful as a candidate-generation proxy and would require exact FP32
//! reranking before results are returned. That traversal path is not part of
//! this module.
#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};

/// Number of levels representable in 4 bits.
pub const SQ4_LEVELS: usize = 16;

/// Maximum magnitude of a quantized query weight. Kept below `i8::MAX` so that
/// `w * code` (max `127 * 15 = 1905`) stays far inside `i16` after the
/// pairwise widening performed by `pmaddubsw` / `sdot`.
const WEIGHT_LIMIT: f32 = 127.0;

/// Per-dimension 4-bit scalar quantization tables.
///
/// Build once per immutable corpus with [`Sq4Codec::train`]. The codec is
/// immutable after construction and can be shared across threads with an
/// [`std::sync::Arc`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sq4Codec {
    dim: usize,
    /// Per-dimension lower bound of the quantization range.
    min: Vec<f32>,
    /// Per-dimension step: `(max_d - min_d) / 15`. Exactly `0.0` for constant
    /// dimensions, which makes every code `0` and contributes nothing.
    scale: Vec<f32>,
    /// Packed bytes per vector: `ceil(dim / 2)`.
    codes_per_lane: usize,
    /// Observed reconstruction RMSE across the training vectors.
    rmse: f32,
}

impl Sq4Codec {
    /// Packed code length in bytes for `dim` dimensions.
    #[inline]
    pub fn packed_len(dim: usize) -> usize {
        dim.div_ceil(2)
    }

    /// Derive per-dimension ranges from `vectors` and encode nothing yet.
    ///
    /// `vectors` is a flat `n * dim` row-major buffer. Returns `None` when the
    /// buffer is empty, contains a non-finite value, or its length is not a
    /// multiple of `dim`.
    pub fn train(dim: usize, vectors: &[f32]) -> Option<Self> {
        if dim == 0
            || vectors.is_empty()
            || !vectors.len().is_multiple_of(dim)
            || vectors.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let n = vectors.len() / dim;
        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        for row in 0..n {
            let base = row * dim;
            for d in 0..dim {
                let v = vectors[base + d];
                if v < min[d] {
                    min[d] = v;
                }
                if v > max[d] {
                    max[d] = v;
                }
            }
        }
        let mut scale = vec![0.0f32; dim];
        for d in 0..dim {
            let range = max[d] - min[d];
            // Degenerate (constant) dimensions get scale 0 → code 0 → no
            // contribution to any score, which is the correct behaviour.
            scale[d] = if range > 0.0 && range.is_finite() {
                range / (SQ4_LEVELS - 1) as f32
            } else {
                0.0
            };
        }

        let mut codec = Self {
            dim,
            min,
            scale,
            codes_per_lane: Self::packed_len(dim),
            rmse: 0.0,
        };
        codec.rmse = codec.measure_rmse(vectors);
        Some(codec)
    }

    /// Build a codec that maps `[-1, 1]` onto `[0, 15]` uniformly in every
    /// dimension. Useful for normalized embeddings and for tests that need a
    /// data-independent codec.
    pub fn symmetric(dim: usize) -> Self {
        let min = vec![-1.0f32; dim];
        let scale = vec![2.0 / (SQ4_LEVELS - 1) as f32; dim];
        Self {
            dim,
            min,
            scale,
            codes_per_lane: Self::packed_len(dim),
            rmse: 0.0,
        }
    }

    fn measure_rmse(&self, vectors: &[f32]) -> f32 {
        let n = vectors.len() / self.dim;
        if n == 0 {
            return 0.0;
        }
        let mut packed = vec![0u8; self.codes_per_lane];
        let mut sum_sq = 0.0f64;
        let mut count = 0u64;
        for row in 0..n {
            let v = &vectors[row * self.dim..(row + 1) * self.dim];
            self.encode(v, &mut packed);
            for d in 0..self.dim {
                let recon = self.reconstruct_dim(d, code_at(&packed, d));
                let err = f64::from(v[d] - recon);
                sum_sq += err * err;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum_sq / count as f64).sqrt() as f32
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Packed bytes produced by [`Self::encode`] for one vector.
    pub fn codes_per_lane(&self) -> usize {
        self.codes_per_lane
    }

    /// Root-mean-square reconstruction error over the training set.
    pub fn rmse(&self) -> f32 {
        self.rmse
    }

    pub fn min(&self) -> &[f32] {
        &self.min
    }

    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// 4-bit code for dimension `d` of `v`.
    #[inline]
    fn code(&self, v: &[f32], d: usize) -> u8 {
        let scale = self.scale[d];
        if scale == 0.0 {
            return 0;
        }
        let q = (v[d] - self.min[d]) / scale;
        // `round` then clamp: out-of-range values saturate rather than wrap.
        let c = q.round() as i32;
        c.clamp(0, (SQ4_LEVELS - 1) as i32) as u8
    }

    /// Approximate value of dimension `d` given its 4-bit code.
    #[inline]
    pub fn reconstruct_dim(&self, d: usize, code: u8) -> f32 {
        self.min[d] + f32::from(code) * self.scale[d]
    }

    /// Pack `v` into `out`, which must be at least [`Self::codes_per_lane`]
    /// bytes long. Byte `j` holds dimension `2j` in its low nibble and
    /// dimension `2j + 1` in its high nibble.
    ///
    /// # Panics
    /// Panics in debug builds when `v.len() != dim` or `out` is too short.
    pub fn encode(&self, v: &[f32], out: &mut [u8]) {
        assert_eq!(v.len(), self.dim, "sq4 encode: dim mismatch");
        assert!(
            v.iter().all(|value| value.is_finite()),
            "sq4 encode: values must be finite"
        );
        assert!(
            out.len() >= self.codes_per_lane,
            "sq4 encode: output too short"
        );
        let mut d = 0;
        for j in 0..self.codes_per_lane {
            let lo = self.code(v, d);
            let hi = if d + 1 < self.dim {
                self.code(v, d + 1)
            } else {
                0
            };
            out[j] = lo | (hi << 4);
            d += 2;
        }
    }

    /// Encode a flat `n * dim` row-major buffer into `n * codes_per_lane`
    /// packed bytes.
    pub fn encode_all(&self, vectors: &[f32]) -> Vec<u8> {
        assert_eq!(
            vectors.len() % self.dim,
            0,
            "sq4 encode_all: input is not a whole number of vectors"
        );
        let n = vectors.len() / self.dim;
        let mut out = vec![0u8; n * self.codes_per_lane];
        for row in 0..n {
            let v = &vectors[row * self.dim..(row + 1) * self.dim];
            self.encode(v, &mut out[row * self.codes_per_lane..]);
        }
        out
    }

    /// Decode packed codes back to `dim` floats. Lossy; used by tests and by
    /// any path that needs an approximate vector rather than a score.
    pub fn decode(&self, packed: &[u8]) -> Vec<f32> {
        assert!(
            packed.len() >= self.codes_per_lane,
            "sq4 decode: packed buffer too short ({} < {})",
            packed.len(),
            self.codes_per_lane
        );
        (0..self.dim)
            .map(|d| self.reconstruct_dim(d, code_at(packed, d)))
            .collect()
    }

    /// Prepare `q` for integer scoring.
    ///
    /// The returned [`Sq4Query`] holds the per-dimension weights split into the
    /// even/odd order in which the packed codes store them, so the kernel needs
    /// no shuffling on the hot path.
    ///
    /// # Panics
    ///
    /// Panics when `q` has the wrong dimension or contains a non-finite value.
    pub fn prepare_query(&self, q: &[f32]) -> Sq4Query {
        assert_eq!(q.len(), self.dim, "sq4 query: dim mismatch");
        assert!(
            q.iter().all(|value| value.is_finite()),
            "sq4 query: values must be finite"
        );
        let mut w = vec![0.0f32; self.dim];
        let mut max_abs = 0.0f32;
        for d in 0..self.dim {
            let wd = q[d] * self.scale[d];
            w[d] = wd;
            let a = wd.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        // Scale the whole weight vector so the largest magnitude lands on
        // WEIGHT_LIMIT. When every weight is zero (constant corpus) alpha is
        // irrelevant and all scores come out 0.
        let alpha = if max_abs > 0.0 {
            WEIGHT_LIMIT / max_abs
        } else {
            0.0
        };

        // Both weight arrays are padded to `codes_per_lane` so a kernel can
        // walk `even`, `odd` and `packed` in one fused pass. For odd `dim` the
        // trailing odd weight is 0, which pairs with the zero padding nibble
        // the encoder emits — the contribution is 0 either way.
        let n = self.codes_per_lane;
        let mut even = vec![0i8; n];
        let mut odd = vec![0i8; n];
        for j in 0..n {
            let d = 2 * j;
            even[j] = quantize_weight(w[d] * alpha);
            if d + 1 < self.dim {
                odd[j] = quantize_weight(w[d + 1] * alpha);
            }
        }

        // Constant term dropped from the integer score; retained so callers can
        // map an integer score back onto the approximate true dot product.
        let bias: f32 = (0..self.dim).map(|d| q[d] * self.min[d]).sum();

        Sq4Query {
            dim: self.dim,
            codes_per_lane: self.codes_per_lane,
            even,
            odd,
            alpha,
            bias,
        }
    }
}

/// Round a scaled weight to `i8`, saturating instead of wrapping.
#[inline]
fn quantize_weight(w: f32) -> i8 {
    if !w.is_finite() {
        return 0;
    }
    let r = w.round();
    if r >= WEIGHT_LIMIT {
        WEIGHT_LIMIT as i8
    } else if r <= -WEIGHT_LIMIT {
        -(WEIGHT_LIMIT as i8)
    } else {
        r as i8
    }
}

/// Extract the 4-bit code for dimension `d` from a packed buffer.
#[inline]
pub fn code_at(packed: &[u8], d: usize) -> u8 {
    let byte = packed[d / 2];
    if d.is_multiple_of(2) {
        byte & 0x0F
    } else {
        byte >> 4
    }
}

/// A query prepared for `i8 x u4` integer scoring against [`Sq4Codec`] codes.
///
/// Weights are pre-split into the even/odd dimension order used by the packed
/// layout, so the kernel can consume a packed byte buffer directly:
///
/// ```text
/// score(q, packed) = sum_j  even[j] * (packed[j] & 0x0F)
///                  + sum_j  odd[j]  * (packed[j] >> 4)
/// ```
#[derive(Clone, Debug)]
pub struct Sq4Query {
    dim: usize,
    codes_per_lane: usize,
    /// Weights for even dimensions (`2j`), one per packed byte.
    even: Vec<i8>,
    /// Weights for odd dimensions (`2j + 1`), one per packed byte.
    odd: Vec<i8>,
    /// Weight scaling applied during preparation; divide an integer score by
    /// this to recover the alpha-scaled true value.
    alpha: f32,
    /// The dropped constant term `sum_d q_d * min_d`.
    bias: f32,
}

impl Sq4Query {
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Packed bytes per vector this query expects.
    pub fn codes_per_lane(&self) -> usize {
        self.codes_per_lane
    }

    #[inline]
    pub fn even(&self) -> &[i8] {
        &self.even
    }

    #[inline]
    pub fn odd(&self) -> &[i8] {
        &self.odd
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Integer score against one packed code buffer. Higher means closer for
    /// dot-shaped metrics.
    ///
    /// # Panics
    /// Panics when `packed` is shorter than [`Self::codes_per_lane`].
    #[inline]
    pub fn score(&self, packed: &[u8]) -> i32 {
        assert!(
            packed.len() >= self.codes_per_lane,
            "sq4 score: packed buffer too short ({} < {})",
            packed.len(),
            self.codes_per_lane
        );
        crate::vector::kernels::dot_i8_u4_deinterleaved(&self.even, &self.odd, packed)
    }

    /// Approximate `dot(q, v)` recovered from an integer score.
    ///
    /// Accurate up to quantization error in both the codes and the weights.
    /// Meaningful for [`DistanceMetric::Cosine`](crate::DistanceMetric::Cosine)
    /// and [`Dot`](crate::DistanceMetric::Dot); for Euclidean it is only a
    /// monotone-ish proxy and callers must rerank exactly.
    #[inline]
    pub fn approximate_dot(&self, integer_score: i32) -> f32 {
        if self.alpha == 0.0 {
            return self.bias;
        }
        self.bias + integer_score as f32 / self.alpha
    }
}

/// Scalar reference implementation of the deinterleaved `i8 x u4` dot product.
///
/// Every SIMD kernel must agree with this exactly; it is the oracle used by the
/// kernel tests.
#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift stream in `[-1, 1)`.
    fn generate(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn encode_decode_roundtrip_is_lossy_but_close() {
        let dim = 64;
        let data = generate(1, dim * 32);
        let codec = Sq4Codec::train(dim, &data).unwrap();
        let packed = codec.encode_all(&data);
        assert_eq!(packed.len(), 32 * codec.codes_per_lane());

        let mut max_err = 0.0f32;
        for row in 0..32 {
            let v = &data[row * dim..(row + 1) * dim];
            let recon = codec.decode(&packed[row * codec.codes_per_lane()..]);
            for d in 0..dim {
                max_err = max_err.max((v[d] - recon[d]).abs());
            }
        }
        // Worst-case error is half a step; allow rounding slack.
        let half_step = codec.scale().iter().cloned().fold(0.0f32, f32::max) / 2.0;
        assert!(
            max_err <= half_step + 1e-4,
            "max_err {max_err} exceeds half step {half_step}"
        );
    }

    #[test]
    fn codes_are_nibbles_and_packing_is_little_nibble_first() {
        let dim = 4;
        // Deliberately asymmetric values so even/odd nibble order is visible.
        let data: Vec<f32> = vec![-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0];
        let codec = Sq4Codec::train(dim, &data).unwrap();
        let packed = codec.encode_all(&data);
        assert_eq!(packed.len(), 4);
        // dim 0 spans [-1, 1]: -1 -> 0, +1 -> 15.
        assert_eq!(packed[0] & 0x0F, 0, "dim0 of row0");
        assert_eq!(packed[0] >> 4, 15, "dim1 of row0");
        assert_eq!(packed[2] & 0x0F, 15, "dim0 of row1");
        assert_eq!(packed[2] >> 4, 0, "dim1 of row1");
    }

    #[test]
    fn odd_dimension_leaves_high_nibble_zero() {
        let dim = 3;
        let data = vec![0.0f32, 0.5, -0.5, 0.25, -0.25, 0.75];
        let codec = Sq4Codec::train(dim, &data).unwrap();
        assert_eq!(codec.codes_per_lane(), 2);
        let packed = codec.encode_all(&data);
        // Last byte's high nibble is padding and must be zero so it cannot
        // contribute a spurious score when a kernel reads whole bytes.
        assert_eq!(packed[1] >> 4, 0);
        assert_eq!(packed[3] >> 4, 0);
    }

    #[test]
    fn constant_dimensions_contribute_nothing() {
        let dim = 4;
        let data: Vec<f32> = vec![
            0.5, 0.0, -1.0, 1.0, // dim1 is constant 0 across both rows
            0.25, 0.0, -0.5, 0.5,
        ];
        let codec = Sq4Codec::train(dim, &data).unwrap();
        assert_eq!(codec.scale()[1], 0.0);
        let q = Sq4Query {
            dim,
            codes_per_lane: codec.codes_per_lane(),
            even: vec![10, 0],
            odd: vec![10, 0],
            alpha: 1.0,
            bias: 0.0,
        };
        let packed = codec.encode_all(&data);
        // Constant dimension encodes to 0 in both rows, so it cannot move the score.
        let s0 = q.score(&packed[0..]);
        let s1 = q.score(&packed[2..]);
        assert_ne!(s0, s1, "distinct vectors must score differently");
    }

    #[test]
    fn integer_score_preserves_dot_ranking() {
        let dim = 128;
        let n = 256;
        let raw = generate(7, dim * n);
        let data: Vec<f32> = (0..n)
            .flat_map(|row| unit(&raw[row * dim..(row + 1) * dim]))
            .collect();
        let codec = Sq4Codec::train(dim, &data).unwrap();
        let packed = codec.encode_all(&data);

        let q = unit(&generate(99, dim));
        let sq4q = codec.prepare_query(&q);

        // Exact dot products (the ranking oracle).
        let mut exact: Vec<(usize, f32)> = (0..n)
            .map(|row| {
                let v = &data[row * dim..(row + 1) * dim];
                let dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
                (row, dot)
            })
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut approx: Vec<(usize, i32)> = (0..n)
            .map(|row| (row, sq4q.score(&packed[row * codec.codes_per_lane()..])))
            .collect();
        approx.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        // Top-10 overlap is the property traversal actually depends on.
        let top = 10;
        let exact_top: std::collections::HashSet<usize> =
            exact.iter().take(top).map(|e| e.0).collect();
        let hits = approx
            .iter()
            .take(top)
            .filter(|a| exact_top.contains(&a.0))
            .count();
        assert!(
            hits >= top - 2,
            "SQ4 top-{top} overlap with exact was only {hits}"
        );
    }

    #[test]
    fn approximate_dot_recovers_scale() {
        let dim = 32;
        let raw = generate(3, dim * 16);
        let data: Vec<f32> = (0..16)
            .flat_map(|row| unit(&raw[row * dim..(row + 1) * dim]))
            .collect();
        let codec = Sq4Codec::train(dim, &data).unwrap();
        let packed = codec.encode_all(&data);
        let q = unit(&generate(4, dim));
        let sq4q = codec.prepare_query(&q);

        for row in 0..16 {
            let v = &data[row * dim..(row + 1) * dim];
            let true_dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
            let est = sq4q.approximate_dot(sq4q.score(&packed[row * codec.codes_per_lane()..]));
            // This is a scale sanity check for normalized embeddings. Ranking
            // behavior is covered separately; SQ4 is not an exact scorer.
            let tol = 0.10;
            assert!(
                (est - true_dot).abs() <= tol,
                "row {row}: est {est} vs true {true_dot}"
            );
        }
    }

    #[test]
    fn scalar_kernel_matches_reference_loop() {
        let dim = 65; // odd, exercises the padding nibble
        let data = generate(11, dim * 8);
        let codec = Sq4Codec::train(dim, &data).unwrap();
        let packed = codec.encode_all(&data);
        let q = generate(12, dim);
        let sq4q = codec.prepare_query(&q);

        for row in 0..8 {
            let p = &packed[row * codec.codes_per_lane()..(row + 1) * codec.codes_per_lane()];
            let got = sq4q.score(p);
            let want =
                crate::vector::kernels::dot_i8_u4_deinterleaved_scalar(sq4q.even(), sq4q.odd(), p);
            assert_eq!(got, want, "row {row}");
        }
    }

    #[test]
    fn train_rejects_mismatched_input() {
        assert!(Sq4Codec::train(0, &[1.0]).is_none());
        assert!(Sq4Codec::train(4, &[]).is_none());
        assert!(Sq4Codec::train(4, &[1.0, 2.0, 3.0]).is_none());
        assert!(Sq4Codec::train(2, &[0.0, f32::NAN]).is_none());
        assert!(Sq4Codec::train(2, &[0.0, f32::INFINITY]).is_none());
    }

    #[test]
    fn symmetric_codec_maps_unit_range() {
        let codec = Sq4Codec::symmetric(4);
        let v = vec![-1.0f32, -1.0 / 3.0, 1.0 / 3.0, 1.0];
        let mut packed = [0u8; 2];
        codec.encode(&v, &mut packed);
        let recon = codec.decode(&packed);
        assert_eq!(recon[0], -1.0);
        assert_eq!(recon[3], 1.0);
        assert!((recon[1] - v[1]).abs() < 0.15);
    }

    #[test]
    fn zero_weights_produce_zero_scores() {
        let dim = 8;
        let data = generate(5, dim * 4);
        let codec = Sq4Codec::train(dim, &data).unwrap();
        let packed = codec.encode_all(&data);
        let sq4q = codec.prepare_query(&vec![0.0; dim]);
        assert_eq!(sq4q.alpha(), 0.0);
        for row in 0..4 {
            assert_eq!(sq4q.score(&packed[row * codec.codes_per_lane()..]), 0);
        }
    }
}
