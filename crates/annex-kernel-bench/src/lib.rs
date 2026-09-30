//! Shared data generation for the kernel benchmarks in `benches/`.
//! Commands and reporting rules: docs/benchmarks.md.

#[cfg(target_arch = "x86_64")]
pub mod legacy;

/// Deterministic xorshift64 stream in [-1, 1).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    pub fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }

    pub fn unit_vector(&mut self, dim: usize) -> Vec<f32> {
        let v = self.vector(dim);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.into_iter().map(|x| x / norm).collect()
    }
}

/// SQ8 codes as stored by the HNSW index: `round(x * 127.5 + 128)`.
pub fn sq8_encode(unit: &[f32]) -> Vec<u8> {
    unit.iter()
        .map(|&x| (x * 127.5 + 128.0).clamp(0.0, 255.0).round() as u8)
        .collect()
}

/// SQ8 query quantisation used by `HNSWIndex::quantize_query_i8`.
pub fn sq8_query(unit: &[f32]) -> Vec<i8> {
    unit.iter()
        .map(|&x| (x * 127.5).clamp(-128.0, 127.0).round() as i8)
        .collect()
}

/// Plain Rust with 16 independent lanes: what the auto-vectoriser produces
/// without explicit intrinsics (with `-C target-cpu=native`).
pub fn dot_autovec(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; 16];
    let ((ca, ta), (cb, tb)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + ta.iter().zip(tb).map(|(x, y)| x * y).sum::<f32>()
}

pub fn l2_autovec(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; 16];
    let ((ca, ta), (cb, tb)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..16 {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    acc.iter().sum::<f32>()
        + ta.iter()
            .zip(tb)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
}

/// Row-major `rows x cols` matrix from `rows` unit vectors.
pub fn unit_matrix(rng: &mut Rng, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows).map(|_| rng.unit_vector(cols)).collect()
}

#[cfg(feature = "blas")]
pub mod blas {
    //! Minimal CBLAS binding (system OpenBLAS, linked by build.rs).
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    unsafe extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        fn openblas_set_num_threads(n: i32);
    }

    pub fn single_threaded() {
        unsafe { openblas_set_num_threads(1) }
    }

    /// `out[nq x nd] = query[nq x dim] * doc[nd x dim]^T`.
    pub fn sgemm_nt(query: &[f32], doc: &[f32], nq: usize, nd: usize, dim: usize, out: &mut [f32]) {
        assert!(query.len() >= nq * dim && doc.len() >= nd * dim && out.len() >= nq * nd);
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                TRANS,
                nq as i32,
                nd as i32,
                dim as i32,
                1.0,
                query.as_ptr(),
                dim as i32,
                doc.as_ptr(),
                dim as i32,
                0.0,
                out.as_mut_ptr(),
                nd as i32,
            )
        }
    }
}
