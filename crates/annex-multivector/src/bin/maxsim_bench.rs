//! Synthetic end-to-end MaxSim rescoring bench that mirrors the BEIR/FiQA
//! workload dimensions (~10K docs × 200 tokens × 128 dims + 32-token queries)
//! without needing the BEIR dataset. Used to profile the MaxSim rescoring
//! kernel in isolation and provide a stable before/after signal for kernel
//! optimizations.
use multivector::maxsim;
use rayon::prelude::*;
use std::time::Instant;

const DIM: usize = 128;
const DOC_TOKENS: usize = 200;
const QUERY_TOKENS: usize = 32;
const DOCS: usize = 10_000;
const QUERIES: usize = 100;
const CANDIDATES: usize = 250;
const WARMUP: usize = 3;

fn xorshift(state: &mut u64) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x as u32
}

fn gen_normalized(state: &mut u64, dim: usize, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| {
            let raw: Vec<f32> = (0..dim)
                .map(|_| (xorshift(state) as f32 / u32::MAX as f32) * 2.0 - 1.0)
                .collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            raw.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

fn flat(document: &[Vec<f32>]) -> Vec<f32> {
    document.iter().flat_map(|v| v.iter().copied()).collect()
}

fn maxsim_flat_scalar(query: &[Vec<f32>], document: &[f32], dimension: usize) -> f32 {
    query
        .iter()
        .map(|q| {
            document
                .chunks_exact(dimension)
                .map(|d| {
                    let mut s = 0.0f32;
                    for i in 0..dimension {
                        s += q[i] * d[i];
                    }
                    s
                })
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_neon_128(a: *const f32, b: *const f32) -> f32 {
    use std::arch::aarch64::*;
    // Four accumulators to hide FMA latency (~4 cycles on M-series).
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

#[cfg(target_arch = "aarch64")]
fn maxsim_flat_neon(query: &[Vec<f32>], document: &[f32], dimension: usize) -> f32 {
    assert_eq!(dimension, 128);
    query
        .iter()
        .map(|q| {
            let qp = q.as_ptr();
            let mut best = f32::NEG_INFINITY;
            for doc in document.chunks_exact(dimension) {
                // SAFETY: q has dimension = 128 lanes, doc has 128 lanes, both
                // are contiguous slices with aligned f32 elements.
                let s = unsafe { dot_neon_128(qp, doc.as_ptr()) };
                if s > best {
                    best = s;
                }
            }
            best
        })
        .sum()
}

fn run_one(
    label: &str,
    kernel: impl Fn(&[Vec<f32>], &[f32], usize) -> f32 + Sync,
    docs: &[Vec<f32>],
    queries: &[Vec<Vec<f32>>],
) {
    let mut per_query_ms: Vec<f64> = Vec::with_capacity(queries.len());
    let mut accumulator: f32 = 0.0;
    let wall = Instant::now();
    for query in queries.iter() {
        let start = Instant::now();
        let sum: f32 = docs[..CANDIDATES]
            .par_iter()
            .map(|doc| kernel(query, doc, DIM))
            .sum();
        accumulator += sum;
        per_query_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let total_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let p50 = percentile(per_query_ms.clone(), 0.50);
    let p95 = percentile(per_query_ms.clone(), 0.95);
    let p99 = percentile(per_query_ms.clone(), 0.99);
    let mean: f64 = per_query_ms.iter().sum::<f64>() / per_query_ms.len() as f64;
    let dot_ops_per_query =
        CANDIDATES as f64 * DOC_TOKENS as f64 * QUERY_TOKENS as f64 * DIM as f64;
    let gflops = (dot_ops_per_query * 2.0) / (mean * 1e-3) / 1e9;
    println!(
        "{label:<8}  mean={mean:>7.2}ms  p50={p50:>7.2}ms  p95={p95:>7.2}ms  p99={p99:>7.2}ms  wall={total_ms:>8.1}ms  GFLOP/s(FMA)={gflops:>6.2}  checksum={accumulator:>14.4}"
    );
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((xs.len() - 1) as f64 * p) as usize;
    xs[idx]
}

fn main() {
    let mut rng_state = 0x_dead_beef_cafe_babe_u64;

    eprintln!(
        "gen: {DOCS} docs × {DOC_TOKENS} tokens × {DIM} dims, {QUERIES} queries × {QUERY_TOKENS} tokens"
    );
    let docs: Vec<Vec<f32>> = (0..DOCS)
        .map(|_| flat(&gen_normalized(&mut rng_state, DIM, DOC_TOKENS)))
        .collect();
    let queries: Vec<Vec<Vec<f32>>> = (0..QUERIES)
        .map(|_| gen_normalized(&mut rng_state, DIM, QUERY_TOKENS))
        .collect();

    // Warmup: prime the thread pool and page caches.
    for _ in 0..WARMUP {
        let query = &queries[0];
        let _: f32 = docs[..CANDIDATES]
            .par_iter()
            .map(|doc| maxsim_flat_scalar(query, doc, DIM))
            .sum();
    }

    println!(
        "maxsim rescore   candidates={CANDIDATES}  doc_tokens={DOC_TOKENS}  query_tokens={QUERY_TOKENS}  dim={DIM}"
    );
    println!(
        "{:<8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}",
        "kernel", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "wall_ms", "GFLOP/s"
    );
    run_one("scalar", maxsim_flat_scalar, &docs, &queries);
    #[cfg(target_arch = "aarch64")]
    run_one("neon-bench", maxsim_flat_neon, &docs, &queries);
    // Also measure the crate-level dispatched maxsim_flat so kernel-level
    // improvements in fde.rs (loop reorder, LUT changes, etc.) show up
    // here without having to re-copy the code into this bench.
    run_one(
        "crate",
        |q, d, dim| multivector::maxsim_flat(q, d, dim),
        &docs,
        &queries,
    );

    // Correctness check: NEON kernel must agree with scalar per-query.
    #[cfg(target_arch = "aarch64")]
    {
        let q = &queries[0];
        let d = &docs[0];
        let s = maxsim_flat_scalar(q, d, DIM);
        let n = maxsim_flat_neon(q, d, DIM);
        let diff = (s - n).abs();
        eprintln!("correctness: scalar={s:.6} neon={n:.6} abs_diff={diff:.2e}  tolerance=1e-3");
        assert!(diff < 1e-3, "NEON MaxSim disagreed with scalar reference");
    }
    let _ = maxsim;
}
