//! Criterion micro-benchmarks for the annex-multivector kernels.
//!
//! Establishes reproducible perf baselines so future kernel edits either
//! ratchet the numbers up or fail loudly. Complements the ad-hoc
//! `src/bin/maxsim_bench.rs` which prints a one-shot number.
//!
//! Run with:
//!     cargo bench -p annex-multivector --bench kernels
//!
//! To gate CI on no regression, add `--save-baseline main` on the ref run
//! and `--baseline main` on PR runs.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use multivector::maxsim_flat;

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

fn bench_maxsim_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxsim_flat");
    // ColBERTv2 shape: 32 query tokens x ~200 doc tokens x 128 dims. Sizes
    // chosen to match the FiQA/Scifact rescoring workload.
    for &(dim, doc_tokens, query_tokens) in &[
        (128usize, 200usize, 32usize),
        (128, 100, 32),
        (384, 200, 32), // E5/MPNet dim
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

fn bench_dot(c: &mut Criterion) {
    use multivector::maxsim;
    let mut group = c.benchmark_group("maxsim_naive");
    // A tiny 8-token x 8-token maxsim to bound the naive (non-flat) path.
    let mut rng_state = 0xfeed_face_dead_beef_u64;
    let query = gen_normalized(&mut rng_state, 128, 8);
    let doc = gen_normalized(&mut rng_state, 128, 8);
    group.throughput(Throughput::Elements(8 * 8 * 128 * 2));
    group.bench_function("dim=128/q=8/d=8", |b| {
        b.iter(|| maxsim(black_box(&query), black_box(&doc)))
    });
    group.finish();
}

criterion_group!(kernels, bench_maxsim_flat, bench_dot);
criterion_main!(kernels);
