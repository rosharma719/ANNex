//! Kernel measurements; commands and reporting rules: docs/benchmarks.md.

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
    // Covers standard ColBERT/E5/MPNet/instructor embedding dimensions.
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

fn bench_dot(c: &mut Criterion) {
    use multivector::maxsim;
    let mut group = c.benchmark_group("maxsim_naive");
    let mut rng_state = 0xfeed_face_dead_beef_u64;
    let query = gen_normalized(&mut rng_state, 128, 8);
    let doc = gen_normalized(&mut rng_state, 128, 8);
    group.throughput(Throughput::Elements(8 * 8 * 128 * 2));
    group.bench_function("dim=128/q=8/d=8", |b| {
        b.iter(|| maxsim(black_box(&query), black_box(&doc)))
    });
    group.finish();
}

fn bench_screen_dot(c: &mut Criterion) {
    use annex::bench_access::{screen_dot_dispatch, screen_dot_scalar};

    let mut group = c.benchmark_group("screen_dot");

    for &dim in &[128usize, 256, 768] {
        let mut rng = 0xdead_beef_u64;
        let query_i8: Vec<i8> = (0..dim)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng as i8).wrapping_add(1)
            })
            .collect();
        let stored_u8: Vec<u8> = (0..dim)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng as u8
            })
            .collect();

        group.throughput(Throughput::Elements(dim as u64));

        // Scalar reference.
        let q = query_i8.clone();
        let s = stored_u8.clone();
        let label = format!("scalar/dim={dim}");
        group.bench_function(&label, |b| {
            b.iter(|| screen_dot_scalar(black_box(&q), black_box(&s)))
        });

        // Full dispatch (routes to best available kernel: VNNI on Ice Lake, AVX2 elsewhere).
        let q2 = query_i8.clone();
        let s2 = stored_u8.clone();
        let label = format!("dispatch/dim={dim}");
        group.bench_function(&label, |b| {
            b.iter(|| screen_dot_dispatch(black_box(&q2), black_box(&s2)))
        });
    }
    group.finish();
}

fn bench_maxsim_flat_bf16(c: &mut Criterion) {
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = c;
        println!("[bench_maxsim_flat_bf16] not x86_64 — skipping");
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx512bf16") {
        let _ = c;
        println!("[bench_maxsim_flat_bf16] avx512bf16 not available on this host — skipping");
        return;
    }
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

criterion_group!(
    kernels,
    bench_maxsim_flat,
    bench_dot,
    bench_screen_dot,
    bench_maxsim_flat_bf16
);
criterion_main!(kernels);
