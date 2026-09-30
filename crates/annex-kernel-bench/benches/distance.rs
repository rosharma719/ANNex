//! Single-pair distance kernels: ANNex (per ISA and dispatched), the legacy
//! pre-`kernels` x86 code, SimSIMD and auto-vectorised Rust.
//!
//! Each iteration scores one query against a 256-vector working set (L2
//! resident up to 1536 dims), so throughput is reported per score.
//! Commands and reporting rules: docs/benchmarks.md.

use std::hint::black_box;
use std::time::Duration;

use annex::vector::kernels::{self, Isa, Kernels};
use annex_kernel_bench::{Rng, dot_autovec, l2_autovec, sq8_encode, sq8_query};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use simsimd::SpatialSimilarity;

const DIMS: &[usize] = &[96, 100, 128, 256, 384, 768, 960, 1536];
const SET: usize = 256;

fn filter(name: &str) -> bool {
    // ANNEX_BENCH_ISAS=avx2,avx512 limits the per-ISA series.
    match std::env::var("ANNEX_BENCH_ISAS") {
        Ok(v) if !v.is_empty() => v.split(',').any(|s| s.trim() == name),
        _ => true,
    }
}

fn tables() -> Vec<Kernels> {
    Kernels::all_supported()
        .into_iter()
        .filter(|k| k.isa() != Isa::Scalar && filter(k.isa().name()))
        // Avx512 and Avx512Vnni share the f32 kernels; SQ8 keeps both.
        .collect()
}

fn f32_group(
    c: &mut Criterion,
    name: &str,
    annex: fn(&[f32], &[f32]) -> f32,
    table: fn(&Kernels, &[f32], &[f32]) -> f32,
    #[allow(unused_variables)] legacy: fn(&[f32], &[f32]) -> f32,
    simsimd: fn(&[f32], &[f32]) -> Option<f64>,
    autovec: fn(&[f32], &[f32]) -> f32,
) {
    let mut group = c.benchmark_group(name);
    for &dim in DIMS {
        let mut rng = Rng::new(dim as u64);
        let query = rng.vector(dim);
        let set: Vec<Vec<f32>> = (0..SET).map(|_| rng.vector(dim)).collect();
        group.throughput(Throughput::Elements(SET as u64));
        let mut run = |id: &str, f: &dyn Fn(&[f32], &[f32]) -> f32| {
            group.bench_function(BenchmarkId::new(id, dim), |b| {
                b.iter(|| {
                    let mut acc = 0.0f32;
                    for v in &set {
                        acc += f(black_box(&query), black_box(v));
                    }
                    acc
                })
            });
        };
        run("annex", &|a, b| annex(a, b));
        for k in tables() {
            if k.isa() == Isa::Avx512Vnni {
                continue;
            }
            run(&format!("annex-{}", k.isa()), &|a, b| table(&k, a, b));
        }
        #[cfg(target_arch = "x86_64")]
        run("legacy", &|a, b| legacy(a, b));
        run("simsimd", &|a, b| simsimd(a, b).unwrap() as f32);
        run("autovec", &|a, b| autovec(a, b));
        run("scalar", &|a, b| {
            table(&Kernels::for_isa(Isa::Scalar).unwrap(), a, b)
        });
    }
    group.finish();
}

#[cfg(target_arch = "x86_64")]
use annex_kernel_bench::legacy;

#[cfg(not(target_arch = "x86_64"))]
mod legacy {
    pub fn dot_product(_: &[f32], _: &[f32]) -> f32 {
        unreachable!()
    }
    pub fn l2_squared(_: &[f32], _: &[f32]) -> f32 {
        unreachable!()
    }
    pub fn screen_dot(_: &[i8], _: &[u8]) -> i32 {
        unreachable!()
    }
}

fn bench_dot(c: &mut Criterion) {
    f32_group(
        c,
        "dot_f32",
        kernels::dot,
        Kernels::dot,
        legacy::dot_product,
        f32::dot,
        dot_autovec,
    );
}

fn bench_l2(c: &mut Criterion) {
    f32_group(
        c,
        "l2sq_f32",
        kernels::l2_squared,
        Kernels::l2_squared,
        legacy::l2_squared,
        f32::l2sq,
        l2_autovec,
    );
}

fn bench_sq8(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_i8xu8");
    for &dim in DIMS {
        let mut rng = Rng::new(dim as u64 ^ 0x5a);
        let query = sq8_query(&rng.unit_vector(dim));
        let set: Vec<Vec<u8>> = (0..SET)
            .map(|_| sq8_encode(&rng.unit_vector(dim)))
            .collect();
        // SimSIMD has no u8-with-offset kernel; its i8 x i8 dot does the same
        // amount of work on the recentred codes.
        let set_i8: Vec<Vec<i8>> = set
            .iter()
            .map(|v| v.iter().map(|&b| (b ^ 0x80) as i8).collect())
            .collect();
        group.throughput(Throughput::Elements(SET as u64));
        let mut run = |id: &str, f: &dyn Fn(&[i8], &[u8]) -> i32| {
            group.bench_function(BenchmarkId::new(id, dim), |b| {
                b.iter(|| {
                    let mut acc = 0i64;
                    for v in &set {
                        acc += f(black_box(&query), black_box(v)) as i64;
                    }
                    acc
                })
            });
        };
        run("annex", &kernels::dot_i8_u8_centered);
        for k in tables() {
            run(&format!("annex-{}", k.isa()), &|q, s| {
                k.dot_i8_u8_centered(q, s)
            });
        }
        #[cfg(target_arch = "x86_64")]
        run("legacy", &legacy::screen_dot);
        run("scalar", &kernels::dot_i8_u8_centered_scalar);
        group.bench_function(BenchmarkId::new("simsimd-i8", dim), |b| {
            b.iter(|| {
                let mut acc = 0.0f64;
                for v in &set_i8 {
                    acc += i8::dot(black_box(&query), black_box(v)).unwrap();
                }
                acc
            })
        });
    }
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(1200))
        .sample_size(30)
}

criterion_group! {
    name = distance;
    config = config();
    targets = bench_dot, bench_l2, bench_sq8
}
criterion_main!(distance);
