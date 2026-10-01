//! ColBERT MaxSim (`sum_q max_d q . d`): ANNex packed kernel (per call and
//! with a prepared query), the legacy scalar x86 path, SimSIMD pairwise dots,
//! and GEMM + row-max through ndarray/matrixmultiply and (with
//! `--features blas`) single-threaded OpenBLAS.
//!
//! Throughput counts 2 * nq * nd * dim flops per score.
//! Commands and reporting rules: docs/benchmarks.md.

use std::hint::black_box;
use std::time::Duration;

use annex_kernel_bench::{Rng, unit_matrix};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use multivector::{MaxSimQuery, maxsim_flat};
use ndarray::{Array2, ArrayView2, linalg::general_mat_mul};
use simsimd::SpatialSimilarity;

/// (dim, doc tokens, query tokens). ColBERTv2 is 128-d with 32 query tokens;
/// 96 is answerai-colbert-small; 384 covers E5/MiniLM token embeddings.
const SHAPES: &[(usize, usize, usize)] = &[
    (128, 200, 32),
    (128, 100, 32),
    (128, 300, 32),
    (96, 200, 32),
    (384, 200, 32),
    (128, 200, 8),
];

fn row_max_sum(scores: &[f32], nd: usize) -> f32 {
    scores
        .chunks_exact(nd)
        .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .sum()
}

fn bench_maxsim(c: &mut Criterion) {
    #[cfg(feature = "blas")]
    annex_kernel_bench::blas::single_threaded();
    let mut group = c.benchmark_group("maxsim");
    for &(dim, nd, nq) in SHAPES {
        let mut rng = Rng::new((dim * 1000 + nd + nq) as u64);
        let query = unit_matrix(&mut rng, nq, dim);
        let doc_rows = unit_matrix(&mut rng, nd, dim);
        let doc: Vec<f32> = doc_rows.concat();
        let query_flat: Vec<f32> = query.concat();
        let reference = maxsim_flat(&query, &doc, dim);
        let label = format!("d{dim}/n{nd}/q{nq}");
        group.throughput(Throughput::Elements((2 * nq * nd * dim) as u64));

        let check = |name: &str, got: f32| {
            assert!(
                (got - reference).abs() < 1e-3,
                "{name} {label}: {got} vs {reference}"
            );
        };

        group.bench_function(BenchmarkId::new("annex", &label), |b| {
            b.iter(|| maxsim_flat(black_box(&query), black_box(&doc), dim))
        });

        let prepared = MaxSimQuery::new(&query, dim);
        check("prepared", prepared.score(&doc, dim));
        group.bench_function(BenchmarkId::new("annex-prepared", &label), |b| {
            b.iter(|| prepared.score(black_box(&doc), dim))
        });

        #[cfg(target_arch = "x86_64")]
        {
            use annex_kernel_bench::legacy;
            check("legacy", legacy::maxsim_flat(&query, &doc, dim));
            group.bench_function(BenchmarkId::new("legacy", &label), |b| {
                b.iter(|| legacy::maxsim_flat(black_box(&query), black_box(&doc), dim))
            });
        }

        let simsimd = |query: &[Vec<f32>], doc: &[f32]| -> f32 {
            query
                .iter()
                .map(|q| {
                    doc.chunks_exact(dim)
                        .map(|d| f32::dot(q, d).unwrap() as f32)
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .sum()
        };
        check("simsimd", simsimd(&query, &doc));
        group.bench_function(BenchmarkId::new("simsimd", &label), |b| {
            b.iter(|| simsimd(black_box(&query), black_box(&doc)))
        });

        let q_view = ArrayView2::from_shape((nq, dim), &query_flat).unwrap();
        let mut out = Array2::<f32>::zeros((nq, nd));
        let ndarray_gemm = |doc: &[f32], out: &mut Array2<f32>| -> f32 {
            let d_view = ArrayView2::from_shape((nd, dim), doc).unwrap();
            general_mat_mul(1.0, &q_view, &d_view.t(), 0.0, out);
            row_max_sum(out.as_slice().unwrap(), nd)
        };
        check("ndarray", ndarray_gemm(&doc, &mut out));
        group.bench_function(BenchmarkId::new("ndarray-gemm", &label), |b| {
            b.iter(|| ndarray_gemm(black_box(&doc), &mut out))
        });

        #[cfg(feature = "blas")]
        {
            use annex_kernel_bench::blas::sgemm_nt;
            let mut scores = vec![0.0f32; nq * nd];
            let mut run = |doc: &[f32]| {
                sgemm_nt(&query_flat, doc, nq, nd, dim, &mut scores);
                row_max_sum(&scores, nd)
            };
            check("openblas", run(&doc));
            group.bench_function(BenchmarkId::new("openblas-gemm", &label), |b| {
                b.iter(|| run(black_box(&doc)))
            });
        }
    }
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(30)
}

criterion_group! {
    name = maxsim;
    config = config();
    targets = bench_maxsim
}
criterion_main!(maxsim);
