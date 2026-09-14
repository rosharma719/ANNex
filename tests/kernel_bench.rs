/// Microbenchmark for the distance kernel.
/// Measures raw dot-product throughput by running many searches on a small index
/// and timing only the search phase.
///
/// Run with:
///   cargo test --release --test kernel_bench -- --ignored --nocapture
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;
use annex::vector::hnsw::SearchRuntimeOptions;

#[test]
#[ignore]
fn bench_dot_kernel_throughput() {
    const DIM: usize = 256;
    const N: usize = 50_000;
    const NUM_QUERIES: usize = 2_000;
    const TOP_K: usize = 10;
    const EF: usize = 64;
    const ROUNDS: usize = 3;

    let mut rng = StdRng::seed_from_u64(42);
    let mut index = HNSWIndex::new(DistanceMetric::Cosine, 16, 64, 16, DIM);

    // Normalize vectors (mimic NYT cosine workflow).
    let vecs: Vec<Vector> = (0..N)
        .map(|_| {
            let v: Vec<f32> = (0..DIM)
                .map(|_| rand::Rng::random::<f32>(&mut rng) * 2.0 - 1.0)
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect();

    println!("Building index ({N} × {DIM}-d cosine) ...");
    let t_build = Instant::now();
    for (i, v) in vecs.iter().enumerate() {
        index.insert(i as u64, v.clone()).unwrap();
    }
    println!("  build: {:.2}s", t_build.elapsed().as_secs_f64());

    let queries: Vec<Vector> = (0..NUM_QUERIES)
        .map(|_| {
            let v: Vec<f32> = (0..DIM)
                .map(|_| rand::Rng::random::<f32>(&mut rng) * 2.0 - 1.0)
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect();

    let opts = SearchRuntimeOptions {
        ef_search: Some(EF),
        ..Default::default()
    };

    // Warm-up round.
    for q in &queries {
        let _ = index.search_with_options(q, TOP_K, &opts).unwrap();
    }

    println!("\nSearching ({NUM_QUERIES} queries × {ROUNDS} rounds, ef={EF}, top_k={TOP_K}):");

    let mut best_qps = 0.0f64;
    for round in 1..=ROUNDS {
        let t = Instant::now();
        for q in &queries {
            let _ = index.search_with_options(q, TOP_K, &opts).unwrap();
        }
        let elapsed = t.elapsed().as_secs_f64();
        let qps = NUM_QUERIES as f64 / elapsed;
        let avg_us = elapsed * 1e6 / NUM_QUERIES as f64;
        println!("  round {round}: {qps:.0} QPS  {avg_us:.1} µs/query");
        if qps > best_qps {
            best_qps = qps;
        }
    }
    println!("  best: {best_qps:.0} QPS");

    // Also estimate distance calls/second from QPS × avg_visited.
    println!("\nNote: at ef=64 on {N} vectors, each query visits ~1000-2000 nodes.");
    println!(
        "Distance calls/sec ≈ QPS × ~1500 ≈ {:.0}M/s",
        best_qps * 1500.0 / 1e6
    );
}
