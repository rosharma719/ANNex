use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;
use annex::vector::hnsw::SearchRuntimeOptions;
use rand::rngs::StdRng;
/// Tests for multi-entry L0 seeds and adaptive EF routing.
///
/// These tests use synthetic data to measure recall improvement without
/// requiring any external dataset.
use rand::{Rng, SeedableRng};

fn build_random_index(
    n: usize,
    dim: usize,
    m: usize,
    ef_construct: usize,
    seed: u64,
) -> (HNSWIndex, Vec<Vector>) {
    let mut rng = StdRng::seed_from_u64(seed);
    // ef_construct doubles as the ef parameter at construction time.
    let index = HNSWIndex::new(DistanceMetric::Cosine, m, ef_construct, 16, dim);
    let mut vecs: Vec<Vector> = Vec::with_capacity(n);
    for i in 0..n {
        let v: Vector = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        index.insert(i as u64, v.clone()).unwrap();
        vecs.push(v);
    }
    (index, vecs)
}

/// Brute-force exact top-k for recall computation.
fn ground_truth(query: &[f32], vecs: &[Vector], k: usize) -> Vec<usize> {
    let qa = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut scored: Vec<(usize, f32)> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let va: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos_dist = 1.0 - dot / (qa * va + 1e-9);
            (i, cos_dist)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_k(results: &[u64], truth: &[usize]) -> f32 {
    let found = results
        .iter()
        .filter(|&&id| truth.contains(&(id as usize)))
        .count();
    found as f32 / truth.len() as f32
}

/// Verify adaptive dispatch against explicit fixed-budget searches. Recall
/// frontiers belong in the opt-in benchmarks below, not probabilistic CI gates.
#[test]
fn adaptive_routing_matches_explicit_search_budgets() {
    let (index, _) = build_random_index(512, 16, 16, 64, 42);
    assert!(index.current_max_level() >= 1);
    let mut rng = StdRng::seed_from_u64(1234);
    for seeds in [1, 3] {
        for _ in 0..8 {
            let query: Vector = (0..16).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let search = |ef, adaptive, threshold| {
                let options = SearchRuntimeOptions {
                    ef_search: Some(ef),
                    num_entry_seeds: Some(seeds),
                    adaptive_ef_high: Some(if adaptive { 128 } else { ef }),
                    adaptive_ef_score_threshold: Some(threshold),
                    ..Default::default()
                };
                index
                    .search_with_options(&query, 10, &options)
                    .unwrap()
                    .into_iter()
                    .map(|hit| (hit.id, hit.raw_score))
                    .collect::<Vec<_>>()
            };
            let low = search(16, false, 3.0);
            let high = search(128, false, 3.0);
            assert_eq!(
                search(16, true, -1.0),
                high,
                "forced retry must execute high EF"
            );
            assert_eq!(
                search(16, true, 3.0),
                low,
                "disabled retry must retain low EF"
            );
            assert_eq!(high.len(), 10);
            assert_eq!(
                high.iter()
                    .map(|hit| hit.0)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                10
            );
        }
    }
}

/// Sweep seeds=1,2,3,4 at EF=32 and print recall@10 + latency to show the curve.
#[test]
#[ignore]
fn bench_multi_entry_seed_sweep() {
    use std::time::Instant;
    const N: usize = 10_000;
    const DIM: usize = 64;
    const TOP_K: usize = 10;
    const NUM_QUERIES: usize = 500;
    const EF: usize = 32;

    let (index, vecs) = build_random_index(N, DIM, 16, 100, 42);
    println!(
        "N={N} dim={DIM} ef_construct=100 M=16 max_level={}",
        index.current_max_level()
    );

    let mut rng = StdRng::seed_from_u64(1111);
    let queries: Vec<Vector> = (0..NUM_QUERIES)
        .map(|_| (0..DIM).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let truths: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| ground_truth(q, &vecs, TOP_K))
        .collect();

    println!(
        "\n{:<8} {:<6} {:<12} {:<12}",
        "seeds", "ef", "recall@10", "avg_ms"
    );
    println!("{}", "-".repeat(42));
    for seeds in [1usize, 2, 3, 4] {
        let mut recall_sum = 0.0f32;
        let mut elapsed_us = 0u128;
        for (q, truth) in queries.iter().zip(truths.iter()) {
            let opts = SearchRuntimeOptions {
                ef_search: Some(EF),
                num_entry_seeds: Some(seeds),
                ..Default::default()
            };
            let t0 = Instant::now();
            let res = index.search_with_options(q, TOP_K, &opts).unwrap();
            elapsed_us += t0.elapsed().as_micros();
            let ids: Vec<u64> = res.iter().map(|r| r.id).collect();
            recall_sum += recall_at_k(&ids, truth);
        }
        let recall = recall_sum / NUM_QUERIES as f32;
        let avg_ms = elapsed_us as f64 / NUM_QUERIES as f64 / 1000.0;
        println!("{:<8} {:<6} {:<12.4} {:<12.3}", seeds, EF, recall, avg_ms);
    }
}

/// Sweep adaptive EF thresholds to show how recall, latency, and trigger-rate trade off.
#[test]
#[ignore]
fn bench_adaptive_ef_threshold_sweep() {
    use std::time::Instant;
    const N: usize = 10_000;
    const DIM: usize = 64;
    const TOP_K: usize = 10;
    const NUM_QUERIES: usize = 500;
    const BASE_EF: usize = 32;
    const HIGH_EF: usize = 128;

    let (index, vecs) = build_random_index(N, DIM, 16, 100, 42);
    println!("N={N} dim={DIM} ef_construct=100 M=16");

    let mut rng = StdRng::seed_from_u64(2222);
    let queries: Vec<Vector> = (0..NUM_QUERIES)
        .map(|_| (0..DIM).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let truths: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| ground_truth(q, &vecs, TOP_K))
        .collect();

    // Baseline: plain EF=32 (no adaptive)
    let mut base_recall = 0.0f32;
    let mut base_us = 0u128;
    for (q, truth) in queries.iter().zip(truths.iter()) {
        let opts = SearchRuntimeOptions {
            ef_search: Some(BASE_EF),
            ..Default::default()
        };
        let t0 = Instant::now();
        let res = index.search_with_options(q, TOP_K, &opts).unwrap();
        base_us += t0.elapsed().as_micros();
        base_recall += recall_at_k(&res.iter().map(|r| r.id).collect::<Vec<_>>(), truth);
    }
    base_recall /= NUM_QUERIES as f32;
    let base_ms = base_us as f64 / NUM_QUERIES as f64 / 1000.0;

    // Upper bound: plain EF=128 (no adaptive)
    let mut ceil_recall = 0.0f32;
    let mut ceil_us = 0u128;
    for (q, truth) in queries.iter().zip(truths.iter()) {
        let opts = SearchRuntimeOptions {
            ef_search: Some(HIGH_EF),
            ..Default::default()
        };
        let t0 = Instant::now();
        let res = index.search_with_options(q, TOP_K, &opts).unwrap();
        ceil_us += t0.elapsed().as_micros();
        ceil_recall += recall_at_k(&res.iter().map(|r| r.id).collect::<Vec<_>>(), truth);
    }
    ceil_recall /= NUM_QUERIES as f32;
    let ceil_ms = ceil_us as f64 / NUM_QUERIES as f64 / 1000.0;

    println!(
        "\n{:<14} {:<12} {:<10} {:<10} {:<16}",
        "config", "recall@10", "Δrecall", "avg_ms", "triggered"
    );
    println!("{}", "-".repeat(64));
    println!(
        "{:<14} {:<12.4} {:<10} {:<10.3} {:<16}",
        format!("ef={BASE_EF}"),
        base_recall,
        "—",
        base_ms,
        "—"
    );

    for threshold in [0.20f32, 0.30, 0.40, 0.50, 0.60] {
        let mut recall_sum = 0.0f32;
        let mut elapsed_us = 0u128;
        let mut triggered = 0u32;
        for (q, truth) in queries.iter().zip(truths.iter()) {
            let opts = SearchRuntimeOptions {
                ef_search: Some(BASE_EF),
                adaptive_ef_high: Some(HIGH_EF),
                adaptive_ef_score_threshold: Some(threshold),
                ..Default::default()
            };
            let t0 = Instant::now();
            let res = index.search_with_options(q, TOP_K, &opts).unwrap();
            elapsed_us += t0.elapsed().as_micros();
            recall_sum += recall_at_k(&res.iter().map(|r| r.id).collect::<Vec<_>>(), truth);
            if res.first().map(|r| r.sort_key).unwrap_or(0.0) > threshold {
                triggered += 1;
            }
        }
        let recall = recall_sum / NUM_QUERIES as f32;
        let avg_ms = elapsed_us as f64 / NUM_QUERIES as f64 / 1000.0;
        println!(
            "{:<14} {:<12.4} {:<+10.4} {:<10.3} {:<16}",
            format!("t={threshold:.2}"),
            recall,
            recall - base_recall,
            avg_ms,
            format!("{triggered}/{NUM_QUERIES}"),
        );
    }
    println!(
        "{:<14} {:<12.4} {:<+10.4} {:<10.3} {:<16}",
        format!("ef={HIGH_EF}"),
        ceil_recall,
        ceil_recall - base_recall,
        ceil_ms,
        "all"
    );
}
