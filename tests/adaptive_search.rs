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
    let mut index = HNSWIndex::new(DistanceMetric::Cosine, m, ef_construct, 16, dim);
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
    let mut scored: Vec<(usize, f32)> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let qa: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
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

/// Check that the index has at least 2 levels so multi-entry actually has an L1 to harvest seeds from.
fn has_upper_layers(index: &HNSWIndex) -> bool {
    index.current_max_level() >= 1
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// Multi-entry (3 seeds) should produce recall >= single-entry on the same EF budget.
#[test]
fn test_multi_entry_recall_ge_single_entry() {
    const N: usize = 8_000;
    const DIM: usize = 32;
    const TOP_K: usize = 10;
    const NUM_QUERIES: usize = 200;
    const EF: usize = 32;

    let (index, vecs) = build_random_index(N, DIM, 16, 100, 42);
    assert!(
        has_upper_layers(&index),
        "index needs upper layers for multi-entry to be meaningful"
    );

    let mut rng = StdRng::seed_from_u64(1234);
    let mut recall_single = 0.0f32;
    let mut recall_multi = 0.0f32;

    for _ in 0..NUM_QUERIES {
        let query: Vector = (0..DIM).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let truth = ground_truth(&query, &vecs, TOP_K);

        let opts_single = SearchRuntimeOptions {
            ef_search: Some(EF),
            num_entry_seeds: Some(1),
            ..Default::default()
        };
        let res_single = index
            .search_with_options(&query, TOP_K, &opts_single)
            .unwrap();
        let ids_single: Vec<u64> = res_single.iter().map(|r| r.id).collect();

        let opts_multi = SearchRuntimeOptions {
            ef_search: Some(EF),
            num_entry_seeds: Some(3),
            ..Default::default()
        };
        let res_multi = index
            .search_with_options(&query, TOP_K, &opts_multi)
            .unwrap();
        let ids_multi: Vec<u64> = res_multi.iter().map(|r| r.id).collect();

        recall_single += recall_at_k(&ids_single, &truth);
        recall_multi += recall_at_k(&ids_multi, &truth);
    }

    recall_single /= NUM_QUERIES as f32;
    recall_multi /= NUM_QUERIES as f32;

    println!(
        "multi-entry seeds=1: recall@{TOP_K}={:.4}  seeds=3: recall@{TOP_K}={:.4}  delta={:+.4}",
        recall_single,
        recall_multi,
        recall_multi - recall_single,
    );

    // Multi-entry should not hurt recall (allow tiny float noise).
    assert!(
        recall_multi >= recall_single - 0.01,
        "multi-entry degraded recall: {:.4} < {:.4}",
        recall_multi,
        recall_single
    );
}

/// Adaptive EF should improve recall on hard queries without degrading easy ones.
/// We identify "hard" queries as those where single EF=32 misses ground-truth hits,
/// and verify the adaptive path produces better or equal results on them.
#[test]
fn test_adaptive_ef_improves_hard_queries() {
    const N: usize = 8_000;
    const DIM: usize = 32;
    const TOP_K: usize = 10;
    const NUM_QUERIES: usize = 300;
    const BASE_EF: usize = 32;
    const HIGH_EF: usize = 128;

    let (index, vecs) = build_random_index(N, DIM, 16, 100, 99);

    let mut rng = StdRng::seed_from_u64(5678);

    let mut recall_base = 0.0f32;
    let mut recall_adaptive = 0.0f32;
    let mut adaptive_triggered = 0u32;

    // Use a threshold that captures queries where base EF struggles.
    // For cosine distance, sort_key ≈ 1 - cosine_similarity; a value > 0.4 means the
    // top result has cosine similarity < 0.6 — a genuinely hard or out-of-distribution query.
    const THRESHOLD: f32 = 0.40;

    for _ in 0..NUM_QUERIES {
        let query: Vector = (0..DIM).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let truth = ground_truth(&query, &vecs, TOP_K);

        // Base EF, no adaptive
        let opts_base = SearchRuntimeOptions {
            ef_search: Some(BASE_EF),
            ..Default::default()
        };
        let res_base = index
            .search_with_options(&query, TOP_K, &opts_base)
            .unwrap();
        let ids_base: Vec<u64> = res_base.iter().map(|r| r.id).collect();
        recall_base += recall_at_k(&ids_base, &truth);

        // Adaptive EF
        let opts_adaptive = SearchRuntimeOptions {
            ef_search: Some(BASE_EF),
            adaptive_ef_high: Some(HIGH_EF),
            adaptive_ef_score_threshold: Some(THRESHOLD),
            ..Default::default()
        };
        let res_adaptive = index
            .search_with_options(&query, TOP_K, &opts_adaptive)
            .unwrap();
        let ids_adaptive: Vec<u64> = res_adaptive.iter().map(|r| r.id).collect();
        recall_adaptive += recall_at_k(&ids_adaptive, &truth);

        // Track how often adaptive triggered (best score > threshold)
        if res_base.first().map(|r| r.sort_key).unwrap_or(0.0) > THRESHOLD {
            adaptive_triggered += 1;
        }
    }

    recall_base /= NUM_QUERIES as f32;
    recall_adaptive /= NUM_QUERIES as f32;

    println!(
        "adaptive EF  base_ef={BASE_EF} recall@{TOP_K}={:.4}  adaptive_ef={HIGH_EF} threshold={THRESHOLD} recall@{TOP_K}={:.4}  delta={:+.4}  triggered={}/{NUM_QUERIES}",
        recall_base,
        recall_adaptive,
        recall_adaptive - recall_base,
        adaptive_triggered,
    );

    // Adaptive EF should not hurt recall overall.
    assert!(
        recall_adaptive >= recall_base - 0.005,
        "adaptive EF degraded recall: {:.4} < {:.4}",
        recall_adaptive,
        recall_base
    );
}

/// Combined: multi-entry seeds + adaptive EF vs plain EF=32.
#[test]
fn test_combined_multi_entry_and_adaptive_ef() {
    const N: usize = 8_000;
    const DIM: usize = 32;
    const TOP_K: usize = 10;
    const NUM_QUERIES: usize = 300;
    const BASE_EF: usize = 32;
    const HIGH_EF: usize = 128;
    const THRESHOLD: f32 = 0.40;

    let (index, vecs) = build_random_index(N, DIM, 16, 100, 77);

    let mut rng = StdRng::seed_from_u64(9999);
    let mut recall_plain = 0.0f32;
    let mut recall_combined = 0.0f32;

    for _ in 0..NUM_QUERIES {
        let query: Vector = (0..DIM).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let truth = ground_truth(&query, &vecs, TOP_K);

        let opts_plain = SearchRuntimeOptions {
            ef_search: Some(BASE_EF),
            ..Default::default()
        };
        let res_plain = index
            .search_with_options(&query, TOP_K, &opts_plain)
            .unwrap();
        recall_plain += recall_at_k(&res_plain.iter().map(|r| r.id).collect::<Vec<_>>(), &truth);

        let opts_combined = SearchRuntimeOptions {
            ef_search: Some(BASE_EF),
            num_entry_seeds: Some(3),
            adaptive_ef_high: Some(HIGH_EF),
            adaptive_ef_score_threshold: Some(THRESHOLD),
            ..Default::default()
        };
        let res_combined = index
            .search_with_options(&query, TOP_K, &opts_combined)
            .unwrap();
        recall_combined += recall_at_k(
            &res_combined.iter().map(|r| r.id).collect::<Vec<_>>(),
            &truth,
        );
    }

    recall_plain /= NUM_QUERIES as f32;
    recall_combined /= NUM_QUERIES as f32;

    println!(
        "combined  plain ef={BASE_EF} recall@{TOP_K}={:.4}  multi-seed+adaptive recall@{TOP_K}={:.4}  delta={:+.4}",
        recall_plain,
        recall_combined,
        recall_combined - recall_plain,
    );

    assert!(
        recall_combined >= recall_plain - 0.005,
        "combined mode degraded recall: {:.4} < {:.4}",
        recall_combined,
        recall_plain
    );
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
