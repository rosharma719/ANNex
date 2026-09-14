//! Opt-in, warmed NYT search benchmark. Recall and counters are measured outside
//! the timed production search path. See docs/recall-investigation.md.
use std::{env, fs, hint::black_box, time::Instant};

use annex::{
    segment::Segment,
    utils::types::Vector,
    vector::hnsw::{HNSWIndex, SearchRuntimeOptions},
};
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde_json::json;

fn setting(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(default)
}

#[test]
#[ignore]
fn nytimes_warmed_frontier() {
    let path = env::var("VECTORDB_NYT_PERSIST_PATH").unwrap_or_else(|_| {
        "data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin".into()
    });
    let segment = Segment::load_from_path(&path).unwrap();
    let mut index = HNSWIndex::from_snapshot(segment.hnsw().to_snapshot());
    drop(segment);
    // Experimental offline ordering only. Mutable insertion requires distance-sorted
    // adjacency, so do not persist or expose this ordering as a mutable index.
    let order = env::var("VECTORDB_FRONTIER_ORDER").unwrap_or_else(|_| "original".into());
    if order == "diverse" || order == "pruned" {
        let start = Instant::now();
        let mut snapshot = index.to_snapshot();
        let vectors = &snapshot.vectors;
        for (&id, neighbors) in snapshot.layers.get_mut(&0).unwrap() {
            let mut chosen = Vec::new();
            let mut positions = Vec::new();
            for (position, &candidate) in neighbors.iter().enumerate() {
                if candidate == id || chosen.len() == 32 {
                    continue;
                }
                let qdist = cosine_distance(&vectors[&id], &vectors[&candidate]);
                if chosen
                    .iter()
                    .all(|r| cosine_distance(&vectors[r], &vectors[&candidate]) >= qdist)
                {
                    chosen.push(candidate);
                    positions.push(position);
                }
            }
            let tail: Vec<_> = neighbors
                .iter()
                .enumerate()
                .filter(|(position, _)| !positions.contains(position))
                .map(|(_, &id)| id)
                .collect();
            if order == "diverse" {
                chosen.extend(tail);
                assert_eq!(chosen.len(), neighbors.len());
            }
            *neighbors = chosen;
        }
        index = HNSWIndex::from_snapshot(snapshot);
        eprintln!("{order} graph preparation took {:?}", start.elapsed());
    } else {
        assert_eq!(order, "original");
    }
    let queries: Array2<f32> = read_npy("data/nytimes-256-angular/queries.npy").unwrap();
    let truth: Vec<Vec<u64>> =
        serde_json::from_slice(&fs::read("data/nytimes-256-angular/ground_truth.json").unwrap())
            .unwrap();
    let offset = setting("VECTORDB_FRONTIER_OFFSET", 0);
    let count = setting("VECTORDB_QUERIES", 1000);
    assert!(count > 0);
    assert!(offset + count <= queries.nrows());
    let queries: Vec<Vector> = queries
        .rows()
        .into_iter()
        .skip(offset)
        .take(count)
        .map(|r| r.to_vec())
        .collect();
    let rounds = setting("VECTORDB_FRONTIER_ROUNDS", 3);
    let seeds = setting("VECTORDB_NUM_ENTRY_SEEDS", 1);
    let scans: Vec<usize> = env::var("VECTORDB_FRONTIER_SCANS")
        .unwrap_or_else(|_| "32,64,128".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let efs: Vec<usize> = env::var("VECTORDB_EF_SEARCH_LIST")
        .unwrap_or_else(|_| "32,64,128,256,512".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    for scan in scans {
        for &ef in &efs {
            let opts = SearchRuntimeOptions {
                ef_search: Some(ef),
                neighbor_scan_cap_level0: Some(scan),
                num_entry_seeds: Some(seeds),
                ..Default::default()
            };
            let mut hits = 0;
            let mut visited = 0;
            let mut distances = 0;
            // Untimed warm-up, recall, counters and path equivalence.
            for (i, q) in queries.iter().enumerate() {
                let (results, stats) = index.search_with_stats_with_options(q, 20, &opts).unwrap();
                let plain = index.search_with_options(q, 20, &opts).unwrap();
                assert_eq!(
                    results.iter().map(|r| r.id).collect::<Vec<_>>(),
                    plain.iter().map(|r| r.id).collect::<Vec<_>>()
                );
                hits += results
                    .iter()
                    .filter(|r| truth[offset + i][..20].contains(&r.id))
                    .count();
                visited += stats.visited;
                distances += stats.distance_computations;
            }
            for round in 0..rounds {
                let mut times = Vec::with_capacity(count);
                let start = Instant::now();
                for q in &queries {
                    let query_start = Instant::now();
                    black_box(index.search_with_options(black_box(q), 20, &opts).unwrap());
                    times.push(query_start.elapsed().as_secs_f64() * 1000.0);
                }
                let elapsed = start.elapsed().as_secs_f64();
                times.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    json!({"snapshot": path, "order": order, "query_offset":offset, "queries":count, "round":round, "scan":scan, "ef":ef, "seeds":seeds,
                    "recall":hits as f64/(count*20) as f64, "qps":count as f64/elapsed, "mean_ms":elapsed*1000.0/count as f64,
                    "p50_ms": times[count/2], "p99_ms":times[(count*99/100).min(count-1)], "mean_visited": visited as f64/count as f64, "mean_distances":distances as f64/count as f64})
                );
            }
        }
    }
}

fn load_index(path: &str) -> HNSWIndex {
    use annex::segment::Segment;
    let segment = Segment::load_from_path(path).unwrap();
    let index = HNSWIndex::from_snapshot(segment.hnsw().to_snapshot());
    drop(segment);
    index
}

#[test]
#[ignore]
fn nytimes_calibrate_patience() {
    let path = std::env::var("VECTORDB_NYT_PERSIST_PATH").unwrap_or_else(|_| {
        "data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin".into()
    });
    let index = load_index(&path);
    let queries: ndarray::Array2<f32> =
        ndarray_npy::read_npy("data/nytimes-256-angular/queries.npy").unwrap();
    let truth: Vec<Vec<u64>> = serde_json::from_slice(
        &std::fs::read("data/nytimes-256-angular/ground_truth.json").unwrap(),
    )
    .unwrap();
    let count = 1000usize;
    let queries: Vec<annex::utils::types::Vector> = queries
        .rows()
        .into_iter()
        .take(count)
        .map(|r| r.to_vec())
        .collect();

    for early_patience in [0usize, 1, 2, 3, 5, 8] {
        for scan_patience in [0usize, 2, 4, 8] {
            for ef in [32usize, 64, 128, 256] {
                let opts = SearchRuntimeOptions {
                    ef_search: Some(ef),
                    early_exit_patience: Some(early_patience),
                    neighbor_scan_patience: Some(scan_patience),
                    ..Default::default()
                };
                let mut hits = 0usize;
                for (i, q) in queries.iter().enumerate() {
                    let results = index.search_with_options(q, 20, &opts).unwrap();
                    hits += results
                        .iter()
                        .filter(|r| truth[i][..20].contains(&r.id))
                        .count();
                }
                let recall = hits as f64 / (count * 20) as f64;
                let start = std::time::Instant::now();
                for q in &queries {
                    std::hint::black_box(
                        index
                            .search_with_options(std::hint::black_box(q), 20, &opts)
                            .unwrap(),
                    );
                }
                let qps = count as f64 / start.elapsed().as_secs_f64();
                println!(
                    "{}",
                    serde_json::json!({
                        "experiment": "calibrate_patience",
                        "ef": ef,
                        "early_patience": early_patience,
                        "scan_patience": scan_patience,
                        "recall": recall,
                        "qps": qps
                    })
                );
            }
        }
    }
}

#[test]
#[ignore]
fn nytimes_calibrate_adaptive_ef() {
    let path = std::env::var("VECTORDB_NYT_PERSIST_PATH").unwrap_or_else(|_| {
        "data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin".into()
    });
    let index = load_index(&path);
    let queries: ndarray::Array2<f32> =
        ndarray_npy::read_npy("data/nytimes-256-angular/queries.npy").unwrap();
    let truth: Vec<Vec<u64>> = serde_json::from_slice(
        &std::fs::read("data/nytimes-256-angular/ground_truth.json").unwrap(),
    )
    .unwrap();
    let count = 1000usize;
    let queries: Vec<annex::utils::types::Vector> = queries
        .rows()
        .into_iter()
        .take(count)
        .map(|r| r.to_vec())
        .collect();

    // Sweep thresholds: distance values where we retry with higher ef.
    // For cosine, sort_key is 1 - similarity; 0.05 = query has similarity < 0.95 to top result.
    for base_ef in [64usize, 128] {
        for high_ef in [base_ef * 2, base_ef * 4] {
            for threshold in [0.03f32, 0.05, 0.08, 0.12] {
                let opts = SearchRuntimeOptions {
                    ef_search: Some(base_ef),
                    adaptive_ef_high: Some(high_ef),
                    adaptive_ef_score_threshold: Some(threshold),
                    ..Default::default()
                };
                let mut hits = 0usize;
                let start = std::time::Instant::now();
                for (i, q) in queries.iter().enumerate() {
                    let results = index.search_with_options(q, 20, &opts).unwrap();
                    hits += results
                        .iter()
                        .filter(|r| truth[i][..20].contains(&r.id))
                        .count();
                }
                let qps = count as f64 / start.elapsed().as_secs_f64();
                let recall = hits as f64 / (count * 20) as f64;
                println!(
                    "{}",
                    serde_json::json!({
                        "experiment": "calibrate_adaptive_ef",
                        "base_ef": base_ef,
                        "high_ef": high_ef,
                        "threshold": threshold,
                        "recall": recall,
                        "qps": qps
                    })
                );
            }
        }
    }
}

// Snapshot cosine vectors are normalized. Four independent sums keep the
// offline experiment affordable without depending on private SIMD kernels.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0; 4];
    for (aa, bb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for i in 0..4 {
            sums[i] += aa[i] * bb[i];
        }
    }
    assert_eq!(a.len() % 4, 0);
    1.0 - sums.into_iter().sum::<f32>()
}
