use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use annex::segment::segment::Segment;
use annex::utils::types::Vector;
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde_json::from_slice;

#[derive(Clone, Copy)]
struct BenchSummary {
    ef_search: usize,
    qps: f64,
    avg_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    p99_ms: f64,
    recall: f64,
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.replace('_', "").parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_usize_list(name: &str, default: &[usize]) -> Vec<usize> {
    env::var(name)
        .ok()
        .and_then(|raw| {
            let values: Vec<usize> = raw
                .split(',')
                .filter_map(|part| {
                    let trimmed = part.trim().replace('_', "");
                    if trimmed.is_empty() {
                        None
                    } else {
                        trimmed.parse::<usize>().ok()
                    }
                })
                .collect();
            if values.is_empty() {
                None
            } else {
                Some(values)
            }
        })
        .unwrap_or_else(|| default.to_vec())
}

fn env_path(name: &str, default: &str) -> PathBuf {
    env::var(name)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

fn load_vectors(path: &PathBuf) -> Vec<Vector> {
    let arr: Array2<f32> = read_npy(path).expect("failed to read .npy");
    arr.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn load_ground_truth(path: &PathBuf) -> Vec<Vec<u64>> {
    let data = fs::read(path).expect("failed to read ground truth json");
    let raw: Vec<Vec<usize>> = from_slice(&data).expect("failed to parse ground truth json");
    raw.into_iter()
        .map(|ids| ids.into_iter().map(|id| id as u64).collect())
        .collect()
}

fn percentile_ms(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

fn default_nyt_snapshot_path() -> String {
    let m = env_usize("VECTORDB_M", 16);
    let m0 = env_usize("VECTORDB_M0", m * 2);
    let stored = env_usize("VECTORDB_STORED_CAP_L0", m0);
    let efc = env_usize("VECTORDB_EF_CONSTRUCT", 300);
    format!(
        "data/nytimes-256-angular/index_m{}_m0_{}_stored_{}_efc{}.bin",
        m, m0, stored, efc
    )
}

fn main() {
    let snapshot = env_path("VECTORDB_NYT_PERSIST_PATH", &default_nyt_snapshot_path());
    let queries_path = env_path(
        "VECTORDB_NYT_QUERIES",
        "data/nytimes-256-angular/queries.npy",
    );
    let truth_path = env_path(
        "VECTORDB_NYT_GROUND_TRUTH",
        "data/nytimes-256-angular/ground_truth.json",
    );
    let top_k = env_usize("VECTORDB_TOPK", 20);
    let ef_values = env_usize_list("VECTORDB_EF_SEARCH_LIST", &[32, 64, 128, 256, 512]);
    let warmup = env_usize("VECTORDB_WARMUP_QUERIES", 100);
    let num_queries = env_usize("VECTORDB_QUERIES", 1000);

    let mut segment =
        Segment::load_from_path(&snapshot).expect("failed to load persisted NYTimes segment");

    let queries = load_vectors(&queries_path);
    let ground_truth = load_ground_truth(&truth_path);
    assert_eq!(
        queries.len(),
        ground_truth.len(),
        "ground truth length must match queries"
    );
    let warmup_count = warmup.min(queries.len());
    let num_queries = num_queries.min(queries.len());

    println!("snapshot={}", snapshot.display());
    println!("queries_path={}", queries_path.display());
    println!("ground_truth_path={}", truth_path.display());
    println!("top_k={}", top_k);
    println!("warmup_queries={}", warmup_count);
    println!("measured_queries={}", num_queries);
    println!("ef_search_list={:?}", ef_values);

    let mut summaries = Vec::with_capacity(ef_values.len());
    for ef in ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);

        for q in queries.iter().take(warmup_count) {
            let _ = segment.search(q, top_k).expect("warmup search failed");
        }

        let mut per_query_ms = Vec::with_capacity(num_queries);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let start = Instant::now();
        for (query, truth) in queries.iter().zip(&ground_truth).take(num_queries) {
            let q_start = Instant::now();
            let results = segment.search(query, top_k).expect("search failed");
            per_query_ms.push(q_start.elapsed().as_secs_f64() * 1000.0);
            assert!(!results.is_empty(), "search returned empty results");
            let truth_k = truth.len().min(top_k);
            total_targets += truth_k;
            hits += results
                .iter()
                .filter(|result| truth.iter().take(truth_k).any(|&id| id == result.id))
                .count();
        }
        let total = start.elapsed().as_secs_f64();
        let avg_ms = per_query_ms.iter().sum::<f64>() / per_query_ms.len().max(1) as f64;
        let qps = per_query_ms.len() as f64 / total.max(f64::MIN_POSITIVE);
        let recall = hits as f64 / total_targets.max(1) as f64;
        let summary = BenchSummary {
            ef_search,
            qps,
            avg_ms,
            p50_ms: percentile_ms(&per_query_ms, 50.0),
            p90_ms: percentile_ms(&per_query_ms, 90.0),
            p99_ms: percentile_ms(&per_query_ms, 99.0),
            recall,
        };
        println!(
            "ef_search={} qps={:.1} avg_ms={:.3} p50_ms={:.3} p90_ms={:.3} p99_ms={:.3} recall={:.3}",
            summary.ef_search,
            summary.qps,
            summary.avg_ms,
            summary.p50_ms,
            summary.p90_ms,
            summary.p99_ms,
            summary.recall
        );
        summaries.push(summary);
    }

    println!();
    println!("Summary (ef_search -> qps, ms/query, recall):");
    for summary in summaries {
        println!(
            "  {} -> {:.1} qps, {:.3} ms/query, {:.3} recall",
            summary.ef_search, summary.qps, summary.avg_ms, summary.recall
        );
    }
}
