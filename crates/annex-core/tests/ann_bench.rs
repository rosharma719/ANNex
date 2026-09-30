//! Generic ANN benchmark: build any dataset snapshot and sweep ef_search values.
//!
//! Tests:
//!   ann_build_snapshot  — build and persist an HNSW index from base.npy
//!   ann_pareto_sweep    — sweep ef_search, emit one JSON line per (config, ef)
//!
//! Required env vars for both tests:
//!   ANNEX_BENCH_DATA_DIR   — directory containing base.npy / queries.npy / ground_truth.json
//!   VECTORDB_PERSIST_PATH  — snapshot file path (output for build, input for sweep)
//!
//! Build knobs (ann_build_snapshot):
//!   ANNEX_BENCH_METRIC     — euclidean | cosine | dot  (default: cosine)
//!   VECTORDB_M             — HNSW M parameter           (default: 16)
//!   VECTORDB_M0            — HNSW M0 parameter          (default: 2×M)
//!   VECTORDB_EF_CONSTRUCT  — ef_construct               (default: 300)
//!
//! Sweep knobs (ann_pareto_sweep):
//!   ANNEX_BENCH_TOPK       — k for recall@k             (default: 10)
//!   ANNEX_BENCH_QUERIES    — queries per round           (default: 1000)
//!   ANNEX_BENCH_ROUNDS     — timing rounds               (default: 3)
//!   VECTORDB_EF_SEARCH_LIST — comma-separated ef values  (default: 32,64,128,256,512)
//!   ANNEX_BENCH_LABEL      — "config" field in JSON      (default: "annexdb")
//!   ANNEX_BENCH_RCM        — "true" to apply RCM         (default: false)
//!   ANNEX_BENCH_SQ8        — "true" to apply SQ8         (default: false)

use std::{env, fs, hint::black_box, time::Instant};

use annex::{
    segment::Segment,
    utils::types::{DistanceMetric, Vector},
    vector::hnsw::{HNSWIndex, SearchRuntimeOptions},
};
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde_json::json;

fn env_str(key: &str) -> Option<String> {
    env::var(key).ok().filter(|s| !s.is_empty())
}

fn env_usize(key: &str, default: usize) -> usize {
    env_str(key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn parse_metric() -> DistanceMetric {
    match env_str("ANNEX_BENCH_METRIC").as_deref() {
        Some(s) if s.eq_ignore_ascii_case("euclidean") => DistanceMetric::Euclidean,
        Some(s) if s.eq_ignore_ascii_case("dot") => DistanceMetric::Dot,
        _ => DistanceMetric::Cosine,
    }
}

fn parse_efs() -> Vec<usize> {
    env_str("VECTORDB_EF_SEARCH_LIST")
        .unwrap_or_else(|| "32,64,128,256,512".into())
        .split(',')
        .map(|v| v.trim().parse().expect("invalid ef value"))
        .collect()
}

/// Build and persist an HNSW index from ANNEX_BENCH_DATA_DIR/base.npy.
#[test]
#[ignore]
fn ann_build_snapshot() {
    let data_dir = env_str("ANNEX_BENCH_DATA_DIR").expect("ANNEX_BENCH_DATA_DIR is required");
    let snap_path = env_str("VECTORDB_PERSIST_PATH").expect("VECTORDB_PERSIST_PATH is required");
    let metric = parse_metric();
    let m = env_usize("VECTORDB_M", 16);
    let m0 = env_usize("VECTORDB_M0", m * 2);
    let stored_cap_l0 = env_usize("VECTORDB_STORED_CAP_L0", m0);
    let max_level = env_usize("VECTORDB_MAX_LEVEL", 16);
    let ef_construct = env_usize("VECTORDB_EF_CONSTRUCT", 300);

    let base: Array2<f32> =
        read_npy(format!("{data_dir}/base.npy")).expect("failed to read base.npy");
    let dim = base.ncols();
    let n = base.nrows();
    eprintln!(
        "ann_build_snapshot: {n}×{dim} {metric:?} M={m} M0={m0} ef_construct={ef_construct}"
    );

    let mut segment = Segment::new(HNSWIndex::new(metric, m, 64, max_level, dim));
    segment.hnsw_mut().set_m0(m0);
    segment.hnsw_mut().set_stored_cap_l0(stored_cap_l0);
    segment.hnsw_mut().set_ef_construct(ef_construct);

    let entries: Vec<(u64, Vector)> = base
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, r)| (i as u64, r.to_vec()))
        .collect();

    let t0 = Instant::now();
    let chunk = 50_000;
    for (pos, batch) in entries.chunks(chunk).enumerate() {
        segment.bulk_load(batch).expect("bulk_load failed");
        eprintln!(
            "  {} / {n}  ({:.0}s)",
            (pos + 1) * chunk.min(entries.len() - pos * chunk),
            t0.elapsed().as_secs_f64()
        );
    }
    eprintln!("Build complete in {:.1}s", t0.elapsed().as_secs_f64());

    segment.save_to_path(&snap_path).expect("failed to save snapshot");
    let size = fs::metadata(&snap_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("Snapshot: {snap_path}  ({:.0} MiB)", size as f64 / (1 << 20) as f64);
}

/// Sweep ef_search values and emit one JSON line per ef.
#[test]
#[ignore]
fn ann_pareto_sweep() {
    let data_dir = env_str("ANNEX_BENCH_DATA_DIR").expect("ANNEX_BENCH_DATA_DIR is required");
    let snap_path = env_str("VECTORDB_PERSIST_PATH").expect("VECTORDB_PERSIST_PATH is required");
    let label = env_str("ANNEX_BENCH_LABEL").unwrap_or_else(|| "annexdb".into());
    let apply_rcm = env_str("ANNEX_BENCH_RCM").as_deref() == Some("true");
    let apply_sq8 = env_str("ANNEX_BENCH_SQ8").as_deref() == Some("true");
    let top_k = env_usize("ANNEX_BENCH_TOPK", 10);
    let n_queries = env_usize("ANNEX_BENCH_QUERIES", 1000);
    let rounds = env_usize("ANNEX_BENCH_ROUNDS", 3);
    let efs = parse_efs();

    let queries: Array2<f32> =
        read_npy(format!("{data_dir}/queries.npy")).expect("failed to read queries.npy");
    let truth: Vec<Vec<u64>> = serde_json::from_slice(
        &fs::read(format!("{data_dir}/ground_truth.json"))
            .expect("failed to read ground_truth.json"),
    )
    .expect("failed to parse ground_truth.json");

    let qs: Vec<Vector> = queries
        .rows()
        .into_iter()
        .take(n_queries)
        .map(|r| r.to_vec())
        .collect();
    let count = qs.len();

    let segment = Segment::load_from_path(&snap_path).expect("failed to load snapshot");
    let mut index = HNSWIndex::from_snapshot(segment.hnsw().to_snapshot());
    drop(segment);
    if apply_rcm {
        index.reorder_rcm();
    }
    if apply_sq8 {
        index.quantize_all();
    }

    // Cache warm
    let warm = SearchRuntimeOptions { ef_search: Some(64), ..Default::default() };
    for q in &qs {
        let _ = index.search_with_options(q, top_k, &warm);
    }

    for ef in efs {
        let opts = SearchRuntimeOptions {
            ef_search: Some(ef),
            sq8_screen: Some(apply_sq8),
            ..Default::default()
        };

        // Recall pass (outside timing)
        let mut hits = 0usize;
        for (i, q) in qs.iter().enumerate() {
            let r = index.search_with_options(q, top_k, &opts).unwrap();
            hits += r.iter().filter(|x| truth[i][..top_k].contains(&x.id)).count();
        }
        let recall = hits as f64 / (count * top_k) as f64;

        // Timing: ROUNDS passes, all per-query latencies concatenated
        let mut times: Vec<f64> = Vec::with_capacity(count * rounds);
        let wall = Instant::now();
        for _ in 0..rounds {
            for q in &qs {
                let t0 = Instant::now();
                black_box(index.search_with_options(black_box(q), top_k, &opts).unwrap());
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let qps = (count * rounds) as f64 / wall.elapsed().as_secs_f64();
        times.sort_by(f64::total_cmp);
        let n = times.len();

        println!(
            "{}",
            json!({
                "lib": "annexdb",
                "config": label,
                "ef": ef,
                "recall": recall,
                "qps": qps,
                "p50_ms": times[n / 2],
                "p95_ms": times[(n * 95 / 100).min(n - 1)],
                "p99_ms": times[(n * 99 / 100).min(n - 1)],
            })
        );
    }
}
