use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

use ndarray::Array2;
use ndarray_npy::read_npy;
use serde::Serialize;
use serde_json::from_slice;

mod common;
use common::{
    DatasetHarnessConfig, MissStats, QueryLogWriter, QueryStatsAgg, TestLogConfig, log_peak_rss,
    summarize_f64, summarize_usize,
};

use annex::segment::segment::Segment;
use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;
use annex::vector::hnsw::config::{
    disable_early_exit, early_exit_patience, neighbor_scan_cap, neighbor_scan_rotate_enabled,
    neighbor_scan_stride_enabled, search_expansion_cap_override, search_expansion_multiplier,
    set_neighbor_scan_cap_level0_default,
};

// NYTimes benchmark defaults (matches m=16/m0=32/stored_cap=128/efc=300/scan_cap=64).
// Each value is still overridable via env var.
const NYT_DEFAULT_M0: usize = 32;
const NYT_DEFAULT_STORED_CAP_L0: usize = 128;
const NYT_DEFAULT_EF_CONSTRUCT: usize = 300;
const NYT_DEFAULT_NEIGHBOR_SCAN_CAP_L0: usize = 64;

fn default_nytimes_snapshot_path() -> String {
    let m = common::env_usize_first(&["VECTORDB_M", "VECTORDB_NYT_M"]).unwrap_or(16);
    let m0 = common::env_usize_first(&["VECTORDB_M0", "VECTORDB_NYT_M0"]).unwrap_or(NYT_DEFAULT_M0);
    let stored_cap =
        common::env_usize_first(&["VECTORDB_STORED_CAP_L0", "VECTORDB_NYT_STORED_CAP_L0"])
            .unwrap_or(NYT_DEFAULT_STORED_CAP_L0);
    let efc = common::env_usize_first(&["VECTORDB_EF_CONSTRUCT", "VECTORDB_NYT_EF_CONSTRUCT"])
        .unwrap_or(NYT_DEFAULT_EF_CONSTRUCT);
    format!(
        "data/nytimes-256-angular/index_m{}_m0_{}_stored_{}_efc{}.bin",
        m, m0, stored_cap, efc
    )
}

fn apply_nytimes_build_defaults(harness: &mut DatasetHarnessConfig) {
    // Apply NYTimes-specific defaults only when env vars didn't already set them.
    if common::env_usize_first(&["VECTORDB_M0", "VECTORDB_NYT_M0"]).is_none() {
        harness.build.m0 = NYT_DEFAULT_M0;
    }
    if common::env_usize_first(&["VECTORDB_STORED_CAP_L0", "VECTORDB_NYT_STORED_CAP_L0"]).is_none()
    {
        harness.build.stored_cap_l0 = NYT_DEFAULT_STORED_CAP_L0;
    }
    if common::env_usize_first(&["VECTORDB_EF_CONSTRUCT", "VECTORDB_NYT_EF_CONSTRUCT"]).is_none() {
        harness.search.ef_construct = NYT_DEFAULT_EF_CONSTRUCT;
    }
    // Seed the L0 scan cap if the env hasn't already locked it in. This is the query-side half
    // of the stored_cap/scan_cap split: store 128 neighbors, scan only 64 per BFS step.
    if common::env_usize_first(&["VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0"]).is_none() {
        let _ = set_neighbor_scan_cap_level0_default(Some(NYT_DEFAULT_NEIGHBOR_SCAN_CAP_L0));
    }
}

fn load_vectors(path: &Path) -> Vec<Vector> {
    let arr: Array2<f32> = read_npy(path).expect("failed to read .npy");
    arr.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn load_ground_truth(path: &Path) -> Vec<Vec<usize>> {
    let data = fs::read(path).expect("failed to read ground truth json");
    from_slice(&data).expect("failed to parse ground truth json")
}

fn build_nytimes_segment(segment: &mut Segment, base: &[Vector], logs: &TestLogConfig) {
    let outer_chunk = logs.insert_progress_every.max(1000);

    logs.log_info(&format!(
        "🚀 Bulk-loading {} vectors (outer_chunk={})...",
        base.len(),
        outer_chunk,
    ));
    let start_insert = Instant::now();
    let mut last_log = start_insert;
    let mut inserted = 0usize;

    let entries: Vec<(u64, Vector)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.clone()))
        .collect();

    for chunk in entries.chunks(outer_chunk) {
        segment.bulk_load(chunk).unwrap();
        inserted += chunk.len();
        let now = Instant::now();
        logs.log_info(&format!(
            "Inserted {} vectors (+{:?}, chunk={:?})",
            inserted,
            now - start_insert,
            now - last_log,
        ));
        last_log = now;
    }

    let insert_dur = start_insert.elapsed();
    let insert_ms = insert_dur.as_secs_f64() * 1000.0 / base.len().max(1) as f64;
    logs.log_info(&format!(
        "✅ Inserted {} vectors in {:?} (~{:.3} ms/insert)",
        base.len(),
        insert_dur,
        insert_ms
    ));
}

fn persist_segment(segment: &Segment, path: &str, logs: &TestLogConfig) {
    logs.log_info(&format!("💾 Persisting NYTimes segment to {} ...", path));
    let start = Instant::now();
    segment
        .save_to_path(path)
        .expect("failed to persist NYTimes segment");
    let elapsed = start.elapsed();
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    logs.log_info(&format!(
        "✅ Saved NYTimes segment to {} (size={} bytes, elapsed={:?})",
        path, size, elapsed
    ));
}

fn ensure_exists(path: &Path) {
    assert!(
        path.exists(),
        "missing required dataset file: {} (run the download script in README)",
        path.display()
    );
}

fn read_search_caps_from_env() -> (bool, usize, Option<usize>) {
    let expansion_mult = search_expansion_multiplier();
    let expansion_cap = search_expansion_cap_override();
    (disable_early_exit(), expansion_mult, expansion_cap)
}

fn log_search_caps(logs: &TestLogConfig) {
    let (disable_early_exit, expansion_mult, expansion_cap) = read_search_caps_from_env();
    let patience = early_exit_patience();
    let cap = neighbor_scan_cap(0);
    let rotation = neighbor_scan_rotate_enabled();
    let stride = neighbor_scan_stride_enabled();
    let cap_display = expansion_cap
        .map(|v| v.to_string())
        .unwrap_or_else(|| "none".to_string());
    logs.log_info(&format!(
        "⚙️  Query-time knobs: early_exit_disabled={} early_exit_patience={} expansion_mult={} expansion_cap={} neighbor_scan_cap_level0={} rotation={} rotation_stride={}",
        disable_early_exit,
        patience,
        expansion_mult,
        cap_display,
        if cap == usize::MAX {
            "none".to_string()
        } else {
            cap.to_string()
        },
        if rotation { "on" } else { "off" },
        if stride { "enabled" } else { "off" },
    ));
}

#[derive(Serialize)]
struct QueryLogEntry {
    dataset: &'static str,
    query_idx: usize,
    ef_search: usize,
    top_k: usize,
    elapsed_ms: f64,
    visited: Option<usize>,
    expanded: Option<usize>,
    results_len: usize,
    recall: f64,
    misses: usize,
}

/// Uses the ANN-Benchmarks NYTimes 256-d Angular dataset from Hugging Face.
/// Requires local files:
///   base.npy, queries.npy, ground_truth.json in data/nytimes-256-angular (by default).
#[test]
#[ignore]
fn nytimes_256_angular_perf_and_recall() {
    run_nytimes_perf_and_recall(TestMode::Default);
}

#[test]
#[ignore]
fn nytimes_build_and_persist_snapshot_only() {
    run_nytimes_perf_and_recall(TestMode::BuildOnly);
}

#[test]
#[ignore]
fn nytimes_recall_from_snapshot() {
    run_nytimes_recall_only();
}

#[test]
#[ignore]
fn nytimes_qps_latency_curve() {
    run_nytimes_qps_latency_curve();
}

#[derive(Clone, Copy)]
enum TestMode {
    Default,
    BuildOnly,
}

fn run_nytimes_perf_and_recall(mode: TestMode) {
    let t0 = Instant::now();
    let logs = TestLogConfig::from_env();
    let mut harness = DatasetHarnessConfig::from_env(
        "NYT",
        "data/nytimes-256-angular",
        &default_nytimes_snapshot_path(),
        &[32, 64, 128, 256, 512],
        1000,
        16,
        16,
    );
    apply_nytimes_build_defaults(&mut harness);
    if matches!(mode, TestMode::BuildOnly) {
        harness.snapshot.use_snapshot = false;
        harness.snapshot.allow_build = true;
        harness.snapshot.save_snapshot = true;
    }
    let top_k = harness.search.top_k.unwrap_or(20);

    let base_path = Path::new(&harness.data_dir).join("base.npy");
    let queries_path = Path::new(&harness.data_dir).join("queries.npy");
    let truth_path = Path::new(&harness.data_dir).join("ground_truth.json");
    ensure_exists(&base_path);
    ensure_exists(&queries_path);
    ensure_exists(&truth_path);
    log_search_caps(&logs);

    logs.log_info(&format!(
        "\n📚 Loading NYTimes dataset from {} (top_k={}, ef_search_list={:?}, max_queries={}, base_cap={:?}, ef_construct={})",
        harness.data_dir,
        top_k,
        harness.search.ef_values,
        harness.search.queries_cap,
        harness.search.base_cap,
        harness.search.ef_construct
    ));

    let mut base = Vec::new();
    if !harness.snapshot.use_snapshot || harness.snapshot.allow_build {
        base = load_vectors(&base_path);
        if let Some(cap) = harness.search.base_cap {
            if cap < base.len() {
                base.truncate(cap);
            }
        }
    }
    let queries = load_vectors(&queries_path);
    let ground_truth = load_ground_truth(&truth_path);
    assert_eq!(
        ground_truth.len(),
        queries.len(),
        "ground truth length must match queries"
    );
    logs.log_info(&format!("⏱️  Data loaded in {:?}", t0.elapsed()));

    let mut segment = if harness.snapshot.use_snapshot {
        if !Path::new(&harness.snapshot.persist_path).exists() {
            if harness.snapshot.allow_build {
                logs.log_info("💾 Snapshot missing; building a fresh segment...");
                let dim = base.first().map(|v| v.len()).unwrap_or(0);
                assert_eq!(dim, 256, "expected 256-d vectors");
                let metric = DistanceMetric::Cosine;
                let mut segment = Segment::new(HNSWIndex::new(
                    metric,
                    harness.build.m,
                    harness
                        .search
                        .ef_values
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(1)
                        .max(top_k),
                    harness.build.max_level,
                    dim,
                ));
                segment.hnsw_mut().set_m0(harness.build.m0);
                segment
                    .hnsw_mut()
                    .set_stored_cap_l0(harness.build.stored_cap_l0);
                segment
                    .hnsw_mut()
                    .set_ef_construct(harness.search.ef_construct);
                build_nytimes_segment(&mut segment, &base, &logs);
                if harness.snapshot.save_snapshot {
                    persist_segment(&segment, &harness.snapshot.persist_path, &logs);
                }
                segment
            } else {
                panic!(
                    "missing snapshot at {} (set VECTORDB_NYT_ALLOW_BUILD=1 or VECTORDB_ALLOW_BUILD=1 to build)",
                    harness.snapshot.persist_path
                );
            }
        } else {
            logs.log_info(&format!(
                "💾 Loading persisted NYTimes segment from {} ...",
                harness.snapshot.persist_path
            ));
            let (segment, metadata) =
                Segment::load_from_path_with_metadata(&harness.snapshot.persist_path)
                    .expect("failed to load persisted NYTimes segment");
            if let Some(meta) = metadata {
                logs.log_info(&format!(
                    "🧾 Snapshot metadata: created_at_ms={} points={} payloads={}",
                    meta.created_at_ms, meta.points, meta.payloads
                ));
            }
            segment
        }
    } else {
        let dim = base.first().map(|v| v.len()).unwrap_or(0);
        assert_eq!(dim, 256, "expected 256-d vectors");
        let metric = DistanceMetric::Cosine;
        let mut segment = Segment::new(HNSWIndex::new(
            metric,
            harness.build.m,
            harness
                .search
                .ef_values
                .iter()
                .copied()
                .max()
                .unwrap_or(1)
                .max(top_k),
            harness.build.max_level,
            dim,
        ));
        segment.hnsw_mut().set_m0(harness.build.m0);
        segment
            .hnsw_mut()
            .set_stored_cap_l0(harness.build.stored_cap_l0);
        segment
            .hnsw_mut()
            .set_ef_construct(harness.search.ef_construct);
        build_nytimes_segment(&mut segment, &base, &logs);
        if harness.snapshot.save_snapshot {
            persist_segment(&segment, &harness.snapshot.persist_path, &logs);
        }
        segment
    };

    let num_queries = queries.len().min(harness.search.queries_cap);
    let mut query_logger = QueryLogWriter::new(logs.query_log_path.clone(), logs.query_log_every);
    logs.log_info(&format!(
        "🔍 Sweeping ef_search over {:?} for {} queries (top_k={})...",
        harness.search.ef_values, num_queries, top_k
    ));

    let mut summary: Vec<(usize, f64, f64)> = Vec::new();
    for &ef in &harness.search.ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let mut stats = QueryStatsAgg::new(num_queries);
        let mut miss_stats = MissStats::default();
        let start_search = Instant::now();
        for (qi, q) in queries.iter().take(num_queries).enumerate() {
            let q_start = Instant::now();
            let (approx, qstats) = segment.search_with_stats(q, top_k).unwrap();
            let truth = &ground_truth[qi];
            let truth_k = truth.len().min(top_k);
            let truth_set: HashSet<_> = truth
                .iter()
                .take(truth_k)
                // Ground truth IDs are 0-based row indices; we insert with the same IDs.
                .filter(|&&id| (id as usize) < segment.hnsw().len())
                .map(|&id| id as u64)
                .collect();
            total_targets += truth_set.len();
            let query_hits = approx.iter().filter(|r| truth_set.contains(&r.id)).count();
            hits += query_hits;
            let recall = query_hits as f64 / truth_set.len().max(1) as f64;
            let misses = truth_set.len().saturating_sub(query_hits);
            let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            stats.record(
                elapsed_ms,
                Some(qstats.visited),
                Some(qstats.expanded),
                Some(qstats.adjacency_reads),
                Some(qstats.distance_computations),
                Some(qstats.cap_breaks),
                Some(qstats.patience_breaks),
                recall,
            );
            for miss_id in truth_set
                .iter()
                .filter(|id| !approx.iter().any(|r| r.id == **id))
            {
                if let Some(level) = segment.hnsw().point_level(*miss_id) {
                    let degree = segment.hnsw().point_degree(*miss_id, 0);
                    miss_stats.record(level, degree);
                }
            }
            query_logger.write(
                qi,
                &QueryLogEntry {
                    dataset: "nytimes-256-angular",
                    query_idx: qi,
                    ef_search,
                    top_k,
                    elapsed_ms,
                    visited: Some(qstats.visited),
                    expanded: Some(qstats.expanded),
                    results_len: approx.len(),
                    recall,
                    misses,
                },
            );
            if logs.level.allows_debug()
                && ((qi + 1) % logs.progress_every == 0 || qi + 1 == num_queries)
            {
                let partial_recall = hits as f64 / total_targets.max(1) as f64;
                logs.log_debug(&format!(
                    "  progress: query {}/{} (ef_search={}) cumulative recall={:.3}",
                    qi + 1,
                    num_queries,
                    ef_search,
                    partial_recall
                ));
            }
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries as f64;
        let recall = hits as f64 / total_targets.max(1) as f64;
        logs.log_info(&format!(
            "🎯 [ef_search={}] recall@{}: {:.3} over {} queries (hits {}/{}) | avg {:.3} ms/query",
            ef_search, top_k, recall, num_queries, hits, total_targets, avg_ms
        ));
        let latency = summarize_f64(&stats.elapsed_ms);
        let visited = summarize_usize(&stats.visited);
        let expanded = summarize_usize(&stats.expanded);
        let adjacency_reads = summarize_usize(&stats.adjacency_reads);
        let distance_computations = summarize_usize(&stats.distance_computations);
        let cap_breaks = summarize_usize(&stats.cap_breaks);
        let patience_breaks = summarize_usize(&stats.patience_breaks);
        logs.log_info(&format!(
            "[query_stats] ef_search={} ms(p50/p90/p99)={:.3}/{:.3}/{:.3} visited(p50/p90/p99)={:.0}/{:.0}/{:.0} expanded(p50/p90/p99)={:.0}/{:.0}/{:.0} adjacency_reads(p50/p90/p99)={:.0}/{:.0}/{:.0} distance_computations(p50/p90/p99)={:.0}/{:.0}/{:.0} cap_breaks(p50/p90/p99)={:.0}/{:.0}/{:.0} patience_breaks(p50/p90/p99)={:.0}/{:.0}/{:.0}",
            ef_search,
            latency.p50,
            latency.p90,
            latency.p99,
            visited.p50,
            visited.p90,
            visited.p99,
            expanded.p50,
            expanded.p90,
            expanded.p99,
            adjacency_reads.p50,
            adjacency_reads.p90,
            adjacency_reads.p99,
            distance_computations.p50,
            distance_computations.p90,
            distance_computations.p99,
            cap_breaks.p50,
            cap_breaks.p90,
            cap_breaks.p99,
            patience_breaks.p50,
            patience_breaks.p90,
            patience_breaks.p99
        ));
        if miss_stats.total > 0 {
            let degree = summarize_usize(&miss_stats.degree_samples);
            let levels: Vec<String> = miss_stats
                .level_counts
                .iter()
                .enumerate()
                .map(|(lvl, count)| format!("L{}={}", lvl, count))
                .collect();
            logs.log_info(&format!(
                "[miss_stats] ef_search={} total={} levels={} degree(p50/p90/p99)={:.0}/{:.0}/{:.0}",
                ef_search,
                miss_stats.total,
                levels.join(","),
                degree.p50,
                degree.p90,
                degree.p99
            ));
        }
        // Flush any unfiltered search stats for this sweep so logging is emitted promptly.
        segment.hnsw().flush_unfiltered_search_stats();
        summary.push((ef_search, recall, avg_ms));
    }

    logs.log_info("\nSummary (ef_search -> recall, ms/query):");
    for (ef, recall, ms) in summary {
        logs.log_info(&format!("  {} -> {:.3}, {:.3} ms/query", ef, recall, ms));
    }
}

fn run_nytimes_recall_only() {
    let t0 = Instant::now();
    let logs = TestLogConfig::from_env();
    let mut harness = DatasetHarnessConfig::from_env(
        "NYT",
        "data/nytimes-256-angular",
        &default_nytimes_snapshot_path(),
        &[32, 64, 128, 256, 512],
        1000,
        16,
        16,
    );
    apply_nytimes_build_defaults(&mut harness);
    let top_k = harness.search.top_k.unwrap_or(20);

    let queries_path = Path::new(&harness.data_dir).join("queries.npy");
    let truth_path = Path::new(&harness.data_dir).join("ground_truth.json");
    ensure_exists(&queries_path);
    ensure_exists(&truth_path);
    log_search_caps(&logs);

    let queries = load_vectors(&queries_path);
    let ground_truth = load_ground_truth(&truth_path);
    assert_eq!(
        ground_truth.len(),
        queries.len(),
        "ground truth length must match queries"
    );
    logs.log_info(&format!(
        "⏱️  Queries and truth loaded in {:?}",
        t0.elapsed()
    ));
    log_peak_rss("nytimes_qps_loaded_queries");
    log_peak_rss("nytimes_loaded_queries");

    if !harness.snapshot.use_snapshot {
        panic!("nytimes_recall_from_snapshot requires VECTORDB_USE_SNAPSHOT=1");
    }
    if !Path::new(&harness.snapshot.persist_path).exists() {
        panic!(
            "missing snapshot at {} (run nytimes_build_and_persist_snapshot_only first)",
            harness.snapshot.persist_path
        );
    }
    logs.log_info(&format!(
        "💾 Loading persisted NYTimes segment from {} ...",
        harness.snapshot.persist_path
    ));
    let (mut segment, metadata) =
        Segment::load_from_path_with_metadata(&harness.snapshot.persist_path)
            .expect("failed to load persisted NYTimes segment");
    let cfg = segment.hnsw().config_summary();
    logs.log_info(&format!(
        "✅ Loaded persisted segment with {} vectors (payloads={})",
        segment.hnsw().len(),
        segment.payloads().len()
    ));
    log_peak_rss("nytimes_qps_loaded_snapshot");
    log_peak_rss("nytimes_loaded_snapshot");
    logs.log_info(&format!(
        "🧭 HNSW config: metric={:?} dim={} m={} m0={} ef={} ef_construct={} level_cap={} level_scale={:.3} max_level={} exact_fallback={} threshold={}",
        cfg.metric,
        cfg.dim,
        cfg.m,
        cfg.m0,
        cfg.ef,
        cfg.ef_construct,
        cfg.max_level_cap,
        cfg.level_scale,
        cfg.current_max_level,
        cfg.exact_fallback_enabled,
        cfg.exact_fallback_threshold
    ));
    if let Some(meta) = metadata {
        logs.log_info(&format!(
            "🧾 Snapshot metadata: created_at_ms={} points={} payloads={}",
            meta.created_at_ms, meta.points, meta.payloads
        ));
    }

    let num_queries = queries.len().min(harness.search.queries_cap);
    let mut query_logger = QueryLogWriter::new(logs.query_log_path.clone(), logs.query_log_every);
    logs.log_info(&format!(
        "🔍 Sweeping ef_search over {:?} for {} queries (top_k={})...",
        harness.search.ef_values, num_queries, top_k
    ));
    log_peak_rss("nytimes_qps_before_sweep");
    log_peak_rss("nytimes_before_sweep");

    let mut summary: Vec<(usize, f64, f64)> = Vec::new();
    for &ef in &harness.search.ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let mut stats = QueryStatsAgg::new(num_queries);
        let mut miss_stats = MissStats::default();
        let start_search = Instant::now();
        for (qi, q) in queries.iter().take(num_queries).enumerate() {
            let q_start = Instant::now();
            let (approx, qstats) = segment.search_with_stats(q, top_k).unwrap();
            let truth = &ground_truth[qi];
            let truth_k = truth.len().min(top_k);
            let truth_set: HashSet<_> = truth
                .iter()
                .take(truth_k)
                // Ground truth IDs are 0-based row indices.
                .map(|&id| id as u64)
                .collect();
            total_targets += truth_set.len();
            let query_hits = approx.iter().filter(|r| truth_set.contains(&r.id)).count();
            hits += query_hits;
            let recall = query_hits as f64 / truth_set.len().max(1) as f64;
            let misses = truth_set.len().saturating_sub(query_hits);
            let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            stats.record(
                elapsed_ms,
                Some(qstats.visited),
                Some(qstats.expanded),
                Some(qstats.adjacency_reads),
                Some(qstats.distance_computations),
                Some(qstats.cap_breaks),
                Some(qstats.patience_breaks),
                recall,
            );
            for miss_id in truth_set
                .iter()
                .filter(|id| !approx.iter().any(|r| r.id == **id))
            {
                if let Some(level) = segment.hnsw().point_level(*miss_id) {
                    let degree = segment.hnsw().point_degree(*miss_id, 0);
                    miss_stats.record(level, degree);
                }
            }
            query_logger.write(
                qi,
                &QueryLogEntry {
                    dataset: "nytimes-256-angular",
                    query_idx: qi,
                    ef_search,
                    top_k,
                    elapsed_ms,
                    visited: Some(qstats.visited),
                    expanded: Some(qstats.expanded),
                    results_len: approx.len(),
                    recall,
                    misses,
                },
            );
            if logs.level.allows_debug()
                && ((qi + 1) % logs.progress_every == 0 || qi + 1 == num_queries)
            {
                let partial_recall = hits as f64 / total_targets.max(1) as f64;
                logs.log_debug(&format!(
                    "  progress: query {}/{} (ef_search={}) cumulative recall={:.3}",
                    qi + 1,
                    num_queries,
                    ef_search,
                    partial_recall
                ));
            }
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries as f64;
        let recall = hits as f64 / total_targets.max(1) as f64;
        logs.log_info(&format!(
            "🎯 [ef_search={}] recall@{}: {:.3} over {} queries (hits {}/{}) | avg {:.3} ms/query",
            ef_search, top_k, recall, num_queries, hits, total_targets, avg_ms
        ));
        let latency = summarize_f64(&stats.elapsed_ms);
        let visited = summarize_usize(&stats.visited);
        let expanded = summarize_usize(&stats.expanded);
        let adjacency_reads = summarize_usize(&stats.adjacency_reads);
        let distance_computations = summarize_usize(&stats.distance_computations);
        let cap_breaks = summarize_usize(&stats.cap_breaks);
        let patience_breaks = summarize_usize(&stats.patience_breaks);
        logs.log_info(&format!(
            "[query_stats] ef_search={} ms(p50/p90/p99)={:.3}/{:.3}/{:.3} visited(p50/p90/p99)={:.0}/{:.0}/{:.0} expanded(p50/p90/p99)={:.0}/{:.0}/{:.0} adjacency_reads(p50/p90/p99)={:.0}/{:.0}/{:.0} distance_computations(p50/p90/p99)={:.0}/{:.0}/{:.0} cap_breaks(p50/p90/p99)={:.0}/{:.0}/{:.0} patience_breaks(p50/p90/p99)={:.0}/{:.0}/{:.0}",
            ef_search,
            latency.p50,
            latency.p90,
            latency.p99,
            visited.p50,
            visited.p90,
            visited.p99,
            expanded.p50,
            expanded.p90,
            expanded.p99,
            adjacency_reads.p50,
            adjacency_reads.p90,
            adjacency_reads.p99,
            distance_computations.p50,
            distance_computations.p90,
            distance_computations.p99,
            cap_breaks.p50,
            cap_breaks.p90,
            cap_breaks.p99,
            patience_breaks.p50,
            patience_breaks.p90,
            patience_breaks.p99
        ));
        if miss_stats.total > 0 {
            let degree = summarize_usize(&miss_stats.degree_samples);
            let levels: Vec<String> = miss_stats
                .level_counts
                .iter()
                .enumerate()
                .map(|(lvl, count)| format!("L{}={}", lvl, count))
                .collect();
            logs.log_info(&format!(
                "[miss_stats] ef_search={} total={} levels={} degree(p50/p90/p99)={:.0}/{:.0}/{:.0}",
                ef_search,
                miss_stats.total,
                levels.join(","),
                degree.p50,
                degree.p90,
                degree.p99
            ));
        }
        segment.hnsw().flush_unfiltered_search_stats();
        summary.push((ef_search, recall, avg_ms));
    }

    logs.log_info("\nSummary (ef_search -> recall, ms/query):");
    for (ef, recall, ms) in summary {
        logs.log_info(&format!("  {} -> {:.3}, {:.3} ms/query", ef, recall, ms));
    }
    log_peak_rss("nytimes_recall_complete");
}

fn run_nytimes_qps_latency_curve() {
    let t0 = Instant::now();
    let logs = TestLogConfig::from_env();
    let mut harness = DatasetHarnessConfig::from_env(
        "NYT",
        "data/nytimes-256-angular",
        &default_nytimes_snapshot_path(),
        &[32, 64, 128, 256, 512],
        1000,
        16,
        16,
    );
    apply_nytimes_build_defaults(&mut harness);
    let top_k = harness.search.top_k.unwrap_or(20);

    let queries_path = Path::new(&harness.data_dir).join("queries.npy");
    let truth_path = Path::new(&harness.data_dir).join("ground_truth.json");
    ensure_exists(&queries_path);
    ensure_exists(&truth_path);
    log_search_caps(&logs);

    let queries = load_vectors(&queries_path);
    let ground_truth = load_ground_truth(&truth_path);
    assert_eq!(
        ground_truth.len(),
        queries.len(),
        "ground truth length must match queries"
    );
    logs.log_info(&format!(
        "⏱️  Queries and truth loaded in {:?}",
        t0.elapsed()
    ));

    if !harness.snapshot.use_snapshot {
        panic!("nytimes_qps_latency_curve requires VECTORDB_USE_SNAPSHOT=1");
    }
    if !Path::new(&harness.snapshot.persist_path).exists() {
        panic!(
            "missing snapshot at {} (run nytimes_build_and_persist_snapshot_only first)",
            harness.snapshot.persist_path
        );
    }
    logs.log_info(&format!(
        "💾 Loading persisted NYTimes segment from {} ...",
        harness.snapshot.persist_path
    ));
    let (mut segment, metadata) =
        Segment::load_from_path_with_metadata(&harness.snapshot.persist_path)
            .expect("failed to load persisted NYTimes segment");
    let cfg = segment.hnsw().config_summary();
    logs.log_info(&format!(
        "✅ Loaded persisted segment with {} vectors (payloads={})",
        segment.hnsw().len(),
        segment.payloads().len()
    ));
    logs.log_info(&format!(
        "🧭 HNSW config: metric={:?} dim={} m={} m0={} ef={} ef_construct={} level_cap={} level_scale={:.3} max_level={} exact_fallback={} threshold={}",
        cfg.metric,
        cfg.dim,
        cfg.m,
        cfg.m0,
        cfg.ef,
        cfg.ef_construct,
        cfg.max_level_cap,
        cfg.level_scale,
        cfg.current_max_level,
        cfg.exact_fallback_enabled,
        cfg.exact_fallback_threshold
    ));
    if let Some(meta) = metadata {
        logs.log_info(&format!(
            "🧾 Snapshot metadata: created_at_ms={} points={} payloads={}",
            meta.created_at_ms, meta.points, meta.payloads
        ));
    }

    let num_queries = queries.len().min(harness.search.queries_cap);
    let mut query_logger = QueryLogWriter::new(logs.query_log_path.clone(), logs.query_log_every);
    logs.log_info(&format!(
        "🔍 Sweeping ef_search over {:?} for {} queries (top_k={})...",
        harness.search.ef_values, num_queries, top_k
    ));

    let mut summary: Vec<(usize, f64, f64, f64)> = Vec::new();
    for &ef in &harness.search.ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let mut stats = QueryStatsAgg::new(num_queries);
        let start_search = Instant::now();
        for (qi, q) in queries.iter().take(num_queries).enumerate() {
            let q_start = Instant::now();
            let (approx, qstats) = segment.search_with_stats(q, top_k).unwrap();
            let truth = &ground_truth[qi];
            let truth_k = truth.len().min(top_k);
            let truth_set: HashSet<_> = truth.iter().take(truth_k).map(|&id| id as u64).collect();
            total_targets += truth_set.len();
            let query_hits = approx.iter().filter(|r| truth_set.contains(&r.id)).count();
            hits += query_hits;
            let recall = query_hits as f64 / truth_set.len().max(1) as f64;
            let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            stats.record(
                elapsed_ms,
                Some(qstats.visited),
                Some(qstats.expanded),
                Some(qstats.adjacency_reads),
                Some(qstats.distance_computations),
                Some(qstats.cap_breaks),
                Some(qstats.patience_breaks),
                recall,
            );
            query_logger.write(
                qi,
                &QueryLogEntry {
                    dataset: "nytimes-256-angular",
                    query_idx: qi,
                    ef_search,
                    top_k,
                    elapsed_ms,
                    visited: Some(qstats.visited),
                    expanded: Some(qstats.expanded),
                    results_len: approx.len(),
                    recall,
                    misses: truth_set.len().saturating_sub(query_hits),
                },
            );
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries as f64;
        let qps = num_queries as f64 / search_dur.as_secs_f64();
        let recall = hits as f64 / total_targets.max(1) as f64;
        let latency = summarize_f64(&stats.elapsed_ms);
        logs.log_info(&format!(
            "📈 [ef_search={}] qps={:.1} avg_ms={:.3} p50/p90/p99={:.3}/{:.3}/{:.3} recall@{}={:.3}",
            ef_search,
            qps,
            avg_ms,
            latency.p50,
            latency.p90,
            latency.p99,
            top_k,
            recall
        ));
        summary.push((ef_search, qps, avg_ms, recall));
    }

    logs.log_info("\nSummary (ef_search -> qps, ms/query, recall):");
    for (ef, qps, ms, recall) in summary {
        logs.log_info(&format!(
            "  {} -> {:.1} qps, {:.3} ms/query, {:.3} recall",
            ef, qps, ms, recall
        ));
    }
    log_peak_rss("nytimes_qps_complete");
}

/// Sweeps multi-entry seeds and adaptive EF on the NYT dataset.
/// Builds a fresh index if no snapshot exists at the default path.
/// Run with:
///   VECTORDB_NYT_ALLOW_BUILD=1 cargo test --release --test nytimes \
///     nytimes_adaptive_search_sweep -- --ignored --nocapture
#[test]
#[ignore]
fn nytimes_adaptive_search_sweep() {
    use annex::vector::hnsw::SearchRuntimeOptions;

    let logs = TestLogConfig::from_env();
    let mut harness = DatasetHarnessConfig::from_env(
        "NYT",
        "data/nytimes-256-angular",
        &default_nytimes_snapshot_path(),
        &[32, 64, 128, 256, 512],
        1000,
        16,
        16,
    );
    apply_nytimes_build_defaults(&mut harness);
    let top_k = harness.search.top_k.unwrap_or(20);
    let num_queries = harness.search.queries_cap;

    let base_path = Path::new(&harness.data_dir).join("base.npy");
    let queries_path = Path::new(&harness.data_dir).join("queries.npy");
    let truth_path = Path::new(&harness.data_dir).join("ground_truth.json");
    ensure_exists(&queries_path);
    ensure_exists(&truth_path);

    let queries = load_vectors(&queries_path);
    let ground_truth = load_ground_truth(&truth_path);
    let num_queries = num_queries.min(queries.len());

    // Load or build the segment.
    let segment = if Path::new(&harness.snapshot.persist_path).exists() {
        logs.log_info(&format!(
            "💾 Loading snapshot from {} ...", harness.snapshot.persist_path
        ));
        let (seg, _) = Segment::load_from_path_with_metadata(&harness.snapshot.persist_path)
            .expect("failed to load snapshot");
        seg
    } else {
        ensure_exists(&base_path);
        logs.log_info("🔨 No snapshot found; building index ...");
        let base = load_vectors(&base_path);
        let dim = base[0].len();
        let mut seg = Segment::new(HNSWIndex::new(
            DistanceMetric::Cosine,
            harness.build.m,
            512.max(top_k),
            harness.build.max_level,
            dim,
        ));
        seg.hnsw_mut().set_m0(harness.build.m0);
        seg.hnsw_mut().set_stored_cap_l0(harness.build.stored_cap_l0);
        seg.hnsw_mut().set_ef_construct(harness.search.ef_construct);
        build_nytimes_segment(&mut seg, &base, &logs);
        seg.save_to_path(&harness.snapshot.persist_path).ok();
        seg
    };

    let cfg = segment.hnsw().config_summary();
    logs.log_info(&format!(
        "index: {} vectors  M={}  M0={}  ef_construct={}  max_level={}",
        segment.hnsw().len(), cfg.m, cfg.m0, cfg.ef_construct, cfg.current_max_level,
    ));
    logs.log_info(&format!("queries: {}  top_k: {}\n", num_queries, top_k));

    // Helper: run a config over all queries, return (recall, avg_ms, p50_ms, p90_ms, p99_ms).
    let run_config = |segment: &Segment, opts: &SearchRuntimeOptions| -> (f64, f64, f64, f64, f64) {
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let mut latencies: Vec<f64> = Vec::with_capacity(num_queries);
        for (qi, q) in queries.iter().take(num_queries).enumerate() {
            let t0 = Instant::now();
            let approx = segment.search_with_options(q, top_k, opts).unwrap();
            latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            let truth = &ground_truth[qi];
            let truth_set: HashSet<u64> = truth.iter().take(top_k).map(|&id| id as u64).collect();
            total_targets += truth_set.len();
            hits += approx.iter().filter(|r| truth_set.contains(&r.id)).count();
        }
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = latencies.len();
        let avg_ms = latencies.iter().sum::<f64>() / n as f64;
        let p50 = latencies[(n as f64 * 0.50) as usize];
        let p90 = latencies[(n as f64 * 0.90) as usize];
        let p99 = latencies[(n as f64 * 0.99) as usize];
        let recall = hits as f64 / total_targets.max(1) as f64;
        (recall, avg_ms, p50, p90, p99)
    };

    // ── Baseline EF sweep ─────────────────────────────────────────────────
    println!("{:<36} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "config", "recall", "avg_ms", "p50_ms", "p90_ms", "p99_ms");
    println!("{}", "─".repeat(80));

    for &ef in &[32usize, 64, 128, 256] {
        let opts = SearchRuntimeOptions { ef_search: Some(ef), ..Default::default() };
        let (recall, avg_ms, p50, p90, p99) = run_config(&segment, &opts);
        println!("{:<36} {:>8.4} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            format!("plain ef={ef}"), recall, avg_ms, p50, p90, p99);
    }

    println!();

    // ── Multi-entry seeds sweep at EF=64 ─────────────────────────────────
    for seeds in [1usize, 2, 3] {
        let opts = SearchRuntimeOptions {
            ef_search: Some(64),
            num_entry_seeds: Some(seeds),
            ..Default::default()
        };
        let (recall, avg_ms, p50, p90, p99) = run_config(&segment, &opts);
        println!("{:<36} {:>8.4} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            format!("ef=64 seeds={seeds}"), recall, avg_ms, p50, p90, p99);
    }

    println!();

    // ── Adaptive EF: base=32 → high=128 at various thresholds ────────────
    // First measure what fraction of queries fire at each threshold.
    let thresholds = [0.30f32, 0.40, 0.50, 0.55];
    for threshold in thresholds {
        let opts = SearchRuntimeOptions {
            ef_search: Some(32),
            adaptive_ef_high: Some(128),
            adaptive_ef_score_threshold: Some(threshold),
            ..Default::default()
        };
        let (recall, avg_ms, p50, p90, p99) = run_config(&segment, &opts);
        // Count trigger rate in a separate pass (cheap).
        let triggered: usize = queries.iter().take(num_queries).map(|q| {
            let base_opts = SearchRuntimeOptions { ef_search: Some(32), ..Default::default() };
            let res = segment.search_with_options(q, top_k, &base_opts).unwrap();
            if res.first().map(|r| r.sort_key).unwrap_or(0.0) > threshold { 1 } else { 0 }
        }).sum();
        println!("{:<36} {:>8.4} {:>8.3} {:>8.3} {:>8.3} {:>8.3}  triggered={}/{}",
            format!("adapt 32→128 t={threshold:.2}"),
            recall, avg_ms, p50, p90, p99,
            triggered, num_queries);
    }

    println!();

    // ── Adaptive EF: base=64 → high=256 at threshold=0.40 ────────────────
    {
        let opts = SearchRuntimeOptions {
            ef_search: Some(64),
            adaptive_ef_high: Some(256),
            adaptive_ef_score_threshold: Some(0.40),
            ..Default::default()
        };
        let (recall, avg_ms, p50, p90, p99) = run_config(&segment, &opts);
        println!("{:<36} {:>8.4} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            "adapt 64→256 t=0.40", recall, avg_ms, p50, p90, p99);
    }

    // ── Combined: seeds=3 + adaptive 32→128 t=0.45 ───────────────────────
    {
        let opts = SearchRuntimeOptions {
            ef_search: Some(32),
            num_entry_seeds: Some(3),
            adaptive_ef_high: Some(128),
            adaptive_ef_score_threshold: Some(0.45),
            ..Default::default()
        };
        let (recall, avg_ms, p50, p90, p99) = run_config(&segment, &opts);
        println!("{:<36} {:>8.4} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            "seeds=3 + adapt 32→128 t=0.45", recall, avg_ms, p50, p90, p99);
    }

    // Make sure the segment is not dropped while closures reference it.
    drop(segment);
}
