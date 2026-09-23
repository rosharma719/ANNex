use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use ndarray::Array2;
use ndarray_npy::read_npy;
use ordered_float::OrderedFloat;
use serde::Deserialize;
use serde::Serialize;
use serde_json::{Value, from_slice};

use annex::payload_storage::filters::Filter;
use annex::segment::segment::Segment;
use annex::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;

mod common;
use common::{
    DatasetHarnessConfig, MissStats, QueryLogWriter, QueryStatsAgg, TestLogConfig, log_peak_rss,
    summarize_f64, summarize_usize,
};

fn load_vectors(path: &Path) -> Vec<Vector> {
    let arr: Array2<f32> = read_npy(path).expect("failed to read vectors.npy");
    arr.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn json_to_payload_value(v: &Value) -> Option<PayloadValue> {
    match v {
        Value::Null => None,
        Value::Bool(b) => Some(PayloadValue::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(PayloadValue::Int(i))
            } else {
                n.as_f64().map(|f| PayloadValue::Float(OrderedFloat(f)))
            }
        }
        Value::String(s) => Some(PayloadValue::Str(s.clone())),
        Value::Array(arr) => {
            if arr.is_empty() {
                return Some(PayloadValue::ListStr(Vec::new()));
            }
            if arr.iter().all(|x| x.as_i64().is_some()) {
                return Some(PayloadValue::ListInt(
                    arr.iter().map(|x| x.as_i64().unwrap()).collect(),
                ));
            }
            if arr.iter().all(|x| x.as_f64().is_some()) {
                return Some(PayloadValue::ListFloat(
                    arr.iter()
                        .map(|x| OrderedFloat(x.as_f64().unwrap()))
                        .collect(),
                ));
            }
            if arr.iter().all(|x| x.as_str().is_some()) {
                return Some(PayloadValue::ListStr(
                    arr.iter()
                        .map(|x| x.as_str().unwrap().to_string())
                        .collect(),
                ));
            }
            if arr.iter().all(|x| x.as_bool().is_some()) {
                return Some(PayloadValue::ListBool(
                    arr.iter().map(|x| x.as_bool().unwrap()).collect(),
                ));
            }
            None
        }
        Value::Object(_) => None,
    }
}

fn json_obj_to_payload(obj: &serde_json::Map<String, Value>) -> Payload {
    let mut payload = Payload::default();
    for (k, v) in obj {
        if let Some(pv) = json_to_payload_value(v) {
            payload.set(k, pv);
        }
    }
    payload
}

fn load_payloads(path: &Path) -> Vec<Payload> {
    let file = File::open(path).expect("failed to open payloads.jsonl");
    let reader = BufReader::new(file);
    reader
        .lines()
        .enumerate()
        .map(|(idx, line)| {
            let line = line.expect("failed to read payload line");
            let value: Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("payload line {} failed to parse: {}", idx, e));
            let obj = value.as_object().unwrap_or_else(|| {
                panic!("payload line {} is not an object payload", idx);
            });
            json_obj_to_payload(obj)
        })
        .collect()
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RawTestCase {
    query: Vec<f32>,
    conditions: Value,
    closest_ids: Vec<usize>,
    closest_scores: Vec<f32>,
}

fn parse_range_filter(key: &str, range: &serde_json::Map<String, Value>) -> Result<Filter, String> {
    let mut parts = Vec::new();
    for (k, v) in range {
        let pv = json_to_payload_value(v)
            .ok_or_else(|| format!("unsupported range value for {key}:{k}"))?;
        let op = match k.as_str() {
            "gt" => ScalarComparisonOp::Gt,
            "gte" => ScalarComparisonOp::Gte,
            "lt" => ScalarComparisonOp::Lt,
            "lte" => ScalarComparisonOp::Lte,
            _ => return Err(format!("unsupported range operator: {}", k)),
        };
        parts.push(Filter::Compare {
            key: key.to_string(),
            op,
            value: pv,
        });
    }
    match parts.len() {
        0 => Err(format!("range filter for {} had no operators", key)),
        1 => Ok(parts.into_iter().next().unwrap()),
        _ => Ok(Filter::And(parts)),
    }
}

fn parse_condition(node: &Value) -> Result<Filter, String> {
    let obj = node
        .as_object()
        .ok_or_else(|| "condition node must be an object".to_string())?;

    if let Some(and) = obj.get("and") {
        let clauses = and
            .as_array()
            .ok_or_else(|| "`and` must be an array".to_string())?
            .iter()
            .map(parse_condition)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Filter::And(clauses));
    }

    if let Some(or) = obj.get("or") {
        let clauses = or
            .as_array()
            .ok_or_else(|| "`or` must be an array".to_string())?
            .iter()
            .map(parse_condition)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Filter::Or(clauses));
    }

    if let Some(not) = obj.get("not") {
        let inner = parse_condition(not)?;
        return Ok(Filter::Not(Box::new(inner)));
    }

    let (field, inner) = obj
        .iter()
        .next()
        .ok_or_else(|| "empty condition object".to_string())?;
    let inner_obj = inner
        .as_object()
        .ok_or_else(|| format!("condition for {} must be an object", field))?;

    if let Some(m) = inner_obj.get("match") {
        let mv = m
            .get("value")
            .ok_or_else(|| format!("match for {} missing `value`", field))?;
        let value = json_to_payload_value(mv)
            .ok_or_else(|| format!("unsupported match value for key {}", field))?;
        return Ok(Filter::Match {
            key: field.clone(),
            value,
        });
    }

    if let Some(rng) = inner_obj.get("range") {
        let range_obj = rng
            .as_object()
            .ok_or_else(|| format!("range clause for {} must be object", field))?;
        return parse_range_filter(field, range_obj);
    }

    if inner_obj.get("geo").is_some() {
        return Err(format!("geo filters are not supported (field {})", field));
    }

    Err(format!("unsupported condition format for key {}", field))
}

fn load_test_cases(path: &Path) -> Vec<RawTestCase> {
    let file = File::open(path).expect("failed to open tests.jsonl");
    let reader = BufReader::new(file);
    reader
        .lines()
        .enumerate()
        .map(|(idx, line)| {
            let line = line.expect("failed to read test case line");
            from_slice::<RawTestCase>(line.as_bytes())
                .unwrap_or_else(|e| panic!("tests.jsonl line {} failed to parse: {}", idx, e))
        })
        .collect()
}

#[derive(Clone)]
struct PreparedCase {
    query: Vector,
    filter: Option<Filter>,
    truth_ids: Vec<u64>,
}

fn build_hnm_segment(
    segment: &mut Segment,
    base: &[Vector],
    payloads: &[Payload],
    logs: &TestLogConfig,
) {
    logs.log_info(&format!(
        "🚀 Inserting {} vectors with dataset-aligned IDs...",
        base.len()
    ));
    let start_insert = Instant::now();
    for (i, (v, p)) in base.iter().zip(payloads.iter()).enumerate() {
        let dataset_id = i as u64;
        segment
            .insert_with_id(dataset_id, v.clone(), Some(p.clone()))
            .unwrap();
        if logs.level.allows_debug() && i != 0 && i % 5000 == 0 {
            logs.log_debug(&format!(
                "Inserted {} vectors (+{:?})",
                i,
                start_insert.elapsed()
            ));
        }
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

fn persist_hnm_segment(segment: &Segment, path: &str, logs: &TestLogConfig) {
    logs.log_info(&format!("💾 Persisting H&M segment to {} ...", path));
    segment
        .save_to_path(path)
        .expect("failed to persist H&M segment");
    logs.log_info(&format!("✅ Saved H&M segment to {}", path));
}

#[derive(Serialize)]
struct QueryLogEntry {
    dataset: &'static str,
    query_idx: usize,
    ef_search: usize,
    top_k: usize,
    elapsed_ms: f64,
    results_len: usize,
    recall: f64,
    misses: usize,
}

#[test]
#[ignore]
fn hnm_filtered_cosine_recall() {
    run_hnm_filtered_cosine_recall(TestMode::Default);
}

#[test]
#[ignore]
fn hnm_build_and_persist_snapshot_only() {
    // Build the index and persist it, skip recall to save time.
    run_hnm_filtered_cosine_recall(TestMode::BuildOnly);
}

#[test]
#[ignore]
fn hnm_recall_from_snapshot() {
    run_hnm_recall_only();
}

#[derive(Clone, Copy)]
enum TestMode {
    Default,
    BuildOnly,
}

fn run_hnm_filtered_cosine_recall(mode: TestMode) {
    let t0 = Instant::now();
    let logs = TestLogConfig::from_env();
    let mut harness = DatasetHarnessConfig::from_env(
        "HNM",
        "data/hnm",
        "data/hnm/index_filtered.bin",
        &[32, 64, 128, 256, 512],
        1000,
        16,
        16,
    );
    if matches!(mode, TestMode::BuildOnly) {
        harness.snapshot.use_snapshot = false;
        harness.snapshot.allow_build = true;
        harness.snapshot.save_snapshot = true;
    }

    let vectors_path = Path::new(&harness.data_dir).join("vectors.npy");
    let payloads_path = Path::new(&harness.data_dir).join("payloads.jsonl");
    let tests_path = Path::new(&harness.data_dir).join("tests.jsonl");
    assert!(vectors_path.exists(), "missing {}", vectors_path.display());
    assert!(
        payloads_path.exists(),
        "missing {}",
        payloads_path.display()
    );
    assert!(tests_path.exists(), "missing {}", tests_path.display());

    logs.log_info(&format!(
        "\n🧪 H&M filtered recall: dir={}, ef_search={:?}, ef_construct={}, base_cap={:?}, queries_cap={}",
        harness.data_dir,
        harness.search.ef_values,
        harness.search.ef_construct,
        harness.search.base_cap,
        harness.search.queries_cap
    ));

    let mut base = Vec::new();
    let mut payloads = Vec::new();
    if !harness.snapshot.use_snapshot || harness.snapshot.allow_build {
        base = load_vectors(&vectors_path);
        if let Some(cap) = harness.search.base_cap {
            if cap < base.len() {
                base.truncate(cap);
            }
        }
        payloads = load_payloads(&payloads_path);
        if payloads.len() > base.len() {
            payloads.truncate(base.len());
        }
        assert_eq!(
            base.len(),
            payloads.len(),
            "vectors and payloads must have the same length"
        );
    }
    let raw_tests = load_test_cases(&tests_path);
    logs.log_info(&format!("⏱️  Data loaded in {:?}", t0.elapsed()));

    let default_topk = raw_tests.first().map(|c| c.closest_ids.len()).unwrap_or(10);
    let top_k = harness.search.top_k.unwrap_or(default_topk);

    let prepared_cases: Vec<PreparedCase> = raw_tests
        .into_iter()
        .take(harness.search.queries_cap)
        .map(|raw| {
            let filter = if raw.conditions.is_null() {
                None
            } else {
                Some(parse_condition(&raw.conditions).expect("failed to parse filter conditions"))
            };
            let truth_ids = raw
                .closest_ids
                .into_iter()
                .take(top_k)
                .map(|id| id as u64)
                .collect();
            PreparedCase {
                query: raw.query,
                filter,
                truth_ids,
            }
        })
        .collect();

    let mut segment = if harness.snapshot.use_snapshot {
        if !Path::new(&harness.snapshot.persist_path).exists() {
            if harness.snapshot.allow_build {
                logs.log_info("💾 Snapshot missing; building a fresh segment...");
                let dim = base.first().map(|v| v.len()).unwrap_or(0);
                assert_eq!(dim, 2048, "expected 2048-d vectors");
                let metric = DistanceMetric::Cosine;
                let max_ef = harness
                    .search
                    .ef_values
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(1)
                    .max(top_k);
                let mut segment = Segment::new(HNSWIndex::new(
                    metric,
                    harness.build.m,
                    max_ef,
                    harness.build.max_level,
                    dim,
                ));
                segment.hnsw_mut().set_m0(harness.build.m0);
                segment
                    .hnsw_mut()
                    .set_ef_construct(harness.search.ef_construct);
                build_hnm_segment(&mut segment, &base, &payloads, &logs);
                if harness.snapshot.save_snapshot {
                    persist_hnm_segment(&segment, &harness.snapshot.persist_path, &logs);
                }
                segment
            } else {
                panic!(
                    "missing snapshot at {} (set VECTORDB_HNM_ALLOW_BUILD=1 or VECTORDB_ALLOW_BUILD=1 to build)",
                    harness.snapshot.persist_path
                );
            }
        } else {
            logs.log_info(&format!(
                "💾 Loading persisted H&M segment from {} ...",
                harness.snapshot.persist_path
            ));
            Segment::load_from_path(&harness.snapshot.persist_path)
                .expect("failed to load persisted H&M segment")
        }
    } else {
        let dim = base.first().map(|v| v.len()).unwrap_or(0);
        assert_eq!(dim, 2048, "expected 2048-d vectors");
        let metric = DistanceMetric::Cosine;
        let max_ef = harness
            .search
            .ef_values
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .max(top_k);
        let mut segment = Segment::new(HNSWIndex::new(
            metric,
            harness.build.m,
            max_ef,
            harness.build.max_level,
            dim,
        ));
        segment.hnsw_mut().set_m0(harness.build.m0);
        segment
            .hnsw_mut()
            .set_ef_construct(harness.search.ef_construct);
        build_hnm_segment(&mut segment, &base, &payloads, &logs);
        if harness.snapshot.save_snapshot {
            persist_hnm_segment(&segment, &harness.snapshot.persist_path, &logs);
        }
        segment
    };

    let num_queries = prepared_cases.len();
    let collection_len = segment.hnsw().len();
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
        let mut per_query_recalls: Vec<f64> = Vec::with_capacity(num_queries);
        let mut query_stats = QueryStatsAgg::new(num_queries);
        let mut miss_stats = MissStats::default();
        let start_search = Instant::now();
        for (qi, case) in prepared_cases.iter().enumerate() {
            let q_start = Instant::now();
            let res = segment
                .search_with_filter(&case.query, top_k, case.filter.as_ref())
                .unwrap();
            let truth: HashSet<_> = case
                .truth_ids
                .iter()
                .filter(|id| (**id as usize) < collection_len)
                .copied()
                .collect();
            total_targets += truth.len();
            let query_hits = res.iter().filter(|r| truth.contains(&r.id)).count();
            hits += query_hits;
            let recall = query_hits as f64 / truth.len().max(1) as f64;
            per_query_recalls.push(recall);
            let misses = truth.len().saturating_sub(query_hits);
            let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            query_stats.record(elapsed_ms, None, None, None, None, None, None, recall);
            for miss_id in truth.iter().filter(|id| !res.iter().any(|r| r.id == **id)) {
                if let Some(level) = segment.hnsw().point_level(*miss_id) {
                    let degree = segment.hnsw().point_degree(*miss_id, 0);
                    miss_stats.record(level, degree);
                }
            }
            query_logger.write(
                qi,
                &QueryLogEntry {
                    dataset: "hnm-2048-cosine",
                    query_idx: qi,
                    ef_search,
                    top_k,
                    elapsed_ms,
                    results_len: res.len(),
                    recall,
                    misses,
                },
            );
            if logs.level.allows_debug()
                && ((qi + 1) % logs.progress_every == 0 || qi + 1 == num_queries)
            {
                let partial = hits as f64 / total_targets.max(1) as f64;
                logs.log_debug(&format!(
                    "  progress: query {}/{} (ef_search={}) cumulative recall={:.3}",
                    qi + 1,
                    num_queries,
                    ef_search,
                    partial
                ));
            }
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries.max(1) as f64;
        let recall = hits as f64 / total_targets.max(1) as f64;
        logs.log_info(&format!(
            "🎯 [ef_search={}] recall@{}: {:.3} over {} queries (hits {}/{}) | avg {:.3} ms/query",
            ef_search, top_k, recall, num_queries, hits, total_targets, avg_ms
        ));

        // Per-query recall stats to understand distribution.
        per_query_recalls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let recall_pct = |p: f64| -> f64 {
            if per_query_recalls.is_empty() {
                0.0
            } else {
                let idx = ((p / 100.0) * (per_query_recalls.len() as f64 - 1.0)).round() as usize;
                per_query_recalls[idx]
            }
        };
        let mean =
            per_query_recalls.iter().copied().sum::<f64>() / per_query_recalls.len().max(1) as f64;
        let p50 = recall_pct(50.0);
        let p90 = recall_pct(90.0);
        let p99 = recall_pct(99.0);
        let min = *per_query_recalls.first().unwrap_or(&0.0);
        let max = *per_query_recalls.last().unwrap_or(&0.0);
        let full = per_query_recalls
            .iter()
            .filter(|&&r| (r - 1.0).abs() < 1e-6)
            .count();
        let ge_08 = per_query_recalls.iter().filter(|&&r| r >= 0.8).count();
        let ge_05 = per_query_recalls.iter().filter(|&&r| r >= 0.5).count();
        logs.log_info(&format!(
            "[recall_stats] ef_search={} queries={} mean={:.3} p50={:.3} p90={:.3} p99={:.3} min={:.3} max={:.3} full={} ge_0.8={} ge_0.5={}",
            ef_search,
            num_queries,
            mean,
            p50,
            p90,
            p99,
            min,
            max,
            full,
            ge_08,
            ge_05
        ));
        let latency = summarize_f64(&query_stats.elapsed_ms);
        logs.log_info(&format!(
            "[query_stats] ef_search={} ms(p50/p90/p99)={:.3}/{:.3}/{:.3}",
            ef_search, latency.p50, latency.p90, latency.p99
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

        summary.push((ef_search, recall, avg_ms));
    }

    logs.log_info("\nSummary (ef_search -> recall, ms/query):");
    for (ef, recall, ms) in summary {
        logs.log_info(&format!("  {} -> {:.3}, {:.3} ms/query", ef, recall, ms));
    }
}

fn run_hnm_recall_only() {
    let t0 = Instant::now();
    let logs = TestLogConfig::from_env();
    let harness = DatasetHarnessConfig::from_env(
        "HNM",
        "data/hnm",
        "data/hnm/index_filtered.bin",
        &[32, 64, 128, 256, 512],
        1000,
        16,
        16,
    );

    let tests_path = Path::new(&harness.data_dir).join("tests.jsonl");
    assert!(tests_path.exists(), "missing {}", tests_path.display());

    // Load only the test cases; vectors/payloads come from the persisted segment.
    let raw_tests = load_test_cases(&tests_path);
    logs.log_info(&format!("⏱️  Tests loaded in {:?}", t0.elapsed()));
    log_peak_rss("hnm_loaded_tests");

    let default_topk = raw_tests.first().map(|c| c.closest_ids.len()).unwrap_or(10);
    let top_k = harness.search.top_k.unwrap_or(default_topk);

    let prepared_cases: Vec<PreparedCase> = raw_tests
        .into_iter()
        .take(harness.search.queries_cap)
        .map(|raw| {
            let filter = if raw.conditions.is_null() {
                None
            } else {
                Some(parse_condition(&raw.conditions).expect("failed to parse filter conditions"))
            };
            let truth_ids = raw
                .closest_ids
                .into_iter()
                .take(top_k)
                .map(|id| id as u64)
                .collect();
            PreparedCase {
                query: raw.query,
                filter,
                truth_ids,
            }
        })
        .collect();

    if !harness.snapshot.use_snapshot {
        panic!("hnm_recall_from_snapshot requires VECTORDB_USE_SNAPSHOT=1");
    }
    if !Path::new(&harness.snapshot.persist_path).exists() {
        panic!(
            "missing snapshot at {} (run hnm_build_and_persist_snapshot_only first)",
            harness.snapshot.persist_path
        );
    }
    logs.log_info(&format!(
        "💾 Loading persisted H&M segment from {} ...",
        harness.snapshot.persist_path
    ));
    let (mut segment, metadata) =
        Segment::load_from_path_with_metadata(&harness.snapshot.persist_path)
            .expect("failed to load persisted H&M segment");
    logs.log_info(&format!(
        "✅ Loaded persisted segment with {} payloads, ef_search={}",
        segment.payloads().len(),
        segment.hnsw().ef()
    ));
    log_peak_rss("hnm_loaded_snapshot");
    let cfg = segment.hnsw().config_summary();
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
    let collection_len = segment.hnsw().len();

    let num_queries = prepared_cases.len();
    let mut summary: Vec<(usize, f64, f64)> = Vec::new();
    let mut query_logger = QueryLogWriter::new(logs.query_log_path.clone(), logs.query_log_every);
    log_peak_rss("hnm_before_sweep");
    for &ef in &harness.search.ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let mut per_query_recalls: Vec<f64> = Vec::with_capacity(num_queries);
        let mut query_stats = QueryStatsAgg::new(num_queries);
        let mut miss_stats = MissStats::default();
        let start_search = Instant::now();
        for (qi, case) in prepared_cases.iter().enumerate() {
            let q_start = Instant::now();
            let res = segment
                .search_with_filter(&case.query, top_k, case.filter.as_ref())
                .unwrap();
            let truth: HashSet<_> = case
                .truth_ids
                .iter()
                .filter(|id| (**id as usize) < collection_len)
                .copied()
                .collect();
            total_targets += truth.len();
            let query_hits = res.iter().filter(|r| truth.contains(&r.id)).count();
            hits += query_hits;
            let recall = query_hits as f64 / truth.len().max(1) as f64;
            per_query_recalls.push(recall);
            let misses = truth.len().saturating_sub(query_hits);
            let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            query_stats.record(elapsed_ms, None, None, None, None, None, None, recall);
            for miss_id in truth.iter().filter(|id| !res.iter().any(|r| r.id == **id)) {
                if let Some(level) = segment.hnsw().point_level(*miss_id) {
                    let degree = segment.hnsw().point_degree(*miss_id, 0);
                    miss_stats.record(level, degree);
                }
            }
            query_logger.write(
                qi,
                &QueryLogEntry {
                    dataset: "hnm-2048-cosine",
                    query_idx: qi,
                    ef_search,
                    top_k,
                    elapsed_ms,
                    results_len: res.len(),
                    recall,
                    misses,
                },
            );
            if logs.level.allows_debug()
                && ((qi + 1) % logs.progress_every == 0 || qi + 1 == num_queries)
            {
                let partial = hits as f64 / total_targets.max(1) as f64;
                logs.log_debug(&format!(
                    "  progress: query {}/{} (ef_search={}) cumulative recall={:.3}",
                    qi + 1,
                    num_queries,
                    ef_search,
                    partial
                ));
            }
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries as f64;
        let recall = hits as f64 / total_targets.max(1) as f64;
        let mean = per_query_recalls.iter().copied().sum::<f64>() / num_queries.max(1) as f64;
        let mut sorted = per_query_recalls.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = |p: f64| -> usize {
            let n = sorted.len();
            (((p / 100.0) * (n.saturating_sub(1) as f64)).round() as usize).min(n.saturating_sub(1))
        };
        let p50 = sorted.get(idx(50.0)).copied().unwrap_or(0.0);
        let p90 = sorted.get(idx(90.0)).copied().unwrap_or(0.0);
        let p99 = sorted.get(idx(99.0)).copied().unwrap_or(0.0);
        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let full = per_query_recalls
            .iter()
            .filter(|&&r| (r - 1.0).abs() < 1e-6)
            .count();
        let ge_08 = per_query_recalls.iter().filter(|&&r| r >= 0.8).count();
        let ge_05 = per_query_recalls.iter().filter(|&&r| r >= 0.5).count();
        logs.log_info(&format!(
            "[recall_stats] ef_search={} queries={} mean={:.3} p50={:.3} p90={:.3} p99={:.3} min={:.3} max={:.3} full={} ge_0.8={} ge_0.5={}",
            ef_search,
            num_queries,
            mean,
            p50,
            p90,
            p99,
            min,
            max,
            full,
            ge_08,
            ge_05
        ));
        logs.log_info(&format!(
            "🎯 [ef_search={}] recall@{}: {:.3} over {} queries (hits {}/{}) | avg {:.3} ms/query",
            ef_search, top_k, recall, num_queries, hits, total_targets, avg_ms
        ));
        let latency = summarize_f64(&query_stats.elapsed_ms);
        logs.log_info(&format!(
            "[query_stats] ef_search={} ms(p50/p90/p99)={:.3}/{:.3}/{:.3}",
            ef_search, latency.p50, latency.p90, latency.p99
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
        summary.push((ef_search, recall, avg_ms));
    }

    logs.log_info("\nSummary (ef_search -> recall, ms/query):");
    for (ef, recall, ms) in summary {
        logs.log_info(&format!("  {} -> {:.3}, {:.3} ms/query", ef, recall, ms));
    }
    log_peak_rss("hnm_recall_complete");
}
