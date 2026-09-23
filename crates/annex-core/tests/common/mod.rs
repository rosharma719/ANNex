#![allow(dead_code)]

use std::convert::TryInto;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use serde::Serialize;
use serde_json;

use annex::payload_storage::filters::Filter;
use annex::segment::segment::Segment;
use annex::utils::env::{
    env_bool_first as shared_env_bool_first, env_string,
    env_string_first as shared_env_string_first, env_usize_first as shared_env_usize_first,
    env_usize_list_first as shared_env_usize_list_first,
};
use annex::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
use annex::utils::telemetry::{HarnessTelemetryConfig, HumanLogLevel};
use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;

/// Return the first present env var (by key) parsed as usize.
pub fn env_usize_first(keys: &[&str]) -> Option<usize> {
    shared_env_usize_first(keys)
}

/// Return the first present env var (by key) parsed as a comma list of usize.
pub fn env_usize_list_first(keys: &[&str]) -> Option<Vec<usize>> {
    shared_env_usize_list_first(keys)
}

pub fn env_string_first(keys: &[&str]) -> Option<String> {
    shared_env_string_first(keys)
}

pub fn env_bool_first(keys: &[&str]) -> Option<bool> {
    shared_env_bool_first(keys)
}

#[cfg(unix)]
pub fn peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let res = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if res != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let raw = usage.ru_maxrss as u64;
    #[cfg(target_os = "macos")]
    {
        Some(raw)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Some(raw.saturating_mul(1024))
    }
}

#[cfg(not(unix))]
pub fn peak_rss_bytes() -> Option<u64> {
    None
}

pub fn log_peak_rss(label: &str) {
    if let Some(bytes) = peak_rss_bytes() {
        let mb = bytes as f64 / (1024.0 * 1024.0);
        println!("🧠 peak_rss {} = {:.2} MiB", label, mb);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TestLogLevel {
    Quiet,
    Info,
    Debug,
}

impl TestLogLevel {
    pub fn from_env() -> Self {
        match HumanLogLevel::from_env_var("VECTORDB_TEST_LOG", "info") {
            HumanLogLevel::Quiet => TestLogLevel::Quiet,
            HumanLogLevel::Info => TestLogLevel::Info,
            HumanLogLevel::Debug => TestLogLevel::Debug,
        }
    }

    pub fn allows_info(self) -> bool {
        self >= TestLogLevel::Info
    }

    pub fn allows_debug(self) -> bool {
        self >= TestLogLevel::Debug
    }
}

#[derive(Clone, Debug)]
pub struct TestLogConfig {
    pub level: TestLogLevel,
    pub progress_every: usize,
    pub insert_progress_every: usize,
    pub query_log_path: Option<String>,
    pub query_log_every: usize,
}

impl TestLogConfig {
    pub fn from_env() -> Self {
        let telemetry = HarnessTelemetryConfig::from_env();
        let level = match telemetry.level {
            HumanLogLevel::Quiet => TestLogLevel::Quiet,
            HumanLogLevel::Info => TestLogLevel::Info,
            HumanLogLevel::Debug => TestLogLevel::Debug,
        };
        Self {
            level,
            progress_every: telemetry.progress_every,
            insert_progress_every: telemetry.insert_progress_every,
            query_log_path: telemetry.query_log_path,
            query_log_every: telemetry.query_log_every,
        }
    }

    pub fn log_info(&self, msg: &str) {
        if self.level.allows_info() {
            println!("{}", msg);
        }
    }

    pub fn log_debug(&self, msg: &str) {
        if self.level.allows_debug() {
            println!("{}", msg);
        }
    }
}

#[derive(Clone, Debug)]
pub struct DatasetBuildConfig {
    pub m: usize,
    pub m0: usize,
    pub stored_cap_l0: usize,
    pub max_level: usize,
}

impl DatasetBuildConfig {
    pub fn from_env(prefix: &str, default_m: usize, default_max_level: usize) -> Self {
        let m =
            env_usize_first(&["VECTORDB_M", &format!("VECTORDB_{prefix}_M")]).unwrap_or(default_m);
        let m0 =
            env_usize_first(&["VECTORDB_M0", &format!("VECTORDB_{prefix}_M0")]).unwrap_or(m * 2);
        let stored_cap_l0 = env_usize_first(&[
            "VECTORDB_STORED_CAP_L0",
            &format!("VECTORDB_{prefix}_STORED_CAP_L0"),
        ])
        .unwrap_or(m0);
        let max_level = env_usize_first(&[
            "VECTORDB_MAX_LEVEL",
            &format!("VECTORDB_{prefix}_MAX_LEVEL"),
        ])
        .unwrap_or(default_max_level);
        Self {
            m,
            m0,
            stored_cap_l0,
            max_level,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DatasetHarnessConfig {
    pub data_dir: String,
    pub search: SearchConfig,
    pub snapshot: SnapshotConfig,
    pub build: DatasetBuildConfig,
}

impl DatasetHarnessConfig {
    pub fn from_env(
        prefix: &str,
        default_data_dir: &str,
        default_snapshot_path: &str,
        default_ef_values: &[usize],
        default_queries: usize,
        default_m: usize,
        default_max_level: usize,
    ) -> Self {
        Self {
            data_dir: env_string(&format!("VECTORDB_{prefix}_DATA_DIR"))
                .unwrap_or_else(|| default_data_dir.to_string()),
            search: SearchConfig::from_env(prefix, default_ef_values, default_queries),
            snapshot: SnapshotConfig::from_env(prefix, default_snapshot_path),
            build: DatasetBuildConfig::from_env(prefix, default_m, default_max_level),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SnapshotConfig {
    pub persist_path: String,
    pub use_snapshot: bool,
    pub allow_build: bool,
    pub save_snapshot: bool,
}

impl SnapshotConfig {
    pub fn from_env(prefix: &str, default_path: &str) -> Self {
        let persist_path = env_string_first(&[
            "VECTORDB_PERSIST_PATH",
            &format!("VECTORDB_{prefix}_PERSIST_PATH"),
        ])
        .unwrap_or_else(|| default_path.to_string());
        let use_snapshot = env_bool_first(&[
            "VECTORDB_USE_SNAPSHOT",
            &format!("VECTORDB_{prefix}_USE_SNAPSHOT"),
        ])
        .unwrap_or(true);
        let allow_build = env_bool_first(&[
            "VECTORDB_ALLOW_BUILD",
            &format!("VECTORDB_{prefix}_ALLOW_BUILD"),
        ])
        .unwrap_or(false);
        let save_snapshot = env_bool_first(&[
            "VECTORDB_SAVE_SNAPSHOT",
            &format!("VECTORDB_{prefix}_SAVE_SNAPSHOT"),
        ])
        .unwrap_or(false);
        Self {
            persist_path,
            use_snapshot,
            allow_build,
            save_snapshot,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub top_k: Option<usize>,
    pub ef_values: Vec<usize>,
    pub queries_cap: usize,
    pub base_cap: Option<usize>,
    pub ef_construct: usize,
}

impl SearchConfig {
    pub fn from_env(prefix: &str, default_ef_values: &[usize], default_queries: usize) -> Self {
        let top_k = env_usize_first(&["VECTORDB_TOPK", &format!("VECTORDB_{prefix}_TOPK")]);
        let ef_values = env_usize_list_first(&[
            "VECTORDB_EF_SEARCH_LIST",
            &format!("VECTORDB_{prefix}_EF_SEARCH_LIST"),
        ])
        .or_else(|| {
            env_usize_first(&[
                "VECTORDB_EF_SEARCH",
                &format!("VECTORDB_{prefix}_EF_SEARCH"),
            ])
            .map(|v| vec![v])
        })
        .unwrap_or_else(|| default_ef_values.to_vec());
        let queries_cap =
            env_usize_first(&["VECTORDB_QUERIES", &format!("VECTORDB_{prefix}_QUERIES")])
                .unwrap_or(default_queries);
        let base_cap = env_usize_first(&[
            "VECTORDB_BASE_LIMIT",
            &format!("VECTORDB_{prefix}_BASE_LIMIT"),
        ]);
        let ef_construct = env_usize_first(&[
            "VECTORDB_EF_CONSTRUCT",
            &format!("VECTORDB_{prefix}_EF_CONSTRUCT"),
        ])
        .unwrap_or(200);
        Self {
            top_k,
            ef_values,
            queries_cap,
            base_cap,
            ef_construct,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StatSummary {
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
}

pub fn summarize_f64(vals: &[f64]) -> StatSummary {
    if vals.is_empty() {
        return StatSummary {
            mean: 0.0,
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        sorted[idx]
    };
    StatSummary {
        mean,
        p50: pct(50.0),
        p90: pct(90.0),
        p99: pct(99.0),
        min: *sorted.first().unwrap(),
        max: *sorted.last().unwrap(),
    }
}

pub fn summarize_usize(vals: &[usize]) -> StatSummary {
    let floats: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
    summarize_f64(&floats)
}

pub struct QueryStatsAgg {
    pub elapsed_ms: Vec<f64>,
    pub visited: Vec<usize>,
    pub expanded: Vec<usize>,
    pub adjacency_reads: Vec<usize>,
    pub distance_computations: Vec<usize>,
    pub cap_breaks: Vec<usize>,
    pub patience_breaks: Vec<usize>,
    pub recall: Vec<f64>,
}

impl QueryStatsAgg {
    pub fn new(capacity: usize) -> Self {
        Self {
            elapsed_ms: Vec::with_capacity(capacity),
            visited: Vec::with_capacity(capacity),
            expanded: Vec::with_capacity(capacity),
            adjacency_reads: Vec::with_capacity(capacity),
            distance_computations: Vec::with_capacity(capacity),
            cap_breaks: Vec::with_capacity(capacity),
            patience_breaks: Vec::with_capacity(capacity),
            recall: Vec::with_capacity(capacity),
        }
    }

    pub fn record(
        &mut self,
        elapsed_ms: f64,
        visited: Option<usize>,
        expanded: Option<usize>,
        adjacency_reads: Option<usize>,
        distance_computations: Option<usize>,
        cap_breaks: Option<usize>,
        patience_breaks: Option<usize>,
        recall: f64,
    ) {
        self.elapsed_ms.push(elapsed_ms);
        if let Some(v) = visited {
            self.visited.push(v);
        }
        if let Some(e) = expanded {
            self.expanded.push(e);
        }
        if let Some(value) = adjacency_reads {
            self.adjacency_reads.push(value);
        }
        if let Some(value) = distance_computations {
            self.distance_computations.push(value);
        }
        if let Some(value) = cap_breaks {
            self.cap_breaks.push(value);
        }
        if let Some(value) = patience_breaks {
            self.patience_breaks.push(value);
        }
        self.recall.push(recall);
    }
}

#[derive(Default)]
pub struct MissStats {
    pub total: usize,
    pub level_counts: Vec<usize>,
    pub degree_samples: Vec<usize>,
}

impl MissStats {
    pub fn record(&mut self, level: usize, degree: Option<usize>) {
        if self.level_counts.len() <= level {
            self.level_counts.resize(level + 1, 0);
        }
        self.level_counts[level] += 1;
        self.total += 1;
        if let Some(d) = degree {
            self.degree_samples.push(d);
        }
    }
}

pub struct QueryLogWriter {
    writer: Option<BufWriter<File>>,
    every: usize,
}

impl QueryLogWriter {
    pub fn new(path: Option<String>, every: usize) -> Self {
        let writer = path.and_then(|p| File::create(p).ok()).map(BufWriter::new);
        Self {
            writer,
            every: every.max(1),
        }
    }

    pub fn write<T: Serialize>(&mut self, idx: usize, entry: &T) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        if idx % self.every != 0 {
            return;
        }
        if serde_json::to_writer(&mut *writer, entry).is_ok() {
            let _ = writer.write_all(b"\n");
        }
    }
}

pub fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}

pub fn generate_vector(i: usize) -> Vector {
    vecf(&[
        (i as f32).sin() * 5.0,
        ((i * 3) as f32).cos() * 3.0,
        ((i % 7) as f32).sqrt(),
    ])
}

pub fn generate_vector_dim(i: usize, dim: usize) -> Vector {
    // Simple deterministic vector of given dimension (sinusoidal manifold)
    (0..dim).map(|d| ((i + d) as f32).sin()).collect()
}

pub fn generate_payload(i: usize) -> Payload {
    let mut payload = Payload::default();
    payload.set("index", PayloadValue::Int(i.try_into().unwrap()));
    payload.set(
        "animal",
        PayloadValue::Str(
            match i % 4 {
                0 => "dog",
                1 => "cat",
                2 => "bird",
                _ => "fish",
            }
            .to_string(),
        ),
    );
    payload.set("age", PayloadValue::Int((i % 8 + 1).try_into().unwrap()));
    payload.set(
        "score",
        PayloadValue::Float((60.0 + (i % 40) as f64).into()),
    );
    payload.set(
        "tags",
        PayloadValue::ListStr(if i % 2 == 0 {
            vec!["cheap".to_string(), "small".to_string()]
        } else {
            vec!["expensive".to_string(), "large".to_string()]
        }),
    );
    payload.set("active", PayloadValue::Bool(i % 3 == 0));
    payload
}

// === Insertion Benchmark ===
pub fn bench_insertion(metric: DistanceMetric, size: usize) -> Segment {
    println!("\n🛠️ Inserting {} points with {:?} metric", size, metric);
    let hnsw = HNSWIndex::new(metric, 16, 64, 16, 3);
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(100);

    let start = Instant::now();
    for i in 0..size {
        let vec = generate_vector(i);
        let payload = generate_payload(i);
        segment.insert(vec, Some(payload)).unwrap();
    }
    println!("✅ Insertion took {:?}", start.elapsed());
    segment
}

// === Deletion Benchmark ===
pub fn bench_deletion(segment: &mut Segment, size: usize) {
    println!("\n❌ Deleting every 7th point...");
    let start = Instant::now();
    for i in (0..size).step_by(7) {
        let _ = segment.delete((i + 1) as u64); // IDs start at 1
    }
    println!("Sparse deletion took {:?}", start.elapsed());

    println!("🧹 Full deletion...");
    let start = Instant::now();
    for i in 0..size {
        let _ = segment.delete((i + 1) as u64);
    }
    println!("Full deletion (purge) took {:?}", start.elapsed());
}

pub fn bench_search(segment: &Segment, query: &Vector) {
    println!("\n🔍 Basic search...");
    let start = Instant::now();
    let _ = segment.search(query, 10).unwrap();
    println!("Basic search took {:?}", start.elapsed());

    let filter = Filter::And(vec![
        Filter::Match {
            key: "animal".into(),
            value: PayloadValue::Str("dog".into()),
        },
        Filter::Compare {
            key: "age".into(),
            op: ScalarComparisonOp::Gte,
            value: PayloadValue::Int(6),
        },
        Filter::Compare {
            key: "score".into(),
            op: ScalarComparisonOp::Lt,
            value: PayloadValue::Float(90.0.into()),
        },
    ]);

    println!("🧃 Filtered search...");
    let start = Instant::now();
    let _ = segment
        .search_with_filter(query, 10, Some(&filter))
        .unwrap();
    println!("Filtered search took {:?}", start.elapsed());

    let tag_filter = Filter::Compare {
        key: "tags".into(),
        op: ScalarComparisonOp::Eq,
        value: PayloadValue::Str("cheap".into()),
    };

    println!("📦 List match search...");
    let start = Instant::now();
    let _ = segment
        .search_with_filter(query, 10, Some(&tag_filter))
        .unwrap();
    println!("List filter took {:?}", start.elapsed());
}
