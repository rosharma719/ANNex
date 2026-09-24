use std::fs::File;
use std::io::BufWriter;
use std::sync::Mutex;

use crate::utils::env::{env_bool, env_string, env_usize, env_usize_nonzero};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HumanLogLevel {
    Quiet,
    Info,
    Debug,
}

impl HumanLogLevel {
    pub fn from_env_var(key: &str, default: &str) -> Self {
        let raw = env_string(key).unwrap_or_else(|| default.to_string());
        match raw.trim().to_ascii_lowercase().as_str() {
            "quiet" | "silent" | "off" => HumanLogLevel::Quiet,
            "debug" | "trace" => HumanLogLevel::Debug,
            _ => HumanLogLevel::Info,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HarnessTelemetryConfig {
    pub level: HumanLogLevel,
    pub progress_every: usize,
    pub insert_progress_every: usize,
    pub query_log_path: Option<String>,
    pub query_log_every: usize,
}

impl HarnessTelemetryConfig {
    pub fn from_env() -> Self {
        Self {
            level: HumanLogLevel::from_env_var("VECTORDB_TEST_LOG", "info"),
            progress_every: env_usize("VECTORDB_PROGRESS_EVERY")
                .filter(|&value| value > 0)
                .unwrap_or(100),
            insert_progress_every: env_usize("VECTORDB_INSERT_PROGRESS_EVERY").unwrap_or(1000),
            query_log_path: env_string("VECTORDB_QUERY_LOG"),
            query_log_every: env_usize("VECTORDB_QUERY_LOG_EVERY")
                .filter(|&value| value > 0)
                .unwrap_or(1),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HnswTelemetryConfig {
    pub log_unfiltered_search: bool,
    pub unfiltered_log_every: usize,
    pub filter_seed_log_every: usize,
    pub filter_search_log_path: Option<String>,
    pub search_trace_log_path: Option<String>,
    pub insert_trace_log_path: Option<String>,
    pub trace_every: usize,
}

impl HnswTelemetryConfig {
    pub fn from_env() -> Self {
        Self {
            log_unfiltered_search: env_bool("VECTORDB_LOG_UNFILTERED_SEARCH").unwrap_or(false),
            unfiltered_log_every: env_usize_nonzero("VECTORDB_LOG_UNFILTERED_EVERY")
                .unwrap_or(1000),
            filter_seed_log_every: env_usize_nonzero("VECTORDB_LOG_FILTER_SEED_EVERY")
                .unwrap_or(100),
            filter_search_log_path: env_string("VECTORDB_FILTER_SEARCH_LOG"),
            search_trace_log_path: env_string("VECTORDB_SEARCH_TRACE_LOG"),
            insert_trace_log_path: env_string("VECTORDB_INSERT_TRACE_LOG"),
            trace_every: env_usize_nonzero("VECTORDB_TRACE_EVERY").unwrap_or(100),
        }
    }
}

pub fn jsonl_sink(path: Option<&str>) -> Option<Mutex<BufWriter<File>>> {
    path.and_then(|value| File::create(value).ok())
        .map(|file| Mutex::new(BufWriter::new(file)))
}
