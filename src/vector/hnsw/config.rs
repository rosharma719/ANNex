use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::utils::env::{env_bool, env_f32, env_usize, env_usize_nonzero};
use crate::utils::telemetry::{HnswTelemetryConfig, jsonl_sink};

pub(crate) const VERBOSE: bool = false;
pub(crate) const DEFAULT_EXACT_FALLBACK_THRESHOLD: usize = 256;
pub(crate) const FILTER_EDGE_LOG_CHUNK: usize = 10_000;

static LOG_UNFILTERED_SEARCH: OnceLock<bool> = OnceLock::new();
static UNFILTERED_LOG_CHUNK: OnceLock<usize> = OnceLock::new();
static FILTER_SEED_LOG_CHUNK: OnceLock<usize> = OnceLock::new();
static HNSW_TELEMETRY: OnceLock<HnswTelemetryConfig> = OnceLock::new();
static DISABLE_EARLY_EXIT: OnceLock<bool> = OnceLock::new();
static SEARCH_EXPANSION_MULT: OnceLock<usize> = OnceLock::new();
static SEARCH_EXPANSION_CAP: OnceLock<Option<usize>> = OnceLock::new();
static EARLY_EXIT_PATIENCE: OnceLock<usize> = OnceLock::new();
static FILTER_SEARCH_LOG: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
static FILTER_EXPANSION_CAP: OnceLock<Option<usize>> = OnceLock::new();
static FILTER_PASSING_BUDGET: OnceLock<Option<usize>> = OnceLock::new();
static FILTER_FAILING_BUDGET: OnceLock<Option<usize>> = OnceLock::new();
static FILTER_SEARCH_SEQ: AtomicU64 = AtomicU64::new(0);
static SEARCH_TRACE_LOG: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
static INSERT_TRACE_LOG: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
static TRACE_EVERY: OnceLock<usize> = OnceLock::new();
static SEARCH_TRACE_SEQ: AtomicU64 = AtomicU64::new(0);
static INSERT_TRACE_SEQ: AtomicU64 = AtomicU64::new(0);
static EXACT_FALLBACK_ENABLED: OnceLock<Option<bool>> = OnceLock::new();
static EXACT_FALLBACK_THRESHOLD: OnceLock<Option<usize>> = OnceLock::new();
static FILTER_ENTRY_CANDIDATES: OnceLock<Option<usize>> = OnceLock::new();
static NEIGHBOR_SCAN_CAP_LEVEL0: OnceLock<Option<usize>> = OnceLock::new();
static NEIGHBOR_SCAN_ROTATE: OnceLock<Option<bool>> = OnceLock::new();
static NEIGHBOR_SCAN_STRIDE: OnceLock<Option<bool>> = OnceLock::new();
static NEIGHBOR_SCAN_PATIENCE: OnceLock<Option<usize>> = OnceLock::new();
static NEIGHBOR_SCAN_STATE_LOGGED: OnceLock<()> = OnceLock::new();
static ENFORCE_NEIGHBOR_CAPS: OnceLock<Option<bool>> = OnceLock::new();
static NUM_ENTRY_SEEDS: OnceLock<Option<usize>> = OnceLock::new();
static ADAPTIVE_EF_HIGH: OnceLock<Option<usize>> = OnceLock::new();
static ADAPTIVE_EF_SCORE_THRESHOLD: OnceLock<Option<f32>> = OnceLock::new();
static DIVERSITY_ALPHA: OnceLock<Option<f32>> = OnceLock::new();
static DIVERSITY_ALPHA_LOW: OnceLock<Option<f32>> = OnceLock::new();
static DIVERSITY_ALPHA_HIGH: OnceLock<Option<f32>> = OnceLock::new();
static DIVERSITY_PRUNE_FLOOR: OnceLock<Option<usize>> = OnceLock::new();

fn hnsw_telemetry() -> &'static HnswTelemetryConfig {
    HNSW_TELEMETRY.get_or_init(HnswTelemetryConfig::from_env)
}

pub(crate) fn log_unfiltered_enabled() -> bool {
    *LOG_UNFILTERED_SEARCH.get_or_init(|| {
        let enabled = hnsw_telemetry().log_unfiltered_search;
        if enabled {
            let chunk = unfiltered_log_chunk();
            log::info!(
                target: "vector::hnsw",
                "[unfiltered_search_stats] enabled (chunk={}, set VECTORDB_LOG_UNFILTERED_EVERY to change)",
                chunk
            );
        }
        enabled
    })
}

pub(crate) fn unfiltered_log_chunk() -> usize {
    *UNFILTERED_LOG_CHUNK.get_or_init(|| hnsw_telemetry().unfiltered_log_every)
}

pub(crate) fn filter_seed_log_chunk() -> usize {
    *FILTER_SEED_LOG_CHUNK.get_or_init(|| hnsw_telemetry().filter_seed_log_every)
}

pub fn disable_early_exit() -> bool {
    *DISABLE_EARLY_EXIT.get_or_init(|| env_bool("VECTORDB_DISABLE_EARLY_EXIT").unwrap_or(false))
}

pub fn search_expansion_multiplier() -> usize {
    *SEARCH_EXPANSION_MULT
        .get_or_init(|| env_usize_nonzero("VECTORDB_SEARCH_EXPANSION_MULT").unwrap_or(2))
}

pub fn search_expansion_cap_override() -> Option<usize> {
    *SEARCH_EXPANSION_CAP.get_or_init(|| {
        env_usize("VECTORDB_SEARCH_EXPANSION_CAP")
            .and_then(|value| if value == 0 { None } else { Some(value) })
    })
}

pub fn early_exit_patience() -> usize {
    *EARLY_EXIT_PATIENCE.get_or_init(|| env_usize("VECTORDB_EARLY_EXIT_PATIENCE").unwrap_or(2))
}

pub(crate) fn filter_expansion_cap() -> Option<usize> {
    // Specific cap for filtered search; 0 means unbounded.
    *FILTER_EXPANSION_CAP.get_or_init(|| {
        env_usize("VECTORDB_FILTER_EXPANSION_CAP")
            .map(|value| if value == 0 { usize::MAX } else { value })
    })
}

pub(crate) fn filter_passing_budget(m: usize) -> usize {
    (*FILTER_PASSING_BUDGET.get_or_init(|| env_usize("VECTORDB_FILTER_PASSING_BUDGET")))
        .unwrap_or_else(|| std::cmp::max(8, m.saturating_mul(2)))
}

pub(crate) fn filter_failing_budget(m: usize) -> usize {
    (*FILTER_FAILING_BUDGET.get_or_init(|| env_usize("VECTORDB_FILTER_FAILING_BUDGET")))
        .unwrap_or_else(|| std::cmp::max(1, m / 8))
}

pub(crate) fn filter_search_logger() -> Option<&'static Mutex<BufWriter<File>>> {
    FILTER_SEARCH_LOG
        .get_or_init(|| jsonl_sink(hnsw_telemetry().filter_search_log_path.as_deref()))
        .as_ref()
}

pub(crate) fn next_filter_search_seq() -> u64 {
    FILTER_SEARCH_SEQ.fetch_add(1, Ordering::Relaxed) + 1
}

pub(crate) fn search_trace_logger() -> Option<&'static Mutex<BufWriter<File>>> {
    SEARCH_TRACE_LOG
        .get_or_init(|| jsonl_sink(hnsw_telemetry().search_trace_log_path.as_deref()))
        .as_ref()
}

pub(crate) fn insert_trace_logger() -> Option<&'static Mutex<BufWriter<File>>> {
    INSERT_TRACE_LOG
        .get_or_init(|| jsonl_sink(hnsw_telemetry().insert_trace_log_path.as_deref()))
        .as_ref()
}

pub(crate) fn trace_every() -> usize {
    *TRACE_EVERY.get_or_init(|| hnsw_telemetry().trace_every)
}

pub(crate) fn next_search_trace_seq() -> u64 {
    SEARCH_TRACE_SEQ.fetch_add(1, Ordering::Relaxed) + 1
}

pub(crate) fn next_insert_trace_seq() -> u64 {
    INSERT_TRACE_SEQ.fetch_add(1, Ordering::Relaxed) + 1
}

pub(crate) fn exact_fallback_enabled_override() -> Option<bool> {
    *EXACT_FALLBACK_ENABLED.get_or_init(|| env_bool("VECTORDB_EXACT_FALLBACK_ENABLED"))
}

pub(crate) fn exact_fallback_threshold_override() -> Option<usize> {
    *EXACT_FALLBACK_THRESHOLD.get_or_init(|| env_usize("VECTORDB_EXACT_FALLBACK_THRESHOLD"))
}

pub(crate) fn filter_entry_candidates() -> Option<usize> {
    *FILTER_ENTRY_CANDIDATES.get_or_init(|| env_usize_nonzero("VECTORDB_FILTER_ENTRY_CANDIDATES"))
}

pub fn neighbor_scan_cap(level: usize) -> usize {
    if level == 0 {
        NEIGHBOR_SCAN_CAP_LEVEL0
            .get_or_init(|| {
                env_usize("VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0")
                    .map(|value| if value == 0 { usize::MAX } else { value })
            })
            .unwrap_or(usize::MAX)
    } else {
        usize::MAX
    }
}

/// Seed the L0 neighbor-scan cap if it hasn't already been initialized (e.g., from env var
/// or a prior `neighbor_scan_cap` call). Returns true if the value was applied.
/// Use `Some(0)` or `None` to mean "no cap".
pub fn set_neighbor_scan_cap_level0_default(cap: Option<usize>) -> bool {
    let normalized = cap.and_then(|v| if v == 0 { None } else { Some(v) });
    NEIGHBOR_SCAN_CAP_LEVEL0.set(normalized).is_ok()
}

pub fn neighbor_scan_patience() -> usize {
    (*NEIGHBOR_SCAN_PATIENCE.get_or_init(|| env_usize_nonzero("VECTORDB_NEIGHBOR_SCAN_PATIENCE")))
        .unwrap_or(0)
}

pub fn neighbor_scan_rotate_enabled() -> bool {
    NEIGHBOR_SCAN_ROTATE
        .get_or_init(|| env_bool("VECTORDB_NEIGHBOR_SCAN_ROTATE"))
        .unwrap_or(false)
}

pub fn neighbor_scan_stride_enabled() -> bool {
    NEIGHBOR_SCAN_STRIDE
        .get_or_init(|| env_bool("VECTORDB_NEIGHBOR_SCAN_STRIDE"))
        .unwrap_or(false)
}

pub fn enforce_neighbor_caps() -> bool {
    ENFORCE_NEIGHBOR_CAPS
        .get_or_init(|| env_bool("VECTORDB_ENFORCE_NEIGHBOR_CAPS"))
        .unwrap_or(true)
}

pub(crate) fn log_neighbor_scan_state(expansion_mult: usize, expansion_cap: Option<usize>) {
    NEIGHBOR_SCAN_STATE_LOGGED.get_or_init(|| {
        let cap = neighbor_scan_cap(0);
        let rotate = neighbor_scan_rotate_enabled();
        let stride = neighbor_scan_stride_enabled();
        let cap_display = if cap == usize::MAX {
            "none".to_string()
        } else {
            cap.to_string()
        };
        log::info!(
            target: "vector::hnsw",
            "Neighbor scan cap L0={} rotation={} rotation_stride={} expansion_mult={} expansion_cap={}",
            cap_display,
            if rotate { "on" } else { "off" },
            if stride { "enabled" } else { "off" },
            expansion_mult,
            expansion_cap.map(|v| v.to_string()).unwrap_or_else(|| "none".into())
        );
    });
}

pub fn num_entry_seeds_default() -> Option<usize> {
    *NUM_ENTRY_SEEDS.get_or_init(|| env_usize_nonzero("VECTORDB_NUM_ENTRY_SEEDS"))
}

pub fn adaptive_ef_high_default() -> Option<usize> {
    *ADAPTIVE_EF_HIGH.get_or_init(|| env_usize_nonzero("VECTORDB_ADAPTIVE_EF_HIGH"))
}

pub fn adaptive_ef_score_threshold_default() -> Option<f32> {
    *ADAPTIVE_EF_SCORE_THRESHOLD.get_or_init(|| {
        env_f32("VECTORDB_ADAPTIVE_EF_SCORE_THRESHOLD").filter(|v| v.is_finite() && *v > 0.0)
    })
}

pub(crate) fn diversity_alpha_for_level(level: usize) -> f32 {
    if let Some(alpha) = DIVERSITY_ALPHA.get_or_init(|| {
        env_f32("VECTORDB_DIVERSITY_ALPHA").filter(|value| value.is_finite() && *value > 0.0)
    }) {
        return *alpha;
    }
    let low = DIVERSITY_ALPHA_LOW
        .get_or_init(|| {
            env_f32("VECTORDB_DIVERSITY_ALPHA_LOW")
                .filter(|value| value.is_finite() && *value > 0.0)
        })
        .unwrap_or(1.0);
    let high = DIVERSITY_ALPHA_HIGH
        .get_or_init(|| {
            env_f32("VECTORDB_DIVERSITY_ALPHA_HIGH")
                .filter(|value| value.is_finite() && *value > 0.0)
        })
        .unwrap_or(1.0);
    if level == 0 { low } else { high }
}

pub(crate) fn diversity_prune_floor() -> usize {
    DIVERSITY_PRUNE_FLOOR
        .get_or_init(|| env_usize("VECTORDB_DIVERSITY_PRUNE_FLOOR"))
        .unwrap_or(0)
}
