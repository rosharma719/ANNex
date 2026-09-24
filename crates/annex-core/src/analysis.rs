use std::path::PathBuf;

use anyhow::{Context, Result};
use ndarray::{Array2, Axis};
use ndarray_npy::read_npy;
use rand::prelude::SliceRandom;
use rand::rng;
use serde::Serialize;

use crate::segment::Segment;
use crate::utils::types::{DistanceMetric, Vector};
use crate::vector::hnsw::HNSWIndex;
use crate::vector::metric::score;

const DEFAULT_BASE: &str = "data/nytimes-256-angular/base.npy";
const DEFAULT_QUERIES: &str = "data/nytimes-256-angular/queries.npy";

/// Lightweight config used by the analyzer/sweeper.
#[derive(Clone)]
pub struct AnalyzerConfig {
    pub snapshot_path: PathBuf,
    pub base_path: PathBuf,
    pub queries_path: PathBuf,
    pub top_k: usize,
    pub num_queries: usize,
    pub sample_size: usize,
    pub neighbor_scan_cap: usize,
}

impl AnalyzerConfig {
    pub fn from_paths(
        snapshot: PathBuf,
        base: Option<PathBuf>,
        queries: Option<PathBuf>,
        top_k: usize,
        num_queries: usize,
        sample_size: usize,
        neighbor_scan_cap: usize,
    ) -> Self {
        AnalyzerConfig {
            snapshot_path: snapshot,
            base_path: base.unwrap_or_else(|| DEFAULT_BASE.into()),
            queries_path: queries.unwrap_or_else(|| DEFAULT_QUERIES.into()),
            top_k,
            num_queries,
            sample_size,
            neighbor_scan_cap,
        }
    }
}

#[derive(Serialize)]
pub struct DegreeStats {
    pub level: usize,
    pub nodes: usize,
    pub min: usize,
    pub p50: usize,
    pub p90: usize,
    pub p99: usize,
    pub max: usize,
    pub avg: f64,
    pub tail_ratio: f64,
}

#[derive(Serialize)]
pub struct PercentileStatsF64 {
    pub min: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
    pub avg: f64,
}

#[derive(Serialize)]
pub struct PercentileStatsF32 {
    pub min: f32,
    pub p50: f32,
    pub p90: f32,
    pub p99: f32,
    pub max: f32,
    pub avg: f32,
}

#[derive(Serialize)]
pub struct QueryStats {
    pub visited: PercentileStatsF64,
    pub expanded: PercentileStatsF64,
    pub visit_to_expansion: PercentileStatsF64,
    pub best_score: PercentileStatsF32,
    pub worst_score: PercentileStatsF32,
    pub neighbor_scan: NeighborScanStats,
    pub top_k: usize,
}

#[derive(Serialize)]
pub struct NeighborScanStats {
    pub adjacency_reads: PercentileStatsF64,
    pub distance_computations: PercentileStatsF64,
    pub cap_breaks: PercentileStatsF64,
    pub patience_breaks: PercentileStatsF64,
}

#[derive(Serialize)]
pub struct DistanceStats {
    pub min: f32,
    pub p50: f32,
    pub p90: f32,
    pub p99: f32,
    pub max: f32,
    pub avg: f32,
}

#[derive(Serialize)]
pub struct AnalysisResult {
    pub snapshot: String,
    pub metric: DistanceMetric,
    pub config: AnalyzerConfigSummary,
    pub degrees: Vec<DegreeStats>,
    pub query_stats: QueryStats,
    pub distance_stats: DistanceStats,
}

#[derive(Serialize)]
pub struct AnalyzerConfigSummary {
    pub top_k: usize,
    pub num_queries: usize,
    pub sample_size: usize,
    pub neighbor_scan_cap: usize,
}

pub fn analyze_snapshot(config: AnalyzerConfig) -> Result<AnalysisResult> {
    let snapshot_name = config
        .snapshot_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("snapshot")
        .to_string();
    let segment = Segment::load_from_path(&config.snapshot_path)
        .with_context(|| format!("failed to load snapshot {:?}", config.snapshot_path))?;
    let hnsw = segment.hnsw();
    let metric = hnsw.metric();

    let base_vectors = load_vectors(&config.base_path)?;
    let queries = load_vectors(&config.queries_path)?;

    let normalized_base_vectors = if metric == DistanceMetric::Cosine {
        Some(
            base_vectors
                .iter()
                .map(|v| hnsw.maybe_normalize(v))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let normalized_base_slice = normalized_base_vectors.as_ref().map(|vec| vec.as_slice());

    let degrees = compute_degree_stats(hnsw, config.neighbor_scan_cap);

    let (query_stats, distance_stats) = run_query_stats(
        &segment,
        hnsw,
        metric,
        &queries,
        &base_vectors,
        normalized_base_slice,
        &config,
    )?;

    Ok(AnalysisResult {
        snapshot: snapshot_name,
        metric,
        config: AnalyzerConfigSummary {
            top_k: config.top_k,
            num_queries: config.num_queries,
            sample_size: config.sample_size,
            neighbor_scan_cap: config.neighbor_scan_cap,
        },
        degrees,
        query_stats,
        distance_stats,
    })
}

fn compute_degree_stats(hnsw: &HNSWIndex, cap: usize) -> Vec<DegreeStats> {
    let total = hnsw.len();
    let current_max = hnsw.current_max_level();
    let mut stats = Vec::new();
    for level in 0..=current_max {
        let mut degrees = Vec::with_capacity(total);
        for idx in 0..total {
            let degree = hnsw
                .layer_neighbors(level, idx)
                .map(|neighbors| neighbors.len())
                .unwrap_or(0);
            degrees.push(degree);
        }
        degrees.sort_unstable();
        if degrees.is_empty() {
            continue;
        }
        let nodes = degrees.len();
        let tail = if cap == 0 {
            0.0
        } else {
            degrees.iter().filter(|&&deg| deg > cap).count() as f64 / nodes as f64
        };
        stats.push(DegreeStats {
            level,
            nodes,
            min: *degrees.first().unwrap_or(&0),
            max: *degrees.last().unwrap_or(&0),
            p50: quantile_usize(&degrees, 50.0),
            p90: quantile_usize(&degrees, 90.0),
            p99: quantile_usize(&degrees, 99.0),
            avg: degrees.iter().copied().sum::<usize>() as f64 / nodes as f64,
            tail_ratio: tail,
        });
    }
    stats
}

fn run_query_stats(
    segment: &Segment,
    hnsw: &HNSWIndex,
    metric: DistanceMetric,
    queries: &[Vector],
    base_vectors: &[Vector],
    normalized_base_vectors: Option<&[Vector]>,
    config: &AnalyzerConfig,
) -> Result<(QueryStats, DistanceStats)> {
    let mut visited = Vec::with_capacity(config.num_queries);
    let mut expanded = Vec::with_capacity(config.num_queries);
    let mut best_scores = Vec::with_capacity(config.num_queries);
    let mut worst_scores = Vec::with_capacity(config.num_queries);
    let mut distance_samples = Vec::with_capacity(config.num_queries * config.sample_size);
    let mut visit_exp_ratio = Vec::with_capacity(config.num_queries);
    let mut adjacency_reads = Vec::with_capacity(config.num_queries);
    let mut distance_computations = Vec::with_capacity(config.num_queries);
    let mut cap_breaks = Vec::with_capacity(config.num_queries);
    let mut patience_breaks = Vec::with_capacity(config.num_queries);

    let sampled_indexes = sample_indexes(base_vectors.len(), config.sample_size);

    for (qi, query) in queries.iter().take(config.num_queries).enumerate() {
        let prepared_query = if metric == DistanceMetric::Cosine {
            hnsw.maybe_normalize(query)
        } else {
            query.clone()
        };

        let (_, stats) = segment.search_with_stats(&prepared_query, config.top_k)?;
        visited.push(stats.visited as f64);
        expanded.push(stats.expanded as f64);
        best_scores.push(stats.best_score);
        worst_scores.push(stats.worst_score);
        adjacency_reads.push(stats.adjacency_reads as f64);
        distance_computations.push(stats.distance_computations as f64);
        cap_breaks.push(stats.cap_breaks as f64);
        patience_breaks.push(stats.patience_breaks as f64);
        let visit_to_expansion = if stats.expanded == 0 {
            stats.visited as f64
        } else {
            stats.visited as f64 / stats.expanded as f64
        };
        visit_exp_ratio.push(visit_to_expansion);

        for &idx in &sampled_indexes {
            let sample = if let Some(norm_bases) = normalized_base_vectors {
                &norm_bases[idx]
            } else {
                &base_vectors[idx]
            };
            let raw = score(&prepared_query, sample, metric);
            distance_samples.push(raw);
        }

        if (qi + 1) % 20 == 0 {
            println!("  processed {}/{} queries", qi + 1, config.num_queries);
        }
    }

    let distance_percentiles = percentiles_f32(&distance_samples);
    Ok((
        QueryStats {
            visited: percentiles_f64(&visited),
            expanded: percentiles_f64(&expanded),
            visit_to_expansion: percentiles_f64(&visit_exp_ratio),
            best_score: percentiles_f32(&best_scores),
            worst_score: percentiles_f32(&worst_scores),
            neighbor_scan: NeighborScanStats {
                adjacency_reads: percentiles_f64(&adjacency_reads),
                distance_computations: percentiles_f64(&distance_computations),
                cap_breaks: percentiles_f64(&cap_breaks),
                patience_breaks: percentiles_f64(&patience_breaks),
            },
            top_k: config.top_k,
        },
        DistanceStats {
            min: distance_percentiles.min,
            p50: distance_percentiles.p50,
            p90: distance_percentiles.p90,
            p99: distance_percentiles.p99,
            max: distance_percentiles.max,
            avg: distance_percentiles.avg,
        },
    ))
}

fn load_vectors(path: &PathBuf) -> Result<Vec<Vector>> {
    let array: Array2<f32> =
        read_npy(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(array.axis_iter(Axis(0)).map(|row| row.to_vec()).collect())
}

fn sample_indexes(dataset_len: usize, sample_size: usize) -> Vec<usize> {
    let mut rng = rng();
    let mut indexes: Vec<usize> = (0..dataset_len).collect();
    indexes.shuffle(&mut rng);
    indexes.truncate(sample_size.min(indexes.len()));
    indexes
}

fn percentiles_f64(values: &[f64]) -> PercentileStatsF64 {
    if values.is_empty() {
        return PercentileStatsF64 {
            min: 0.0,
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
            max: 0.0,
            avg: 0.0,
        };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    PercentileStatsF64 {
        min: *sorted.first().unwrap(),
        p50: quantile_f64(&sorted, 50.0),
        p90: quantile_f64(&sorted, 90.0),
        p99: quantile_f64(&sorted, 99.0),
        max: *sorted.last().unwrap(),
        avg: sorted.iter().copied().sum::<f64>() / sorted.len() as f64,
    }
}

fn percentiles_f32(values: &[f32]) -> PercentileStatsF32 {
    if values.is_empty() {
        return PercentileStatsF32 {
            min: 0.0,
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
            max: 0.0,
            avg: 0.0,
        };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    PercentileStatsF32 {
        min: *sorted.first().unwrap(),
        p50: quantile_f32(&sorted, 50.0),
        p90: quantile_f32(&sorted, 90.0),
        p99: quantile_f32(&sorted, 99.0),
        max: *sorted.last().unwrap(),
        avg: sorted.iter().copied().sum::<f32>() / sorted.len() as f32,
    }
}

fn quantile_usize(values: &[usize], pct: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let idx = ((pct / 100.0) * (values.len() - 1) as f64).round() as usize;
    values[idx]
}

fn quantile_f64(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * (values.len() - 1) as f64).round() as usize;
    values[idx]
}

fn quantile_f32(values: &[f32], pct: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * (values.len() - 1) as f64).round() as usize;
    values[idx]
}
