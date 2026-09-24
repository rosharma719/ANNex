use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use rand::Rng;
use serde::Serialize;

use annex::segment::Segment;
use annex::utils::types::DistanceMetric;
use annex::vector::hnsw::HNSWIndex;
use annex::vector::metric::score;

#[derive(Default)]
struct Cli {
    snapshot: Option<PathBuf>,
    snapshots_file: Option<PathBuf>,
    pair_samples: usize,
}

impl Cli {
    fn from_args() -> Result<Self> {
        let mut cli = Cli {
            snapshot: None,
            snapshots_file: None,
            pair_samples: 5_000,
        };

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--snapshot" => {
                    cli.snapshot = Some(PathBuf::from(
                        args.next().context("expected path after --snapshot")?,
                    ));
                }
                "--snapshots-file" => {
                    cli.snapshots_file = Some(PathBuf::from(
                        args.next()
                            .context("expected path after --snapshots-file")?,
                    ));
                }
                "--pair-samples" => {
                    cli.pair_samples = args
                        .next()
                        .context("expected usize after --pair-samples")?
                        .replace('_', "")
                        .parse()
                        .context("failed to parse --pair-samples")?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown option {other}"),
            }
        }

        if cli.snapshot.is_none() && cli.snapshots_file.is_none() {
            return Err(anyhow!("either --snapshot or --snapshots-file is required"));
        }
        Ok(cli)
    }
}

#[derive(Serialize)]
struct IndexStatsRow {
    snapshot: String,
    snapshot_path: String,
    file_size_bytes: u64,

    metric: DistanceMetric,
    points: usize,
    deleted_count: usize,
    deleted_frac: f64,
    dim: usize,
    m: usize,
    m0: usize,
    ef: usize,
    ef_construct: usize,
    max_level_cap: usize,
    level_scale: f64,
    current_max_level: usize,

    level_avg: f64,
    level_p50: usize,
    level_p90: usize,
    level_p99: usize,

    l0_deg_avg: f64,
    l0_deg_p50: usize,
    l0_deg_p90: usize,
    l0_deg_p99: usize,
    l0_deg_max: usize,
    l0_deg_tail_over_32: f64,
    l0_deg_tail_over_64: f64,

    vec_norm_avg: f64,
    vec_norm_p50: f64,
    vec_norm_p90: f64,
    vec_norm_p99: f64,

    pair_score_avg: f64,
    pair_score_p50: f64,
    pair_score_p90: f64,
    pair_score_p99: f64,
}

fn main() -> Result<()> {
    let cli = Cli::from_args()?;

    let snapshots = if let Some(p) = cli.snapshot {
        vec![p]
    } else {
        let file = fs::File::open(cli.snapshots_file.unwrap()).context("open snapshots file")?;
        let reader = io::BufReader::new(file);
        reader
            .lines()
            .enumerate()
            .filter_map(|(i, line)| match line {
                Ok(s) => {
                    let s = s.trim();
                    if s.is_empty() || s.starts_with('#') {
                        None
                    } else {
                        Some(Ok(PathBuf::from(s)))
                    }
                }
                Err(e) => Some(Err(anyhow!("read snapshots file line {}: {}", i + 1, e))),
            })
            .collect::<Result<Vec<_>>>()?
    };

    println!("{}", csv_header());
    for path in snapshots {
        let row = compute_stats(&path, cli.pair_samples)
            .with_context(|| format!("compute stats for {:?}", path))?;
        println!("{}", row.to_csv());
    }
    Ok(())
}

fn compute_stats(snapshot_path: &Path, pair_samples: usize) -> Result<IndexStatsRow> {
    let meta = fs::metadata(snapshot_path)
        .with_context(|| format!("stat snapshot {:?}", snapshot_path))?;
    let file_size_bytes = meta.len();

    let (segment, metadata) = Segment::load_from_path_with_metadata(snapshot_path)
        .with_context(|| format!("load snapshot {:?}", snapshot_path))?;
    let hnsw = segment.hnsw();

    let snapshot = snapshot_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("snapshot")
        .to_string();

    let deleted_count = hnsw.deleted_count();
    let deleted_frac = hnsw.deleted_fraction();
    let points = hnsw.len();

    let cfg = if let Some(m) = metadata {
        m.hnsw
    } else {
        hnsw.config_summary()
    };

    let active_levels = hnsw.iter_active_levels().collect::<Vec<_>>();
    let (level_avg, level_p50, level_p90, level_p99) = level_stats(&active_levels);

    let l0_degrees = level_degrees(hnsw, 0);
    let (l0_deg_avg, l0_deg_p50, l0_deg_p90, l0_deg_p99, l0_deg_max) = degree_stats(&l0_degrees);
    let l0_deg_tail_over_32 = tail_ratio(&l0_degrees, 32);
    let l0_deg_tail_over_64 = tail_ratio(&l0_degrees, 64);

    let vectors = hnsw
        .iter_active_vectors()
        .map(|(_, v)| v)
        .collect::<Vec<_>>();
    let norms = vectors.iter().map(|v| l2_norm(v)).collect::<Vec<_>>();
    let (vec_norm_avg, vec_norm_p50, vec_norm_p90, vec_norm_p99) = f64_stats(&norms);

    let pair_scores = sample_pair_scores(&vectors, cfg.metric, pair_samples);
    let (pair_score_avg, pair_score_p50, pair_score_p90, pair_score_p99) = f64_stats(&pair_scores);

    Ok(IndexStatsRow {
        snapshot,
        snapshot_path: snapshot_path.display().to_string(),
        file_size_bytes,

        metric: cfg.metric,
        points,
        deleted_count,
        deleted_frac,
        dim: cfg.dim,
        m: cfg.m,
        m0: cfg.m0,
        ef: cfg.ef,
        ef_construct: cfg.ef_construct,
        max_level_cap: cfg.max_level_cap,
        level_scale: cfg.level_scale,
        current_max_level: cfg.current_max_level,

        level_avg,
        level_p50,
        level_p90,
        level_p99,

        l0_deg_avg,
        l0_deg_p50,
        l0_deg_p90,
        l0_deg_p99,
        l0_deg_max,
        l0_deg_tail_over_32,
        l0_deg_tail_over_64,

        vec_norm_avg,
        vec_norm_p50,
        vec_norm_p90,
        vec_norm_p99,

        pair_score_avg,
        pair_score_p50,
        pair_score_p90,
        pair_score_p99,
    })
}

fn level_degrees(hnsw: &HNSWIndex, level: usize) -> Vec<usize> {
    let mut degrees = Vec::with_capacity(hnsw.len());
    for point_id in hnsw.iter_active_vectors().map(|(id, _)| id) {
        let deg = hnsw
            .point_degree(point_id, level)
            .unwrap_or(0)
            .saturating_sub(1);
        degrees.push(deg);
    }
    degrees.sort_unstable();
    degrees
}

fn level_stats(levels: &[usize]) -> (f64, usize, usize, usize) {
    if levels.is_empty() {
        return (0.0, 0, 0, 0);
    }
    let mut sorted = levels.to_vec();
    sorted.sort_unstable();
    let avg = sorted.iter().copied().sum::<usize>() as f64 / sorted.len() as f64;
    (
        avg,
        quantile_usize(&sorted, 50.0),
        quantile_usize(&sorted, 90.0),
        quantile_usize(&sorted, 99.0),
    )
}

fn degree_stats(degrees: &[usize]) -> (f64, usize, usize, usize, usize) {
    if degrees.is_empty() {
        return (0.0, 0, 0, 0, 0);
    }
    let avg = degrees.iter().copied().sum::<usize>() as f64 / degrees.len() as f64;
    (
        avg,
        quantile_usize(degrees, 50.0),
        quantile_usize(degrees, 90.0),
        quantile_usize(degrees, 99.0),
        *degrees.last().unwrap_or(&0),
    )
}

fn tail_ratio(degrees: &[usize], cap: usize) -> f64 {
    if degrees.is_empty() {
        return 0.0;
    }
    if cap == 0 {
        return 0.0;
    }
    degrees.iter().filter(|&&d| d > cap).count() as f64 / degrees.len() as f64
}

fn l2_norm(v: &[f32]) -> f64 {
    let sum = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
    sum.sqrt()
}

fn sample_pair_scores(vectors: &[&[f32]], metric: DistanceMetric, samples: usize) -> Vec<f64> {
    if vectors.len() < 2 || samples == 0 {
        return vec![];
    }
    let mut rng = rand::rng();
    let mut out = Vec::with_capacity(samples.min(100_000));
    for _ in 0..samples {
        let i = rng.random_range(0..vectors.len());
        let mut j = rng.random_range(0..(vectors.len() - 1));
        if j >= i {
            j += 1;
        }
        out.push(score(vectors[i], vectors[j], metric) as f64);
    }
    out.sort_by(|x, y| x.partial_cmp(y).unwrap());
    out
}

fn f64_stats(values: &[f64]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let avg = values.iter().copied().sum::<f64>() / values.len() as f64;
    (
        avg,
        quantile_f64(values, 50.0),
        quantile_f64(values, 90.0),
        quantile_f64(values, 99.0),
    )
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

fn csv_header() -> &'static str {
    "snapshot,snapshot_path,file_size_bytes,metric,points,deleted_count,deleted_frac,dim,m,m0,ef,ef_construct,max_level_cap,level_scale,current_max_level,level_avg,level_p50,level_p90,level_p99,l0_deg_avg,l0_deg_p50,l0_deg_p90,l0_deg_p99,l0_deg_max,l0_deg_tail_over_32,l0_deg_tail_over_64,vec_norm_avg,vec_norm_p50,vec_norm_p90,vec_norm_p99,pair_score_avg,pair_score_p50,pair_score_p90,pair_score_p99"
}

impl IndexStatsRow {
    fn to_csv(&self) -> String {
        let mut parts = Vec::new();
        parts.push(escape_csv(&self.snapshot));
        parts.push(escape_csv(&self.snapshot_path));
        parts.push(self.file_size_bytes.to_string());
        parts.push(escape_csv(&format!("{:?}", self.metric)));
        parts.push(self.points.to_string());
        parts.push(self.deleted_count.to_string());
        parts.push(format!("{:.6}", self.deleted_frac));
        parts.push(self.dim.to_string());
        parts.push(self.m.to_string());
        parts.push(self.m0.to_string());
        parts.push(self.ef.to_string());
        parts.push(self.ef_construct.to_string());
        parts.push(self.max_level_cap.to_string());
        parts.push(format!("{:.6}", self.level_scale));
        parts.push(self.current_max_level.to_string());
        parts.push(format!("{:.6}", self.level_avg));
        parts.push(self.level_p50.to_string());
        parts.push(self.level_p90.to_string());
        parts.push(self.level_p99.to_string());

        parts.push(format!("{:.6}", self.l0_deg_avg));
        parts.push(self.l0_deg_p50.to_string());
        parts.push(self.l0_deg_p90.to_string());
        parts.push(self.l0_deg_p99.to_string());
        parts.push(self.l0_deg_max.to_string());
        parts.push(format!("{:.6}", self.l0_deg_tail_over_32));
        parts.push(format!("{:.6}", self.l0_deg_tail_over_64));

        parts.push(format!("{:.6}", self.vec_norm_avg));
        parts.push(format!("{:.6}", self.vec_norm_p50));
        parts.push(format!("{:.6}", self.vec_norm_p90));
        parts.push(format!("{:.6}", self.vec_norm_p99));

        parts.push(format!("{:.6}", self.pair_score_avg));
        parts.push(format!("{:.6}", self.pair_score_p50));
        parts.push(format!("{:.6}", self.pair_score_p90));
        parts.push(format!("{:.6}", self.pair_score_p99));

        parts.join(",")
    }
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

fn print_help() {
    println!(
        "\
Usage:
  cargo run --bin index_stats -- --snapshot <path> [--pair-samples N]
  cargo run --bin index_stats -- --snapshots-file <file> [--pair-samples N]

Output:
  CSV to stdout (one row per snapshot).

Notes:
  - Reads vectors + graph structure from the snapshot; no external .npy required.
  - pair-score stats use random vector pairs (query-agnostic dataset geometry proxy).
"
    );
}
