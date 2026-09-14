use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::to_string_pretty;

use annex::analysis::{AnalyzerConfig, analyze_snapshot};

const DEFAULT_BASE: &str = "data/nytimes-256-angular/base.npy";
const DEFAULT_QUERIES: &str = "data/nytimes-256-angular/queries.npy";

fn main() -> Result<()> {
    let cli = AnalyzerCli::from_args()?;
    let config = AnalyzerConfig::from_paths(
        cli.snapshot,
        Some(cli.base),
        Some(cli.queries),
        cli.top_k,
        cli.num_queries,
        cli.sample_size,
        cli.neighbor_scan_cap,
    );
    let result = analyze_snapshot(config)?;
    println!("{}", to_string_pretty(&result)?);
    Ok(())
}

struct AnalyzerCli {
    snapshot: PathBuf,
    base: PathBuf,
    queries: PathBuf,
    top_k: usize,
    num_queries: usize,
    sample_size: usize,
    neighbor_scan_cap: usize,
}

impl AnalyzerCli {
    fn from_args() -> Result<Self> {
        let mut snapshot = None;
        let mut base = PathBuf::from(DEFAULT_BASE);
        let mut queries = PathBuf::from(DEFAULT_QUERIES);
        let mut top_k = 10;
        let mut num_queries = 100;
        let mut sample_size = 500;
        let mut neighbor_scan_cap = 128;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--snapshot" => {
                    snapshot = Some(PathBuf::from(
                        args.next().context("expected path after --snapshot")?,
                    ));
                }
                "--base" => {
                    base = PathBuf::from(args.next().context("expected path after --base")?);
                }
                "--queries" => {
                    queries = PathBuf::from(args.next().context("expected path after --queries")?);
                }
                "--top-k" => {
                    top_k = args
                        .next()
                        .context("expected usize after --top-k")?
                        .parse()
                        .context("failed to parse top-k")?;
                }
                "--num-queries" => {
                    num_queries = args
                        .next()
                        .context("expected usize after --num-queries")?
                        .parse()
                        .context("failed to parse num-queries")?;
                }
                "--sample-size" => {
                    sample_size = args
                        .next()
                        .context("expected usize after --sample-size")?
                        .parse()
                        .context("failed to parse sample-size")?;
                }
                "--neighbor-scan-cap" => {
                    neighbor_scan_cap = args
                        .next()
                        .context("expected usize after --neighbor-scan-cap")?
                        .parse()
                        .context("failed to parse neighbor scan cap")?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    bail!("unknown option {}", other);
                }
            }
        }

        let snapshot = snapshot.ok_or_else(|| anyhow!("--snapshot is required"))?;
        Ok(AnalyzerCli {
            snapshot,
            base,
            queries,
            top_k,
            num_queries,
            sample_size,
            neighbor_scan_cap,
        })
    }
}

fn print_help() {
    println!(
        "\
Usage:
  cargo run --bin index_analyzer -- --snapshot <path> [options]

Options:
  --snapshot <path>            Path to snapshot (required)
  --base <path>                Base `.npy` (default data/nytimes-256-angular/base.npy)
  --queries <path>             Query `.npy` (default data/nytimes-256-angular/queries.npy)
  --top-k <usize>              Number of top candidates (default 10)
  --num-queries <usize>        Number of queries (default 100)
  --sample-size <usize>        Sample size for dataset distances (default 500)
  --neighbor-scan-cap <usize>  Cap used when computing level-0 tail ratio (default 128)
  --help, -h                   Print this help text
"
    );
}
