//! Planner regret benchmark.
//!
//! Synthesises a corpus, enumerates feasible plans, and computes regret of
//! ANNex's chosen plan vs an exhaustive exact-scan oracle.
//!
//! Usage:
//!   cargo run --release --bin regret-bench -- [OPTIONS]
//!   cargo run --release --bin regret-bench -- --json
use clap::Parser;
use multivector::{
    Durability, IndexConfig, MultiVectorIndex, UpsertDocument,
    regret::{PlanEvaluation, compute_regret, evaluate_plan, pareto_frontier, recall_at_k},
};
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Planner regret benchmark against an exact oracle")]
struct Args {
    #[arg(long, default_value_t = 50)]
    queries: usize,
    #[arg(long, default_value_t = 500)]
    corpus: usize,
    #[arg(long, default_value_t = 32)]
    dim: usize,
    /// Latency budget in ms (0 = no budget)
    #[arg(long, default_value_t = 0.0)]
    budget_ms: f64,
    /// Comma-separated ef_search values to sweep
    #[arg(long, default_value = "64,256,1024")]
    ef: String,
    #[arg(long)]
    json: bool,
    /// Working directory for the index (default: OS temp)
    #[arg(long)]
    workdir: Option<PathBuf>,
}

fn xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as f32 / u64::MAX as f32) * 2.0 - 1.0
}

fn gen_vector(state: &mut u64, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|_| xorshift(state)).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    raw.into_iter().map(|x| x / norm).collect()
}

fn build_index(corpus: usize, dim: usize, seed: u64, path: &std::path::Path) -> MultiVectorIndex {
    // Always start from scratch.
    if path.exists() {
        fs::remove_dir_all(path).unwrap();
    }
    fs::create_dir_all(path).unwrap();
    let config = IndexConfig::new(dim);
    let index =
        MultiVectorIndex::open_with_durability(path, config, Durability::Buffered).unwrap();
    let mut state = seed;
    let train_n = corpus.min(256).max(64);
    let samples: Vec<Vec<f32>> = (0..train_n).map(|_| gen_vector(&mut state, dim)).collect();
    index.train(&samples, 5).unwrap();
    let docs: Vec<_> = (0..corpus)
        .map(|i| UpsertDocument {
            id: format!("doc{i}"),
            vectors: vec![gen_vector(&mut state, dim)],
            metadata: json!({"cluster": i % 10}),
        })
        .collect();
    index.upsert_batch(docs).unwrap();
    index.build_fde_ann(16, 64).unwrap();
    index
}

fn main() {
    let args = Args::parse();
    let ef_values: Vec<usize> = args
        .ef
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let budget = if args.budget_ms > 0.0 {
        Some(args.budget_ms)
    } else {
        None
    };
    let limit = 10_usize;

    let workdir = args.workdir.unwrap_or_else(|| {
        let p = std::env::temp_dir().join("annex_regret_bench");
        fs::create_dir_all(&p).unwrap();
        p
    });
    fs::create_dir_all(&workdir).unwrap();

    if !args.json {
        eprintln!(
            "Building corpus: {} docs, dim={}, {} queries",
            args.corpus, args.dim, args.queries
        );
    }
    let index = build_index(args.corpus, args.dim, 0x1234_5678_9abc_def0, &workdir);

    let mut state = 0xdead_beef_cafe_0000_u64;
    let queries: Vec<Vec<f32>> = (0..args.queries)
        .map(|_| gen_vector(&mut state, args.dim))
        .collect();

    struct QueryResult {
        quality_regret: f32,
        eps_pareto_hit: bool,
        budget_violated: bool,
        budget_overrun_frac: f64,
    }
    let mut results: Vec<QueryResult> = Vec::with_capacity(args.queries);
    let total_start = Instant::now();

    for (qi, query_vec) in queries.iter().enumerate() {
        if !args.json && qi % 10 == 0 {
            eprint!(".");
        }

        // Oracle: exhaustive exact scan
        let oracle_req = serde_json::from_value(json!({
            "prefetch": [{"kind":"multivector","vectors":[query_vec],"limit": limit*4,"backend":"exact"}],
            "limit": limit
        })).unwrap();
        let oracle_resp = index.retrieve(&oracle_req).unwrap();
        let oracle_ids: Vec<String> =
            oracle_resp.matches.iter().map(|h| h.id.clone()).collect();
        let oracle_eval = PlanEvaluation {
            description: "oracle".into(),
            recall: 1.0,
            latency_ms: oracle_resp.trace.elapsed_ms,
            is_oracle: true,
        };

        // Enumerate HNSW plans across ef values
        let mut evals: Vec<PlanEvaluation> = vec![oracle_eval.clone()];
        for &ef in &ef_values {
            let req = serde_json::from_value(json!({
                "prefetch": [{"kind":"multivector","vectors":[query_vec],"limit": limit*2,"backend":"hnsw","ef_search":ef}],
                "limit": limit
            })).unwrap();
            if let Ok(eval) = evaluate_plan(&index, &req, &oracle_ids, format!("hnsw_ef{ef}"), false) {
                evals.push(eval);
            }
        }

        // ANNex planner choice
        let mut auto_obj = serde_json::Map::new();
        if args.budget_ms > 0.0 {
            auto_obj.insert("latency_budget_ms".into(), json!(args.budget_ms));
        }
        let auto_req = serde_json::from_value(json!({
            "prefetch": [{"kind":"multivector","vectors":[query_vec],"limit": limit*2,"backend":"auto"}],
            "limit": limit,
            "objective": auto_obj
        })).unwrap();
        let planner_resp = index.retrieve(&auto_req).unwrap();
        let planner_ids: Vec<String> = planner_resp.matches.iter().map(|h| h.id.clone()).collect();
        let planner_eval = PlanEvaluation {
            description: "planner".into(),
            recall: recall_at_k(&oracle_ids, &planner_ids, limit),
            latency_ms: planner_resp.trace.elapsed_ms,
            is_oracle: false,
        };
        evals.push(planner_eval.clone());

        // Pareto frontier from enumerated plans (excluding oracle)
        let candidates: Vec<PlanEvaluation> =
            evals.iter().filter(|e| !e.is_oracle).cloned().collect();
        let frontier = pareto_frontier(&candidates);

        let report = compute_regret(&oracle_eval, &planner_eval, &frontier, budget);
        let overrun =
            budget.map_or(0.0, |b| (planner_eval.latency_ms / b - 1.0).max(0.0));
        results.push(QueryResult {
            quality_regret: report.quality_regret,
            eps_pareto_hit: report.eps_pareto_hit,
            budget_violated: report.budget_violated,
            budget_overrun_frac: overrun,
        });
    }
    if !args.json {
        eprintln!();
    }

    let elapsed = total_start.elapsed().as_secs_f64();
    let n = results.len() as f32;
    let mut regrets: Vec<f32> = results.iter().map(|r| r.quality_regret).collect();
    regrets.sort_unstable_by(|a, b| a.total_cmp(b));
    let median_regret = regrets[regrets.len() / 2];
    let eps_hit_rate = results.iter().filter(|r| r.eps_pareto_hit).count() as f32 / n;
    let budget_violation_rate =
        results.iter().filter(|r| r.budget_violated).count() as f32 / n;
    let mut overruns: Vec<f64> = results.iter().map(|r| r.budget_overrun_frac).collect();
    overruns.sort_unstable_by(|a, b| a.total_cmp(b));
    let p95_overrun = overruns[((overruns.len() as f32 * 0.95) as usize).min(overruns.len() - 1)];

    if args.json {
        println!(
            "{}",
            json!({
                "queries": args.queries,
                "corpus": args.corpus,
                "dim": args.dim,
                "ef_values": ef_values,
                "budget_ms": budget,
                "median_quality_regret": median_regret,
                "eps_pareto_hit_rate": eps_hit_rate,
                "budget_violation_rate": budget_violation_rate,
                "p95_budget_overrun": p95_overrun,
                "total_elapsed_s": elapsed
            })
        );
    } else {
        println!("=== Planner Regret Report ===");
        println!("Corpus: {} docs, dim={}", args.corpus, args.dim);
        println!("Queries: {}", args.queries);
        println!("ef sweep: {ef_values:?}");
        println!("Budget: {budget:?} ms");
        println!();
        println!("Median quality regret:  {median_regret:.4}");
        println!("eps-Pareto hit rate:    {:.1}%", eps_hit_rate * 100.0);
        println!("Budget violation rate:  {:.1}%", budget_violation_rate * 100.0);
        println!("P95 budget overrun:     {:.1}%", p95_overrun * 100.0);
        println!("Total elapsed:          {elapsed:.2}s");
    }

    // Cleanup workdir
    let _ = fs::remove_dir_all(&workdir);
}
