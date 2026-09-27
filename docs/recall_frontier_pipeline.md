# Recall frontier pipeline

Development analysis of NYT search policies. Reporting requirements live in [BENCHMARK_POLICY.md](../BENCHMARK_POLICY.md).

## 1. Inputs

1. `logs/nyt_query_suite.jsonl` – produced by `scripts/run_query_grid.py`, contains both per-query entries (`type="query"`) and per-run summaries (`type="summary"`).
   Rows record runtime settings and repository state:
   `search_expansion_mult`, `search_expansion_cap`, `disable_early_exit`,
   `neighbor_rotate`, `neighbor_stride`, `neighbor_scan_patience`,
   `git_commit`, and `git_dirty`.
2. `logs/nyt_query_suite_analysis/` (produced) – derived tables + reports from `scripts/analyze_nyt_query_suite.py`.

## 2. Script `scripts/analyze_nyt_query_suite.py`

Running:
```sh
python scripts/analyze_nyt_query_suite.py \
  --in-jsonl logs/nyt_query_suite.jsonl \
  --out-dir logs/nyt_query_suite_analysis \
  --thresholds 0.8,0.85,0.9,0.95,0.97,0.99 \
  --latency-metric ms_p90 \
  --weights per_m \
  --recompute-from-queries
```

- Reads the `type="summary"` rows and builds:
  - `policy_table.csv` (one row per snapshot × query policy)
  - `run_summaries.csv` (optional; recomputed from `type="query"` rows when you pass `--recompute-from-queries`)
  - `frontier_table.csv` / `global_frontier.csv` (best latency at each recall threshold)
  - `effects_report.md` (linear + quadratic coefficient summaries, reweighted to address sampling imbalance).
  - `bottleneck_report.md` (latency-vs-work counter fits + ratio diagnostics like ms-per-visit)
  - `paired_knob_effects.csv` (paired deltas within a snapshot holding other knobs fixed)
  - `frontier_winners.csv` (which query-knob settings win per threshold across snapshots)
  - `marginal_returns.csv` (within a snapshot+cap+patience, EF step → Δrecall, Δlatency, and ratios)
  - `query_hardness.csv` (which `query_idx` are hardest on average / in the tail)
  - `query_knob_effects.csv` (per-`query_idx` linear knob coefficients for `recall`, `elapsed_ms`, `visited`, `expanded`)

## 3. Workflow

1. Run `scripts/run_query_grid.py` to produce `logs/nyt_query_suite.jsonl`.
   The runner pins runtime settings in each record.
2. Run `scripts/analyze_nyt_query_suite.py` to emit the frontier tables and knob-effect report.
