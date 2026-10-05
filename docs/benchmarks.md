# Benchmarks And Harnesses

Commands run from the workspace root. Reporting rules live in [BENCHMARK_POLICY.md](../BENCHMARK_POLICY.md); historical corrections live in [RESULTS.md](../crates/annex-multivector/benchmark/RESULTS.md).

## Counterfactual physical-plan evaluation

Run a declared request grid against a quiescent service and replay its durable
observations into Pareto/regret reports. See [plan evaluation](plan-evaluation.md)
for inputs, commands, failure accounting and measurement limits.

## NYTimes (256-D Angular)

Download instructions live in [data-download.md](./data-download.md).

Run ANNex and the available competitor libraries, then generate the comparison:

```bash
crates/annex-core/bench/nyt256/run_all.sh
```

The shared harness accepts `M_VALUES` and `EF_SEARCH_LIST`; pass
`--skip-competitors` to reuse existing competitor output:

```bash
M_VALUES="16 32" EF_SEARCH_LIST="32,64,128,256" \
  crates/annex-core/bench/nyt256/run_all.sh --skip-competitors
```

Analyze a snapshot:

```bash
cargo run --bin index_analyzer -- \
  --snapshot data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
  --base data/nytimes-256-angular/base.npy \
  --queries data/nytimes-256-angular/queries.npy \
  --top-k 10 \
  --num-queries 100 \
  --sample-size 500
```

Sweep multiple snapshots:

```bash
cargo run --bin snapshot_sweeper -- \
  --snapshots data/nytimes-256-angular/index_m16_m0_16_efc100.bin,\
data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
  --output logs/nyt_snapshot_sweep.jsonl \
  --top-k 10 \
  --num-queries 100 \
  --sample-size 500 \
  --neighbor-scan-cap 128
```

For canonical naming and build manifests, see [index-construction.md](./index-construction.md).

## H&M filtered retrieval

```sh
cargo test --release -p annex --test hnm hnm_filtered_cosine_recall -- --ignored --nocapture
```

See [dataset setup](data-download.md) and [harness configuration](test-config.md).

## Synthetic performance diagnostics

One parameterized harness covers corpus size and dimension; these measurements
do not establish real-data recall or production performance.

```sh
VECTORDB_BENCH_SIZE=20000 VECTORDB_BENCH_DIM=1536 \
cargo test --release -p annex --test perf_unfiltered -- --ignored --nocapture
cargo bench -p annex-multivector --bench kernels
cargo bench -p annex-multivector --bench planner
```

Use `VECTORDB_BENCH_SIZE=1000000` for the same synthetic harness at one million
vectors. Kernel benchmarks measure kernels, not database throughput. Dedicated
hardware is required for performance comparisons; CI only checks compilation.

The `planner` Criterion target measures planning overhead at 100 and 10,000
documents for manual dense, manual hybrid, filtered hybrid, and automatic-policy
requests. It includes document eligibility and statistics collection performed by
`MultiVectorIndex::plan`; it does not execute retrieval or measure planner quality
regret. Use the real-query protocol in `docs/query-planner-spec.md` for quality,
budget, and Pareto-frontier claims.

The manual `Dedicated x86 benchmark` workflow runs on a self-hosted runner with
the `annex-benchmark` label. It requires AVX2 and FMA, validates each explicit
SIMD path against the scalar oracle, and uploads machine identity plus raw
Criterion output. Treat those kernel results as diagnostics under the reporting
policy, not as database or RAG comparisons.

## RAG retrieval

The [multivector benchmark guide](../crates/annex-multivector/benchmark/README.md)
owns encoding, exact/ANN comparisons, frozen operating points and result artifacts.
