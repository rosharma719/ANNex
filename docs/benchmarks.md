# Benchmarks And Harnesses

Commands run from the workspace root. Reporting rules live in [BENCHMARK_POLICY.md](../BENCHMARK_POLICY.md); historical corrections live in [RESULTS.md](../crates/annex-multivector/benchmark/RESULTS.md).

## NYTimes (256-D Angular)

Download instructions live in [data-download.md](./data-download.md).

Run the main harness:

```bash
cargo test --release nytimes_256_angular_perf_and_recall -- --ignored --nocapture
```

Run the QPS/latency curve from a persisted snapshot:

```bash
VECTORDB_USE_SNAPSHOT=1 \
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
cargo test --release nytimes_qps_latency_curve -- --ignored --nocapture
```

Build a snapshot with trace logging:

```bash
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
VECTORDB_NYT_M=16 \
VECTORDB_NYT_M0=32 \
VECTORDB_NYT_EF_CONSTRUCT=100 \
VECTORDB_NYT_EF_SEARCH_LIST=100 \
VECTORDB_DIVERSITY_ALPHA=1 \
VECTORDB_INSERT_TRACE_LOG=logs/nytimes_insert_m16_m0_32_efc100.jsonl \
VECTORDB_NYT_ALLOW_BUILD=1 \
VECTORDB_NYT_SAVE_SNAPSHOT=1 \
cargo test --release nytimes_build_and_persist_snapshot_only -- --ignored --nocapture
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
```

Use `VECTORDB_BENCH_SIZE=1000000` for the same synthetic harness at one million
vectors. Kernel benchmarks measure kernels, not database throughput. Dedicated
hardware is required for performance comparisons; CI only checks compilation.

The manual `Dedicated x86 benchmark` workflow runs on a self-hosted runner with
the `annex-benchmark` label. It requires AVX2 and FMA, validates each explicit
SIMD path against the scalar oracle, and uploads machine identity plus raw
Criterion output. Treat those kernel results as diagnostics under the reporting
policy, not as database or RAG comparisons.

## RAG retrieval

The [multivector benchmark guide](../crates/annex-multivector/benchmark/README.md)
owns encoding, exact/ANN comparisons, frozen operating points and result artifacts.
