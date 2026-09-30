# Benchmark Environment

## Hardware
- CPU: Apple M2
- Architecture: aarch64 (ARM64)
- SIMD: NEON + FEAT_DotProd (vdotq_s32)

## Software
- Rust: 1.98.1 (stable)
- Build profile: `--release`
- Compile flags: `RUSTFLAGS` unset (default release + lto=thin from Cargo.toml)
- Python: 3.x (venv at `/tmp/annbench_env`)
- hnswlib, usearch, faiss-cpu, annoy via pip

## Dataset: LastFM-64-Dot
- Source: https://ann-benchmarks.com/lastfm-64-dot.hdf5
- Vectors: 292,000 base, 50,000 queries, 64-dim float32, dot product (inner product) metric
- Ground truth: top-100 true neighbours per query
- File layout: `data/lastfm-64-dot/{base,queries}.npy`, `ground_truth.json`
- Description: Audio feature vectors extracted from the LastFM dataset. This is the
  only dot-product (inner product) benchmark in the ANNex suite; all other datasets
  use angular (cosine) or Euclidean metrics.

### Dataset checksum (sha256)
Run `sha256sum data/lastfm-64-dot/base.npy` and record here.

## Measurement methodology
- **Single query, single thread**: no batching, no parallelism
- **Cache warm**: one full pass over all 1000 query vectors at ef=64 before timing
- **Timing**: per-query `Instant::now()` in Rust, `perf_counter()` in Python
- **Reported**: p50 latency over 1000 queries × 3 rounds
- **Recall@10**: true positives in top-10 results vs ground-truth top-10, averaged over 1000 queries
- Queries: first 1000 rows of `queries.npy` (dataset has 50K total; limited to 1000 via
  `ANNEX_BENCH_QUERIES=1000` for consistent comparison with other benchmarks)

## Index parameters (ANNex default benchmark index)
- M=16, M0=32, ef_construction=300
- Snapshot: `data/lastfm-64-dot/annexdb_m16_efc300.bin`

## Comparison library parameters
- hnswlib: M=16, ef_construction=300, space="ip" (inner product)
- usearch: connectivity=16, expansion_add=300
- faiss-hnsw: M=32, ef_construction=300, METRIC_INNER_PRODUCT
- faiss-ivfpq: IVF4096, PQ8x8, METRIC_INNER_PRODUCT
