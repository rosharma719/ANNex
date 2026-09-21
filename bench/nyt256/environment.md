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

## Dataset: NYT-256-Angular
- Source: https://ann-benchmarks.com/nytimes-256-angular.hdf5
- Vectors: 290,000 base, 10,000 queries, 256-dim float32, angular (cosine) metric
- Ground truth: top-1000 true neighbours per query
- File layout: `data/nytimes-256-angular/{base,queries}.npy`, `ground_truth.json`

### Dataset checksum (sha256)
Run `sha256sum data/nytimes-256-angular/base.npy` and record here.

## Measurement methodology
- **Single query, single thread**: no batching, no parallelism
- **Cache warm**: one full pass over all 1000 query vectors at ef=64 before timing
- **Timing**: per-query `Instant::now()` in Rust, `perf_counter()` in Python
- **Reported**: p50 latency over 1000 queries × 1 round (from feature_isolation test)
- **Recall@20**: true positives in top-20 results vs ground-truth top-20, averaged over 1000 queries
- Queries: first 1000 rows of `queries.npy`

## Index parameters (ANNex default benchmark index)
- M=16, M0=32, stored_cap_l0=128, ef_construction=300
- Snapshot: `data/nytimes-256-angular/index_m16_m0_32_stored_128_efc300.bin`

## Comparison library parameters
- hnswlib: M=16, ef_construction=300, space="cosine"
- usearch: connectivity=16, expansion_add=300
- faiss-hnsw: M=32, ef_construction=300, METRIC_INNER_PRODUCT on L2-normalised vectors
- faiss-ivfpq: IVF4096, PQ32x8
