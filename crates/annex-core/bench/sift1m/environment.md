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

## Dataset: SIFT-1M-Euclidean
- Source: https://ann-benchmarks.com/sift-128-euclidean.hdf5
- Vectors: 1,000,000 base, 10,000 queries, 128-dim float32, Euclidean (L2) metric
- Ground truth: top-100 true neighbours per query
- File layout: `data/sift-128-euclidean/{base,queries}.npy`, `ground_truth.json`

### Dataset checksum (sha256)
Run `sha256sum data/sift-128-euclidean/base.npy` and record here.

## Measurement methodology
- **Single query, single thread**: no batching, no parallelism
- **Cache warm**: one full pass over all 1000 query vectors at ef=64 before timing
- **Timing**: per-query `Instant::now()` in Rust, `perf_counter()` in Python
- **Reported**: p50 latency over 1000 queries × 3 rounds (median of rounds)
- **Recall@10**: true positives in top-10 results vs ground-truth top-10, averaged over 1000 queries
- Queries: first 1000 rows of `queries.npy`

## Index parameters (ANNex default benchmark index)
- M=16, M0=32, stored_cap_l0=128, ef_construction=300
- Snapshot: `data/sift-128-euclidean/annexdb_m16_efc300.bin`

## Comparison library parameters
- hnswlib: M=16, ef_construction=300, space="l2"
- usearch: connectivity=16, expansion_add=300
- faiss-hnsw: M=32, ef_construction=300, METRIC_L2
- faiss-ivfpq: IVF4096, PQ16x8
