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

## Dataset: MNIST-784-Euclidean
- Source: https://ann-benchmarks.com/mnist-784-euclidean.hdf5
- Vectors: 60,000 base, 10,000 queries, 784-dim float32, Euclidean (L2) metric
- Ground truth: top-100 true neighbours per query
- Description: handwritten digit pixel vectors from the MNIST dataset; each
  vector is a flattened 28×28 grayscale image. Tests high-dimensional Euclidean
  behavior at small scale — the 784-D space and compact 60K base set make this
  a clean stress test for distance accuracy and ef_search sensitivity.
- File layout: `data/mnist-784-euclidean/{base,queries}.npy`, `ground_truth.json`

### Dataset checksum (sha256)
Run `sha256sum data/mnist-784-euclidean/base.npy` and record here.

## Measurement methodology
- **Single query, single thread**: no batching, no parallelism
- **Cache warm**: one full pass over all 1000 query vectors at ef=64 before timing
- **Timing**: per-query `Instant::now()` in Rust, `perf_counter()` in Python
- **Reported**: p50 latency over 1000 queries × 3 rounds
- **Recall@10**: true positives in top-10 results vs ground-truth top-10, averaged over 1000 queries
- Queries: first 1000 rows of `queries.npy`

## Index parameters (ANNex benchmark index)
- M=16, M0=32, ef_construction=300
- Snapshot: `data/mnist-784-euclidean/annexdb_m16_efc300.bin`

## Comparison library parameters
- hnswlib: M=16, ef_construction=300, space="l2"
- usearch: connectivity=16, expansion_add=300, metric="l2sq"
- faiss-hnsw: M=16, ef_construction=300, METRIC_L2
- faiss-ivfpq: IVF1024, PQ32x8
