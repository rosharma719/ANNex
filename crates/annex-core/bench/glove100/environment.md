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

## Dataset: GloVe-100-Angular
- Source: https://ann-benchmarks.com/glove-100-angular.hdf5
- Vectors: 1,183,514 base, 10,000 queries, 100-dim float32, angular (cosine) metric
- Ground truth: top-100 true neighbours per query
- Description: GloVe word embeddings (trained on Twitter + Wikipedia 2014 + Gigaword 5)
- File layout: `data/glove-100-angular/{base,queries}.npy`, `ground_truth.json`

### Dataset checksum (sha256)
Run `sha256sum data/glove-100-angular/base.npy` and record here.

## Measurement methodology
- **Single query, single thread**: no batching, no parallelism
- **Cache warm**: one full pass over all 1000 query vectors at ef=64 before timing
- **Timing**: per-query `Instant::now()` in Rust, `perf_counter()` in Python
- **Reported**: p50 latency over 1000 queries × 3 rounds
- **Recall@10**: true positives in top-10 results vs ground-truth top-10, averaged over 1000 queries
- Queries: first 1000 rows of `queries.npy`

## Index parameters (ANNex default benchmark index)
- M=16, M0=32, stored_cap_l0=32, ef_construction=300
- Snapshot: `data/glove-100-angular/annexdb_m16_efc300.bin`

## Comparison library parameters
- hnswlib: M=16, ef_construction=300, space="cosine"
- usearch: connectivity=16, expansion_add=300
- faiss-hnsw: M=32, ef_construction=300, METRIC_INNER_PRODUCT on L2-normalised vectors
- faiss-ivfpq: IVF4096, PQ32x8
