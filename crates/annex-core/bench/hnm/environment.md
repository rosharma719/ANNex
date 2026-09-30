# H&M Benchmark Environment

## Hardware
- CPU: Apple M2
- Architecture: aarch64 (ARM64)

## Software
- Rust: release build
- Build profile: `--release`

## Dataset: H&M Filtered Cosine
- Source: Qdrant filtered ANN benchmark
- Vectors: 105,100 base, 10,000 test cases, 2048-dim float32, cosine metric
- Each test case: query vector + metadata filter conditions + ground-truth IDs
- Payloads: product attributes (price, category, colour, department, etc.)
- File layout: `data/hnm/{vectors.npy, payloads.jsonl, tests.jsonl, filters.json}`

## Measurement methodology
- **Single query, single thread**: no batching, no parallelism
- **Index**: M=16, ef_construct=200, persisted to `data/hnm/index_filtered.bin`
- **Queries**: first 1,000 test cases from `tests.jsonl`
- **Recall**: filtered recall@k — fraction of true filtered neighbors found
  (truth set restricted to vectors within the index at query time)
- **Latency**: p50/p90/p99 per-query wall time including filter evaluation

## Notes
- No competitor comparison: hnswlib, usearch, and faiss do not support
  metadata filtering natively. This benchmark measures ANNex filtered-search
  quality and speed in isolation.
- Recall is reported as mean over queries plus per-query p50/p90/p99 to
  surface filter-hardness distribution.
