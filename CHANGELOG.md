# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Multi-entry L0 seeds** (`SearchRuntimeOptions::num_entry_seeds`): after the
  upper-layer greedy descent, run a small BFS at L1 to collect N candidate entry
  points and seed all of them into the L0 search. Reduces sensitivity to routing
  quality in upper layers. Controlled at runtime via `VECTORDB_NUM_ENTRY_SEEDS`.
- **Adaptive EF routing** (`SearchRuntimeOptions::adaptive_ef_high` +
  `adaptive_ef_score_threshold`): re-runs the L0 search with a higher EF budget
  when the top-1 result score exceeds a threshold, targeting only the hard
  queries that actually need extra work. Controlled at runtime via
  `VECTORDB_ADAPTIVE_EF_HIGH` and `VECTORDB_ADAPTIVE_EF_SCORE_THRESHOLD`.
- `tests/adaptive_search.rs`: recall + latency harness for multi-entry and
  adaptive EF on synthetic data, with `#[ignore]` sweep benchmarks.
- `tests/kernel_bench.rs`: distance-kernel throughput microbenchmark.
- `nytimes_adaptive_search_sweep` benchmark: side-by-side recall/latency table
  across plain EF, multi-entry, and adaptive EF configs on 290k NYT-256-Angular.

### Changed

- **4× SIMD kernel unrolling across all platforms** (`dot_neon`, `l2_neon`,
  `dot_avx2_fma`, `l2_avx2_fma`): changed from a single accumulator to four
  independent accumulators (16 floats/iteration on NEON, 32 on AVX2), breaking
  the FMA latency chain and saturating the CPU's dual-issue FMA pipeline.

### Benchmark results (NYT-256-Angular, 290k vectors, M=16, ef_construct=300)

Measured on Apple Silicon (aarch64 NEON), 1000 queries, top_k=20:

| config | recall@20 | avg ms/query | vs prior |
| --- | ---: | ---: | ---: |
| plain ef=32 | 0.855 | 0.418 | −23% latency |
| plain ef=64 | 0.886 | 0.542 | −27% latency |
| plain ef=128 | 0.910 | 0.922 | −28% latency |
| plain ef=256 | 0.934 | 1.604 | −34% latency |
| ef=64, seeds=3 | 0.887 | 0.493 | — |
| adaptive 32→128, t=0.55 | 0.878 | 0.479 | 225/1000 retried |
| adaptive 64→256, t=0.40 | 0.902 | 1.116 | — |

Index build time also improved ~42% due to the kernel being on the hot path
during parallel graph construction.

Prior baseline (single-accumulator kernels, same hardware and index):

| ef | recall@20 | avg ms/query |
| --- | ---: | ---: |
| 32 | 0.855 | 0.541 |
| 64 | 0.886 | 0.745 |
| 128 | 0.910 | 1.284 |
| 256 | 0.934 | 2.422 |

## [0.1.0] - 2025-09-14

Initial public release of **ANNex** (crate name `annex`).

### Added

- HNSW-based approximate nearest-neighbor index with cosine, dot, and
  euclidean metrics.
- Per-point payloads (`Payload`, `PayloadValue`) with scalar and homogeneous
  list types.
- Boolean filters (`Filter::Match`, `Compare`, `And`, `Or`, `Not`) with
  inverted-index acceleration on scalar fields.
- In-place filtered search with configurable budget routing and optional
  exact fallback for small filter sets.
- Tombstone deletion and threshold-triggered purge/rebuild.
- Snapshot persistence with Adler32 checksums, atomic rename, and versioned
  on-disk format (V1 → V3).
- Write-ahead log with configurable fsync policy and replay on load.
- Background snapshotter (`start_background_snapshots`) that triggers on
  elapsed time or op count.
- Utility binaries: `snapshot_info`, `index_analyzer`, `snapshot_sweeper`,
  `index_stats`, `nyt_search_bench`.
- Parallel bulk insert with SIMD FMA on supported targets.

### Public API

The crate root (`annex::*`) exposes the intended stable surface:
`Segment`, `Filter`, `Payload`, `PayloadValue`, `ScalarComparisonOp`,
`DistanceMetric`, `PointId`, `Vector`, `Score`, `ScoredPoint`,
`SearchRuntimeOptions`, `SnapshotMetadata`, `SnapshotConfig`,
`SnapshotterHandle`, `SharedSegment`, `WalConfig`,
`start_background_snapshots`, and `DBError`. Everything else is considered an
implementation detail.
