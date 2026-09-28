# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Durable hybrid retrieval with named dense/sparse/multivector fields, metadata
  predicates, BM25/RRF, optional MaxSim reranking and context selection.
- Independent collection namespaces and named dense ANN graphs.
- Python/NumPy snapshot search with validated inputs, GIL release, and threaded
  batches; Linux/macOS CI installs and tests the built wheel.
- Immutable query generations with retained mappings, sealed vector segments
  and live-record compaction. Format-3 manifests remain readable by this version
  alongside formats 1/2; older binaries reject newly written format-3 indexes.

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
- `nytimes_adaptive_search_sweep` benchmark: side-by-side recall/latency table
  across plain EF, multi-entry, and adaptive EF configs on 290k NYT-256-Angular.

### Changed

- **4× SIMD kernel unrolling across all platforms** (`dot_neon`, `l2_neon`,
  `dot_avx2_fma`, `l2_avx2_fma`): changed from a single accumulator to four
  independent accumulators (16 floats/iteration on NEON, 32 on AVX2), breaking
  the FMA latency chain and saturating the CPU's dual-issue FMA pipeline.

### Maintenance and verification

- Removed generated benchmark output, superseded plans and retired exploratory
  scripts from the active tree; historical evidence and corrections are linked
  from `benchmark/RESULTS.md`.
- Consolidated duplicate performance harnesses and documentation. Persistence
  tests retain arena-boundary coverage with smaller fixtures; filtered tests
  require nonempty matches.
- Multivector HTTP rejects unknown fields, unsupported options and excessive
  work parameters; added single-document deletion.
- Head-to-head runs record actual exact/ANN backends, full training/ingestion/index
  costs and durable per-query outcomes. Failed/interrupted requests remain visible.
- Embedding caches verify file checksums. Diagnostic corpus slices preserve
  original relevance denominators and avoid materializing the full source prefix.
- CI exercises the actual HTTP process alongside the retrieval/transaction oracles.
- Full-corpus dense/BM25/hybrid comparisons against native Qdrant Server and
  LanceDB retain every query and freeze configuration before test evaluation.
- Shared nDCG evaluation now uses BEIR/trec_eval linear relevance gains.
- Release the directory lock explicitly on index drop so inherited/duplicated
  descriptors cannot delay reopening; the existing lock test covers this case.

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
