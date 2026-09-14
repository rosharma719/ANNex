# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
