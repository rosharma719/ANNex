# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-04

### Added

- **Python: build and save indexes** (`annex-py`). `annexdb.Index.build(vectors, *, ids=None,
  metric="cosine", m=16, ef_construct=200, level_cap=16, quantize=False)` builds an index from a
  float32 NumPy array of shape `[n, dim]` (row `i` gets id `i` unless `ids` is given; the metric
  also accepts the aliases `angular`, `ip` and `l2`; `quantize=True` builds the SQ8 codes used by
  `sq8_screen` searches). `Index.save(path)` writes
  a snapshot that `Index(path)` and Rust's `Segment::load_from_path` both load, and
  `Index.metric()` reports the metric. Python no longer needs a Rust-built snapshot to get
  started. Dense search only: hybrid retrieval, payload filters and multivector search remain in
  the Rust API and the `annex-multivector` HTTP server.
- **Query planner for hybrid retrieval** (`annex-multivector`). `POST /v1/plan` compiles a
  `/v1/retrieve` request into the physical plan the engine would run, without scoring documents:
  each channel's operator and the reason it was chosen, limits, filter strategy and
  eligible-document count. Retrieval traces embed the same plan and add `per_stage_actual_ms`.
  `/v1/retrieve` accepts `planning_mode` (`manual`, the default; `auto`; `auto_with_overrides`),
  a `query` object holding the query's representations for auto planning, and an `objective`
  with `latency_budget_ms`, `context_budget_tokens` and `quality` (`fast`, `balanced`, `high`).
  Auto planning classifies the query's intent (lexical, semantic or hybrid) to allocate retrieval
  budgets, learns per-operator costs from runtime observations, and measures field coverage
  within filters. Design: `docs/query-planner-spec.md`.
- **Calibration persistence.** The planner's learned cost calibration is saved with the index
  (atomically alongside the manifest, and autosaved every 50 observations) and restored when the
  index is opened. `flush_calibration()` forces a save.
- **ANN graph persistence** (`annex-multivector`). HNSW graphs built through `/v1/dense/index` and
  `/v1/fde/index` (`build_dense_ann`, `build_fde_ann`) are saved to `<root>/ann/` and reloaded
  on open when the stored graph matches the index generation exactly; otherwise retrieval falls
  back to exact scoring until the graph is rebuilt. `auto_compact_dense_ann` and
  `auto_compact_fde_ann` rebuild a graph only when its unmerged delta exceeds a given fraction of
  the base (0.20 recommended).
- **`MultiVectorIndex::retrieve_batch`** retrieves several requests against one consistent
  index generation.
- **`annex::VectorIndex`**: a single-file persistent index for direct embedding queries such as
  hard-negative mining, without the planner. `build(path, entries, m, ef_construct)` takes
  `(String, Vec<f32>)` entries; `open`, `save`, `search`, `search_with_ef`, `len`, `id_at` and
  `position_of` follow. Scores are dot products and ids are strings. Writes `index.ann` and
  `index.ids`.

### Changed

- **Distance kernels**: AVX-512 (with VNNI for SQ8) and AVX2+FMA kernels for f32 dot, L2 and SQ8
  scoring, selected at runtime with a scalar fallback; NEON is unchanged. HNSW scoring and the
  multivector x86 dot path use them. MaxSim gets a register-tiled kernel over a packed query
  (`MaxSimQuery`, exported from `annex-multivector`) so a query is packed once per rescoring call.
- **BF16 compute is opt-in**: set `VECTORDB_BF16=1` to use an AVX-512 BF16 dot kernel for FDE
  scoring and MaxSim on CPUs that report AVX-512 BF16 (Sapphire Rapids, Zen 4). It is off by
  default because BF16 rounds each operand to bfloat16: scores differ from f32 by up to about
  2^-7 of the vectors' magnitudes, and the exact backend (BF16) and the ANN backend (f32) then
  disagree about a document's score. Compute only: vectors are still stored as f32.
  `MaxSimQuery::score` now always equals `maxsim_flat`; with BF16 on it skips the packed f32
  kernel so both use the same BF16 path.
- Batched four-vector dot scoring and SIMD normalization in HNSW search, and cached collection
  statistics in the planner.
- Python: `Index(path)` keeps the loaded segment instead of copying its graph, so loading a
  snapshot no longer holds two copies of it at once.

### Fixed

- Python README: cosine scores were documented as higher-is-better. They are reported as
  `1 - similarity`, so lower is closer; dot product is a similarity (higher is closer) and
  Euclidean is squared distance (lower is closer).

## [0.2.0] - 2026-09-30

### Added

- **`SearchRuntimeOptions::sq8_score_only`** (env: `VECTORDB_SQ8_SCORE_ONLY`): SQ8-quantized
  L0 traversal without a subsequent float32 rerank pass. Cosine metric only — the integer dot
  product maps to cosine similarity by a fixed 127.5² scale with no equivalent for Euclidean.
  Traverses at `ef` (no extended pool), rescales results to the real cosine-distance domain, and
  returns approximate scores at lower latency than the rerank path. Has no effect unless
  `quantize_all()` has been called on the index.

### Added

- Durable hybrid retrieval with named dense/sparse/multivector fields, metadata
  predicates, BM25/RRF, optional MaxSim reranking and context selection.
- Independent collection namespaces and named dense ANN graphs.
- Persisted lexical analyzer policies, including an English preset with accent
  folding, stop-word removal, Snowball stemming and query term frequencies.
- Default RRF `k` reduced from 60 to 10 after one global five-corpus development
  sweep; benchmark comparators receive the same setting.
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
