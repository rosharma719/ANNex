# Runtime Operations

This contract covers `annex::Segment`. The multivector engine has a separate [durability contract](multivector-durability.md).

## Snapshot notes

- Snapshots are written atomically and include a checksum footer for corruption detection.
- New snapshots load with `Segment::load_from_path`.
- Older snapshots without a footer are still supported.
- Background snapshotting is opt-in.
- Inspect a snapshot with `cargo run --bin snapshot_info -- <path>`.

## Autosave

```rust
use std::sync::{Arc, RwLock};
use std::time::Duration;
use annex::segment::{Segment, SnapshotConfig, start_background_snapshots};
use annex::vector::hnsw::HNSWIndex;
use annex::utils::types::DistanceMetric;

let segment = Arc::new(RwLock::new(Segment::new(
    HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2),
)));

let mut cfg = SnapshotConfig::new("data/segment_autosave.bin");
cfg.interval = Duration::from_secs(30);
cfg.max_ops = 5_000;

let snapshotter = start_background_snapshots(segment.clone(), cfg);
// keep `snapshotter` alive; call snapshotter.stop() on shutdown
```

## WAL

Enable a WAL to make inserts, deletes, and payload updates durable between snapshots. WAL replay is automatic on `Segment::load_from_path` when `<snapshot>.wal` exists.

```rust
use annex::segment::Segment;

let mut segment = Segment::new(hnsw);
segment.enable_wal("data/segment_snapshot.bin.wal")?;

segment.insert_with_id(1, vec![1.0, 0.0], None)?;
segment.save_to_path("data/segment_snapshot.bin")?;

let restored = Segment::load_from_path("data/segment_snapshot.bin")?;
```

WAL-related environment variables:

- `VECTORDB_WAL_PATH`
- `VECTORDB_WAL_DIR`
- `VECTORDB_WAL_AUTO_REPLAY`
- `VECTORDB_WAL_FSYNC`
- `VECTORDB_WAL_FSYNC_EVERY`
- `VECTORDB_WAL_FSYNC_MS`

Use `save_to_path_and_checkpoint` to save a snapshot and truncate the WAL after a successful write.

## RSS memory cap

Set `VECTORDB_MAX_RSS_MB` to cap resident memory and `VECTORDB_OOM_SNAPSHOT_PATH` to control where the emergency snapshot is written. When the cap triggers, the segment snapshots, unloads the in-memory graph, and rejects new inserts until reload.

## Deletion purge

By default, deletions remain as tombstones. Set `VECTORDB_PURGE_DELETIONS=1` to enable automatic purge/rebuild after the configured threshold.

During a rebuild, queries return a `SearchError` indicating that rebuild is in progress.
