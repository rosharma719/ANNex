# ANNex

[![CI](https://github.com/rosharma719/annex/actions/workflows/ci.yml/badge.svg)](https://github.com/rosharma719/annex/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/annex.svg)](https://crates.io/crates/annex)
[![Docs.rs](https://docs.rs/annex/badge.svg)](https://docs.rs/annex)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**ANNex** — ANN + index — is an in-memory vector search engine in Rust. It
implements HNSW-based approximate nearest-neighbor search, in-place payload
filtering, snapshot persistence, and WAL-backed recovery.

- **MSRV:** Rust 1.85 (2024 edition).
- **Scope:** Rust library. There is no HTTP/gRPC server or client SDK — callers
  embed `Segment` inside their own service.

## Features

- HNSW indexing for approximate nearest-neighbor search
- Distance metrics: cosine, euclidean, dot
- Schema-agnostic per-point payloads (ints, floats, strings, bools, homogeneous lists)
- Equality and comparison filters with boolean composition (`And` / `Or` / `Not`)
- Inverted index acceleration on scalar payload fields
- Tombstone deletion with automatic purge/rebuild past a configurable threshold
- Snapshot persistence, background snapshotting, and WAL replay on load

## Install

```toml
[dependencies]
annex = "0.1"
```

## Quickstart

```rust
use std::collections::HashMap;
use annex::{DistanceMetric, Filter, Payload, PayloadValue, Segment};

fn main() {
    // 128-dim cosine index, m=16, ef=64, max level cap 16.
    let mut seg = Segment::with_config(DistanceMetric::Cosine, 16, 64, 16, 128);

    for id in 0..4u64 {
        let vector = vec![id as f32; 128];
        let mut payload = Payload(HashMap::new());
        payload.set(
            "category",
            PayloadValue::Str(if id % 2 == 0 { "even".into() } else { "odd".into() }),
        );
        seg.insert_with_id(id, vector, Some(payload)).unwrap();
    }

    let query = vec![0.1_f32; 128];
    let filter = Filter::Match {
        key: "category".into(),
        value: PayloadValue::Str("even".into()),
    };

    let hits = seg.search_with_filter(&query, 2, Some(&filter)).unwrap();
    for hit in hits {
        println!("id={} score={}", hit.id, hit.raw_score);
    }
}
```

## Thread safety

A single `Segment` is not internally synchronised for writes. For
multi-threaded workloads, wrap it in `Arc<RwLock<Segment>>` (aliased as
`SharedSegment`) and coordinate access through the lock:

- take a read guard for `search*` calls,
- take a write guard for `insert*`, `delete`, `update_payload`, and `purge`.

Background snapshotting (see `start_background_snapshots`) takes only read
locks and is safe to run alongside concurrent readers.

## Filtering model

- Payloads are per-point key/value maps (`Payload`).
- Missing fields evaluate to false for match/compare checks.
- Supported scalar types: `Int`, `Float`, `Str`, `Bool`, plus homogeneous lists.
- Filters compose via `Filter::Match`, `Filter::Compare`, `Filter::And`,
  `Filter::Or`, and `Filter::Not`.
- Scalar payload fields can be routed through the inverted index for fast
  exact-match filtering.

## Configuration

Runtime tuning is driven by `VECTORDB_*` environment variables read once at
startup (search budgets, purge thresholds, telemetry paths, dataset paths for
the benchmark harnesses). The prefix predates the ANNex rename and is
retained for compatibility. See [`.env.example`](.env.example) for the full
list and [`docs/test-config.md`](docs/test-config.md) for a per-test matrix.

## Project layout

- `src/segment/` — `Segment` lifecycle, persistence, WAL, background snapshotting
- `src/vector/hnsw/` — HNSW index internals
- `src/payload_storage/` — payload evaluation and inverted index
- `src/bin/` — snapshot inspection and analysis utilities
- `tests/` — correctness, persistence, recall, and dataset-driven harnesses
- `scripts/` — experiment pipeline
- `docs/` — operational notes, benchmark workflows, dataset setup

## Docs

- Dataset setup: [docs/data-download.md](docs/data-download.md)
- Test and env-var matrix: [docs/test-config.md](docs/test-config.md)
- Snapshot naming and build manifests: [docs/index-construction.md](docs/index-construction.md)
- Runtime persistence and operations: [docs/operations.md](docs/operations.md)
- Benchmark commands and recorded results: [docs/benchmarks.md](docs/benchmarks.md)
- Recall frontier pipeline: [docs/recall_frontier_pipeline.md](docs/recall_frontier_pipeline.md)

## Utilities

```bash
cargo run --bin snapshot_info -- <path>
cargo run --bin index_analyzer -- --help
cargo run --bin snapshot_sweeper -- --help
cargo run --bin index_stats -- --help
```

## Logging

Logging uses the `log` crate with targets such as `vector::hnsw`, `segment`,
`payload`, and `filter`. Wire up any `log` backend (e.g. `env_logger`,
`tracing-log`) and control verbosity through `RUST_LOG`.

## Roadmap

- `ANNEX_*` env-var prefix alongside `VECTORDB_*`
- Optional Python bindings
- Additional distance metrics
- Multi-segment collection layer

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). By
participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
To report a vulnerability privately, see [SECURITY.md](SECURITY.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual-licensed as above, without any additional terms or
conditions.
