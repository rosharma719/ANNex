//! # ANNex
//!
//! In-memory ANN (HNSW) vector search engine with payload filtering, snapshot
//! persistence, and WAL replay.
//!
//! # Overview
//!
//! [`Segment`] is the main entry point. It wraps an HNSW index, per-point
//! payloads, an inverted index for filter acceleration, and an optional
//! write-ahead log. A single `Segment` is not internally synchronised for
//! writes — callers running multi-threaded workloads should wrap it in an
//! [`std::sync::Arc`]`<`[`parking_lot::RwLock`]`<Segment>>` (a
//! [`SharedSegment`]) and coordinate through the lock.
//!
//! # Quickstart
//!
//! ```no_run
//! use std::collections::HashMap;
//! use annex::{DistanceMetric, Filter, Payload, PayloadValue, Segment};
//!
//! // 128-dim cosine index, m=16, ef=64, max level cap 16.
//! let mut seg = Segment::with_config(DistanceMetric::Cosine, 16, 64, 16, 128);
//!
//! for id in 0..4u64 {
//!     let vector = vec![id as f32; 128];
//!     let mut payload = Payload(HashMap::new());
//!     payload.set(
//!         "category",
//!         PayloadValue::Str(if id % 2 == 0 { "even".into() } else { "odd".into() }),
//!     );
//!     seg.insert_with_id(id, vector, Some(payload)).unwrap();
//! }
//!
//! let query = vec![0.1_f32; 128];
//! let filter = Filter::Match {
//!     key: "category".into(),
//!     value: PayloadValue::Str("even".into()),
//! };
//! let hits = seg.search_with_filter(&query, 2, Some(&filter)).unwrap();
//! for hit in hits {
//!     println!("id={} score={}", hit.id, hit.raw_score);
//! }
//! ```
//!
//! # Configuration
//!
//! Runtime tuning (search budgets, purge thresholds, telemetry paths, and so
//! on) is driven by `VECTORDB_*` environment variables read once at startup
//! (the prefix predates the ANNex rename and is retained for compatibility).
//! See `.env.example` in the repository for the full list.
//!
//! # Feature status
//!
//! Implemented: HNSW index, deletion + purge, payload storage, boolean +
//! comparison filters, inverted-index acceleration, snapshot persistence,
//! WAL replay, background snapshotting.
//!
//! Everything below the `Segment` layer (the `vector`, `payload_storage`, and
//! `analysis` modules) is exposed but considered an implementation detail
//! and may change between minor versions.

#[doc(hidden)]
pub mod analysis;
#[doc(hidden)]
pub mod payload_storage;
pub mod segment;
pub mod utils;
#[doc(hidden)]
pub mod vector;

pub use crate::payload_storage::filters::Filter;
pub use crate::segment::segment::{Segment, SnapshotMetadata};
pub use crate::segment::{
    SharedSegment, SnapshotConfig, SnapshotterHandle, WalConfig, start_background_snapshots,
};
pub use crate::utils::errors::DBError;
pub use crate::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
pub use crate::utils::types::{DistanceMetric, PointId, Score, Vector};
pub use crate::vector::hnsw::arena::{
    ChunkedArray, ChunkedArrayView, VectorArena, VectorArenaView,
};
pub use crate::vector::hnsw::{ScoredPoint, SearchRuntimeOptions};
