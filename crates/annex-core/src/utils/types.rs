use serde::{Deserialize, Serialize};

/// The unique identifier for a point in the vector database.
pub type PointId = u64;

/// The vector representation of a point.
pub type Vector = Vec<f32>;

/// The similarity score between a query vector and a stored point.
pub type Score = f32;

/// The unique identifier for a segment (within a shard).
pub type SegmentId = u32;

/// The unique identifier for a shard (within a collection).
pub type ShardId = u32;

/// Optional: a basic alias for collection names.
pub type CollectionName = String;

/// Describes the type of distance metric used for similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Dot,
    Euclidean,
}
