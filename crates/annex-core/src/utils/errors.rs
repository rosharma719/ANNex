use crate::utils::types::PointId;
use std::io;
use thiserror::Error;

/// Central error enum for the vector database.
#[derive(Error, Debug)]
pub enum DBError {
    #[error("Point with ID {0} not found")]
    NotFound(PointId),

    #[error("Point with ID {0} already exists")]
    DuplicatePointId(PointId),

    #[error("Vector length mismatch: expected {expected}, got {actual}")]
    VectorLengthMismatch { expected: usize, actual: usize },

    #[error("I/O error: {0}")]
    IOError(#[from] io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] anyhow::Error),

    #[error("WAL corruption: {0}")]
    WALCorrupt(String),

    #[error("Invalid payload: {0}")]
    InvalidPayload(String),

    #[error("Search failed: {0}")]
    SearchError(String),

    #[error("Memory cap exceeded: {0}")]
    MemoryCapExceeded(String),
}
