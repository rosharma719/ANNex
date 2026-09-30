//! A persistent two-stage late-interaction retrieval engine.

mod analyzer;
pub use analyzer::{Stopwords, TextAnalyzer};
mod collections;
pub use collections::Collections;
mod engine;
mod fde;
mod muvera;
mod storage;

pub use engine::{
    AdaptiveRerank, CandidateHit, Channel, Chunk, ContextHit, ContextOptions, Durability, Fusion,
    Hit, IndexConfig, IndexError, IndexStats, MultiVectorIndex, Predicate, Representation, Rerank,
    RetrievalDocument, RetrievalResponse, RetrievalTrace, RetrieveRequest, UpsertDocument,
};
pub use fde::{maxsim, maxsim_flat};
