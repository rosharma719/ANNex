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
    AdaptiveRerank, CandidateHit, Channel, Chunk, ContextHit, ContextOptions, Durability,
    EscalationPredicate, ConditionalStage, ContextOperator, ContextPlan, FieldStats,
    FilterStrategy, FusionOperator, Fusion, Hit, IndexConfig, IndexError, IndexStats,
    LogicalChannel, LogicalChannelKind, LogicalFusion, LogicalPlan, MultiVectorIndex,
    PhysicalOperator, PlanEstimate, PlanReason, PlanStage, PlannerStats, PlannedChannel,
    Predicate, QualityPreference, Representation, Rerank, RerankPlan, RepresentationKind,
    RetrievalDocument, RetrievalObjective, RetrievalPlan, RetrievalResponse, RetrievalTrace,
    RetrieveRequest, UpsertDocument,
};
pub use fde::{maxsim, maxsim_flat};
