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
    AdaptiveRerank, CalibrationEntry, CalibrationKey, CalibrationSnapshot, CandidateHit, Channel,
    Chunk, ContextHit, ContextOperator, ContextOptions, ContextPlan, Durability, FieldStats,
    FilterStats, FilterStrategy, Fusion, FusionOperator, Hit, IndexConfig, IndexError, IndexStats,
    LogicalChannel, LogicalChannelKind, LogicalFusion, LogicalPlan, MultiVectorIndex,
    PhysicalOperator, PlanEstimate, PlanReason, PlanStage, PlannedChannel, PlannerStats, Predicate,
    QualityPreference, QueryIntent, RankingSignals, Representation, RepresentationKind, Rerank,
    RerankPlan, RetrievalDocument, RetrievalObjective, RetrievalPlan, RetrievalResponse,
    RetrievalTrace, RetrieveRequest, UpsertDocument,
};
pub use fde::{MaxSimQuery, maxsim, maxsim_flat};
