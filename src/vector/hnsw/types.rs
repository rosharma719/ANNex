use crate::utils::types::{PointId, Score};

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredPoint {
    pub id: PointId,
    pub raw_score: Score,
    pub sort_key: Score,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct SearchRuntimeOptions {
    /// Overrides the HNSW `ef_search` for this call. If set, it will be clamped to at least `top_k`.
    pub ef_search: Option<usize>,
    /// Overrides `VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0` for this call.
    /// - `Some(0)` means "no cap" (treated as `usize::MAX`), matching env semantics.
    pub neighbor_scan_cap_level0: Option<usize>,
    /// Overrides `VECTORDB_NEIGHBOR_SCAN_PATIENCE` for this call.
    pub neighbor_scan_patience: Option<usize>,
    /// Overrides `VECTORDB_EARLY_EXIT_PATIENCE` for this call.
    pub early_exit_patience: Option<usize>,
    /// Overrides `VECTORDB_SEARCH_EXPANSION_MULT` for this call.
    /// Insert uses this with `Some(1)` so candidate-pool expansion only happens at query time.
    pub expansion_mult: Option<usize>,
    /// Number of seeds to collect from L1 before entering L0 search. When >1 and the index
    /// has upper layers, a small BFS at L1 collects this many candidates and all are used as
    /// L0 entry points, reducing sensitivity to routing quality in upper layers.
    /// Overrides `VECTORDB_NUM_ENTRY_SEEDS`. Default is 1 (standard single-entry behavior).
    pub num_entry_seeds: Option<usize>,
    /// When set together with `adaptive_ef_score_threshold`, re-runs L0 with this EF for
    /// queries whose best result score exceeds the threshold (hard queries).
    /// Overrides `VECTORDB_ADAPTIVE_EF_HIGH`.
    pub adaptive_ef_high: Option<usize>,
    /// Score threshold that triggers an adaptive EF retry. Queries whose top-1 sort_key
    /// (lower = better) exceeds this value are re-run with `adaptive_ef_high`.
    /// Overrides `VECTORDB_ADAPTIVE_EF_SCORE_THRESHOLD`.
    pub adaptive_ef_score_threshold: Option<f32>,
    /// Enable triangle-inequality neighbor skip at L0. When true and the index has
    /// edge distances stored, neighbors whose TI lower bound exceeds `worst_score` are
    /// skipped before computing `fast_score`. Not valid for Dot metric (no bounded metric).
    /// Overrides `VECTORDB_TI_SKIP`.
    pub use_ti_skip: Option<bool>,
    /// When > 0, use SQ8 quantized scoring during L0 traversal and collect
    /// `ef_search * sq8_rerank_factor` candidates, then rerank with full f32.
    /// Set to 0 to disable SQ8 traversal. Overrides `VECTORDB_SQ8_RERANK_FACTOR`.
    /// Has no effect if `quantize_all()` has not been called on the index.
    pub sq8_rerank_factor: Option<usize>,
    /// Screen each L0 neighbor with a fast integer dot product before prefetching its
    /// f32 vector. Neighbors whose SQ8 score is clearly below the current worst result
    /// are skipped, saving the f32 prefetch + distance computation. Effective at high ef
    /// where most candidates are rejected. Requires quantize_all() to be called first.
    /// Overrides `VECTORDB_SQ8_SCREEN`.
    pub sq8_screen: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct NodeCandidate {
    pub(crate) idx: usize,
    pub(crate) raw_score: Score,
    pub(crate) sort_key: Score,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeRoutingEntry {
    pub(crate) node: NodeCandidate,
    pub(crate) passes_filter: bool,
    pub(crate) budget: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NodeResult(pub(crate) NodeCandidate);

impl PartialEq for NodeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key == other.sort_key
    }
}

impl Eq for NodeCandidate {}

impl PartialOrd for NodeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Invert the ordering so that lower scores (better) are considered "greater" for the BinaryHeap.
        other.sort_key.partial_cmp(&self.sort_key)
    }
}

impl Ord for NodeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialEq for NodeRoutingEntry {
    fn eq(&self, other: &Self) -> bool {
        self.node.sort_key == other.node.sort_key
    }
}

impl Eq for NodeRoutingEntry {}

impl PartialOrd for NodeRoutingEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Lower scores are better; invert for max-heap behavior.
        other.node.sort_key.partial_cmp(&self.node.sort_key)
    }
}

impl Ord for NodeRoutingEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Eq for NodeResult {}

impl PartialOrd for NodeResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Normal ordering: lower score is better, so when used in a max-heap the worst (largest score) will be at the top.
        self.0.sort_key.partial_cmp(&other.0.sort_key)
    }
}

impl Ord for NodeResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.sort_key.partial_cmp(&other.0.sort_key).unwrap()
    }
}
