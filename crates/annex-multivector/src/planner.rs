//! Deterministic physical planning for the retrieval API.
use super::*;

// ── Request-level objective ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityPreference {
    Fast,
    #[default]
    Balanced,
    High,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalObjective {
    pub latency_budget_ms: Option<f64>,
    pub context_budget_tokens: Option<usize>,
    #[serde(default)]
    pub quality: QualityPreference,
}

// ── Logical plan (intent, before physical decisions) ─────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalChannelKind {
    Bm25,
    Sparse,
    Dense,
    Multivector,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LogicalChannel {
    pub index: usize,
    pub kind: LogicalChannelKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    pub candidate_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalFusion {
    ReciprocalRank,
    Weighted,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LogicalPlan {
    pub objective: RetrievalObjective,
    pub channels: Vec<LogicalChannel>,
    pub fusion: LogicalFusion,
    pub filtered: bool,
    pub rerank: bool,
    pub context_selection: bool,
}

// ── Physical operators ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalOperator {
    Bm25,
    SparseDot,
    ExactDense,
    HnswDense,
    ExactMaxsim,
    ExactFde,
    HnswFde,
    /// HNSW with post-filter overfetch + exact fallback. Not the same as
    /// filter-aware graph traversal (which requires HNSW library changes).
    HnswPostFilter,
    /// Exact scan over the eligible set (filter applied first).
    ExactEligibleScan,
    /// Bitmap intersection over indexed categorical fields (Phase 2A).
    FilterBitmap,
}

impl PhysicalOperator {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bm25 => "bm25",
            Self::SparseDot => "sparse_dot",
            Self::ExactDense => "exact_dense",
            Self::HnswDense => "hnsw_dense",
            Self::ExactMaxsim => "exact_maxsim",
            Self::ExactFde => "exact_fde",
            Self::HnswFde => "hnsw_fde",
            Self::HnswPostFilter => "hnsw_post_filter",
            Self::ExactEligibleScan => "exact_eligible_scan",
            Self::FilterBitmap => "filter_bitmap",
        }
    }
}

// ── Plan reasons (machine-readable, with structured values) ──────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanReason {
    /// Index over a lexical/BM25 posting list.
    LexicalIndex,
    /// Index over a sparse inverted index.
    SparseIndex,
    /// Caller explicitly requested exact execution.
    RequestedExact,
    /// ANN graph available and selected by availability heuristic.
    AnnReady,
    /// ANN graph not built; falling back to exact scan.
    AnnUnavailable,
    /// Synonym for AnnUnavailable; prefer IndexUnavailable in new code.
    IndexUnavailable,
    /// Filter present; selectivity too low or post-filter disabled.
    FilterRequiresExact,
    /// Filter selectivity below threshold for HnswPostFilter.
    FilterTooSelective,
    /// Operator does not support ANN (e.g. named multivector exact-only).
    OperatorRequiresExact,
    /// Cost model estimated exact cheaper than HNSW for this corpus size.
    LowerEstimatedCost,
    /// ef_search adapted to fit latency budget.
    LatencyBudgetConstraint,
}

// ── Filter strategy ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterStrategy {
    None,
    MetadataScan,
    BitmapIntersection, // Phase 2A
}

// ── Planner statistics (corpus facts, one per generation) ────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationKind {
    Dense,
    Multivector,
    Sparse,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FieldStats {
    pub kind: RepresentationKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension: Option<usize>,
    pub documents: usize,
    pub graph_ready: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PlannerStats {
    pub generation: u64,
    pub documents: usize,
    pub token_documents: usize,
    pub token_vectors: usize,
    pub fde_dimension: usize,
    pub fde_graph_ready: bool,
    pub fields: BTreeMap<String, FieldStats>,
}

// ── Per-channel execution plans ───────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlannedChannel {
    pub index: usize,
    pub operator: PhysicalOperator,
    pub reason: PlanReason,
    pub limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<usize>,
    /// Raw cost units for this operator (not calibrated to ms yet).
    pub estimated_cost_units: f64,
}

// ── Fusion, rerank, context plans ────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionOperator {
    NativeScore,
    ReciprocalRank,
    WeightedScore,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RerankPlan {
    pub operator: PhysicalOperator,
    pub candidate_limit: usize,
    pub adaptive: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextOperator {
    Ranked,
    Mmr,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ContextPlan {
    pub operator: ContextOperator,
    pub result_limit: usize,
    pub context_budget_tokens: Option<usize>,
}

// ── Conditional escalation (Phase 2D) ────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConditionalStage {
    pub predicate: EscalationPredicate,
    pub additional_channels: Vec<PlannedChannel>,
    pub re_fuse: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EscalationPredicate {
    LowChannelAgreement { threshold: f32 },
    InsufficientEligibleHits { min_count: usize },
    LowTopScore { threshold: f32 },
}

// ── Plan IR ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlanStage {
    Parallel(Vec<PlannedChannel>),
    Fusion(FusionOperator),
    ConditionalEscalation(ConditionalStage),
    Rerank(RerankPlan),
    Context(ContextPlan),
}

/// Cost estimate for the full plan.
/// - `estimated_wall_ms`: critical-path wall time (parallel channels counted once).
/// - `estimated_cpu_work`: aggregate cpu-equivalent ms (sum of all operator costs).
///
/// For a 2-channel parallel plan: wall ≈ max(ch₁, ch₂) while cpu_work = ch₁ + ch₂.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlanEstimate {
    pub estimated_wall_ms: f64,
    pub estimated_cpu_work: f64,
}

// ── Full retrieval plan ───────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RetrievalPlan {
    pub logical: LogicalPlan,
    pub stats: PlannerStats,
    pub eligible_documents: usize,
    pub filter: FilterStrategy,
    pub stages: Vec<PlanStage>,
    pub estimate: PlanEstimate,
}

impl RetrievalPlan {
    /// Returns the `PlannedChannel` slice from the first `Parallel` stage.
    pub fn parallel_channels(&self) -> &[PlannedChannel] {
        self.stages
            .iter()
            .find_map(|s| match s {
                PlanStage::Parallel(ch) => Some(ch.as_slice()),
                _ => None,
            })
            .unwrap_or(&[])
    }

    /// Returns the fusion operator from the plan, if present.
    pub fn fusion_operator(&self) -> Option<FusionOperator> {
        self.stages.iter().find_map(|s| match s {
            PlanStage::Fusion(op) => Some(*op),
            _ => None,
        })
    }

    /// Returns the rerank plan, if present.
    pub fn rerank_plan(&self) -> Option<&RerankPlan> {
        self.stages.iter().find_map(|s| match s {
            PlanStage::Rerank(r) => Some(r),
            _ => None,
        })
    }

    /// Returns the context plan.
    pub fn context_plan(&self) -> Option<&ContextPlan> {
        self.stages.iter().find_map(|s| match s {
            PlanStage::Context(c) => Some(c),
            _ => None,
        })
    }
}

// ── Request validation ────────────────────────────────────────────────────────

fn invalid(message: impl Into<String>) -> IndexError {
    IndexError::Invalid(message.into())
}

pub(super) fn validate_request(request: &RetrieveRequest) -> Result<(), IndexError> {
    if request.prefetch.is_empty()
        || request.prefetch.len() > 8
        || request.limit == 0
        || request.limit > 10_000
        || request.context.neighbors > 8
        || request.context.per_parent == Some(0)
        || request
            .context
            .mmr
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(invalid("invalid retrieval or context budgets"));
    }
    if request
        .objective
        .latency_budget_ms
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
        || request.objective.context_budget_tokens == Some(0)
    {
        return Err(invalid("invalid retrieval objective"));
    }
    if let Some(filter) = &request.filter {
        filter.validate(0)?;
    }
    Ok(())
}

// ── Plan compilation ──────────────────────────────────────────────────────────

impl MultiVectorIndex {
    /// Compile a request without scoring documents.
    pub fn plan(&self, request: &RetrieveRequest) -> Result<RetrievalPlan, IndexError> {
        validate_request(request)?;
        let state = self.snapshot();
        let eligible = state
            .documents
            .values()
            .filter(|document| {
                request
                    .filter
                    .as_ref()
                    .is_none_or(|filter| filter.matches(&document.metadata))
            })
            .count();
        self.compile_plan(&state, request, eligible)
    }

    pub(super) fn compile_plan(
        &self,
        state: &State,
        request: &RetrieveRequest,
        eligible_documents: usize,
    ) -> Result<RetrievalPlan, IndexError> {
        validate_request(request)?;

        let stats = planner_stats(state, self.fde.output_dimension());

        let logical_channels = request
            .prefetch
            .iter()
            .enumerate()
            .map(|(index, channel)| {
                let (kind, field, candidate_limit) = match channel {
                    Channel::Bm25 { limit, .. } => (LogicalChannelKind::Bm25, None, *limit),
                    Channel::Sparse { field, limit, .. } => {
                        (LogicalChannelKind::Sparse, Some(field.clone()), *limit)
                    }
                    Channel::Dense { field, limit, .. } => {
                        (LogicalChannelKind::Dense, Some(field.clone()), *limit)
                    }
                    Channel::Multivector { field, limit, .. } => {
                        (LogicalChannelKind::Multivector, field.clone(), *limit)
                    }
                };
                LogicalChannel {
                    index,
                    kind,
                    field,
                    candidate_limit,
                }
            })
            .collect();

        let logical = LogicalPlan {
            objective: request.objective.clone(),
            channels: logical_channels,
            fusion: match request.fusion {
                Fusion::Rrf { .. } => LogicalFusion::ReciprocalRank,
                Fusion::Weighted { .. } => LogicalFusion::Weighted,
            },
            filtered: request.filter.is_some(),
            rerank: request.rerank.is_some(),
            context_selection: request.context != ContextOptions::default(),
        };

        // ── Physical channel planning ─────────────────────────────────────────
        let mut planned_channels = Vec::with_capacity(request.prefetch.len());
        for (index, channel) in request.prefetch.iter().enumerate() {
            let (operator, reason, limit, ef_search, cost) = match channel {
                Channel::Bm25 { text, limit, k1, b } => {
                    if text.len() > 65_536
                        || !k1.is_finite()
                        || *k1 < 0.0
                        || !b.is_finite()
                        || !(0.0..=1.0).contains(b)
                    {
                        return Err(invalid("invalid BM25 query or parameters"));
                    }
                    let cost = stats.token_documents as f64 * 10.0;
                    (PhysicalOperator::Bm25, PlanReason::LexicalIndex, *limit, None, cost)
                }
                Channel::Sparse {
                    field,
                    vector,
                    limit,
                } => {
                    if !state.retrieval.has_sparse_field(field) {
                        return Err(invalid(format!("unknown sparse field {field:?}")));
                    }
                    vector
                        .canonicalized()
                        .map_err(|error| invalid(error.to_string()))?;
                    let cost = stats.token_documents as f64 * 5.0;
                    (PhysicalOperator::SparseDot, PlanReason::SparseIndex, *limit, None, cost)
                }
                Channel::Dense {
                    field,
                    vector,
                    limit,
                    backend,
                    ef_search,
                } => {
                    retrieval::validate_matrix(std::slice::from_ref(vector))?;
                    if state.retrieval.dense_dimension(field)? != vector.len()
                        || !["auto", "exact", "hnsw"].contains(&backend.as_str())
                        || *ef_search == 0
                        || *ef_search > 65_536
                    {
                        return Err(invalid("invalid dense query dimension or backend"));
                    }
                    let ann_ready = state.named_ann.contains_key(field);
                    if backend == "hnsw" && !ann_ready {
                        return Err(invalid("dense ANN not built"));
                    }
                    let dim = state.retrieval.dense_dimension(field).unwrap_or(1) as f64;
                    let ef = *ef_search as f64;
                    let n = stats.documents.max(1) as f64;
                    let f = eligible_documents.max(1) as f64;
                    let cost_exact = f * dim * 2.0;
                    let cost_hnsw = ef * n.log2().ceil().max(1.0) * dim * 3.0;
                    let (operator, reason) = choose_ann_with_cost(
                        backend,
                        request.filter.is_some(),
                        ann_ready,
                        PhysicalOperator::ExactDense,
                        PhysicalOperator::HnswDense,
                        cost_exact,
                        cost_hnsw,
                    );
                    let cost = if operator == PhysicalOperator::HnswDense {
                        cost_hnsw
                    } else {
                        cost_exact
                    };
                    (operator, reason, *limit, Some(*ef_search), cost)
                }
                Channel::Multivector {
                    field: Some(field),
                    vectors,
                    limit,
                    backend,
                    ..
                } => {
                    retrieval::validate_matrix(vectors)?;
                    let expected = FieldSchema::Multivector {
                        dimension: vectors[0].len(),
                    };
                    if state.retrieval.schema().get(field) != Some(&expected) {
                        return Err(invalid(format!(
                            "unknown field or query dimension/kind mismatch: {field:?}"
                        )));
                    }
                    if backend != "auto" && backend != "exact" {
                        return Err(invalid("named multivectors support exact or auto"));
                    }
                    let reason = if backend == "exact" {
                        PlanReason::RequestedExact
                    } else {
                        PlanReason::OperatorRequiresExact
                    };
                    let cost = estimate_channel_cost(
                        PhysicalOperator::ExactMaxsim,
                        eligible_documents,
                        stats.documents,
                        None,
                        &stats,
                        channel,
                    );
                    (PhysicalOperator::ExactMaxsim, reason, *limit, None, cost)
                }
                Channel::Multivector {
                    field: None,
                    vectors,
                    limit,
                    backend,
                    ef_search,
                } => {
                    self.validate(vectors)?;
                    if vectors.len() > 1024
                        || *ef_search == 0
                        || *ef_search > 65_536
                        || !["auto", "exact", "hnsw"].contains(&backend.as_str())
                    {
                        return Err(invalid("invalid multivector backend or budget"));
                    }
                    let ann_ready = state.fde_ann.is_some();
                    if backend == "hnsw" && !ann_ready {
                        return Err(invalid("FDE ANN not built"));
                    }
                    let fde_dim = stats.fde_dimension.max(1) as f64;
                    let ef = *ef_search as f64;
                    let n = stats.documents.max(1) as f64;
                    let f = eligible_documents.max(1) as f64;
                    let cost_exact = f * fde_dim * 2.0;
                    let cost_hnsw = ef * n.log2().ceil().max(1.0) * fde_dim * 3.0;
                    let (operator, reason) = choose_ann_with_cost(
                        backend,
                        request.filter.is_some(),
                        ann_ready,
                        PhysicalOperator::ExactFde,
                        PhysicalOperator::HnswFde,
                        cost_exact,
                        cost_hnsw,
                    );
                    let cost = if operator == PhysicalOperator::HnswFde {
                        cost_hnsw
                    } else {
                        cost_exact
                    };
                    (operator, reason, *limit, Some(*ef_search), cost)
                }
            };
            if limit == 0 || limit > 100_000 {
                return Err(invalid("channel limit must be in 1..=100000"));
            }
            planned_channels.push(PlannedChannel {
                index,
                operator,
                reason,
                limit,
                ef_search,
                estimated_cost_units: cost,
            });
        }

        // ── Validate fusion ───────────────────────────────────────────────────
        match &request.fusion {
            Fusion::Rrf { k } if !k.is_finite() || *k < 0.0 => {
                return Err(invalid("RRF k must be finite and nonnegative"));
            }
            Fusion::Weighted { weights }
                if weights.len() != planned_channels.len()
                    || weights
                        .iter()
                        .any(|weight| !weight.is_finite() || *weight < 0.0)
                    || weights.iter().all(|weight| *weight == 0.0) =>
            {
                return Err(invalid(
                    "weighted fusion requires one nonnegative finite weight per channel",
                ));
            }
            _ => {}
        }

        // ── Validate rerank ───────────────────────────────────────────────────
        if let Some(rerank) = &request.rerank {
            if rerank.limit < request.limit || rerank.limit > 100_000 {
                return Err(invalid(
                    "rerank limit must cover result limit and be <=100000",
                ));
            }
            if let Some(policy) = &rerank.adaptive
                && (policy.min_candidates < request.limit
                    || policy.min_candidates > rerank.limit
                    || !policy.agreement_threshold.is_finite()
                    || !(0.0..=1.0).contains(&policy.agreement_threshold)
                    || planned_channels.len() < 2)
            {
                return Err(invalid(
                    "invalid adaptive rerank policy; needs multiple channels",
                ));
            }
            retrieval::validate_matrix(&rerank.vectors)?;
            if let Some(field) = &rerank.field {
                let expected = FieldSchema::Multivector {
                    dimension: rerank.vectors[0].len(),
                };
                if state.retrieval.schema().get(field) != Some(&expected) {
                    return Err(invalid(format!(
                        "unknown field or query dimension/kind mismatch: {field:?}"
                    )));
                }
            } else {
                self.validate(&rerank.vectors)?;
            }
        }

        // ── Validate context ──────────────────────────────────────────────────
        match (
            request.context.mmr,
            request.context.diversity_field.as_deref(),
        ) {
            (Some(_), Some(field)) => {
                state.retrieval.dense_dimension(field)?;
            }
            (Some(_), None) => return Err(invalid("MMR requires a dense diversity_field")),
            (None, Some(_)) => return Err(invalid("diversity_field requires mmr")),
            (None, None) => {}
        }

        // ── Assemble stages ───────────────────────────────────────────────────
        let fusion_op = match request.fusion {
            Fusion::Rrf { .. } if planned_channels.len() == 1 => FusionOperator::NativeScore,
            Fusion::Rrf { .. } => FusionOperator::ReciprocalRank,
            Fusion::Weighted { .. } => FusionOperator::WeightedScore,
        };

        let rerank_plan = request.rerank.as_ref().map(|rerank| RerankPlan {
            operator: PhysicalOperator::ExactMaxsim,
            candidate_limit: rerank.limit,
            adaptive: rerank.adaptive.is_some(),
        });

        let context_plan = ContextPlan {
            operator: if request.context.mmr.is_some() {
                ContextOperator::Mmr
            } else {
                ContextOperator::Ranked
            },
            result_limit: request.limit,
            context_budget_tokens: request.objective.context_budget_tokens,
        };

        // ── Cost estimation ───────────────────────────────────────────────────
        // Per-channel costs in ms (using static 1ns/unit calibration for Phase 1A).
        const UNITS_PER_MS: f64 = 1_000_000.0;
        let channel_costs_ms: Vec<f64> = planned_channels
            .iter()
            .map(|ch| ch.estimated_cost_units / UNITS_PER_MS)
            .collect();
        let parallel_wall_ms = channel_costs_ms
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(0.001);
        let parallel_cpu_ms: f64 = channel_costs_ms.iter().sum::<f64>().max(0.001);
        let fusion_ms = (planned_channels.len() as f64 * 5.0) / UNITS_PER_MS;
        let rerank_ms = rerank_plan.as_ref().map_or(0.0, |r| {
            let candidates = r.candidate_limit as f64;
            let avg_tokens = (stats.token_vectors as f64
                / stats.token_documents.max(1) as f64)
                .max(1.0);
            let dim = self.config.dimension as f64;
            (candidates * avg_tokens * dim * 2.0) / UNITS_PER_MS
        });
        let context_ms = (request.limit as f64 * 2.0) / UNITS_PER_MS;

        let estimated_wall_ms = parallel_wall_ms + fusion_ms + rerank_ms + context_ms;
        let estimated_cpu_work = parallel_cpu_ms + fusion_ms + rerank_ms + context_ms;

        // ── Build stage list ──────────────────────────────────────────────────
        let mut stages = Vec::with_capacity(4);
        stages.push(PlanStage::Parallel(planned_channels));
        stages.push(PlanStage::Fusion(fusion_op));
        if let Some(rp) = rerank_plan {
            stages.push(PlanStage::Rerank(rp));
        }
        stages.push(PlanStage::Context(context_plan));

        Ok(RetrievalPlan {
            logical,
            stats,
            eligible_documents,
            filter: if request.filter.is_some() {
                FilterStrategy::MetadataScan
            } else {
                FilterStrategy::None
            },
            stages,
            estimate: PlanEstimate {
                estimated_wall_ms,
                estimated_cpu_work,
            },
        })
    }
}

// ── Statistics snapshot ───────────────────────────────────────────────────────

pub(super) fn planner_stats(state: &State, fde_dimension: usize) -> PlannerStats {
    let fields = state
        .retrieval
        .schema()
        .iter()
        .map(|(name, schema)| {
            let (kind, dimension) = match schema {
                FieldSchema::Dense { dimension } => {
                    (RepresentationKind::Dense, Some(*dimension))
                }
                FieldSchema::Multivector { dimension } => {
                    (RepresentationKind::Multivector, Some(*dimension))
                }
                FieldSchema::Sparse => (RepresentationKind::Sparse, None),
            };
            let documents = state
                .documents
                .values()
                .filter(|document| document.fields.has_representation(name))
                .count();
            (
                name.clone(),
                FieldStats {
                    kind,
                    dimension,
                    documents,
                    graph_ready: state.named_ann.contains_key(name),
                },
            )
        })
        .collect();
    PlannerStats {
        generation: state.generation,
        documents: state.documents.len(),
        token_documents: state
            .documents
            .values()
            .filter(|document| document.tokens > 0)
            .count(),
        token_vectors: state.documents.values().map(|document| document.tokens).sum(),
        fde_dimension,
        fde_graph_ready: state.fde_ann.is_some(),
        fields,
    }
}

// ── Cost-based operator selection (Phase 1C) ─────────────────────────────────

/// Choose between exact and ANN operator using the cost model.
/// `cost_exact` and `cost_hnsw` are in cost units (use `estimate_channel_cost`
/// or the inline formulas in `compile_plan`).
fn choose_ann_with_cost(
    backend: &str,
    filtered: bool,
    ann_ready: bool,
    exact: PhysicalOperator,
    ann: PhysicalOperator,
    cost_exact: f64,
    cost_hnsw: f64,
) -> (PhysicalOperator, PlanReason) {
    if backend == "exact" {
        return (exact, PlanReason::RequestedExact);
    }
    if filtered {
        return (exact, PlanReason::FilterRequiresExact);
    }
    if !ann_ready {
        return (exact, PlanReason::AnnUnavailable);
    }
    // ANN is available — choose by cost model.
    if cost_exact <= cost_hnsw {
        (exact, PlanReason::LowerEstimatedCost)
    } else {
        (ann, PlanReason::LowerEstimatedCost)
    }
}

// ── Per-operator cost estimation (Phase 1A static calibration) ───────────────

fn estimate_channel_cost(
    operator: PhysicalOperator,
    eligible: usize,
    corpus: usize,
    ef_search: Option<usize>,
    stats: &PlannerStats,
    channel: &Channel,
) -> f64 {
    let ef = ef_search.unwrap_or(256) as f64;
    let n = corpus.max(1) as f64;
    let f = eligible.max(1) as f64;
    let log2_n = n.log2().ceil().max(1.0);

    match operator {
        PhysicalOperator::Bm25 => stats.token_documents as f64 * 10.0,
        PhysicalOperator::SparseDot => stats.token_documents as f64 * 5.0,
        PhysicalOperator::ExactDense => {
            let dim = channel_dim(channel, stats).max(1) as f64;
            f * dim * 2.0
        }
        PhysicalOperator::HnswDense => {
            let dim = channel_dim(channel, stats).max(1) as f64;
            ef * log2_n * dim * 3.0
        }
        PhysicalOperator::ExactFde | PhysicalOperator::ExactEligibleScan => {
            let fde_dim = stats.fde_dimension.max(1) as f64;
            f * fde_dim * 2.0
        }
        PhysicalOperator::HnswFde => {
            let fde_dim = stats.fde_dimension.max(1) as f64;
            ef * log2_n * fde_dim * 3.0
        }
        PhysicalOperator::ExactMaxsim => {
            // Approximate: scale FDE cost by token-to-vector ratio
            let fde_dim = stats.fde_dimension.max(1) as f64;
            let avg_tokens = (stats.token_vectors as f64 / stats.token_documents.max(1) as f64)
                .max(1.0)
                .min(512.0);
            f * fde_dim * avg_tokens * 2.0
        }
        PhysicalOperator::HnswPostFilter => {
            let fde_dim = stats.fde_dimension.max(1) as f64;
            let selectivity = (f / n).max(0.05);
            let ef_amp = (ef / selectivity).min(ef * 4.0);
            ef_amp * log2_n * fde_dim * 3.0
        }
        PhysicalOperator::FilterBitmap => f * 4.0,
    }
}

fn channel_dim(channel: &Channel, stats: &PlannerStats) -> usize {
    match channel {
        Channel::Dense { field, .. } => stats
            .fields
            .get(field)
            .and_then(|f| f.dimension)
            .unwrap_or(1),
        Channel::Multivector {
            field: Some(field), ..
        } => stats
            .fields
            .get(field)
            .and_then(|f| f.dimension)
            .unwrap_or(1),
        _ => 1,
    }
}
