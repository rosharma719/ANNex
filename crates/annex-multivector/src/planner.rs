//! Deterministic physical planning for the retrieval API.
use super::*;

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalOperator {
    Bm25,
    SparseDot,
    ExactDense,
    HnswDense,
    ExactMaxsim,
    ExactFde,
    HnswFde,
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
        }
    }
}

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
    /// Filter present; selectivity too low or post-filter disabled.
    FilterRequiresExact,
    /// Operator does not support ANN (e.g. named multivector exact-only).
    OperatorRequiresExact,
    /// Cost model estimated exact cheaper than HNSW for this corpus size.
    LowerEstimatedCost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterStrategy {
    None,
    MetadataIndex,
    MetadataScan,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FilterStats {
    /// Fraction of the current generation that passed the predicate.
    pub selectivity: f32,
    /// Physical strategy used to evaluate the filter.
    pub filter_operator: FilterStrategy,
}

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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlannerStats {
    pub generation: u64,
    pub documents: usize,
    pub text_documents: usize,
    pub token_documents: usize,
    pub token_vectors: usize,
    pub fde_dimension: usize,
    pub fde_graph_ready: bool,
    pub fields: BTreeMap<String, FieldStats>,
    /// Populated when a filter is present; None for unfiltered plans.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter_stats: Option<FilterStats>,
}

/// Collection-wide statistics maintained with each immutable state generation.
/// Query planning must not scan the document map to rediscover these values.
#[derive(Clone, Debug, Default)]
pub(super) struct CachedPlannerStats {
    text_documents: usize,
    token_documents: usize,
    token_vectors: usize,
    field_documents: BTreeMap<String, usize>,
}

impl CachedPlannerStats {
    pub(super) fn from_documents(
        documents: &HashMap<String, DocumentRecord>,
        schema: &BTreeMap<String, FieldSchema>,
    ) -> Self {
        let mut stats = Self {
            field_documents: schema.keys().map(|name| (name.clone(), 0)).collect(),
            ..Self::default()
        };
        for document in documents.values() {
            stats.add(document, schema);
        }
        stats
    }

    pub(super) fn add(
        &mut self,
        document: &DocumentRecord,
        schema: &BTreeMap<String, FieldSchema>,
    ) {
        self.text_documents += usize::from(document.fields.has_text());
        self.token_documents += usize::from(document.tokens > 0);
        self.token_vectors += document.tokens;
        for name in schema.keys() {
            let count = self.field_documents.entry(name.clone()).or_default();
            *count += usize::from(document.fields.has_representation(name));
        }
    }

    pub(super) fn remove(
        &mut self,
        document: &DocumentRecord,
        schema: &BTreeMap<String, FieldSchema>,
    ) {
        self.text_documents = self
            .text_documents
            .checked_sub(usize::from(document.fields.has_text()))
            .expect("cached text-document count is consistent");
        self.token_documents = self
            .token_documents
            .checked_sub(usize::from(document.tokens > 0))
            .expect("cached token-document count is consistent");
        self.token_vectors = self
            .token_vectors
            .checked_sub(document.tokens)
            .expect("cached token-vector count is consistent");
        for name in schema.keys() {
            let decrement = usize::from(document.fields.has_representation(name));
            let count = self.field_documents.entry(name.clone()).or_default();
            *count = count
                .checked_sub(decrement)
                .expect("cached field-document count is consistent");
        }
    }
}

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

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum PlanStage {
    Parallel(Vec<PlannedChannel>),
    Fusion(FusionOperator),
    Rerank(RerankPlan),
    Context(ContextPlan),
}

/// Relative work estimate. These units compare plans; they are not latency.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlanEstimate {
    pub critical_path_cost: f64,
    pub total_cost: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RetrievalPlan {
    pub logical: LogicalPlan,
    pub stats: PlannerStats,
    pub eligible_documents: usize,
    pub filter: FilterStrategy,
    pub stages: Vec<PlanStage>,
    pub estimate: PlanEstimate,
    /// Set when planning_mode != Manual; contains the auto-generated channel list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<PolicyPlan>,
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

fn invalid(message: impl Into<String>) -> IndexError {
    IndexError::Invalid(message.into())
}

pub(super) fn validate_request(request: &RetrieveRequest) -> Result<(), IndexError> {
    // In auto modes, prefetch is generated by the policy planner — allow empty.
    let prefetch_empty = request.prefetch.is_empty();
    let is_auto = request.planning_mode != PlanningMode::Manual;
    if is_auto && request.query.is_none() {
        return Err(invalid(
            "planning_mode auto requires a query field with representations",
        ));
    }
    if prefetch_empty && !is_auto
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
    if request.objective.latency_budget_ms.is_some() {
        return Err(invalid(
            "latency_budget_ms requires calibrated planning and is not supported",
        ));
    }
    if request.objective.context_budget_tokens == Some(0) {
        return Err(invalid("invalid retrieval objective"));
    }
    if let Some(filter) = &request.filter {
        filter.validate(0)?;
    }
    Ok(())
}

impl MultiVectorIndex {
    pub(super) fn prepare_request<'a>(
        &self,
        state: &State,
        request: &'a RetrieveRequest,
        eligible_documents: Option<(usize, FilterStrategy)>,
    ) -> (std::borrow::Cow<'a, RetrieveRequest>, Option<PolicyPlan>) {
        if request.planning_mode == PlanningMode::Manual {
            return (std::borrow::Cow::Borrowed(request), None);
        }
        let stats = planner_stats(state, self.fde.output_dimension(), eligible_documents);
        let mut policy = policy::generate_policy_prefetch(
            request.query.as_ref().expect("auto query validated"),
            &stats,
            request.limit,
            request.objective.quality,
            state.retrieval.schema(),
        );
        if request.planning_mode == PlanningMode::AutoWithOverrides {
            policy::apply_overrides(&mut policy, &request.prefetch);
        }
        let mut effective = request.clone();
        effective.prefetch = policy.generated_prefetch.clone();
        (std::borrow::Cow::Owned(effective), Some(policy))
    }

    /// Compile a request without scoring documents.
    pub fn plan(&self, request: &RetrieveRequest) -> Result<RetrievalPlan, IndexError> {
        validate_request(request)?;
        let state = self.snapshot();
        let (eligible, filter_strategy) = request.filter.as_ref().map_or(
            (state.documents.len(), FilterStrategy::None),
            |filter| {
                state.retrieval.indexed_filter_count(filter).map_or_else(
                    || {
                        (
                            state
                                .documents
                                .values()
                                .filter(|document| filter.matches(&document.metadata))
                                .count(),
                            FilterStrategy::MetadataScan,
                        )
                    },
                    |count| (count, FilterStrategy::MetadataIndex),
                )
            },
        );
        let planner_filter = request.filter.as_ref().map(|_| (eligible, filter_strategy));
        let (effective_request, policy_plan) =
            self.prepare_request(&state, request, planner_filter);
        let request = effective_request.as_ref();
        let mut plan = self.compile_plan(&state, request, eligible, filter_strategy)?;
        plan.policy = policy_plan;
        Ok(plan)
    }

    pub(super) fn compile_plan(
        &self,
        state: &State,
        request: &RetrieveRequest,
        eligible_documents: usize,
        filter_strategy: FilterStrategy,
    ) -> Result<RetrievalPlan, IndexError> {
        validate_request(request)?;
        if request.prefetch.is_empty() {
            return Err(invalid(
                "auto planning found no usable query representation",
            ));
        }

        let stats = planner_stats(
            state,
            self.fde.output_dimension(),
            request
                .filter
                .as_ref()
                .map(|_| (eligible_documents, filter_strategy)),
        );

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
                    (
                        PhysicalOperator::Bm25,
                        PlanReason::LexicalIndex,
                        *limit,
                        None,
                        cost,
                    )
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
                    (
                        PhysicalOperator::SparseDot,
                        PlanReason::SparseIndex,
                        *limit,
                        None,
                        cost,
                    )
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
                    let filtered = request.filter.is_some();
                    let filter_supported = request
                        .filter
                        .as_ref()
                        .is_none_or(|filter| filter.ann_filter().is_some());
                    if backend == "hnsw" && !filter_supported {
                        return Err(invalid("filter is not supported by dense HNSW"));
                    }
                    // EXACT_SCALE: benchmarks on a 10k×128 fixture show exact
                    // is ~26× slower than HNSW unfiltered, but the raw op counts
                    // imply only ~2×. The 13× multiplier closes that gap so the
                    // model's crossover (~4% selectivity) matches empirical data.
                    // In-place filtered HNSW (annex-core) does not degrade with
                    // selectivity the way post-filter HNSW does, so no selectivity
                    // penalty is applied to cost_hnsw.
                    let cost_exact = f * dim * 2.0 * 13.0;
                    let cost_hnsw = ef * n.log2().ceil().max(1.0) * dim * 3.0;
                    let _ = filtered; // crossover via calibrated cost_exact
                    let (operator, reason) = choose_ann_with_cost(
                        backend,
                        ann_ready,
                        filter_supported,
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
                    let filtered = request.filter.is_some();
                    let filter_supported = request
                        .filter
                        .as_ref()
                        .is_none_or(|filter| filter.ann_filter().is_some());
                    if backend == "hnsw" && !filter_supported {
                        return Err(invalid("filter is not supported by FDE HNSW"));
                    }
                    let cost_exact = f * fde_dim * 2.0 * 13.0;
                    let cost_hnsw = ef * n.log2().ceil().max(1.0) * fde_dim * 3.0;
                    let _ = filtered;
                    let (operator, reason) = choose_ann_with_cost(
                        backend,
                        ann_ready,
                        filter_supported,
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

        let channel_costs = planned_channels
            .iter()
            .map(|channel| channel.estimated_cost_units)
            .collect::<Vec<_>>();
        let parallel_critical_path = channel_costs
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let parallel_total = channel_costs.iter().sum::<f64>().max(1.0);
        let fusion_cost = planned_channels.len() as f64 * 5.0;
        let rerank_cost = rerank_plan.as_ref().map_or(0.0, |r| {
            let candidates = r.candidate_limit as f64;
            let avg_tokens =
                (stats.token_vectors as f64 / stats.token_documents.max(1) as f64).max(1.0);
            let dim = self.config.dimension as f64;
            candidates * avg_tokens * dim * 2.0
        });
        let context_cost = request.limit as f64 * 2.0;

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
            filter: filter_strategy,
            stages,
            estimate: PlanEstimate {
                critical_path_cost: parallel_critical_path
                    + fusion_cost
                    + rerank_cost
                    + context_cost,
                total_cost: parallel_total + fusion_cost + rerank_cost + context_cost,
            },
            policy: None, // populated by plan()/retrieve() for non-Manual modes
        })
    }
}

pub(super) fn planner_stats(
    state: &State,
    fde_dimension: usize,
    eligible_documents: Option<(usize, FilterStrategy)>,
) -> PlannerStats {
    let fields = state
        .retrieval
        .schema()
        .iter()
        .map(|(name, schema)| {
            let (kind, dimension) = match schema {
                FieldSchema::Dense { dimension } => (RepresentationKind::Dense, Some(*dimension)),
                FieldSchema::Multivector { dimension } => {
                    (RepresentationKind::Multivector, Some(*dimension))
                }
                FieldSchema::Sparse => (RepresentationKind::Sparse, None),
            };
            let documents = state
                .planner_stats
                .field_documents
                .get(name)
                .copied()
                .unwrap_or(0);
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
    let n = state.documents.len();
    let filter_stats = eligible_documents.map(|(eligible, filter_operator)| {
        let selectivity = if n == 0 {
            1.0_f32
        } else {
            (eligible as f32) / (n as f32)
        };
        FilterStats {
            selectivity: selectivity.clamp(0.0, 1.0),
            filter_operator,
        }
    });
    PlannerStats {
        generation: state.generation,
        documents: n,
        text_documents: state.planner_stats.text_documents,
        token_documents: state.planner_stats.token_documents,
        token_vectors: state.planner_stats.token_vectors,
        fde_dimension,
        fde_graph_ready: state.fde_ann.is_some(),
        fields,
        filter_stats,
    }
}

/// Choose between exact and ANN operator using the cost model.
/// `cost_exact` and `cost_hnsw` are in cost units (use `estimate_channel_cost`
/// or the inline formulas in `compile_plan`).
fn choose_ann_with_cost(
    backend: &str,
    ann_ready: bool,
    ann_supported: bool,
    exact: PhysicalOperator,
    ann: PhysicalOperator,
    cost_exact: f64,
    cost_hnsw: f64,
) -> (PhysicalOperator, PlanReason) {
    if backend == "exact" {
        return (exact, PlanReason::RequestedExact);
    }
    if !ann_ready {
        return (exact, PlanReason::AnnUnavailable);
    }
    if !ann_supported {
        return (exact, PlanReason::FilterRequiresExact);
    }
    if backend == "hnsw" {
        return (ann, PlanReason::AnnReady);
    }
    // ANN is available — choose by cost model.
    if cost_exact <= cost_hnsw {
        (exact, PlanReason::LowerEstimatedCost)
    } else {
        (ann, PlanReason::LowerEstimatedCost)
    }
}

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
            f * dim * 2.0 * 13.0
        }
        PhysicalOperator::HnswDense => {
            let dim = channel_dim(channel, stats).max(1) as f64;
            ef * log2_n * dim * 3.0
        }
        PhysicalOperator::ExactFde => {
            let fde_dim = stats.fde_dimension.max(1) as f64;
            f * fde_dim * 2.0 * 13.0
        }
        PhysicalOperator::HnswFde => {
            let fde_dim = stats.fde_dimension.max(1) as f64;
            ef * log2_n * fde_dim * 3.0
        }
        PhysicalOperator::ExactMaxsim => {
            // Approximate: scale FDE cost by token-to-vector ratio
            let fde_dim = stats.fde_dimension.max(1) as f64;
            let avg_tokens = (stats.token_vectors as f64 / stats.token_documents.max(1) as f64)
                .clamp(1.0, 512.0);
            f * fde_dim * avg_tokens * 2.0
        }
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
