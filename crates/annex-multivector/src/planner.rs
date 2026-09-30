//! Deterministic physical planning for the retrieval API.
use super::*;

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
    LexicalIndex,
    SparseIndex,
    RequestedExact,
    AnnReady,
    AnnUnavailable,
    FilterRequiresExact,
    OperatorRequiresExact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterStrategy {
    None,
    MetadataScan,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PlannedChannel {
    pub index: usize,
    pub operator: PhysicalOperator,
    pub reason: PlanReason,
    pub limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RetrievalPlan {
    pub generation: u64,
    pub documents: usize,
    pub eligible_documents: usize,
    pub filter: FilterStrategy,
    pub channels: Vec<PlannedChannel>,
    pub result_limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_limit: Option<usize>,
}

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
    if let Some(filter) = &request.filter {
        filter.validate(0)?;
    }
    Ok(())
}

impl MultiVectorIndex {
    /// Compile a request without scoring documents. The returned plan is also
    /// embedded in the trace produced by [`Self::retrieve`].
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

        let mut channels = Vec::with_capacity(request.prefetch.len());
        for (index, channel) in request.prefetch.iter().enumerate() {
            let (operator, reason, limit, ef_search) = match channel {
                Channel::Bm25 { text, limit, k1, b } => {
                    if text.len() > 65_536
                        || !k1.is_finite()
                        || *k1 < 0.0
                        || !b.is_finite()
                        || !(0.0..=1.0).contains(b)
                    {
                        return Err(invalid("invalid BM25 query or parameters"));
                    }
                    (
                        PhysicalOperator::Bm25,
                        PlanReason::LexicalIndex,
                        *limit,
                        None,
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
                    (
                        PhysicalOperator::SparseDot,
                        PlanReason::SparseIndex,
                        *limit,
                        None,
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
                    let (operator, reason) = choose_ann(
                        backend,
                        request.filter.is_some(),
                        ann_ready,
                        PhysicalOperator::ExactDense,
                        PhysicalOperator::HnswDense,
                    );
                    (operator, reason, *limit, Some(*ef_search))
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
                    (
                        PhysicalOperator::ExactMaxsim,
                        if backend == "exact" {
                            PlanReason::RequestedExact
                        } else {
                            PlanReason::OperatorRequiresExact
                        },
                        *limit,
                        None,
                    )
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
                    let (operator, reason) = choose_ann(
                        backend,
                        request.filter.is_some(),
                        ann_ready,
                        PhysicalOperator::ExactFde,
                        PhysicalOperator::HnswFde,
                    );
                    (operator, reason, *limit, Some(*ef_search))
                }
            };
            if limit == 0 || limit > 100_000 {
                return Err(invalid("channel limit must be in 1..=100000"));
            }
            channels.push(PlannedChannel {
                index,
                operator,
                reason,
                limit,
                ef_search,
            });
        }

        match &request.fusion {
            Fusion::Rrf { k } if !k.is_finite() || *k < 0.0 => {
                return Err(invalid("RRF k must be finite and nonnegative"));
            }
            Fusion::Weighted { weights }
                if weights.len() != channels.len()
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
                    || channels.len() < 2)
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

        Ok(RetrievalPlan {
            generation: state.generation,
            documents: state.documents.len(),
            eligible_documents,
            filter: if request.filter.is_some() {
                FilterStrategy::MetadataScan
            } else {
                FilterStrategy::None
            },
            channels,
            result_limit: request.limit,
            rerank_limit: request.rerank.as_ref().map(|rerank| rerank.limit),
        })
    }
}

fn choose_ann(
    backend: &str,
    filtered: bool,
    ann_ready: bool,
    exact: PhysicalOperator,
    ann: PhysicalOperator,
) -> (PhysicalOperator, PlanReason) {
    if backend == "exact" {
        (exact, PlanReason::RequestedExact)
    } else if filtered {
        (exact, PlanReason::FilterRequiresExact)
    } else if ann_ready {
        (ann, PlanReason::AnnReady)
    } else {
        (exact, PlanReason::AnnUnavailable)
    }
}
