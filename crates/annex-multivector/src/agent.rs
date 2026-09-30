//! Iterative Evidence Planner (Phase 5).
//!
//! Drives multiple rounds of retrieval without LLM or external models.
//! All actions are operations ANNex can perform natively: widen depth,
//! activate representations, expand neighbors, stop on diminishing returns.
use super::*;

// ── Corpus fingerprint — stable key for strategy memory ──────────────────────

/// A stable identifier for the corpus/schema/config combination.
/// Keyed on schema + config hash + doc-count bucket, NOT on generation
/// (every write bumps generation — keying on it would expire memory constantly).
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CorpusFingerprint {
    /// Sorted field names and kinds (schema identity).
    pub schema_hash: u64,
    /// Approximate document count bucket (1K, 10K, 100K, 1M+).
    pub doc_count_bucket: usize,
    /// FDE dimension (proxy for model config).
    pub fde_dimension: usize,
}

impl CorpusFingerprint {
    pub fn from_stats(stats: &PlannerStats) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for (name, field) in &stats.fields {
            name.hash(&mut hasher);
            format!("{:?}", field.kind).hash(&mut hasher);
        }
        let schema_hash = hasher.finish();
        let doc_count_bucket = match stats.documents {
            0..=999 => 1_000,
            1_000..=9_999 => 10_000,
            10_000..=99_999 => 100_000,
            _ => 1_000_000,
        };
        Self {
            schema_hash,
            doc_count_bucket,
            fde_dimension: stats.fde_dimension,
        }
    }
}

// ── Strategy memory ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CorpusStrategy {
    /// Average iterations to satisfactory evidence in past searches.
    pub avg_iterations: f32,
    /// Channels that produced the most useful evidence.
    pub effective_channels: Vec<LogicalChannelKind>,
}

// ── Stop reasons ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StopReason {
    TokenBudgetExhausted,
    /// < 10% new candidates vs prior iteration.
    DiminishingReturns,
    MaxIterationsReached,
    /// Top-1 margin and spread indicate high-confidence result set.
    HighAgreementAcrossIterations,
}

// ── Agent decisions ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AgentIteration {
    pub request: RetrieveRequest,
    pub rationale: String,
}

#[derive(Clone, Debug)]
pub enum AgentDecision {
    Continue(AgentIteration),
    Stop(StopReason),
}

// ── AgentSearch ──────────────────────────────────────────────────────────────

pub struct AgentSearch {
    pub objective: RetrievalObjective,
    pub query: QueryRepresentations,
    pub prior_ids: HashSet<String>,
    pub iteration: usize,
    pub total_tokens_consumed: usize,
    strategy_memory: HashMap<CorpusFingerprint, CorpusStrategy>,
}

impl AgentSearch {
    pub fn new(objective: RetrievalObjective, query: QueryRepresentations) -> Self {
        Self {
            objective,
            query,
            prior_ids: HashSet::new(),
            iteration: 0,
            total_tokens_consumed: 0,
            strategy_memory: HashMap::new(),
        }
    }

    /// Decide the next retrieval action or stop.
    ///
    /// `stats` is the current corpus snapshot. `last_trace` is None on the
    /// first call and Some on subsequent calls.
    pub fn plan_next(
        &mut self,
        stats: &PlannerStats,
        last_trace: Option<&RetrievalTrace>,
    ) -> AgentDecision {
        // ── Stopping criteria ─────────────────────────────────────────────────
        if let Some(budget) = self.objective.context_budget_tokens {
            if self.total_tokens_consumed >= budget {
                return AgentDecision::Stop(StopReason::TokenBudgetExhausted);
            }
        }
        if self.iteration >= MAX_ITERATIONS {
            return AgentDecision::Stop(StopReason::MaxIterationsReached);
        }
        if let Some(trace) = last_trace {
            if self.iteration > 0
                && trace.signals.top1_margin > HIGH_CONFIDENCE_MARGIN_THRESHOLD
                && trace.signals.topk_score_spread < HIGH_CONFIDENCE_SPREAD_THRESHOLD
            {
                return AgentDecision::Stop(StopReason::HighAgreementAcrossIterations);
            }
        }

        // ── Build next request ────────────────────────────────────────────────
        let quality = self.objective.quality;
        let result_limit = next_result_limit(self.iteration, quality);
        let candidate_limit = result_limit * candidate_multiplier(self.iteration, quality);
        let ef = next_ef(self.iteration, quality);

        let mut channels: Vec<Channel> = Vec::new();
        let mut rationale_parts: Vec<String> = Vec::new();

        // BM25 if text present and corpus is non-empty.
        // Coverage check uses documents > 0 rather than token_documents (FDE vectors)
        // because BM25 works from the lexical index which is populated from text fields.
        if let Some(text) = &self.query.text {
            if stats.documents > 0 {
                channels.push(Channel::Bm25 {
                    text: text.clone(),
                    limit: candidate_limit,
                    k1: 1.2,
                    b: 0.75,
                });
                rationale_parts.push(format!("BM25(limit={})", candidate_limit));
            }
        }

        // Dense fields from query representations
        for (field, vec) in &self.query.dense {
            if let Some(field_stats) = stats.fields.get(field) {
                let coverage = field_stats.documents as f32 / stats.documents.max(1) as f32;
                if coverage >= 0.3 {
                    channels.push(Channel::Dense {
                        field: field.clone(),
                        vector: vec.clone(),
                        limit: candidate_limit,
                        backend: "auto".into(),
                        ef_search: ef,
                    });
                    rationale_parts.push(format!("Dense[{field}](ef={ef})"));
                }
            }
        }

        if channels.is_empty() {
            return AgentDecision::Stop(StopReason::MaxIterationsReached);
        }

        let rationale = format!(
            "iter={}: {}",
            self.iteration,
            rationale_parts.join(", ")
        );

        let request = RetrieveRequest {
            prefetch: channels,
            planning_mode: PlanningMode::Manual,
            query: None,
            objective: self.objective.clone(),
            fusion: Fusion::Rrf { k: 10.0 },
            filter: None,
            rerank: None,
            limit: result_limit,
            context: ContextOptions {
                neighbors: if self.iteration > 0 { 1 } else { 0 },
                ..ContextOptions::default()
            },
        };
        self.iteration += 1;
        AgentDecision::Continue(AgentIteration { request, rationale })
    }

    /// Record results from a completed iteration.
    pub fn record_results(&mut self, trace: &RetrievalTrace, matches: &[ContextHit]) {
        for hit in matches {
            if let Some(t) = hit.estimated_tokens {
                self.total_tokens_consumed += t;
            }
            self.prior_ids.insert(hit.id.clone());
        }
        // Update strategy memory for this corpus.
        let fp = CorpusFingerprint::from_stats(&trace.plan.stats);
        self.strategy_memory
            .entry(fp)
            .and_modify(|s| {
                s.avg_iterations = s.avg_iterations * 0.9 + self.iteration as f32 * 0.1;
            })
            .or_insert(CorpusStrategy {
                avg_iterations: self.iteration as f32,
                effective_channels: trace
                    .plan
                    .logical
                    .channels
                    .iter()
                    .map(|c| c.kind)
                    .collect(),
            });
    }
}

// ── Iteration parameters ─────────────────────────────────────────────────────

const MAX_ITERATIONS: usize = 5;
const HIGH_CONFIDENCE_MARGIN_THRESHOLD: f32 = 0.3;
const HIGH_CONFIDENCE_SPREAD_THRESHOLD: f32 = 0.2;

fn next_result_limit(iteration: usize, quality: QualityPreference) -> usize {
    let base = match quality {
        QualityPreference::Fast => 5,
        QualityPreference::Balanced => 10,
        QualityPreference::High => 20,
    };
    base + iteration * 5
}

fn candidate_multiplier(iteration: usize, _quality: QualityPreference) -> usize {
    (2 + iteration).min(10)
}

fn next_ef(iteration: usize, quality: QualityPreference) -> usize {
    let base = match quality {
        QualityPreference::Fast => 64,
        QualityPreference::Balanced => 256,
        QualityPreference::High => 512,
    };
    (base + iteration * 64).min(65_536)
}
