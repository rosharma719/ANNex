//! Retrieval-policy optimizer (Phase 3).
//!
//! Translates an information need + available query representations into a
//! concrete channel list when `planning_mode == Auto`.
use super::*;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueryRepresentations {
    pub text: Option<String>,
    #[serde(default)]
    pub dense: BTreeMap<String, Vector>,
    #[serde(default)]
    pub sparse: BTreeMap<String, annex::vector::sparse::SparseVector>,
    #[serde(default)]
    pub multivector: BTreeMap<String, Vec<Vector>>,
    /// Unnamed late-interaction (FDE/ColBERT) vectors.
    pub fde: Option<Vec<Vector>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct QueryFeatures {
    pub token_count: usize,
    /// Fraction of tokens matching identifier/camelCase/ALLCAPS patterns.
    pub identifier_fraction: f32,
    pub numeric_fraction: f32,
    pub quoted_phrase_count: usize,
    pub has_dense: bool,
    pub has_sparse: bool,
    pub has_multivector: bool,
    pub has_fde: bool,
}

impl QueryFeatures {
    pub fn from_query(query: &QueryRepresentations) -> Self {
        let text = match &query.text {
            Some(t) => t.as_str(),
            None => {
                return Self {
                    has_dense: !query.dense.is_empty(),
                    has_sparse: !query.sparse.is_empty(),
                    has_multivector: !query.multivector.is_empty(),
                    has_fde: query.fde.is_some(),
                    ..Self::default()
                };
            }
        };
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let token_count = tokens.len();
        let numeric = tokens
            .iter()
            .filter(|t| t.chars().any(char::is_numeric))
            .count();
        let identifier = tokens
            .iter()
            .filter(|t| {
                let bytes = t.as_bytes();
                bytes.len() > 2
                    && bytes.iter().any(|b| b.is_ascii_uppercase())
                    && bytes.iter().any(|b| b.is_ascii_lowercase())
            })
            .count();
        let n = token_count.max(1);
        Self {
            token_count,
            identifier_fraction: identifier as f32 / n as f32,
            numeric_fraction: numeric as f32 / n as f32,
            quoted_phrase_count: text.matches('"').count() / 2,
            has_dense: !query.dense.is_empty(),
            has_sparse: !query.sparse.is_empty(),
            has_multivector: !query.multivector.is_empty(),
            has_fde: query.fde.is_some(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PolicyPlan {
    pub channels_selected: Vec<LogicalChannelKind>,
    pub selection_reasons: Vec<String>,
    pub query_features: QueryFeatures,
    pub generated_prefetch: Vec<Channel>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningMode {
    #[default]
    Manual,
    Auto,
    AutoWithOverrides,
}

pub(super) fn apply_overrides(plan: &mut PolicyPlan, overrides: &[Channel]) {
    for channel in overrides {
        if let Some(existing) = plan
            .generated_prefetch
            .iter_mut()
            .find(|existing| same_channel(existing, channel))
        {
            *existing = channel.clone();
        } else {
            plan.generated_prefetch.push(channel.clone());
        }
    }
    plan.channels_selected = plan.generated_prefetch.iter().map(channel_kind).collect();
    if !overrides.is_empty() {
        plan.selection_reasons
            .push(format!("{} caller override(s)", overrides.len()));
    }
}

fn same_channel(left: &Channel, right: &Channel) -> bool {
    match (left, right) {
        (Channel::Bm25 { .. }, Channel::Bm25 { .. }) => true,
        (Channel::Sparse { field: a, .. }, Channel::Sparse { field: b, .. })
        | (Channel::Dense { field: a, .. }, Channel::Dense { field: b, .. }) => a == b,
        (Channel::Multivector { field: a, .. }, Channel::Multivector { field: b, .. }) => a == b,
        _ => false,
    }
}

fn channel_kind(channel: &Channel) -> LogicalChannelKind {
    match channel {
        Channel::Bm25 { .. } => LogicalChannelKind::Bm25,
        Channel::Sparse { .. } => LogicalChannelKind::Sparse,
        Channel::Dense { .. } => LogicalChannelKind::Dense,
        Channel::Multivector { .. } => LogicalChannelKind::Multivector,
    }
}

fn candidate_limit(result_limit: usize, quality: QualityPreference) -> usize {
    match quality {
        QualityPreference::Fast => (result_limit * 2).max(10),
        QualityPreference::Balanced => (result_limit * 5).max(20),
        QualityPreference::High => (result_limit * 10).max(50),
    }
}

fn ef_search(quality: QualityPreference) -> usize {
    match quality {
        QualityPreference::Fast => 64,
        QualityPreference::Balanced => 256,
        QualityPreference::High => 1024,
    }
}

const COVERAGE_THRESHOLD: f32 = 0.5;

pub(super) fn generate_policy_prefetch(
    query: &QueryRepresentations,
    stats: &PlannerStats,
    result_limit: usize,
    quality: QualityPreference,
    schema: &BTreeMap<String, FieldSchema>,
) -> PolicyPlan {
    let limit = candidate_limit(result_limit, quality);
    let ef = ef_search(quality);
    let features = QueryFeatures::from_query(query);
    let mut channels: Vec<Channel> = Vec::new();
    let mut selected_kinds: Vec<LogicalChannelKind> = Vec::new();
    let mut reasons: Vec<String> = Vec::new();

    let bm25_coverage = stats.text_documents as f32 / stats.documents.max(1) as f32;
    let text = query.text.clone().unwrap_or_default();
    if !text.trim().is_empty() && bm25_coverage >= COVERAGE_THRESHOLD {
        channels.push(Channel::Bm25 {
            text: text.clone(),
            limit,
            k1: 1.2,
            b: 0.75,
        });
        selected_kinds.push(LogicalChannelKind::Bm25);
        reasons.push(format!(
            "BM25: text provided, corpus coverage {:.0}%",
            bm25_coverage * 100.
        ));
    }

    for (field, vec) in &query.dense {
        if let Some(field_stats) = stats.fields.get(field) {
            let coverage = field_stats.documents as f32 / stats.documents.max(1) as f32;
            if coverage >= COVERAGE_THRESHOLD
                && matches!(schema.get(field), Some(FieldSchema::Dense { .. }))
            {
                channels.push(Channel::Dense {
                    field: field.clone(),
                    vector: vec.clone(),
                    limit,
                    backend: "auto".into(),
                    ef_search: ef,
                });
                selected_kinds.push(LogicalChannelKind::Dense);
                reasons.push(format!("Dense[{field}]: coverage {:.0}%", coverage * 100.));
            }
        }
    }

    for (field, vec) in &query.sparse {
        if schema.contains_key(field) {
            channels.push(Channel::Sparse {
                field: field.clone(),
                vector: vec.clone(),
                limit,
            });
            selected_kinds.push(LogicalChannelKind::Sparse);
            reasons.push(format!("Sparse[{field}]: index present"));
        }
    }

    for (field, vecs) in &query.multivector {
        if let Some(field_stats) = stats.fields.get(field) {
            let coverage = field_stats.documents as f32 / stats.documents.max(1) as f32;
            if coverage >= COVERAGE_THRESHOLD
                && matches!(schema.get(field), Some(FieldSchema::Multivector { .. }))
            {
                channels.push(Channel::Multivector {
                    field: Some(field.clone()),
                    vectors: vecs.clone(),
                    limit,
                    backend: "auto".into(),
                    ef_search: ef,
                });
                selected_kinds.push(LogicalChannelKind::Multivector);
                reasons.push(format!(
                    "Multivector[{field}]: coverage {:.0}%",
                    coverage * 100.
                ));
            }
        }
    }

    if let Some(vecs) = &query.fde {
        let fde_coverage = stats.token_documents as f32 / stats.documents.max(1) as f32;
        if fde_coverage >= COVERAGE_THRESHOLD {
            channels.push(Channel::Multivector {
                field: None,
                vectors: vecs.clone(),
                limit,
                backend: "auto".into(),
                ef_search: ef,
            });
            selected_kinds.push(LogicalChannelKind::Multivector);
            reasons.push(format!(
                "FDE: late-interaction coverage {:.0}%",
                fde_coverage * 100.
            ));
        }
    }

    if channels.is_empty() && !text.trim().is_empty() {
        channels.push(Channel::Bm25 {
            text: text.clone(),
            limit,
            k1: 1.2,
            b: 0.75,
        });
        selected_kinds.push(LogicalChannelKind::Bm25);
        reasons.push("BM25: fallback (coverage below threshold)".into());
    }

    PolicyPlan {
        channels_selected: selected_kinds,
        selection_reasons: reasons,
        query_features: features,
        generated_prefetch: channels,
    }
}
