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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntent {
    Lexical,
    Semantic,
    #[default]
    Hybrid,
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
        let identifier = tokens.iter().filter(|token| is_identifier(token)).count();
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

    fn intent(&self, has_text: bool) -> QueryIntent {
        let has_semantic = self.has_dense || self.has_multivector || self.has_fde;
        let lexical_signal = self.identifier_fraction >= 0.2
            || self.numeric_fraction >= 0.25
            || self.quoted_phrase_count > 0;
        if has_text && lexical_signal {
            QueryIntent::Lexical
        } else if has_semantic && (!has_text || self.token_count >= 8) {
            QueryIntent::Semantic
        } else {
            QueryIntent::Hybrid
        }
    }
}

fn is_identifier(token: &&str) -> bool {
    let token = token.trim_matches(|c: char| !c.is_alphanumeric() && !"_:/.-".contains(c));
    let bytes = token.as_bytes();
    bytes.len() > 2
        && (token.contains(|c| ['_', ':', '/', '.'].contains(&c))
            || (bytes.iter().any(|b| b.is_ascii_uppercase())
                && bytes.iter().any(|b| b.is_ascii_lowercase()))
            || (bytes.iter().any(|b| b.is_ascii_alphabetic())
                && bytes.iter().all(|b| !b.is_ascii_lowercase())))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PolicyPlan {
    pub channels_selected: Vec<LogicalChannelKind>,
    pub selection_reasons: Vec<String>,
    pub query_features: QueryFeatures,
    pub intent: QueryIntent,
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

fn channel_limit(
    base: usize,
    result_limit: usize,
    eligible: usize,
    intent: QueryIntent,
    kind: LogicalChannelKind,
) -> usize {
    let favored = matches!(
        (intent, kind),
        (
            QueryIntent::Lexical,
            LogicalChannelKind::Bm25 | LogicalChannelKind::Sparse
        ) | (
            QueryIntent::Semantic,
            LogicalChannelKind::Dense | LogicalChannelKind::Multivector
        )
    );
    let budget = if favored {
        base.saturating_mul(2)
    } else {
        base
    };
    budget.max(result_limit).min(eligible.max(1)).min(100_000)
}

fn ef_search(quality: QualityPreference) -> usize {
    match quality {
        QualityPreference::Fast => 64,
        QualityPreference::Balanced => 256,
        QualityPreference::High => 1024,
    }
}

fn channel_ef(base: usize, intent: QueryIntent) -> usize {
    if intent == QueryIntent::Semantic {
        base.saturating_mul(2).min(65_536)
    } else {
        base
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
    let base_limit = candidate_limit(result_limit, quality);
    let base_ef = ef_search(quality);
    let features = QueryFeatures::from_query(query);
    let text = query.text.clone().unwrap_or_default();
    let intent = features.intent(!text.trim().is_empty());
    let vector_ef = channel_ef(base_ef, intent);
    let eligible = stats
        .filter_stats
        .as_ref()
        .map_or(stats.documents, |filter| {
            (filter.selectivity * stats.documents as f32).round() as usize
        });
    let mut channels: Vec<Channel> = Vec::new();
    let mut selected_kinds: Vec<LogicalChannelKind> = Vec::new();
    let mut reasons = vec![format!("query intent: {intent:?}").to_lowercase()];

    let bm25_coverage = stats.text_documents as f32 / stats.documents.max(1) as f32;
    if !text.trim().is_empty() && bm25_coverage >= COVERAGE_THRESHOLD {
        let limit = channel_limit(
            base_limit,
            result_limit,
            eligible,
            intent,
            LogicalChannelKind::Bm25,
        );
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
                let limit = channel_limit(
                    base_limit,
                    result_limit,
                    eligible,
                    intent,
                    LogicalChannelKind::Dense,
                );
                channels.push(Channel::Dense {
                    field: field.clone(),
                    vector: vec.clone(),
                    limit,
                    backend: "auto".into(),
                    ef_search: vector_ef,
                });
                selected_kinds.push(LogicalChannelKind::Dense);
                reasons.push(format!("Dense[{field}]: coverage {:.0}%", coverage * 100.));
            }
        }
    }

    for (field, vec) in &query.sparse {
        if schema.contains_key(field) {
            let limit = channel_limit(
                base_limit,
                result_limit,
                eligible,
                intent,
                LogicalChannelKind::Sparse,
            );
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
                let limit = channel_limit(
                    base_limit,
                    result_limit,
                    eligible,
                    intent,
                    LogicalChannelKind::Multivector,
                );
                channels.push(Channel::Multivector {
                    field: Some(field.clone()),
                    vectors: vecs.clone(),
                    limit,
                    backend: "auto".into(),
                    ef_search: vector_ef,
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
            let limit = channel_limit(
                base_limit,
                result_limit,
                eligible,
                intent,
                LogicalChannelKind::Multivector,
            );
            channels.push(Channel::Multivector {
                field: None,
                vectors: vecs.clone(),
                limit,
                backend: "auto".into(),
                ef_search: vector_ef,
            });
            selected_kinds.push(LogicalChannelKind::Multivector);
            reasons.push(format!(
                "FDE: late-interaction coverage {:.0}%",
                fde_coverage * 100.
            ));
        }
    }

    if channels.is_empty() && !text.trim().is_empty() {
        let limit = channel_limit(
            base_limit,
            result_limit,
            eligible,
            intent,
            LogicalChannelKind::Bm25,
        );
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
        intent,
        generated_prefetch: channels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(selectivity: Option<f32>) -> PlannerStats {
        PlannerStats {
            generation: 1,
            documents: 1_000,
            text_documents: 1_000,
            token_documents: 1_000,
            token_vectors: 16_000,
            fde_dimension: 128,
            fde_graph_ready: true,
            fields: BTreeMap::from([(
                "semantic".into(),
                FieldStats {
                    kind: RepresentationKind::Dense,
                    dimension: Some(2),
                    documents: 1_000,
                    graph_ready: true,
                },
            )]),
            filter_stats: selectivity.map(|selectivity| FilterStats {
                selectivity,
                filter_operator: FilterStrategy::MetadataIndex,
            }),
        }
    }

    fn schema() -> BTreeMap<String, FieldSchema> {
        BTreeMap::from([("semantic".into(), FieldSchema::Dense { dimension: 2 })])
    }

    fn limits(plan: &PolicyPlan) -> (usize, usize, usize) {
        let Channel::Bm25 { limit: lexical, .. } = &plan.generated_prefetch[0] else {
            panic!("expected BM25 first");
        };
        let Channel::Dense {
            limit: dense,
            ef_search,
            ..
        } = &plan.generated_prefetch[1]
        else {
            panic!("expected dense second");
        };
        (*lexical, *dense, *ef_search)
    }

    #[test]
    fn intent_changes_channel_and_ann_budgets() {
        let lexical = QueryRepresentations {
            text: Some("HTTP 404 in parseUserID".into()),
            dense: BTreeMap::from([("semantic".into(), vec![0.0, 1.0])]),
            ..QueryRepresentations::default()
        };
        let semantic = QueryRepresentations {
            text: Some("why does recovery preserve committed writes after a sudden crash".into()),
            dense: lexical.dense.clone(),
            ..QueryRepresentations::default()
        };

        let lexical = generate_policy_prefetch(
            &lexical,
            &stats(None),
            10,
            QualityPreference::Balanced,
            &schema(),
        );
        let semantic = generate_policy_prefetch(
            &semantic,
            &stats(None),
            10,
            QualityPreference::Balanced,
            &schema(),
        );

        assert_eq!(lexical.intent, QueryIntent::Lexical);
        assert_eq!(limits(&lexical), (100, 50, 256));
        assert_eq!(semantic.intent, QueryIntent::Semantic);
        assert_eq!(limits(&semantic), (50, 100, 512));
    }

    #[test]
    fn selective_filter_caps_generated_candidate_budgets() {
        let query = QueryRepresentations {
            text: Some("HTTP 404".into()),
            dense: BTreeMap::from([("semantic".into(), vec![0.0, 1.0])]),
            ..QueryRepresentations::default()
        };
        let plan = generate_policy_prefetch(
            &query,
            &stats(Some(0.012)),
            10,
            QualityPreference::High,
            &schema(),
        );

        assert_eq!(limits(&plan), (12, 12, 1_024));
    }
}
