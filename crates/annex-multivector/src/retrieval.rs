//! Durable document representations and one retrieval plan over a generation.
use super::*;
use annex::vector::sparse::{SparseIndex, SparseVector};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Representation {
    Dense { vector: Vector },
    Multivector { vectors: Vec<Vector> },
    Sparse { vector: SparseVector },
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Chunk {
    pub parent: String,
    pub position: u32,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalDocument {
    pub id: String,
    #[serde(default)]
    pub vectors: Vec<Vector>,
    #[serde(default)]
    pub metadata: Value,
    pub text: Option<String>,
    #[serde(default)]
    pub representations: BTreeMap<String, Representation>,
    pub chunk: Option<Chunk>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) enum StoredRepresentation {
    Dense {
        location: ObjectLocation,
        dimension: usize,
    },
    Multivector {
        location: ObjectLocation,
        dimension: usize,
        tokens: usize,
    },
    Sparse(SparseVector),
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(super) struct Fields {
    text: Option<String>,
    representations: BTreeMap<String, StoredRepresentation>,
    chunk: Option<Chunk>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum FieldSchema {
    Dense { dimension: usize },
    Multivector { dimension: usize },
    Sparse,
}

#[derive(Clone, Default)]
pub(super) struct RetrievalState {
    next_id: u64,
    by_id: HashMap<String, u64>,
    ids: HashMap<u64, String>,
    vocabulary: HashMap<String, u32>,
    lexical: SparseIndex,
    sparse: HashMap<String, SparseIndex>,
    schema: BTreeMap<String, FieldSchema>,
    document_chunks: HashMap<u64, Chunk>,
    chunks: HashMap<String, BTreeMap<u32, BTreeSet<String>>>,
    metadata_eq: HashMap<MetadataKey, HashSet<u64>>,
    metadata_membership: HashMap<MetadataKey, HashSet<u64>>,
    metadata_keys: HashMap<u64, (Vec<MetadataKey>, Vec<MetadataKey>)>,
    analyzer: Analyzer,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct MetadataKey {
    pointer: String,
    value: String,
}

/// Candidate document numbers produced by the metadata index. Sparse matches
/// stay in a sorted vector; dense matches use one bit per allocated document
/// number. This avoids a `HashSet` allocation per boolean-filter operation and
/// keeps membership checks cheap for lexical, sparse, exact, and ANN overlays.
#[derive(Clone, Debug)]
pub(super) enum DocSet {
    Sparse(Vec<u64>),
    Bitmap { words: Vec<u64>, len: usize },
}

impl DocSet {
    fn from_numbers<I>(next_id: u64, numbers: I) -> Self
    where
        I: IntoIterator<Item = u64>,
    {
        let mut ids: Vec<_> = numbers.into_iter().collect();
        let word_count = usize::try_from(next_id.saturating_add(63) / 64).unwrap_or(usize::MAX);
        // A sparse u64 vector costs eight bytes per match. Prefer a bitmap once
        // it is no larger, while avoiding a large bitmap after heavy churn.
        if word_count > 0 && word_count <= ids.len() {
            let mut words = vec![0; word_count];
            let mut len = 0;
            for &id in &ids {
                let word = (id / 64) as usize;
                if word < words.len() {
                    let mask = 1_u64 << (id % 64);
                    if words[word] & mask == 0 {
                        words[word] |= mask;
                        len += 1;
                    }
                }
            }
            Self::Bitmap { words, len }
        } else {
            ids.sort_unstable();
            ids.dedup();
            Self::Sparse(ids)
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Sparse(ids) => ids.len(),
            Self::Bitmap { len, .. } => *len,
        }
    }

    pub(super) fn contains(&self, id: u64) -> bool {
        match self {
            Self::Sparse(ids) => ids.binary_search(&id).is_ok(),
            Self::Bitmap { words, .. } => words
                .get((id / 64) as usize)
                .is_some_and(|word| word & (1_u64 << (id % 64)) != 0),
        }
    }

    pub(super) fn iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match self {
            Self::Sparse(ids) => Box::new(ids.iter().copied()),
            Self::Bitmap { words, .. } => {
                Box::new(words.iter().enumerate().flat_map(|(word_index, &word)| {
                    (0..64).filter_map(move |bit| {
                        (word & (1_u64 << bit) != 0).then_some((word_index as u64) * 64 + bit)
                    })
                }))
            }
        }
    }

    fn intersection(self, other: &Self, next_id: u64) -> Self {
        Self::from_numbers(next_id, self.iter().filter(|id| other.contains(*id)))
    }

    fn union(self, other: Self, next_id: u64) -> Self {
        Self::from_numbers(next_id, self.iter().chain(other.iter()))
    }
}

fn pointer_escape(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

fn predicate_pointer(field: &str) -> String {
    if field.starts_with('/') {
        field.to_owned()
    } else {
        format!("/{}", pointer_escape(field))
    }
}

fn scalar_key(pointer: &str, value: &Value) -> Option<MetadataKey> {
    (!value.is_array() && !value.is_object()).then(|| MetadataKey {
        pointer: pointer.to_owned(),
        value: serde_json::to_string(value).expect("scalar JSON serialization cannot fail"),
    })
}

fn collect_metadata_keys(
    value: &Value,
    pointer: &str,
    equality: &mut Vec<MetadataKey>,
    membership: &mut Vec<MetadataKey>,
) {
    match value {
        Value::Object(object) => {
            for (name, child) in object {
                let child_pointer = format!("{pointer}/{}", pointer_escape(name));
                collect_metadata_keys(child, &child_pointer, equality, membership);
            }
        }
        Value::Array(values) => {
            for child in values {
                if let Some(key) = scalar_key(pointer, child) {
                    membership.push(key);
                }
            }
            for (index, child) in values.iter().enumerate() {
                let child_pointer = format!("{pointer}/{index}");
                collect_metadata_keys(child, &child_pointer, equality, membership);
            }
        }
        _ => {
            if let Some(key) = scalar_key(pointer, value) {
                equality.push(key.clone());
                membership.push(key);
            }
        }
    }
}

fn ann_payload_value(value: &Value) -> Option<annex::PayloadValue> {
    match value {
        Value::Bool(value) => Some(annex::PayloadValue::Bool(*value)),
        Value::Number(value) => value
            .as_i64()
            .map(annex::PayloadValue::Int)
            .or_else(|| {
                value
                    .as_u64()
                    .and_then(|value| i64::try_from(value).ok())
                    .map(annex::PayloadValue::Int)
            })
            .or_else(|| {
                value
                    .as_f64()
                    .map(|value| annex::PayloadValue::Float(value.into()))
            }),
        Value::String(value) => Some(annex::PayloadValue::Str(value.clone())),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

fn collect_ann_payload(value: &Value, pointer: &str, payload: &mut annex::Payload) {
    match value {
        Value::Object(object) => {
            for (name, child) in object {
                let child_pointer = format!("{pointer}/{}", pointer_escape(name));
                collect_ann_payload(child, &child_pointer, payload);
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                let child_pointer = format!("{pointer}/{index}");
                collect_ann_payload(child, &child_pointer, payload);
            }
        }
        _ => {
            if let Some(value) = ann_payload_value(value) {
                payload.set(pointer, value);
            }
        }
    }
}

pub(super) fn metadata_ann_payload(metadata: &Value) -> annex::Payload {
    let mut payload = annex::Payload::default();
    collect_ann_payload(metadata, "", &mut payload);
    payload
}

fn invalid(message: impl Into<String>) -> IndexError {
    IndexError::Invalid(message.into())
}
fn sparse_error(error: annex::vector::sparse::SparseError) -> IndexError {
    invalid(error.to_string())
}

impl RetrievalState {
    pub(super) fn from_schema(
        schema: BTreeMap<String, FieldSchema>,
        analyzer: Analyzer,
    ) -> Result<Self, IndexError> {
        if schema.iter().any(|(name, shape)| {
            name.is_empty()
                || name.len() > 128
                || matches!(
                    shape,
                    FieldSchema::Dense { dimension }
                        | FieldSchema::Multivector { dimension }
                        if !(1..=65_536).contains(dimension)
                )
        }) {
            return Err(invalid("invalid persisted representation schema"));
        }
        Ok(Self {
            schema,
            analyzer,
            ..Self::default()
        })
    }

    pub(super) fn schema(&self) -> &BTreeMap<String, FieldSchema> {
        &self.schema
    }

    pub(super) fn has_sparse_field(&self, field: &str) -> bool {
        self.sparse.contains_key(field)
    }

    pub(super) fn dense_dimension(&self, field: &str) -> Result<usize, IndexError> {
        match self.schema.get(field) {
            Some(FieldSchema::Dense { dimension }) => Ok(*dimension),
            _ => Err(invalid("unknown dense field")),
        }
    }
    pub(super) fn remove(&mut self, id: &str) {
        if let Some(number) = self.by_id.remove(id) {
            if let Some(chunk) = self.document_chunks.remove(&number)
                && let Some(positions) = self.chunks.get_mut(&chunk.parent)
            {
                if let Some(ids) = positions.get_mut(&chunk.position) {
                    ids.remove(id);
                    if ids.is_empty() {
                        positions.remove(&chunk.position);
                    }
                }
                if positions.is_empty() {
                    self.chunks.remove(&chunk.parent);
                }
            }
            self.ids.remove(&number);
            self.lexical.delete(number);
            for index in self.sparse.values() {
                index.delete(number);
            }
            if let Some((equality, membership)) = self.metadata_keys.remove(&number) {
                for key in equality {
                    if let Some(ids) = self.metadata_eq.get_mut(&key) {
                        ids.remove(&number);
                        if ids.is_empty() {
                            self.metadata_eq.remove(&key);
                        }
                    }
                }
                for key in membership {
                    if let Some(ids) = self.metadata_membership.get_mut(&key) {
                        ids.remove(&number);
                        if ids.is_empty() {
                            self.metadata_membership.remove(&key);
                        }
                    }
                }
            }
        }
    }
    pub(super) fn insert(
        &mut self,
        id: &str,
        fields: &Fields,
        metadata: &Value,
    ) -> Result<(), IndexError> {
        self.remove(id);
        let number = self.next_id;
        self.next_id = number
            .checked_add(1)
            .ok_or_else(|| invalid("document IDs exhausted"))?;
        self.by_id.insert(id.to_owned(), number);
        self.ids.insert(number, id.to_owned());
        let mut equality = Vec::new();
        let mut membership = Vec::new();
        collect_metadata_keys(metadata, "", &mut equality, &mut membership);
        equality.sort_by(|a, b| {
            a.pointer
                .cmp(&b.pointer)
                .then_with(|| a.value.cmp(&b.value))
        });
        equality.dedup();
        membership.sort_by(|a, b| {
            a.pointer
                .cmp(&b.pointer)
                .then_with(|| a.value.cmp(&b.value))
        });
        membership.dedup();
        for key in &equality {
            self.metadata_eq
                .entry(key.clone())
                .or_default()
                .insert(number);
        }
        for key in &membership {
            self.metadata_membership
                .entry(key.clone())
                .or_default()
                .insert(number);
        }
        self.metadata_keys.insert(number, (equality, membership));
        if let Some(chunk) = &fields.chunk {
            self.chunks
                .entry(chunk.parent.clone())
                .or_default()
                .entry(chunk.position)
                .or_default()
                .insert(id.to_owned());
            self.document_chunks.insert(number, chunk.clone());
        }
        if let Some(text) = &fields.text {
            let mut pairs = Vec::new();
            for (term, count) in self.analyzer.analyze(text) {
                let next = u32::try_from(self.vocabulary.len())
                    .map_err(|_| invalid("lexical vocabulary exhausted"))?;
                pairs.push((*self.vocabulary.entry(term).or_insert(next), count));
            }
            self.lexical
                .upsert(number, &SparseVector::from_pairs(pairs))
                .map_err(sparse_error)?;
        }
        for (name, representation) in &fields.representations {
            let shape = match representation {
                StoredRepresentation::Dense { dimension, .. } => FieldSchema::Dense {
                    dimension: *dimension,
                },
                StoredRepresentation::Multivector { dimension, .. } => FieldSchema::Multivector {
                    dimension: *dimension,
                },
                StoredRepresentation::Sparse(vector) => {
                    self.sparse
                        .entry(name.clone())
                        .or_default()
                        .upsert(number, vector)
                        .map_err(sparse_error)?;
                    FieldSchema::Sparse
                }
            };
            if self
                .schema
                .get(name)
                .is_some_and(|expected| *expected != shape)
            {
                return Err(invalid(format!(
                    "representation {name:?} has a different kind or dimension"
                )));
            }
            self.schema.insert(name.clone(), shape);
        }
        Ok(())
    }

    fn indexed_filter_numbers(&self, filter: &Predicate) -> Option<DocSet> {
        match filter {
            Predicate::Eq { field, value } => {
                let key = scalar_key(&predicate_pointer(field), value)?;
                Some(DocSet::from_numbers(
                    self.next_id,
                    self.metadata_eq
                        .get(&key)
                        .into_iter()
                        .flat_map(|ids| ids.iter().copied()),
                ))
            }
            Predicate::In { field, values } => {
                let pointer = predicate_pointer(field);
                let mut ids = Vec::new();
                for value in values {
                    let key = scalar_key(&pointer, value)?;
                    if let Some(matches) = self.metadata_membership.get(&key) {
                        ids.extend(matches.iter().copied());
                    }
                }
                Some(DocSet::from_numbers(self.next_id, ids))
            }
            Predicate::And { filters } => {
                let mut children = filters.iter();
                let mut ids = self.indexed_filter_numbers(children.next()?)?;
                for child in children {
                    let matches = self.indexed_filter_numbers(child)?;
                    ids = ids.intersection(&matches, self.next_id);
                }
                Some(ids)
            }
            Predicate::Or { filters } => {
                let mut ids = DocSet::Sparse(Vec::new());
                for child in filters {
                    ids = ids.union(self.indexed_filter_numbers(child)?, self.next_id);
                }
                Some(ids)
            }
            Predicate::Not { filter } => {
                let excluded = self.indexed_filter_numbers(filter)?;
                Some(DocSet::from_numbers(
                    self.next_id,
                    self.ids
                        .keys()
                        .filter(|id| !excluded.contains(**id))
                        .copied(),
                ))
            }
            Predicate::Range { .. } => None,
        }
    }

    pub(super) fn indexed_filter_set(&self, filter: &Predicate) -> Option<DocSet> {
        self.indexed_filter_numbers(filter)
    }

    pub(super) fn number(&self, id: &str) -> Option<u64> {
        self.by_id.get(id).copied()
    }

    pub(super) fn external_id(&self, number: u64) -> Option<&str> {
        self.ids.get(&number).map(String::as_str)
    }

    pub(super) fn contains_external(&self, set: &DocSet, id: &str) -> bool {
        self.number(id).is_some_and(|number| set.contains(number))
    }

    pub(super) fn indexed_filter_count(&self, filter: &Predicate) -> Option<usize> {
        if let Predicate::Eq { field, value } = filter {
            let key = scalar_key(&predicate_pointer(field), value)?;
            return Some(self.metadata_eq.get(&key).map_or(0, HashSet::len));
        }
        self.indexed_filter_numbers(filter).map(|ids| ids.len())
    }

    #[cfg(test)]
    pub(super) fn indexed_filter_ids<'a>(&'a self, filter: &Predicate) -> Option<HashSet<&'a str>> {
        Some(
            self.indexed_filter_numbers(filter)?
                .iter()
                .filter_map(|number| self.ids.get(&number).map(String::as_str))
                .collect(),
        )
    }

    fn neighbors<'a>(&'a self, chunk: &Chunk, radius: u32) -> impl Iterator<Item = &'a str> {
        let start = chunk.position.saturating_sub(radius);
        let end = chunk.position.saturating_add(radius);
        self.chunks
            .get(&chunk.parent)
            .into_iter()
            .flat_map(move |positions| positions.range(start..=end))
            .flat_map(|(_, ids)| ids.iter().map(String::as_str))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Predicate {
    Eq {
        field: String,
        value: Value,
    },
    In {
        field: String,
        values: Vec<Value>,
    },
    Range {
        field: String,
        gte: Option<f64>,
        lte: Option<f64>,
    },
    And {
        filters: Vec<Predicate>,
    },
    Or {
        filters: Vec<Predicate>,
    },
    Not {
        filter: Box<Predicate>,
    },
}
impl Predicate {
    pub(super) fn validate(&self, depth: usize) -> Result<(), IndexError> {
        if depth > 16 {
            return Err(invalid("filter nesting exceeds 16"));
        }
        match self {
            Self::Eq { field, value } => {
                if field.is_empty() || value.is_object() || value.is_array() {
                    return Err(invalid("eq requires a field and scalar value"));
                }
            }
            Self::In { field, values } => {
                if field.is_empty()
                    || values.is_empty()
                    || values.len() > 1024
                    || values.iter().any(|v| v.is_object() || v.is_array())
                {
                    return Err(invalid("in requires 1..=1024 scalar values"));
                }
            }
            Self::Range { field, gte, lte } => {
                if field.is_empty()
                    || (gte.is_none() && lte.is_none())
                    || gte.iter().chain(lte).any(|v| !v.is_finite())
                    || matches!((gte,lte), (Some(a),Some(b)) if a>b)
                {
                    return Err(invalid("invalid numeric range"));
                }
            }
            Self::And { filters } | Self::Or { filters } => {
                if filters.is_empty() || filters.len() > 64 {
                    return Err(invalid("boolean filters require 1..=64 children"));
                }
                for f in filters {
                    f.validate(depth + 1)?;
                }
            }
            Self::Not { filter } => filter.validate(depth + 1)?,
        }
        Ok(())
    }
    pub(super) fn matches(&self, metadata: &Value) -> bool {
        let value = |field: &str| {
            if field.starts_with('/') {
                metadata.pointer(field)
            } else {
                metadata.get(field)
            }
        };
        match self {
            Self::Eq {
                field,
                value: expected,
            } => value(field).is_some_and(|v| v == expected),
            Self::In { field, values } => value(field).is_some_and(|v| match v {
                Value::Array(a) => a.iter().any(|x| values.contains(x)),
                _ => values.contains(v),
            }),
            Self::Range { field, gte, lte } => value(field)
                .and_then(Value::as_f64)
                .is_some_and(|v| gte.is_none_or(|lo| v >= lo) && lte.is_none_or(|hi| v <= hi)),
            Self::And { filters } => filters.iter().all(|f| f.matches(metadata)),
            Self::Or { filters } => filters.iter().any(|f| f.matches(metadata)),
            Self::Not { filter } => !filter.matches(metadata),
        }
    }

    /// Translate the predicate only when the ANN payload evaluator has exactly
    /// the same semantics. Unsupported shapes retain exact filtered execution.
    pub(super) fn ann_filter(&self) -> Option<annex::Filter> {
        match self {
            Self::Eq { field, value } => Some(annex::Filter::Match {
                key: predicate_pointer(field),
                value: ann_payload_value(value)?,
            }),
            Self::And { filters } => Some(annex::Filter::And(
                filters
                    .iter()
                    .map(Self::ann_filter)
                    .collect::<Option<Vec<_>>>()?,
            )),
            Self::Or { filters } => Some(annex::Filter::Or(
                filters
                    .iter()
                    .map(Self::ann_filter)
                    .collect::<Option<Vec<_>>>()?,
            )),
            Self::In { .. } | Self::Range { .. } | Self::Not { .. } => None,
        }
    }
}
fn default_limit() -> usize {
    100
}
fn default_k1() -> f32 {
    1.2
}
fn default_b() -> f32 {
    0.75
}
fn default_ef() -> usize {
    256
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Channel {
    Bm25 {
        text: String,
        #[serde(default = "default_limit")]
        limit: usize,
        #[serde(default = "default_k1")]
        k1: f32,
        #[serde(default = "default_b")]
        b: f32,
    },
    Sparse {
        field: String,
        vector: SparseVector,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    Dense {
        field: String,
        vector: Vector,
        #[serde(default = "default_limit")]
        limit: usize,
        #[serde(default = "default_backend")]
        backend: String,
        #[serde(default = "default_ef")]
        ef_search: usize,
    },
    Multivector {
        field: Option<String>,
        vectors: Vec<Vector>,
        #[serde(default = "default_limit")]
        limit: usize,
        #[serde(default = "default_backend")]
        backend: String,
        #[serde(default = "default_ef")]
        ef_search: usize,
    },
}
fn default_backend() -> String {
    "auto".into()
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Fusion {
    Rrf {
        #[serde(default = "default_rrf")]
        k: f32,
    },
    Weighted {
        weights: Vec<f32>,
    },
}
fn default_rrf() -> f32 {
    10.
}
impl Default for Fusion {
    fn default() -> Self {
        Self::Rrf { k: 10. }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Rerank {
    pub vectors: Vec<Vector>,
    pub field: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub adaptive: Option<AdaptiveRerank>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveRerank {
    pub min_candidates: usize,
    pub agreement_threshold: f32,
}
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextOptions {
    pub per_parent: Option<usize>,
    #[serde(default)]
    pub neighbors: u32,
    #[serde(default)]
    pub deduplicate: bool,
    pub mmr: Option<f32>,
    pub diversity_field: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrieveRequest {
    #[serde(default)]
    pub prefetch: Vec<Channel>,
    #[serde(default)]
    pub planning_mode: PlanningMode,
    /// Query representations for auto-planning (required when planning_mode != Manual).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<QueryRepresentations>,
    #[serde(default)]
    pub objective: RetrievalObjective,
    #[serde(default)]
    pub fusion: Fusion,
    pub filter: Option<Predicate>,
    pub rerank: Option<Rerank>,
    #[serde(default = "ten")]
    pub limit: usize,
    #[serde(default)]
    pub context: ContextOptions,
}
fn ten() -> usize {
    10
}
#[derive(Clone, Debug, Serialize)]
pub struct ContextHit {
    pub id: String,
    pub score: f32,
    pub metadata: Value,
    pub text: Option<String>,
    pub chunk: Option<Chunk>,
    pub sources: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expanded_from: Option<String>,
    /// Approximate token count for context packing (word_count × 1.3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_tokens: Option<usize>,
}
#[derive(Clone, Debug, Serialize)]
pub struct RankingSignals {
    /// top-1 score minus top-2 score (0 if < 2 results).
    pub top1_margin: f32,
    /// top-1 score minus bottom-of-topk score.
    pub topk_score_spread: f32,
    /// Already in trace; duplicated here for convenience.
    pub channel_agreement: Option<f32>,
    /// Unique parent documents / total matches.
    pub source_diversity: f32,
    /// Candidates dropped by text-hash deduplication.
    pub dedup_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RetrievalTrace {
    pub plan: RetrievalPlan,
    pub generation: u64,
    pub eligible_documents: usize,
    pub channels: Vec<Value>,
    pub fused_candidates: usize,
    pub reranked_candidates: usize,
    pub channel_agreement: Option<f32>,
    pub elapsed_ms: f64,
    /// Wall time in ms for each stage, aligned with `plan.stages`.
    pub per_stage_actual_ms: Vec<f64>,
    pub signals: RankingSignals,
}
#[derive(Clone, Debug, Serialize)]
pub struct RetrievalResponse {
    pub matches: Vec<ContextHit>,
    pub trace: RetrievalTrace,
}

#[derive(Default)]
struct ContextSelection {
    matches: Vec<ContextHit>,
    ids: HashSet<String>,
    texts: HashSet<blake3::Hash>,
    parents: HashMap<String, usize>,
    tokens_consumed: usize,
    dedup_count: usize,
}

impl ContextSelection {
    fn accepts(
        &self,
        id: &str,
        fields: &Fields,
        text_key: Option<blake3::Hash>,
        options: &ContextOptions,
        token_budget: Option<usize>,
        candidate_tokens: usize,
    ) -> AcceptResult {
        if self.ids.contains(id) {
            return AcceptResult::AlreadyAdded;
        }
        if text_key.is_some_and(|key| self.texts.contains(&key)) {
            return AcceptResult::Deduplicated;
        }
        if fields.chunk.as_ref().is_some_and(|chunk| {
            options
                .per_parent
                .is_some_and(|limit| self.parents.get(&chunk.parent).copied().unwrap_or(0) >= limit)
        }) {
            return AcceptResult::GroupFull;
        }
        // Skip-and-continue token packing: don't stop on first non-fitting item.
        if let Some(budget) = token_budget
            && self.tokens_consumed.saturating_add(candidate_tokens) > budget
        {
            return AcceptResult::TokenBudgetExceeded;
        }
        AcceptResult::Ok
    }

    fn push(&mut self, hit: ContextHit, text_key: Option<blake3::Hash>) {
        self.ids.insert(hit.id.clone());
        if let Some(key) = text_key {
            self.texts.insert(key);
        }
        if let Some(chunk) = &hit.chunk {
            *self.parents.entry(chunk.parent.clone()).or_default() += 1;
        }
        if let Some(t) = hit.estimated_tokens {
            self.tokens_consumed += t;
        }
        self.matches.push(hit);
    }
}

#[derive(Debug, PartialEq)]
enum AcceptResult {
    Ok,
    AlreadyAdded,
    Deduplicated,
    GroupFull,
    /// Hit doesn't fit in token budget; skip it (caller continues to next).
    TokenBudgetExceeded,
}

impl Fields {
    pub(super) fn has_text(&self) -> bool {
        self.text.is_some()
    }

    pub(super) fn has_representation(&self, field: &str) -> bool {
        self.representations.contains_key(field)
    }

    pub(super) fn has_dense(&self, field: &str) -> bool {
        matches!(
            self.representations.get(field),
            Some(StoredRepresentation::Dense { .. })
        )
    }
    pub(super) fn relocate(
        &mut self,
        source: &[u8],
        destination: &FixedVectorStore,
    ) -> Result<(), IndexError> {
        for representation in self.representations.values_mut() {
            match representation {
                StoredRepresentation::Dense { location, .. }
                | StoredRepresentation::Multivector { location, .. } => {
                    *location = destination.copy_record(source, *location)?;
                }
                StoredRepresentation::Sparse(_) => (),
            }
        }
        Ok(())
    }
    pub(super) fn prepare(
        document: &RetrievalDocument,
        stores: &SegmentStores,
    ) -> Result<Self, IndexError> {
        if document.id.is_empty()
            || document.id.len() > 4096
            || document.representations.len() > 32
            || document.text.as_ref().is_some_and(|s| s.len() > 1_048_576)
        {
            return Err(invalid(
                "document ID, text or representation count exceeds limits",
            ));
        }
        if let Some(chunk) = &document.chunk
            && (chunk.parent.is_empty() || chunk.parent.len() > 4096)
        {
            return Err(invalid("invalid chunk parent"));
        }
        let mut fields = Self {
            text: document.text.clone(),
            chunk: document.chunk.clone(),
            ..Self::default()
        };
        for (name, value) in &document.representations {
            if name.is_empty() || name.len() > 128 {
                return Err(invalid("representation names require 1..=128 bytes"));
            }
            let stored = match value {
                Representation::Sparse { vector } => {
                    if vector.len() > 65_536 {
                        return Err(invalid("sparse representation exceeds 65536 features"));
                    }
                    StoredRepresentation::Sparse(vector.canonicalized().map_err(sparse_error)?)
                }
                Representation::Dense { vector } => {
                    validate_matrix(std::slice::from_ref(vector))?;
                    StoredRepresentation::Dense {
                        location: stores.fde.put(&normalize(vector))?,
                        dimension: vector.len(),
                    }
                }
                Representation::Multivector { vectors } => {
                    validate_matrix(vectors)?;
                    let flat: Vec<_> = vectors.iter().flat_map(|v| normalize(v)).collect();
                    StoredRepresentation::Multivector {
                        location: stores.fde.put(&flat)?,
                        dimension: vectors[0].len(),
                        tokens: vectors.len(),
                    }
                }
            };
            fields.representations.insert(name.clone(), stored);
        }
        Ok(fields)
    }
    pub(super) fn verify(&self, mapped: &[u8]) -> Result<(), IndexError> {
        for representation in self.representations.values() {
            match representation {
                StoredRepresentation::Dense {
                    location,
                    dimension,
                } => {
                    verify_record(mapped, *location, true)?;
                    let v = FixedVectorStore::get(mapped, *location, *dimension)?;
                    validate_matrix(&[v.to_vec()])?;
                }
                StoredRepresentation::Multivector {
                    location,
                    dimension,
                    tokens,
                } => {
                    verify_record(mapped, *location, true)?;
                    let size = dimension
                        .checked_mul(*tokens)
                        .ok_or_else(|| invalid("representation size overflow"))?;
                    let v = FixedVectorStore::get(mapped, *location, size)?;
                    if *dimension == 0 || *tokens == 0 || v.iter().any(|v| !v.is_finite()) {
                        return Err(invalid("invalid multivector record"));
                    }
                }
                StoredRepresentation::Sparse(v) => {
                    v.canonicalized().map_err(sparse_error)?;
                }
            }
        }
        Ok(())
    }
}
pub(super) fn validate_matrix(vectors: &[Vector]) -> Result<(), IndexError> {
    let dimension = vectors.first().map_or(0, Vec::len);
    if vectors.is_empty()
        || vectors.len() > 8192
        || dimension == 0
        || dimension > 65_536
        || vectors
            .len()
            .checked_mul(dimension)
            .is_none_or(|n| n > 16_777_216)
        || vectors
            .iter()
            .any(|v| v.len() != dimension || v.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid("invalid or excessive vector shape"));
    }
    Ok(())
}
fn top(mut scores: Vec<(String, f32)>, limit: usize) -> Vec<(String, f32)> {
    let order =
        |a: &(String, f32), b: &(String, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    if scores.len() > limit {
        scores.select_nth_unstable_by(limit, order);
        scores.truncate(limit);
    }
    scores.sort_unstable_by(order);
    scores
}

impl MultiVectorIndex {
    pub(super) fn dense_vector<'a>(
        &self,
        s: &'a State,
        id: &str,
        field: &str,
    ) -> Result<&'a [f32], IndexError> {
        match s.documents[id].fields.representations.get(field) {
            Some(StoredRepresentation::Dense {
                location,
                dimension,
            }) => Ok(FixedVectorStore::get(
                s.record_fde(&s.documents[id]),
                *location,
                *dimension,
            )?),
            _ => Err(invalid("missing dense vector")),
        }
    }
    pub fn retrieve(&self, request: &RetrieveRequest) -> Result<RetrievalResponse, IndexError> {
        let started = Instant::now();
        planner::validate_request(request)?;
        let s = self.snapshot();
        self.retrieve_with_snapshot(started, s, request)
    }

    fn retrieve_with_snapshot(
        &self,
        started: Instant,
        s: Arc<State>,
        request: &RetrieveRequest,
    ) -> Result<RetrievalResponse, IndexError> {
        // An unfiltered query means every live document is eligible. Keep that
        // state implicit: materializing a HashSet of every ID makes an HNSW
        // query O(collection size) before graph traversal even begins.
        let mut filter_strategy = FilterStrategy::None;
        let eligible: Option<DocSet> = request.filter.as_ref().map(|filter| {
            if let Some(indexed) = s.retrieval.indexed_filter_set(filter) {
                filter_strategy = FilterStrategy::MetadataIndex;
                indexed
            } else {
                filter_strategy = FilterStrategy::MetadataScan;
                DocSet::from_numbers(
                    s.retrieval.next_id,
                    s.documents
                        .iter()
                        .filter(|(_, document)| filter.matches(&document.metadata))
                        .filter_map(|(id, _)| s.retrieval.number(id)),
                )
            }
        });
        let eligible_documents = eligible.as_ref().map_or(s.documents.len(), DocSet::len);
        let planner_filter = request
            .filter
            .as_ref()
            .map(|_| (eligible_documents, filter_strategy));
        let (effective_request, policy_plan) = self.prepare_request(&s, request, planner_filter);
        let request = effective_request.as_ref();
        let ann_filter = request.filter.as_ref().and_then(Predicate::ann_filter);
        let mut plan = self.compile_plan(&s, request, eligible_documents, filter_strategy)?;
        plan.policy = policy_plan;
        let mut per_stage_actual_ms: Vec<f64> = Vec::with_capacity(plan.stages.len());

        // Channels are independent — dispatch them concurrently with Rayon.
        // NOTE: each channel may itself use par_iter internally; nested Rayon
        // work-stealing is safe but may create contention under high concurrency.
        // A per-query parallelism budget is tracked in a later phase.
        let parallel_stage_start = Instant::now();
        type ChannelResult = Result<(Vec<(String, f32)>, Value), IndexError>;
        let channel_results: Vec<ChannelResult> = request
            .prefetch
            .par_iter()
            .zip(plan.parallel_channels().par_iter())
            .map(|(channel, planned)| {
                let at = Instant::now();
                let limit = planned.limit;
                let allowed = |number: u64| {
                    eligible
                        .as_ref()
                        .is_none_or(|eligible| eligible.contains(number))
                };
                let external = |hits: Vec<(u64, f32)>| -> Vec<(String, f32)> {
                    hits.into_iter()
                        .map(|(id, score)| (s.retrieval.ids[&id].clone(), score))
                        .collect()
                };
                let tie_break = |a: u64, b: u64| s.retrieval.ids[&a].cmp(&s.retrieval.ids[&b]);
                let scores = match channel {
                    Channel::Bm25 { text, k1, b, .. } => {
                        let query = SparseVector::from_pairs(
                            s.retrieval.analyzer.analyze(text).into_iter().filter_map(
                                |(term, count)| {
                                    s.retrieval.vocabulary.get(&term).map(|&id| (id, count))
                                },
                            ),
                        );
                        external(
                            s.retrieval
                                .lexical
                                .search_bm25_filtered_by(&query, limit, *k1, *b, allowed, tie_break)
                                .map_err(sparse_error)?,
                        )
                    }
                    Channel::Sparse { field, vector, .. } => {
                        let index =
                            s.retrieval.sparse.get(field).ok_or_else(|| {
                                invalid(format!("unknown sparse field {field:?}"))
                            })?;
                        external(
                            index
                                .search_dot_filtered_by(vector, limit, allowed, tie_break)
                                .map_err(sparse_error)?,
                        )
                    }
                    Channel::Dense {
                        field,
                        vector,
                        ef_search,
                        ..
                    } => {
                        if planned.operator == PhysicalOperator::HnswDense {
                            let q = normalize(vector);
                            let ann = &s.named_ann[field];
                            let expected = limit.min(eligible_documents);
                            let mut scores = self.ann_scores_filtered(
                                &s,
                                ann,
                                &q,
                                limit,
                                *ef_search,
                                ann_filter.as_ref(),
                                eligible.as_ref(),
                            )?;
                            // Progressive widening: if short of expected, retry at
                            // 2× ef before falling through to exact.
                            if scores.len() < expected {
                                let wider = (ef_search * 2).min(4 * limit).max(*ef_search + 1);
                                scores = self.ann_scores_filtered(
                                    &s,
                                    ann,
                                    &q,
                                    limit,
                                    wider,
                                    ann_filter.as_ref(),
                                    eligible.as_ref(),
                                )?;
                            }
                            if scores.len() < expected {
                                self.named_scores(
                                    &s,
                                    field,
                                    std::slice::from_ref(vector),
                                    false,
                                    eligible.as_ref(),
                                    limit,
                                )?
                            } else {
                                scores
                            }
                        } else {
                            self.named_scores(
                                &s,
                                field,
                                std::slice::from_ref(vector),
                                false,
                                eligible.as_ref(),
                                limit,
                            )?
                        }
                    }
                    Channel::Multivector {
                        field: Some(field),
                        vectors,
                        ..
                    } => self.named_scores(&s, field, vectors, true, eligible.as_ref(), limit)?,
                    Channel::Multivector {
                        field: None,
                        vectors,
                        ef_search,
                        ..
                    } => {
                        let normalized: Vec<_> = vectors.iter().map(|v| normalize(v)).collect();
                        if planned.operator == PhysicalOperator::HnswFde {
                            let encoded = self.fde.encode_query(&normalized);
                            let expected = limit.min(eligible_documents);
                            let mut scores = self.ann_fde_scores_filtered(
                                &s,
                                &encoded,
                                limit,
                                *ef_search,
                                ann_filter.as_ref(),
                                eligible.as_ref(),
                            )?;
                            if scores.len() < expected {
                                let wider = (ef_search * 2).min(4 * limit).max(*ef_search + 1);
                                scores = self.ann_fde_scores_filtered(
                                    &s,
                                    &encoded,
                                    limit,
                                    wider,
                                    ann_filter.as_ref(),
                                    eligible.as_ref(),
                                )?;
                            }
                            if scores.len() < expected {
                                self.exact_fde_scores_filtered(
                                    &s,
                                    &normalized,
                                    Some(limit),
                                    eligible.as_ref(),
                                )?
                            } else {
                                scores
                            }
                        } else {
                            self.exact_fde_scores_filtered(
                                &s,
                                &normalized,
                                Some(limit),
                                eligible.as_ref(),
                            )?
                        }
                    }
                };
                let backend = planned.operator.as_str();
                let entry = serde_json::json!({
                    "backend": backend,
                    "candidates": scores.len(),
                    "elapsed_ms": at.elapsed().as_secs_f64() * 1000.
                });
                Ok((scores, entry))
            })
            .collect();
        let (mut lists, channels): (Vec<_>, Vec<_>) = channel_results
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        per_stage_actual_ms.push(parallel_stage_start.elapsed().as_secs_f64() * 1000.);

        let fusion_stage_start = Instant::now();
        let agreement = if lists.len() > 1 {
            let head: HashSet<_> = lists[0]
                .iter()
                .take(request.limit)
                .map(|h| h.0.as_str())
                .collect();
            Some(
                lists[1..]
                    .iter()
                    .map(|list| {
                        let other: HashSet<_> = list
                            .iter()
                            .take(request.limit)
                            .map(|h| h.0.as_str())
                            .collect();
                        let union = head.union(&other).count();
                        if union == 0 {
                            0.
                        } else {
                            head.intersection(&other).count() as f32 / union as f32
                        }
                    })
                    .fold(1., f32::min),
            )
        } else {
            None
        };
        let mut fused: HashMap<String, (f32, Vec<usize>)> = HashMap::new();
        for (channel, list) in lists.iter().enumerate() {
            for (rank, (id, score)) in list.iter().enumerate() {
                let contribution = match &request.fusion {
                    Fusion::Rrf { k } => 1. / (k + rank as f32 + 1.),
                    Fusion::Weighted { weights } => weights[channel] * score,
                };
                let entry = fused.entry(id.clone()).or_default();
                entry.0 += contribution;
                if !entry.0.is_finite() {
                    return Err(invalid("fusion score overflow"));
                }
                entry.1.push(channel);
            }
        }
        // One channel retains its score; fusion only combines independent lists.
        let mut ranked = if lists.len() == 1 && matches!(request.fusion, Fusion::Rrf { .. }) {
            lists.pop().unwrap()
        } else {
            top(
                fused
                    .iter()
                    .map(|(id, (score, _))| (id.clone(), *score))
                    .collect(),
                fused.len(),
            )
        };
        let fused_candidates = ranked.len();
        per_stage_actual_ms.push(fusion_stage_start.elapsed().as_secs_f64() * 1000.);

        let mut reranked = 0;
        if let Some(rerank) = &request.rerank {
            let rerank_stage_start = Instant::now();
            let budget = if let Some(policy) = &rerank.adaptive {
                if agreement.expect("planner requires multiple channels")
                    >= policy.agreement_threshold
                {
                    policy.min_candidates
                } else {
                    rerank.limit
                }
            } else {
                rerank.limit
            };
            ranked.truncate(budget);
            reranked = ranked.len();
            if let Some(field) = &rerank.field {
                let pool = DocSet::from_numbers(
                    s.retrieval.next_id,
                    ranked.iter().filter_map(|(id, _)| s.retrieval.number(id)),
                );
                ranked =
                    self.named_scores(&s, field, &rerank.vectors, true, Some(&pool), rerank.limit)?;
                if ranked.len() != reranked {
                    return Err(invalid("rerank field missing from a candidate"));
                }
            } else {
                self.validate(&rerank.vectors)?;
                if ranked.iter().any(|(id, _)| s.documents[id].tokens == 0) {
                    return Err(invalid(
                        "default multivector missing from a rerank candidate",
                    ));
                }
                let normalized: Vec<_> = rerank.vectors.iter().map(|v| normalize(v)).collect();
                ranked = self
                    .rescore(&s, &normalized, ranked, rerank.limit, Some(rerank.limit))?
                    .into_iter()
                    .map(|h| (h.id, h.score))
                    .collect();
            }
            per_stage_actual_ms.push(rerank_stage_start.elapsed().as_secs_f64() * 1000.);
        }

        let context_stage_start = Instant::now();
        let (matches, signals) =
            self.context(&s, ranked, &fused, eligible.as_ref(), request, agreement)?;
        per_stage_actual_ms.push(context_stage_start.elapsed().as_secs_f64() * 1000.);

        Ok(RetrievalResponse {
            matches,
            trace: RetrievalTrace {
                plan,
                generation: s.generation,
                eligible_documents,
                channels,
                fused_candidates,
                reranked_candidates: reranked,
                channel_agreement: agreement,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
                per_stage_actual_ms,
                signals,
            },
        })
    }

    /// Execute multiple independent queries against a single immutable snapshot.
    ///
    /// All requests share one `Arc<State>` — one lock acquisition for the whole
    /// batch — and run concurrently via Rayon. Order of results matches order of
    /// inputs. All queries in the batch see the same generation of data, which is
    /// the correct semantic for a training step.
    ///
    /// A per-request error is returned in its slot without aborting the batch.
    pub fn retrieve_batch(
        &self,
        requests: &[RetrieveRequest],
    ) -> Vec<Result<RetrievalResponse, IndexError>> {
        use rayon::prelude::*;
        let s = self.snapshot();
        requests
            .par_iter()
            .map(|req| {
                let started = Instant::now();
                planner::validate_request(req)?;
                self.retrieve_with_snapshot(started, Arc::clone(&s), req)
            })
            .collect()
    }

    fn named_scores(
        &self,
        s: &State,
        field: &str,
        query: &[Vector],
        multivector: bool,
        eligible: Option<&DocSet>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        validate_matrix(query)?;
        let expected = if multivector {
            FieldSchema::Multivector {
                dimension: query[0].len(),
            }
        } else {
            FieldSchema::Dense {
                dimension: query[0].len(),
            }
        };
        if s.retrieval.schema.get(field) != Some(&expected) {
            return Err(invalid(format!(
                "unknown field or query dimension/kind mismatch: {field:?}"
            )));
        }
        let normalized: Vec<_> = query.iter().map(|v| normalize(v)).collect();
        let prepared = MaxSimQuery::new(&normalized, query[0].len());
        let score = |id: &str, d: &DocumentRecord| {
            d.fields.representations.get(field).map(|r| {
                let (location, dimension, count) = match r {
                    StoredRepresentation::Dense {
                        location,
                        dimension,
                    } => (*location, *dimension, 1),
                    StoredRepresentation::Multivector {
                        location,
                        dimension,
                        tokens,
                    } => (*location, *dimension, *tokens),
                    _ => unreachable!(),
                };
                let vector = FixedVectorStore::get(s.record_fde(d), location, dimension * count)?;
                let score = if multivector {
                    prepared.score(vector, dimension)
                } else {
                    dot(&normalized[0], vector)
                };
                Ok::<_, IndexError>((id.to_owned(), score))
            })
        };
        let scores = match eligible {
            Some(eligible) => eligible
                .iter()
                .filter_map(|number| s.retrieval.external_id(number))
                .collect::<Vec<_>>()
                .into_par_iter()
                .filter_map(|id| s.documents.get(id).and_then(|d| score(id, d)))
                .collect::<Result<Vec<_>, _>>()?,
            None => s
                .documents
                .par_iter()
                .filter_map(|(id, d)| score(id, d))
                .collect::<Result<Vec<_>, _>>()?,
        };
        Ok(top(scores, limit))
    }

    fn context(
        &self,
        s: &State,
        ranked: Vec<(String, f32)>,
        fused: &HashMap<String, (f32, Vec<usize>)>,
        eligible: Option<&DocSet>,
        request: &RetrieveRequest,
        channel_agreement: Option<f32>,
    ) -> Result<(Vec<ContextHit>, RankingSignals), IndexError> {
        let options = &request.context;
        let token_budget = request.objective.context_budget_tokens;
        let mmr_vectors = if options.mmr.is_some() {
            if ranked.len() > 4096 {
                return Err(invalid("MMR pool exceeds 4096 candidates"));
            }
            let field = options
                .diversity_field
                .as_ref()
                .ok_or_else(|| invalid("MMR requires a dense diversity_field"))?;
            s.retrieval.dense_dimension(field)?;
            Some(
                ranked
                    .iter()
                    .map(|(id, _)| self.dense_vector(s, id, field))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            if options.diversity_field.is_some() {
                return Err(invalid("diversity_field requires mmr"));
            }
            None
        };
        let text_key = |id: &str| {
            if options.deduplicate {
                s.documents[id]
                    .fields
                    .text
                    .as_ref()
                    .map(|text| blake3::hash(text.as_bytes()))
            } else {
                None
            }
        };
        let estimate_tokens = |id: &str| -> Option<usize> {
            token_budget?;
            let word_count = s.documents[id]
                .fields
                .text
                .as_ref()
                .map_or(0, |t| t.split_whitespace().count());
            Some((word_count * 13 / 10).max(1))
        };
        // Cache hashes only for MMR's repeatedly inspected, bounded pool. Plain
        // ranking hashes documents lazily, stopping as soon as context is full.
        let candidate_keys: Vec<_> = if mmr_vectors.is_some() {
            ranked.iter().map(|(id, _)| text_key(id)).collect()
        } else {
            Vec::new()
        };
        let min = ranked
            .iter()
            .map(|(_, score)| f64::from(*score))
            .fold(f64::INFINITY, f64::min);
        let max = ranked
            .iter()
            .map(|(_, score)| f64::from(*score))
            .fold(f64::NEG_INFINITY, f64::max);
        let relevance = |score: f32| {
            if max > min {
                ((f64::from(score) - min) / (max - min)) as f32
            } else {
                1.
            }
        };
        let mut selected = ContextSelection::default();
        let add = |selected: &mut ContextSelection,
                   id: &str,
                   score: f32,
                   expanded_from: Option<String>,
                   key: Option<blake3::Hash>|
         -> bool {
            let d = &s.documents[id];
            if eligible.is_some_and(|eligible| !s.retrieval.contains_external(eligible, id)) {
                return false;
            }
            let est_tokens = estimate_tokens(id).unwrap_or(0);
            match selected.accepts(id, &d.fields, key, options, token_budget, est_tokens) {
                AcceptResult::Ok => {}
                AcceptResult::Deduplicated => {
                    selected.dedup_count += 1;
                    return false;
                }
                // Token budget exceeded: skip this candidate but don't permanently exclude.
                AcceptResult::TokenBudgetExceeded => return false,
                AcceptResult::AlreadyAdded | AcceptResult::GroupFull => return false,
            }
            selected.push(
                ContextHit {
                    id: id.to_owned(),
                    score,
                    metadata: d.metadata.clone(),
                    text: d.fields.text.clone(),
                    chunk: d.fields.chunk.clone(),
                    sources: fused.get(id).map(|v| v.1.clone()).unwrap_or_default(),
                    expanded_from,
                    estimated_tokens: if token_budget.is_some() {
                        Some(est_tokens)
                    } else {
                        None
                    },
                },
                key,
            );
            true
        };
        let mut remaining = vec![true; ranked.len()];
        let mut redundancy = vec![f32::NEG_INFINITY; ranked.len()];
        let mut seeds = 0usize;
        let mut cursor = 0;
        while selected.matches.len() < request.limit {
            let next = if let Some(lambda) = options.mmr {
                let mut best: Option<(usize, f32)> = None;
                for (i, (id, score)) in ranked.iter().enumerate() {
                    if !remaining[i] {
                        continue;
                    }
                    let est_tokens = estimate_tokens(id).unwrap_or(0);
                    let result = if eligible.is_some_and(|eligible| {
                        !s.retrieval.contains_external(eligible, id.as_str())
                    }) {
                        AcceptResult::AlreadyAdded // permanent exclusion
                    } else {
                        selected.accepts(
                            id,
                            &s.documents[id].fields,
                            candidate_keys[i],
                            options,
                            token_budget,
                            est_tokens,
                        )
                    };
                    if matches!(
                        result,
                        AcceptResult::AlreadyAdded
                            | AcceptResult::GroupFull
                            | AcceptResult::Deduplicated
                    ) {
                        // Group and duplicate exclusions cannot become eligible
                        // later; excluded candidates must not affect diversity.
                        remaining[i] = false;
                        continue;
                    }
                    let value = if seeds == 0 {
                        relevance(*score)
                    } else {
                        lambda * relevance(*score) - (1. - lambda) * redundancy[i]
                    };
                    if best.is_none_or(|(_, previous)| value > previous) {
                        best = Some((i, value));
                    }
                }
                best.map(|(i, _)| i)
            } else if cursor < ranked.len() {
                let next = cursor;
                cursor += 1;
                Some(next)
            } else {
                None
            };
            let Some(next) = next else { break };
            remaining[next] = false;
            let (id, score) = &ranked[next];
            let key = if mmr_vectors.is_some() {
                candidate_keys[next]
            } else {
                text_key(id)
            };
            if !add(&mut selected, id, *score, None, key) {
                continue;
            }
            seeds += 1;
            if selected.matches.len() >= request.limit {
                break;
            }
            // MMR diversifies accepted retrieval seeds. Neighbor expansion adds
            // context around those seeds without requiring a dense neighbor field.
            if let Some(vectors) = &mmr_vectors {
                for i in 0..ranked.len() {
                    if remaining[i] {
                        redundancy[i] = redundancy[i].max(dot(vectors[i], vectors[next]));
                    }
                }
            }
            if options.neighbors > 0
                && let Some(chunk) = &s.documents[id].fields.chunk
            {
                for neighbor in s.retrieval.neighbors(chunk, options.neighbors) {
                    if eligible
                        .is_some_and(|eligible| !s.retrieval.contains_external(eligible, neighbor))
                        || selected.ids.contains(neighbor)
                    {
                        continue;
                    }
                    add(
                        &mut selected,
                        neighbor,
                        *score,
                        Some(id.clone()),
                        text_key(neighbor),
                    );
                    if selected.matches.len() >= request.limit {
                        break;
                    }
                }
            }
        }

        let top1_margin = match ranked.as_slice() {
            [a, b, ..] => (a.1 - b.1).max(0.0),
            _ => 0.0,
        };
        let topk_score_spread = ranked
            .first()
            .zip(
                ranked.get(
                    request
                        .limit
                        .saturating_sub(1)
                        .min(ranked.len().saturating_sub(1)),
                ),
            )
            .map_or(0.0, |(top, bot)| (top.1 - bot.1).max(0.0));
        let unique_sources = selected
            .matches
            .iter()
            .filter_map(|h| h.chunk.as_ref().map(|c| c.parent.as_str()))
            .collect::<HashSet<_>>()
            .len();
        let source_diversity = if selected.matches.is_empty() {
            1.0
        } else {
            unique_sources as f32 / selected.matches.len() as f32
        };
        let signals = RankingSignals {
            top1_margin,
            topk_score_spread,
            channel_agreement,
            source_diversity,
            dedup_count: selected.dedup_count,
        };
        Ok((selected.matches, signals))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::planner::FusionOperator, storage::FAIL_COMMIT};
    use serde_json::json;

    fn document(
        id: &str,
        text: &str,
        vector: Vector,
        tenant: &str,
        position: u32,
    ) -> RetrievalDocument {
        serde_json::from_value(json!({"id":id,"text":text,"metadata":{"tenant":tenant,"year":2026},"chunk":{"parent":tenant,"position":position},"representations":{"semantic":{"kind":"dense","vector":vector},"tokens":{"kind":"multivector","vectors":[vector]},"sparse":{"kind":"sparse","vector":{"indices":[u32::MAX],"values":[position as f32+1.]}}}})).unwrap()
    }

    fn schema_document(id: &str, representation: Representation) -> RetrievalDocument {
        RetrievalDocument {
            id: id.into(),
            representations: BTreeMap::from([("semantic".into(), representation)]),
            ..RetrievalDocument::default()
        }
    }

    fn assert_schema_mismatch(error: IndexError) {
        assert!(
            error.to_string().contains("different kind or dimension"),
            "unexpected error: {error}"
        );
    }
    fn request() -> RetrieveRequest {
        serde_json::from_value(json!({"prefetch":[{"kind":"dense","field":"semantic","vector":[1.,0.],"limit":3},{"kind":"bm25","text":"E123 repair","limit":3}],"limit":3,"filter":{"op":"eq","field":"tenant","value":"a"}})).unwrap()
    }
    fn index(path: &Path) -> MultiVectorIndex {
        let index = MultiVectorIndex::open(path, IndexConfig::new(2)).unwrap();
        index
            .upsert_records(vec![
                document("a", "E123 repair", vec![0.8, 0.6], "a", 0),
                document("b", "hardware guide", vec![1., 0.], "a", 1),
                document("c", "E123 E123 repair repair", vec![1., 0.], "denied", 0),
            ])
            .unwrap();
        index
    }

    #[test]
    fn planner_selects_available_operators_and_execution_reports_them() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let mut query = request();
        query.filter = None;
        query
            .prefetch
            .retain(|channel| matches!(channel, Channel::Dense { .. }));

        let exact = index.plan(&query).unwrap();
        assert_eq!(
            exact.parallel_channels()[0].operator,
            PhysicalOperator::ExactDense
        );
        assert_eq!(
            exact.parallel_channels()[0].reason,
            PlanReason::AnnUnavailable
        );

        index.build_dense_ann("semantic", 4, 16).unwrap();
        let ann = index.plan(&query).unwrap();
        // 3-doc corpus, dim=2: cost_exact=156 < cost_hnsw=2435 → exact chosen by cost model.
        assert_eq!(
            ann.parallel_channels()[0].operator,
            PhysicalOperator::ExactDense
        );
        assert_eq!(
            ann.parallel_channels()[0].reason,
            PlanReason::LowerEstimatedCost
        );
        let response = index.retrieve(&query).unwrap();
        assert_eq!(response.trace.plan, ann);
        assert_eq!(
            response.trace.channels[0]["backend"],
            ann.parallel_channels()[0].operator.as_str()
        );

        query.filter = Some(Predicate::Eq {
            field: "tenant".into(),
            value: json!("a"),
        });
        let filtered = index.plan(&query).unwrap();
        assert_eq!(
            filtered.parallel_channels()[0].operator,
            PhysicalOperator::ExactDense
        );
        assert_eq!(
            filtered.parallel_channels()[0].reason,
            PlanReason::LowerEstimatedCost
        );
        assert_eq!(filtered.eligible_documents, 2);
    }

    #[test]
    fn cached_planner_stats_track_replacement_deletion_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let mut query = request();
        query.filter = None;
        query
            .prefetch
            .retain(|channel| matches!(channel, Channel::Dense { .. }));

        let initial = index.plan(&query).unwrap().stats;
        assert_eq!(initial.documents, 3);
        assert_eq!(initial.text_documents, 3);
        assert_eq!(initial.fields["semantic"].documents, 3);
        assert_eq!(initial.fields["tokens"].documents, 3);
        assert_eq!(initial.fields["sparse"].documents, 3);

        index
            .upsert_records(vec![RetrievalDocument {
                id: "a".into(),
                metadata: json!({"tenant": "a"}),
                ..RetrievalDocument::default()
            }])
            .unwrap();
        index.delete("b").unwrap();

        let updated = index.plan(&query).unwrap().stats;
        assert_eq!(updated.documents, 2);
        assert_eq!(updated.text_documents, 1);
        assert_eq!(updated.fields["semantic"].documents, 1);
        assert_eq!(updated.fields["tokens"].documents, 1);
        assert_eq!(updated.fields["sparse"].documents, 1);

        drop(index);
        let reopened = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        assert_eq!(reopened.plan(&query).unwrap().stats, updated);
    }

    #[test]
    fn metadata_index_matches_predicate_evaluation_across_mutations() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let mut nested = document("nested", "indexed metadata", vec![0., 1.], "b", 0);
        nested.metadata = json!({
            "tenant": "b",
            "tags": ["rust", "search"],
            "owner": {"team": "retrieval"},
            "year": 2025
        });
        index.upsert_records(vec![nested]).unwrap();

        let predicates = vec![
            Predicate::Eq {
                field: "tenant".into(),
                value: json!("a"),
            },
            Predicate::In {
                field: "tags".into(),
                values: vec![json!("rust")],
            },
            Predicate::Eq {
                field: "/owner/team".into(),
                value: json!("retrieval"),
            },
            Predicate::And {
                filters: vec![
                    Predicate::Eq {
                        field: "tenant".into(),
                        value: json!("b"),
                    },
                    Predicate::In {
                        field: "tags".into(),
                        values: vec![json!("search")],
                    },
                ],
            },
            Predicate::Or {
                filters: vec![
                    Predicate::Eq {
                        field: "tenant".into(),
                        value: json!("denied"),
                    },
                    Predicate::Eq {
                        field: "tenant".into(),
                        value: json!("b"),
                    },
                ],
            },
            Predicate::Not {
                filter: Box::new(Predicate::Eq {
                    field: "tenant".into(),
                    value: json!("a"),
                }),
            },
        ];

        let assert_index_matches_scan = |index: &MultiVectorIndex| {
            let state = index.snapshot();
            for predicate in &predicates {
                let expected: HashSet<_> = state
                    .documents
                    .iter()
                    .filter(|(_, document)| predicate.matches(&document.metadata))
                    .map(|(id, _)| id.as_str())
                    .collect();
                assert_eq!(
                    state.retrieval.indexed_filter_ids(predicate).unwrap(),
                    expected
                );
            }
            assert!(
                state
                    .retrieval
                    .indexed_filter_ids(&Predicate::Range {
                        field: "year".into(),
                        gte: Some(2025.),
                        lte: None,
                    })
                    .is_none()
            );
        };

        assert_index_matches_scan(&index);
        let mut replacement = document("nested", "replacement", vec![1., 0.], "a", 0);
        replacement.metadata = json!({"tenant": "a", "tags": ["database"]});
        index.upsert_records(vec![replacement]).unwrap();
        index.delete("c").unwrap();
        assert_index_matches_scan(&index);

        drop(index);
        let reopened = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        assert_index_matches_scan(&reopened);
    }

    #[test]
    fn hybrid_filters_before_top_k_and_preserves_named_fields_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let mut query = request();
        for channel in &mut query.prefetch {
            match channel {
                Channel::Dense { limit, .. } | Channel::Bm25 { limit, .. } => *limit = 1,
                _ => (),
            }
        }
        let response = index.retrieve(&query).unwrap();
        assert_eq!(response.trace.eligible_documents, 2);
        assert_eq!(
            response
                .matches
                .iter()
                .map(|h| h.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert!(response.matches.iter().all(|h| h.metadata["tenant"] == "a"));
        let mut sparse = query.clone();
        sparse.prefetch = vec![Channel::Sparse {
            field: "sparse".into(),
            vector: SparseVector::from_pairs([(u32::MAX, 1.)]),
            limit: 10,
        }];
        assert_eq!(index.retrieve(&sparse).unwrap().matches[0].id, "b");
        query.rerank = Some(Rerank {
            field: Some("tokens".into()),
            vectors: vec![vec![1., 0.]],
            limit: 3,
            adaptive: None,
        });
        assert_eq!(index.retrieve(&query).unwrap().matches[0].id, "b");
        let before = serde_json::to_value(index.retrieve(&query).unwrap().matches).unwrap();
        drop(index);
        let reopened = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        assert_eq!(
            serde_json::to_value(reopened.retrieve(&query).unwrap().matches).unwrap(),
            before
        );
        assert!(reopened.delete("b").unwrap());
        assert_eq!(reopened.retrieve(&sparse).unwrap().matches[0].id, "a");
        reopened
            .upsert_records(vec![document("a", "different", vec![0., 1.], "a", 0)])
            .unwrap();
        query.prefetch.retain(|c| matches!(c, Channel::Bm25 { .. }));
        query.rerank = None;
        assert!(reopened.retrieve(&query).unwrap().matches.is_empty());
    }

    fn analyzer_config(analyzer: TextAnalyzer) -> IndexConfig {
        IndexConfig {
            analyzer,
            ..IndexConfig::new(2)
        }
    }
    fn bm25_only(text: &str) -> RetrieveRequest {
        serde_json::from_value(json!({
            "prefetch": [{"kind": "bm25", "text": text, "limit": 10}],
            "limit": 10
        }))
        .unwrap()
    }

    #[test]
    fn bm25_query_term_frequency_weights_repeated_terms() {
        let dir = tempfile::tempdir().unwrap();
        let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        index
            .upsert_records(vec![
                document("alpha", "alpha shared", vec![1., 0.], "t", 0),
                document("beta", "beta shared", vec![0., 1.], "t", 0),
            ])
            .unwrap();
        // Equal-length documents and equal df tie on a single occurrence of
        // each term; the stable tie-break puts "alpha" first.
        let tied = index.retrieve(&bm25_only("alpha beta")).unwrap();
        assert_eq!(tied.matches[0].id, "alpha");
        assert_eq!(tied.matches[0].score, tied.matches[1].score);
        // A repeated query term must double its contribution so "beta" wins.
        let repeated = index.retrieve(&bm25_only("alpha beta beta")).unwrap();
        assert_eq!(repeated.matches[0].id, "beta");
        let beta = repeated.matches.iter().find(|h| h.id == "beta").unwrap();
        let alpha = repeated.matches.iter().find(|h| h.id == "alpha").unwrap();
        assert_eq!(beta.score, tied.matches[0].score * 2.0);
        assert_eq!(alpha.score, tied.matches[0].score);
    }

    #[test]
    fn english_analyzer_stems_queries_and_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let index =
            MultiVectorIndex::open(dir.path(), analyzer_config(TextAnalyzer::english())).unwrap();
        index
            .upsert_records(vec![
                document("runner", "The runner runs fast", vec![1., 0.], "t", 0),
                document("walker", "walking quickly", vec![0., 1.], "t", 0),
            ])
            .unwrap();
        // "running" stems to "run", matching the indexed "runs"; "the" is a
        // stop word on both sides, so it cannot retrieve anything.
        let ranked = |index: &MultiVectorIndex, text: &str| {
            index
                .retrieve(&bm25_only(text))
                .unwrap()
                .matches
                .iter()
                .map(|h| h.id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(ranked(&index, "running"), ["runner"]);
        assert!(ranked(&index, "the").is_empty());
        drop(index);
        let reopened =
            MultiVectorIndex::open(dir.path(), analyzer_config(TextAnalyzer::english())).unwrap();
        assert_eq!(ranked(&reopened, "running"), ["runner"]);
        drop(reopened);
        // The analyzer is part of the persisted configuration contract.
        let Err(mismatch) = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)) else {
            panic!("reopening with a different analyzer must fail");
        };
        assert!(matches!(mismatch, IndexError::Config { .. }));
    }

    #[test]
    fn analyzer_config_is_validated_and_legacy_configs_default_to_plain() {
        let dir = tempfile::tempdir().unwrap();
        let Err(error) = MultiVectorIndex::open(
            dir.path(),
            analyzer_config(TextAnalyzer {
                max_token_length: Some(0),
                ..TextAnalyzer::plain()
            }),
        ) else {
            panic!("invalid analyzer config must be rejected");
        };
        assert!(error.to_string().contains("max_token_length"));
        let legacy: IndexConfig = serde_json::from_value(json!({
            "dimension": 2, "centroids": 2, "residual_bits": 2, "probes": 2,
            "fde_repetitions": 2, "fde_ksim": 2, "fde_projected": 2
        }))
        .unwrap();
        assert_eq!(legacy.analyzer, TextAnalyzer::plain());
        let custom: TextAnalyzer = serde_json::from_value(json!({
            "stem": true, "stopwords": "english",
            "ascii_folding": true, "max_token_length": 32
        }))
        .unwrap();
        assert!(custom.stem && custom.max_token_length == Some(32));
    }

    #[test]
    fn named_field_schema_survives_deleting_every_document_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        index
            .upsert_records(vec![schema_document(
                "dense",
                Representation::Dense {
                    vector: vec![1., 0.],
                },
            )])
            .unwrap();
        assert!(index.delete("dense").unwrap());

        assert_schema_mismatch(
            index
                .upsert_records(vec![schema_document(
                    "wrong-before-reopen",
                    Representation::Sparse {
                        vector: SparseVector::from_pairs([(1, 1.)]),
                    },
                )])
                .unwrap_err(),
        );
        drop(index);

        let reopened = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        assert_eq!(reopened.stats().documents, 0);
        assert_schema_mismatch(
            reopened
                .upsert_records(vec![schema_document(
                    "wrong-after-reopen",
                    Representation::Sparse {
                        vector: SparseVector::from_pairs([(1, 1.)]),
                    },
                )])
                .unwrap_err(),
        );
        reopened
            .upsert_records(vec![schema_document(
                "same-schema",
                Representation::Dense {
                    vector: vec![0., 1.],
                },
            )])
            .unwrap();
    }

    #[test]
    fn legacy_manifest_infers_and_persists_named_field_schema() {
        let dir = tempfile::tempdir().unwrap();
        let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        index
            .upsert_records(vec![schema_document(
                "legacy",
                Representation::Dense {
                    vector: vec![1., 0.],
                },
            )])
            .unwrap();
        drop(index);

        let path = dir.path().join("manifest.json");
        let mut envelope: ManifestEnvelope =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let mut payload: Value = serde_json::from_str(&envelope.manifest).unwrap();
        payload
            .as_object_mut()
            .unwrap()
            .remove("representation_schema");
        envelope.manifest = serde_json::to_string(&payload).unwrap();
        envelope.checksum_blake3 = blake3::hash(envelope.manifest.as_bytes())
            .to_hex()
            .to_string();
        fs::write(&path, serde_json::to_vec(&envelope).unwrap()).unwrap();

        let restored = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        assert!(restored.delete("legacy").unwrap());
        drop(restored);

        let envelope: ManifestEnvelope = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let manifest: Manifest = serde_json::from_str(&envelope.manifest).unwrap();
        assert_eq!(
            manifest.representation_schema.get("semantic"),
            Some(&FieldSchema::Dense { dimension: 2 })
        );

        let reopened = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        assert_schema_mismatch(
            reopened
                .upsert_records(vec![schema_document(
                    "wrong",
                    Representation::Multivector {
                        vectors: vec![vec![1., 0.]],
                    },
                )])
                .unwrap_err(),
        );
    }
    #[test]
    fn context_grouping_expansion_deduplication_and_mmr_obey_scope() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        index
            .upsert_records(vec![document(
                "duplicate",
                "E123 repair",
                vec![0.8, 0.6],
                "a",
                2,
            )])
            .unwrap();
        let mut query = request();
        query.prefetch.retain(|c| matches!(c, Channel::Bm25 { .. }));
        query.context = ContextOptions {
            neighbors: 1,
            deduplicate: true,
            ..Default::default()
        };
        let response = index.retrieve(&query).unwrap();
        assert_eq!(
            response
                .matches
                .iter()
                .map(|h| h.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(response.matches[1].expanded_from.as_deref(), Some("a"));
        query.context.per_parent = Some(1);
        assert_eq!(index.retrieve(&query).unwrap().matches.len(), 1);
        query = request();
        query.context.mmr = Some(0.2);
        query.context.diversity_field = Some("semantic".into());
        assert_eq!(index.retrieve(&query).unwrap().matches.len(), 3);
        query.filter = Some(Predicate::And {
            filters: vec![Predicate::Range {
                field: "year".into(),
                gte: Some(2027.),
                lte: None,
            }],
        });
        assert!(index.retrieve(&query).unwrap().matches.is_empty());
    }
    #[test]
    fn context_constraints_refill_mmr_and_skip_rejected_seed_neighbors() {
        let dir = tempfile::tempdir().unwrap();
        let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        index
            .upsert_records(vec![
                document("a", "same", vec![1., 0.], "p", 0),
                document("b", "same", vec![0.99, 0.01], "p", 1),
                document("c", "different", vec![0.8, 0.2], "q", 0),
            ])
            .unwrap();
        let mut query: RetrieveRequest = serde_json::from_value(json!({
            "prefetch":[{"kind":"dense","field":"semantic","vector":[1.,0.],"limit":3}],
            "limit":2,"context":{"mmr":1.,"diversity_field":"semantic","per_parent":1,"deduplicate":true}
        })).unwrap();
        let result = index.retrieve(&query).unwrap();
        assert_eq!(
            result
                .matches
                .iter()
                .map(|h| h.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "c"]
        );
        index
            .upsert_records(vec![
                document("b", "same", vec![1., 0.], "q", 0),
                document("c", "neighbor", vec![0., 1.], "q", 1),
            ])
            .unwrap();
        query = serde_json::from_value(json!({
            "prefetch":[{"kind":"bm25","text":"same","limit":3}],
            "limit":3,"context":{"neighbors":1,"deduplicate":true}
        }))
        .unwrap();
        // Rejected duplicate b must neither expand c nor retain its old p position.
        assert_eq!(
            index
                .retrieve(&query)
                .unwrap()
                .matches
                .iter()
                .map(|h| h.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a"]
        );
        query.prefetch = vec![Channel::Bm25 {
            text: "same".into(),
            limit: 1,
            k1: 1.2,
            b: 0.75,
        }];
        query.context = ContextOptions::default();
        assert_eq!(index.retrieve(&query).unwrap().matches[0].id, "a");
        drop(index);
        let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        assert_eq!(index.retrieve(&query).unwrap().matches[0].id, "a");
    }

    #[test]
    fn failed_hybrid_batches_and_compaction_keep_whole_generations() {
        for stage in ["fde_partial_write", "manifest_written", "manifest_renamed"] {
            let dir = tempfile::tempdir().unwrap();
            let index = index(dir.path());
            let before = index.stats().generation;
            FAIL_COMMIT.with(|f| f.set(Some((stage, 1))));
            let result = index.upsert_records(vec![
                document("a", "replaced", vec![0., 1.], "a", 0),
                document("b", "replaced", vec![0., 1.], "a", 1),
            ]);
            assert!(result.is_err(), "{stage}");
            let committed = stage == "manifest_renamed";
            assert_eq!(index.stats().generation, before + u64::from(committed));
            let mut q = request();
            q.prefetch.retain(|c| matches!(c, Channel::Bm25 { .. }));
            assert_eq!(index.retrieve(&q).unwrap().matches.is_empty(), committed);
            drop(index);
            let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
            assert_eq!(index.retrieve(&q).unwrap().matches.is_empty(), committed);
        }
        for stage in [
            "compact_partial_write",
            "compaction_copied",
            "manifest_written",
            "manifest_renamed",
            "directory_synced",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let index = index(dir.path());
            let before = index.stats().generation;
            let expected =
                serde_json::to_value(index.retrieve(&request()).unwrap().matches).unwrap();
            FAIL_COMMIT.with(|f| f.set(Some((stage, 1))));
            assert!(index.compact().is_err(), "{stage}");
            assert_eq!(
                index.stats().generation,
                before + u64::from(["manifest_renamed", "directory_synced"].contains(&stage))
            );
            assert_eq!(
                serde_json::to_value(index.retrieve(&request()).unwrap().matches).unwrap(),
                expected
            );
            drop(index);
            let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
            assert_eq!(
                serde_json::to_value(index.retrieve(&request()).unwrap().matches).unwrap(),
                expected
            );
        }
    }
    #[test]
    fn sealed_generations_and_named_ann_keep_overlay_and_adaptive_budgets() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        index.build_dense_ann("semantic", 4, 16).unwrap();
        index.seal().unwrap();
        index
            .upsert_records(vec![document("a", "E123 repair", vec![1., 0.], "a", 0)])
            .unwrap();
        let mut q = request();
        q.filter = None;
        q.prefetch.retain(|c| matches!(c, Channel::Dense { .. }));
        // 3-doc corpus: cost model prefers exact over HNSW (exact=12 < hnsw=3072 units).
        assert_eq!(
            index.retrieve(&q).unwrap().trace.channels[0]["backend"],
            "exact_dense"
        );
        assert!(index.delete("b").unwrap());
        assert!(
            index
                .retrieve(&q)
                .unwrap()
                .matches
                .iter()
                .all(|h| h.id != "b")
        );
        assert_eq!(index.stats().storage_segments, 2);
        index.compact().unwrap();
        assert_eq!(index.stats().storage_segments, 1);
        let before = serde_json::to_value(index.retrieve(&q).unwrap().matches).unwrap();
        drop(index);
        let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        assert_eq!(
            serde_json::to_value(index.retrieve(&q).unwrap().matches).unwrap(),
            before
        );
        assert_eq!(
            index.retrieve(&q).unwrap().trace.channels[0]["backend"],
            "exact_dense"
        );
        let mut q = request();
        q.limit = 1;
        q.rerank = Some(Rerank {
            field: Some("tokens".into()),
            vectors: vec![vec![1., 0.]],
            limit: 3,
            adaptive: Some(AdaptiveRerank {
                min_candidates: 1,
                agreement_threshold: 0.5,
            }),
        });
        let response = index.retrieve(&q).unwrap();
        assert_eq!(response.trace.reranked_candidates, 1);
        assert_eq!(response.trace.channel_agreement, Some(1.));
    }
    #[test]
    fn compaction_reclaims_overwrites_but_pins_active_readers_and_raw_scores() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        for _ in 0..4 {
            index
                .upsert_records(vec![document("a", "E123 repair", vec![0.8, 0.6], "a", 0)])
                .unwrap();
        }
        let snapshot = index.snapshot();
        let pool = DocSet::from_numbers(
            snapshot.retrieval.next_id,
            ["a", "b"]
                .into_iter()
                .filter_map(|id| snapshot.retrieval.number(id)),
        );
        let expected = index
            .named_scores(
                &snapshot,
                "semantic",
                &[vec![1., 0.]],
                false,
                Some(&pool),
                10,
            )
            .unwrap();
        let report = index.compact().unwrap();
        assert!(report["bytes_after"].as_u64().unwrap() < report["bytes_before"].as_u64().unwrap());
        assert!(dir.path().join("fde/fde.bin").exists());
        assert_eq!(
            index
                .named_scores(
                    &snapshot,
                    "semantic",
                    &[vec![1., 0.]],
                    false,
                    Some(&pool),
                    10
                )
                .unwrap(),
            expected
        );
        drop(snapshot);
        assert!(!dir.path().join("fde").exists());
        drop(index);
        let reopened = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        let reopened_snapshot = reopened.snapshot();
        let reopened_pool = DocSet::from_numbers(
            reopened_snapshot.retrieval.next_id,
            ["a", "b"]
                .into_iter()
                .filter_map(|id| reopened_snapshot.retrieval.number(id)),
        );
        assert_eq!(
            reopened
                .named_scores(
                    &reopened_snapshot,
                    "semantic",
                    &[vec![1., 0.]],
                    false,
                    Some(&reopened_pool),
                    10
                )
                .unwrap(),
            expected
        );
        reopened
            .upsert_records(vec![document("new", "E123", vec![1., 0.], "a", 2)])
            .unwrap();
        assert_eq!(reopened.stats().documents, 4);
    }

    #[test]
    fn auto_mode_generates_bm25_channel_from_text_representation() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());

        let query: RetrieveRequest = serde_json::from_value(json!({
            "prefetch": [],
            "planning_mode": "auto",
            "query": {"text": "E123 repair"},
            "limit": 3
        }))
        .unwrap();
        let response = index.retrieve(&query).unwrap();
        let policy = response
            .trace
            .plan
            .policy
            .as_ref()
            .expect("policy plan must be present in auto mode");
        assert!(
            policy.channels_selected.contains(&LogicalChannelKind::Bm25),
            "auto mode should select BM25 when text coverage is sufficient"
        );
        assert!(
            !response.matches.is_empty(),
            "auto mode must return results"
        );
    }

    #[test]
    fn auto_mode_generates_dense_channel_from_representation() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());

        let query: RetrieveRequest = serde_json::from_value(json!({
            "prefetch": [],
            "planning_mode": "auto",
            "query": {"dense": {"semantic": [1.0, 0.0]}},
            "limit": 3
        }))
        .unwrap();
        let plan = index.plan(&query).unwrap();
        assert!(
            !plan.parallel_channels().is_empty(),
            "auto mode with dense query must generate channels"
        );
    }

    #[test]
    fn auto_mode_uses_filter_cardinality_for_candidate_budgets() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let query: RetrieveRequest = serde_json::from_value(json!({
            "prefetch": [],
            "planning_mode": "auto",
            "query": {"text": "E123"},
            "objective": {"quality": "high"},
            "filter": {"op": "eq", "field": "tenant", "value": "a"},
            "limit": 1
        }))
        .unwrap();

        let plan = index.plan(&query).unwrap();
        assert_eq!(plan.eligible_documents, 2);
        assert_eq!(plan.parallel_channels()[0].limit, 2);
        assert_eq!(plan.policy.unwrap().intent, QueryIntent::Lexical);
    }

    #[test]
    fn unsupported_requests_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let query: RetrieveRequest = serde_json::from_value(json!({
            "prefetch": [],
            "limit": 3
        }))
        .unwrap();
        assert!(
            index.plan(&query).is_err(),
            "manual mode with empty prefetch must be invalid"
        );

        let mut query = request();
        query.objective.latency_budget_ms = Some(10.0);
        let error = index.plan(&query).unwrap_err();
        assert!(error.to_string().contains("requires calibrated planning"));
    }

    #[test]
    fn auto_with_overrides_uses_the_caller_channel() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let query: RetrieveRequest = serde_json::from_value(json!({
            "prefetch": [{"kind":"bm25","text":"exact phrase","limit":1}],
            "planning_mode": "auto_with_overrides",
            "query": {"text":"repair"},
            "limit": 1
        }))
        .unwrap();
        let plan = index.plan(&query).unwrap();
        assert_eq!(plan.parallel_channels()[0].limit, 1);
        assert_eq!(plan.policy.unwrap().generated_prefetch, query.prefetch);
    }

    #[test]
    fn token_budget_skips_large_chunks_rather_than_stopping() {
        let dir = tempfile::tempdir().unwrap();
        let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        let long_text = "word ".repeat(200);
        let short_text = "short";
        index
            .upsert_records(vec![
                RetrievalDocument {
                    id: "long".into(),
                    text: Some(long_text.clone()),
                    metadata: serde_json::json!({}),
                    representations: BTreeMap::from([(
                        "s".into(),
                        Representation::Dense {
                            vector: vec![1.0, 0.0],
                        },
                    )]),
                    ..RetrievalDocument::default()
                },
                RetrievalDocument {
                    id: "short1".into(),
                    text: Some(short_text.into()),
                    metadata: serde_json::json!({}),
                    representations: BTreeMap::from([(
                        "s".into(),
                        Representation::Dense {
                            vector: vec![0.9, 0.1],
                        },
                    )]),
                    ..RetrievalDocument::default()
                },
                RetrievalDocument {
                    id: "short2".into(),
                    text: Some(short_text.into()),
                    metadata: serde_json::json!({}),
                    representations: BTreeMap::from([(
                        "s".into(),
                        Representation::Dense {
                            vector: vec![0.8, 0.2],
                        },
                    )]),
                    ..RetrievalDocument::default()
                },
            ])
            .unwrap();

        let query: RetrieveRequest = serde_json::from_value(json!({
            "prefetch": [{"kind":"dense","field":"s","vector":[1.0,0.0],"limit":10}],
            "limit": 5,
            "objective": {"context_budget_tokens": 50}
        }))
        .unwrap();
        let response = index.retrieve(&query).unwrap();
        let ids: Vec<&str> = response.matches.iter().map(|h| h.id.as_str()).collect();
        assert!(
            ids.contains(&"short1") || ids.contains(&"short2"),
            "short docs should be included when large doc is skipped: got {:?}",
            ids
        );
        assert!(
            response
                .matches
                .iter()
                .any(|h| h.estimated_tokens.is_some()),
            "estimated_tokens must be populated when context_budget_tokens is set"
        );
    }

    #[test]
    fn ranking_signals_are_populated_in_trace() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let query = request();

        let response = index.retrieve(&query).unwrap();
        let signals = &response.trace.signals;
        assert!(signals.source_diversity >= 0.0 && signals.source_diversity <= 1.0);
        assert!(signals.topk_score_spread >= 0.0);
    }

    #[test]
    fn plan_stages_structure_matches_request_channels() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let query = request();
        let plan = index.plan(&query).unwrap();

        let PlanStage::Parallel(channels) = &plan.stages[0] else {
            panic!("expected Parallel stage, got {:?}", plan.stages[0]);
        };
        assert_eq!(channels.len(), query.prefetch.len());

        assert!(
            matches!(
                plan.stages[1],
                PlanStage::Fusion(FusionOperator::ReciprocalRank)
            ),
            "expected RRF fusion, got {:?}",
            plan.stages[1]
        );

        assert!(
            matches!(plan.stages.last().unwrap(), PlanStage::Context(_)),
            "last stage must be Context"
        );

        assert!(plan.estimate.critical_path_cost > 0.0);
        assert!(plan.estimate.total_cost > 0.0);
        serde_json::to_value(&plan).expect("plans exposed by HTTP must serialize");
    }

    #[test]
    fn single_channel_plan_uses_native_score_not_rrf() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let mut query = request();
        query.prefetch.retain(|c| matches!(c, Channel::Bm25 { .. }));

        let plan = index.plan(&query).unwrap();
        assert!(
            matches!(
                plan.stages[1],
                PlanStage::Fusion(FusionOperator::NativeScore)
            ),
            "single channel must bypass RRF"
        );
    }

    #[test]
    fn plan_with_rerank_includes_rerank_stage() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let mut query = request();
        query
            .prefetch
            .retain(|c| matches!(c, Channel::Dense { .. }));
        query.rerank = Some(Rerank {
            field: Some("tokens".into()),
            vectors: vec![vec![1., 0.]],
            limit: 3,
            adaptive: None,
        });

        let plan = index.plan(&query).unwrap();
        let has_rerank = plan
            .stages
            .iter()
            .any(|s| matches!(s, PlanStage::Rerank(_)));
        assert!(has_rerank, "rerank stage missing from plan");
    }

    #[test]
    fn retrieve_trace_per_stage_ms_is_aligned_with_plan_stages() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let query = request();

        let response = index.retrieve(&query).unwrap();
        let trace = &response.trace;

        assert_eq!(
            trace.per_stage_actual_ms.len(),
            trace.plan.stages.len(),
            "per_stage_actual_ms must have one entry per plan stage"
        );
        for &ms in &trace.per_stage_actual_ms {
            assert!(ms >= 0.0, "stage duration must be non-negative");
        }
    }

    #[test]
    fn retrieve_trace_plan_matches_dry_run_plan() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let mut query = request();
        query.filter = None;

        let dry_run = index.plan(&query).unwrap();
        let response = index.retrieve(&query).unwrap();
        assert_eq!(
            response.trace.plan, dry_run,
            "plan embedded in trace must match plan() dry-run"
        );
    }

    #[test]
    fn cost_model_selects_exact_for_tiny_corpus_even_with_ann_built() {
        // 3-doc corpus, dim=2: ExactDense cost = 3×2×2=12 units.
        // HnswDense cost = 256 × ceil(log2(3)) × 2 × 3 = 3072 units → exact cheaper.
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        index.build_dense_ann("semantic", 4, 16).unwrap();

        let mut query = request();
        query.filter = None;
        query
            .prefetch
            .retain(|c| matches!(c, Channel::Dense { .. }));

        let plan = index.plan(&query).unwrap();
        let ch = &plan.parallel_channels()[0];
        // With cost model active, exact should win for tiny corpora.
        assert_eq!(ch.operator, PhysicalOperator::ExactDense);
        assert!(
            matches!(
                ch.reason,
                PlanReason::LowerEstimatedCost | PlanReason::AnnReady
            ),
            "expected cost-based reason, got {:?}",
            ch.reason
        );
    }

    #[test]
    fn filter_stats_selectivity_accurate_for_equality_filter() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path()); // 3 docs: 2 with tenant=a, 1 with tenant=denied
        let query = request(); // filter: tenant=a

        let plan = index.plan(&query).unwrap();
        let fs = plan.stats.filter_stats.as_ref().unwrap();
        assert_eq!(plan.filter, FilterStrategy::MetadataIndex);
        assert_eq!(fs.filter_operator, FilterStrategy::MetadataIndex);
        let expected = 2.0 / 3.0_f32;
        assert!(
            (fs.selectivity - expected).abs() < 0.05,
            "selectivity should be ~{expected:.2}, got {:.2}",
            fs.selectivity
        );

        let mut range_query = query;
        range_query.filter = Some(Predicate::Range {
            field: "year".into(),
            gte: Some(2026.0),
            lte: None,
        });
        let range_plan = index.plan(&range_query).unwrap();
        assert_eq!(range_plan.filter, FilterStrategy::MetadataScan);
        assert_eq!(
            range_plan.stats.filter_stats.unwrap().filter_operator,
            FilterStrategy::MetadataScan
        );
    }

    #[test]
    fn cost_model_selects_hnsw_when_cheaper_than_exact() {
        // 50 docs, dim=2, ef=4:
        // hnsw = 4 × ceil(log2(50)) × 2 × 3 = 4×6×6 = 144
        // exact = 50 × 2 × 2 = 200 → HNSW cheaper
        let dir = tempfile::tempdir().unwrap();
        let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        let docs: Vec<_> = (0..50_u32)
            .map(|i| {
                let v = if i % 2 == 0 {
                    vec![1.0_f32, 0.0]
                } else {
                    vec![0.0, 1.0]
                };
                RetrievalDocument {
                    id: format!("d{i}"),
                    representations: BTreeMap::from([(
                        "f".into(),
                        Representation::Dense { vector: v },
                    )]),
                    ..RetrievalDocument::default()
                }
            })
            .collect();
        index.upsert_records(docs).unwrap();
        index.build_dense_ann("f", 4, 16).unwrap();

        let query: RetrieveRequest = serde_json::from_value(json!({
            "prefetch": [{"kind": "dense", "field": "f", "vector": [1.0, 0.0],
                          "limit": 5, "ef_search": 4}],
            "limit": 5
        }))
        .unwrap();
        let plan = index.plan(&query).unwrap();
        assert_eq!(
            plan.parallel_channels()[0].operator,
            PhysicalOperator::HnswDense,
            "HNSW should be cheaper for 50 docs with ef=4"
        );
        assert_eq!(
            plan.parallel_channels()[0].reason,
            PlanReason::LowerEstimatedCost
        );
    }

    #[test]
    fn adaptive_doc_set_uses_sparse_and_bitmap_forms() {
        let sparse = DocSet::from_numbers(4096, [1, 2048]);
        assert!(matches!(sparse, DocSet::Sparse(_)));
        let dense = DocSet::from_numbers(64, 0..64);
        assert!(matches!(dense, DocSet::Bitmap { .. }));
        assert_eq!(dense.len(), 64);
        assert!(dense.contains(37));
        assert!(!dense.contains(64));

        let evens = DocSet::from_numbers(64, (0..64).filter(|id| id % 2 == 0));
        let high = DocSet::from_numbers(64, 32..64);
        let intersection = evens.clone().intersection(&high, 64);
        assert_eq!(
            intersection.iter().collect::<Vec<_>>(),
            (32..64).step_by(2).collect::<Vec<_>>()
        );
        let union = evens.union(high, 64);
        assert_eq!(union.len(), 48);
    }

    #[test]
    fn dense_hnsw_filters_in_traversal_and_respects_overlay_updates() {
        let dir = tempfile::tempdir().unwrap();
        let index = MultiVectorIndex::open(dir.path(), IndexConfig::new(2)).unwrap();
        let docs: Vec<_> = (0..50_u32)
            .map(|i| RetrievalDocument {
                id: format!("d{i}"),
                metadata: json!({"tenant": if i % 2 == 0 { "a" } else { "b" }}),
                representations: BTreeMap::from([(
                    "semantic".into(),
                    Representation::Dense {
                        vector: vec![1.0, i as f32 / 100.0],
                    },
                )]),
                ..RetrievalDocument::default()
            })
            .collect();
        index.upsert_records(docs).unwrap();
        index.build_dense_ann("semantic", 4, 16).unwrap();

        let query = || -> RetrieveRequest {
            serde_json::from_value(json!({
                "prefetch": [{"kind": "dense", "field": "semantic", "vector": [1.0, 0.0],
                              "limit": 10, "ef_search": 32, "backend": "hnsw"}],
                "filter": {"op": "eq", "field": "tenant", "value": "a"},
                "limit": 10
            }))
            .unwrap()
        };
        let first = index.retrieve(&query()).unwrap();
        assert_eq!(first.trace.channels[0]["backend"], "hnsw_dense");
        assert!(
            first
                .matches
                .iter()
                .all(|hit| hit.metadata["tenant"] == "a")
        );

        index
            .upsert_records(vec![RetrievalDocument {
                id: "d0".into(),
                metadata: json!({"tenant": "b"}),
                representations: BTreeMap::from([(
                    "semantic".into(),
                    Representation::Dense {
                        vector: vec![1.0, 0.0],
                    },
                )]),
                ..RetrievalDocument::default()
            }])
            .unwrap();
        let updated = index.retrieve(&query()).unwrap();
        assert!(updated.matches.iter().all(|hit| hit.id != "d0"));
        assert!(
            updated
                .matches
                .iter()
                .all(|hit| hit.metadata["tenant"] == "a")
        );
    }

    #[test]
    fn fde_hnsw_supports_metadata_filters() {
        let dir = tempfile::tempdir().unwrap();
        let config = IndexConfig {
            dimension: 3,
            centroids: 3,
            residual_bits: 2,
            probes: 3,
            fde_repetitions: 2,
            fde_ksim: 2,
            fde_projected: 2,
            analyzer: TextAnalyzer::plain(),
        };
        let index = MultiVectorIndex::open(dir.path(), config).unwrap();
        index
            .train(
                &[
                    vec![1., 0., 0.],
                    vec![0., 1., 0.],
                    vec![0., 0., 1.],
                    vec![0.8, 0.2, 0.],
                    vec![0.2, 0.8, 0.],
                    vec![0.2, 0., 0.8],
                ],
                2,
            )
            .unwrap();
        index
            .upsert_records(
                (0..24)
                    .map(|i| RetrievalDocument {
                        id: format!("m{i}"),
                        vectors: vec![if i % 2 == 0 {
                            vec![1., 0., 0.]
                        } else {
                            vec![0., 1., 0.]
                        }],
                        metadata: json!({"tenant": if i % 3 == 0 { "a" } else { "b" }}),
                        ..RetrievalDocument::default()
                    })
                    .collect(),
            )
            .unwrap();
        index.build_fde_ann(4, 16).unwrap();
        let query: RetrieveRequest = serde_json::from_value(json!({
            "prefetch": [{"kind": "multivector", "vectors": [[1., 0., 0.]],
                          "limit": 8, "ef_search": 32, "backend": "hnsw"}],
            "filter": {"op": "eq", "field": "tenant", "value": "a"},
            "limit": 8
        }))
        .unwrap();
        let response = index.retrieve(&query).unwrap();
        assert_eq!(response.trace.channels[0]["backend"], "hnsw_fde");
        assert!(
            response
                .matches
                .iter()
                .all(|hit| hit.metadata["tenant"] == "a")
        );
    }

    #[test]
    fn retrieve_batch_returns_same_generation_and_correct_order() {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());

        let make_req = |v: f32| -> RetrieveRequest {
            serde_json::from_value(json!({
                "prefetch": [{"kind": "dense", "field": "semantic",
                              "vector": [v, 0.], "limit": 3}],
                "limit": 3
            }))
            .unwrap()
        };
        let requests = vec![make_req(1.0), make_req(0.8), make_req(0.0)];
        let results = index.retrieve_batch(&requests);

        // All three slots succeed.
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok(), "batch slot failed: {:?}", r);
        }

        // Results are in request order: query [1,0] should rank "b" first,
        // query [0,0] should also return results (scalar fallback).
        let top0 = &results[0].as_ref().unwrap().matches;
        let top2 = &results[2].as_ref().unwrap().matches;
        assert!(!top0.is_empty());
        assert!(!top2.is_empty());

        // All results share the same generation — verified by comparing trace generations.
        let gen0 = results[0].as_ref().unwrap().trace.plan.stats.generation;
        let gen2 = results[2].as_ref().unwrap().trace.plan.stats.generation;
        assert_eq!(gen0, gen2, "batch queries saw different generations");
    }
}
