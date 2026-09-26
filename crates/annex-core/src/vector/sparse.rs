//! Sparse-vector index for hybrid retrieval.
//!
//! Companion to the dense HNSW engine: enables BM25 / SPLADE / uniCOIL style
//! learned-sparse and classic-lexical retrieval, which is table stakes for
//! any 2026 RAG stack.
//!
//! Data model:
//! - Each document is a `SparseVector` of `(index, value)` pairs.
//! - The index is an inverted list keyed by `index`, mapping to `(doc, value)`.
//! - Two scoring modes on query:
//!   - `search_dot`: plain sparse dot product between query and document
//!     sparse vectors. Right choice when the caller has SPLADE-shaped
//!     `(token_id, weight)` embeddings and just wants the inner product.
//!   - `search_bm25`: Okapi BM25 with configurable k1, b. Right choice
//!     when the input is `(token_id, term_frequency)` from a classic
//!     lexical pipeline.
//!
//! Storage is in-memory; snapshot / WAL integration is a separate follow-up.
//! Concurrent inserts are serialized behind a single mutex — this crate's
//! current focus is establishing the primitive, not scaling it. The hot
//! path (query) does not take that mutex.

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::utils::types::PointId;

/// Sparse vector: parallel arrays of feature indices and their values.
/// Indices should be sorted ascending; not enforced but assumed by the
/// dot-product path for early termination opportunities in future
/// optimisations.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, f32)>) -> Self {
        let mut v: Vec<_> = pairs.into_iter().collect();
        v.sort_by_key(|(i, _)| *i);
        let (indices, values) = v.into_iter().unzip();
        Self { indices, values }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

impl Default for SparseVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-index inverted posting.
#[derive(Debug, Default)]
struct Posting {
    /// (doc_local_id, value) pairs. Sorted by doc_local_id insertion order.
    entries: Vec<(u32, f32)>,
}

/// In-memory inverted-index sparse retriever.
#[derive(Debug, Default)]
pub struct SparseIndex {
    inner: RwLock<SparseIndexInner>,
}

#[derive(Debug, Default)]
struct SparseIndexInner {
    /// Postings keyed by feature index. Vec is faster than HashMap when
    /// feature indices are dense/contiguous (typical for BPE tokens with
    /// vocab ~30-50k); we resize on demand.
    postings: Vec<Posting>,
    /// Public PointId ↔ internal local u32 id mapping.
    id_to_local: HashMap<PointId, u32>,
    local_to_id: Vec<PointId>,
    /// Sum of values per document — used as "doc length" for BM25.
    doc_lens: Vec<f32>,
    /// Total accumulated doc length; divided at query time to get avgdl.
    total_len: f64,
}

impl SparseIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a document's sparse vector.
    ///
    /// Replacement is O(sum of posting lengths); simple linear scan of each
    /// touched posting. Fine for the append-heavy workloads sparse indexes
    /// typically serve; a real deletion story lives with the compaction /
    /// segments effort.
    pub fn upsert(&self, id: PointId, vec: &SparseVector) {
        let mut inner = self.inner.write().expect("sparse index mutex poisoned");
        let local = if let Some(&existing) = inner.id_to_local.get(&id) {
            // Remove old contributions from postings + doc_len bookkeeping.
            let old_len = inner.doc_lens[existing as usize];
            inner.total_len -= old_len as f64;
            inner.doc_lens[existing as usize] = 0.0;
            let posting_count = inner.postings.len();
            for posting in &mut inner.postings[..posting_count] {
                posting.entries.retain(|&(doc, _)| doc != existing);
            }
            existing
        } else {
            let new_local = inner.local_to_id.len() as u32;
            inner.local_to_id.push(id);
            inner.id_to_local.insert(id, new_local);
            inner.doc_lens.push(0.0);
            new_local
        };

        // Now add new contributions.
        let mut doc_len = 0.0f32;
        for (&feature, &value) in vec.indices.iter().zip(&vec.values) {
            if feature as usize >= inner.postings.len() {
                inner
                    .postings
                    .resize_with(feature as usize + 1, Posting::default);
            }
            inner.postings[feature as usize]
                .entries
                .push((local, value));
            doc_len += value;
        }
        inner.doc_lens[local as usize] = doc_len;
        inner.total_len += doc_len as f64;
    }

    pub fn len(&self) -> usize {
        self.inner
            .read()
            .expect("sparse index mutex poisoned")
            .local_to_id
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sparse dot product scoring. Query and doc are treated as sparse
    /// vectors; per-doc score is the sum of q[i] * d[i] over shared
    /// indices. Right choice for SPLADE-style learned-sparse embeddings.
    pub fn search_dot(&self, query: &SparseVector, top_k: usize) -> Vec<(PointId, f32)> {
        if top_k == 0 || query.is_empty() {
            return Vec::new();
        }
        let inner = self.inner.read().expect("sparse index mutex poisoned");
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for (&feature, &q_value) in query.indices.iter().zip(&query.values) {
            if let Some(posting) = inner.postings.get(feature as usize) {
                for &(doc, d_value) in &posting.entries {
                    *scores.entry(doc).or_insert(0.0) += q_value * d_value;
                }
            }
        }
        let mut ranked: Vec<_> = scores
            .into_iter()
            .map(|(local, score)| (inner.local_to_id[local as usize], score))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(top_k);
        ranked
    }

    /// Okapi BM25 scoring. Assumes query values are 1.0 (term appears in
    /// query) and document values are term frequencies. Standard k1=1.2,
    /// b=0.75. Doc length = sum of values (i.e. total token count for
    /// classic BM25).
    pub fn search_bm25(
        &self,
        query: &SparseVector,
        top_k: usize,
        k1: f32,
        b: f32,
    ) -> Vec<(PointId, f32)> {
        if top_k == 0 || query.is_empty() {
            return Vec::new();
        }
        let inner = self.inner.read().expect("sparse index mutex poisoned");
        let n = inner.local_to_id.len();
        if n == 0 {
            return Vec::new();
        }
        let avgdl = (inner.total_len / n as f64) as f32;
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for (&feature, &q_value) in query.indices.iter().zip(&query.values) {
            let Some(posting) = inner.postings.get(feature as usize) else {
                continue;
            };
            let df = posting.entries.len() as f32;
            if df == 0.0 {
                continue;
            }
            // Standard BM25 IDF variant with +1 smoothing.
            let idf = ((n as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(doc, tf) in &posting.entries {
                let dl = inner.doc_lens[doc as usize].max(1e-6);
                let denom = tf + k1 * (1.0 - b + b * dl / avgdl.max(1e-6));
                let contribution = q_value * idf * (tf * (k1 + 1.0)) / denom;
                *scores.entry(doc).or_insert(0.0) += contribution;
            }
        }
        let mut ranked: Vec<_> = scores
            .into_iter()
            .map(|(local, score)| (inner.local_to_id[local as usize], score))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(top_k);
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_scores_and_ranks_correctly() {
        let idx = SparseIndex::new();
        idx.upsert(1, &SparseVector::from_pairs([(0, 1.0), (2, 1.0)]));
        idx.upsert(2, &SparseVector::from_pairs([(0, 2.0), (5, 1.0)]));
        idx.upsert(3, &SparseVector::from_pairs([(2, 3.0), (5, 1.0)]));
        // Query is [0: 1.0, 2: 1.0]. Doc 1 shares both (score 1+1=2),
        // doc 3 shares only 2 (score 3), doc 2 shares only 0 (score 2).
        // Expected order: 3 (3.0), 1 (2.0), 2 (2.0) — tie broken by id.
        let query = SparseVector::from_pairs([(0, 1.0), (2, 1.0)]);
        let hits = idx.search_dot(&query, 10);
        assert_eq!(hits[0].0, 3);
        assert!((hits[0].1 - 3.0).abs() < 1e-6);
        assert_eq!(hits[1].0, 1);
        assert_eq!(hits[2].0, 2);
    }

    #[test]
    fn bm25_favors_rare_terms() {
        let idx = SparseIndex::new();
        // Term 0 appears in every doc — common word.
        // Term 100 appears only in doc 3 — rare word.
        idx.upsert(1, &SparseVector::from_pairs([(0, 5.0)]));
        idx.upsert(2, &SparseVector::from_pairs([(0, 5.0)]));
        idx.upsert(3, &SparseVector::from_pairs([(0, 1.0), (100, 1.0)]));
        // Query includes both terms.
        let query = SparseVector::from_pairs([(0, 1.0), (100, 1.0)]);
        let hits = idx.search_bm25(&query, 10, 1.2, 0.75);
        // Doc 3 must rank first because of the rare term contribution.
        assert_eq!(hits[0].0, 3);
    }

    #[test]
    fn upsert_replaces_previous_contribution() {
        let idx = SparseIndex::new();
        idx.upsert(1, &SparseVector::from_pairs([(0, 10.0)]));
        idx.upsert(1, &SparseVector::from_pairs([(5, 1.0)]));
        // Query on term 0 must find nothing (doc 1's term-0 contribution
        // was removed by the upsert overwrite).
        let hits = idx.search_dot(&SparseVector::from_pairs([(0, 1.0)]), 10);
        assert!(hits.is_empty());
        // Query on term 5 must find doc 1.
        let hits = idx.search_dot(&SparseVector::from_pairs([(5, 1.0)]), 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn empty_query_or_index_returns_empty() {
        let idx = SparseIndex::new();
        assert!(idx.search_dot(&SparseVector::new(), 10).is_empty());
        idx.upsert(1, &SparseVector::from_pairs([(0, 1.0)]));
        assert!(idx.search_dot(&SparseVector::new(), 10).is_empty());
    }
}
