//! In-memory sparse dot/BM25 retrieval. Weights are finite and nonnegative;
//! feature IDs are arbitrary u32 values. Clone creates an independent snapshot.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::utils::types::PointId;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SparseError {
    #[error("sparse indices and values must have equal lengths")]
    LengthMismatch,
    #[error("sparse weights must be finite and nonnegative")]
    InvalidWeight,
    #[error("sparse weight or score exceeds f32 range")]
    Overflow,
    #[error("BM25 requires finite k1 >= 0 and b in [0, 1]")]
    InvalidBm25,
}

/// Parallel feature/weight arrays. Index operations validate and canonicalize
/// these arrays: sort features, sum duplicates, and omit zero weights.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, f32)>) -> Self {
        let (indices, values) = pairs.into_iter().unzip();
        Self { indices, values }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn canonicalized(&self) -> Result<Self, SparseError> {
        if self.indices.len() != self.values.len() {
            return Err(SparseError::LengthMismatch);
        }
        if self.values.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(SparseError::InvalidWeight);
        }
        let mut pairs: Vec<_> = self
            .indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .collect();
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        let mut result = Self::new();
        let mut i = 0;
        while i < pairs.len() {
            let feature = pairs[i].0;
            let mut weight = 0.0f64;
            while i < pairs.len() && pairs[i].0 == feature {
                weight += f64::from(pairs[i].1);
                i += 1;
            }
            if weight > f64::from(f32::MAX) {
                return Err(SparseError::Overflow);
            }
            if weight > 0.0 {
                result.indices.push(feature);
                result.values.push(weight as f32);
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Debug)]
struct Document {
    vector: SparseVector,
    length: f64,
}

#[derive(Clone, Debug, Default)]
struct SparseIndexInner {
    postings: HashMap<u32, Arc<HashMap<PointId, f32>>>,
    documents: HashMap<PointId, Arc<Document>>,
    // Subtract within similar magnitudes so deleting a large document cannot
    // erase the accumulated lengths of much smaller remaining documents.
    length_bins: BTreeMap<u16, (usize, f64)>,
}

impl SparseIndexInner {
    fn remove(&mut self, id: PointId) -> bool {
        let Some(doc) = self.documents.remove(&id) else {
            return false;
        };
        for &feature in &doc.vector.indices {
            let posting = self.postings.get_mut(&feature).expect("indexed feature");
            Arc::make_mut(posting).remove(&id);
            if posting.is_empty() {
                self.postings.remove(&feature);
            }
        }
        if doc.length > 0.0 {
            let exponent = (doc.length.to_bits() >> 52) as u16;
            let bin = self.length_bins.get_mut(&exponent).expect("indexed length");
            if bin.0 == 1 {
                self.length_bins.remove(&exponent);
            } else {
                bin.0 -= 1;
                bin.1 -= doc.length;
            }
        }
        true
    }
}

/// Memory grows with live documents and nonzero features, never feature IDs.
#[derive(Debug, Default)]
pub struct SparseIndex {
    inner: RwLock<SparseIndexInner>,
}

impl Clone for SparseIndex {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(
                self.inner
                    .read()
                    .expect("sparse index lock poisoned")
                    .clone(),
            ),
        }
    }
}

impl SparseIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validation precedes mutation. Replacement and deletion touch only the
    /// document's own terms. Snapshots share postings until a writer changes them.
    pub fn upsert(&self, id: PointId, vector: &SparseVector) -> Result<(), SparseError> {
        let vector = vector.canonicalized()?;
        let length: f64 = vector.values.iter().map(|&v| f64::from(v)).sum();
        let mut inner = self.inner.write().expect("sparse index lock poisoned");
        inner.remove(id);
        for (&feature, &value) in vector.indices.iter().zip(&vector.values) {
            Arc::make_mut(inner.postings.entry(feature).or_default()).insert(id, value);
        }
        if length > 0.0 {
            let bin = inner
                .length_bins
                .entry((length.to_bits() >> 52) as u16)
                .or_default();
            bin.0 += 1;
            bin.1 += length;
        }
        inner
            .documents
            .insert(id, Arc::new(Document { vector, length }));
        Ok(())
    }

    pub fn delete(&self, id: PointId) -> bool {
        self.inner
            .write()
            .expect("sparse index lock poisoned")
            .remove(id)
    }

    pub fn len(&self) -> usize {
        self.inner
            .read()
            .expect("sparse index lock poisoned")
            .documents
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn search_dot(
        &self,
        query: &SparseVector,
        top_k: usize,
    ) -> Result<Vec<(PointId, f32)>, SparseError> {
        self.search_dot_filtered(query, top_k, |_| true)
    }

    pub fn search_dot_filtered(
        &self,
        query: &SparseVector,
        top_k: usize,
        eligible: impl Fn(PointId) -> bool,
    ) -> Result<Vec<(PointId, f32)>, SparseError> {
        self.search_dot_filtered_by(query, top_k, eligible, |a, b| a.cmp(&b))
    }

    /// Supply a stable external-ID order when local IDs can change on rebuild.
    pub fn search_dot_filtered_by(
        &self,
        query: &SparseVector,
        top_k: usize,
        eligible: impl Fn(PointId) -> bool,
        tie_break: impl Fn(PointId, PointId) -> Ordering,
    ) -> Result<Vec<(PointId, f32)>, SparseError> {
        self.search(query, top_k, None, eligible, tie_break)
    }

    /// BM25 uses global live-document statistics, including empty documents.
    /// Query weights multiply term contributions; use ones for lexical queries.
    pub fn search_bm25(
        &self,
        query: &SparseVector,
        top_k: usize,
        k1: f32,
        b: f32,
    ) -> Result<Vec<(PointId, f32)>, SparseError> {
        self.search_bm25_filtered(query, top_k, k1, b, |_| true)
    }

    /// Eligibility is applied before scoring and top-k, without changing IDF.
    pub fn search_bm25_filtered(
        &self,
        query: &SparseVector,
        top_k: usize,
        k1: f32,
        b: f32,
        eligible: impl Fn(PointId) -> bool,
    ) -> Result<Vec<(PointId, f32)>, SparseError> {
        self.search_bm25_filtered_by(query, top_k, k1, b, eligible, |a, b| a.cmp(&b))
    }

    /// BM25 with eligibility and caller-defined ordering for equal scores.
    pub fn search_bm25_filtered_by(
        &self,
        query: &SparseVector,
        top_k: usize,
        k1: f32,
        b: f32,
        eligible: impl Fn(PointId) -> bool,
        tie_break: impl Fn(PointId, PointId) -> Ordering,
    ) -> Result<Vec<(PointId, f32)>, SparseError> {
        if !k1.is_finite() || k1 < 0.0 || !b.is_finite() || !(0.0..=1.0).contains(&b) {
            return Err(SparseError::InvalidBm25);
        }
        self.search(
            query,
            top_k,
            Some((f64::from(k1), f64::from(b))),
            eligible,
            tie_break,
        )
    }

    fn search(
        &self,
        query: &SparseVector,
        top_k: usize,
        bm25: Option<(f64, f64)>,
        eligible: impl Fn(PointId) -> bool,
        tie_break: impl Fn(PointId, PointId) -> Ordering,
    ) -> Result<Vec<(PointId, f32)>, SparseError> {
        let query = query.canonicalized()?;
        if top_k == 0 || query.is_empty() {
            return Ok(Vec::new());
        }
        let inner = self.inner.read().expect("sparse index lock poisoned");
        let n = inner.documents.len() as f64;
        let total_len: f64 = inner.length_bins.values().map(|bin| bin.1).sum();
        let avgdl = if n > 0.0 { total_len / n } else { 0.0 };
        let mut scores: HashMap<PointId, f64> = HashMap::new();
        for (&feature, &q) in query.indices.iter().zip(&query.values) {
            let Some(posting) = inner.postings.get(&feature) else {
                continue;
            };
            let df = posting.len() as f64;
            let idf = ((n - df + 0.5) / (df + 0.5)).ln_1p();
            for (&id, &value) in posting.iter() {
                if !eligible(id) {
                    continue;
                }
                let tf = f64::from(value);
                let contribution = match bm25 {
                    Some((k1, b)) => {
                        let dl = inner.documents[&id].length;
                        let denom = tf + k1 * (1.0 - b + b * dl / avgdl);
                        f64::from(q) * idf * tf * (k1 + 1.0) / denom
                    }
                    None => f64::from(q) * tf,
                };
                *scores.entry(id).or_default() += contribution;
            }
        }
        let mut ranked = Vec::with_capacity(scores.len());
        for (id, score) in scores {
            if !score.is_finite() || score > f64::from(f32::MAX) {
                return Err(SparseError::Overflow);
            }
            ranked.push((id, score as f32));
        }
        let order = |a: &(PointId, f32), b: &(PointId, f32)| {
            b.1.total_cmp(&a.1).then_with(|| tie_break(a.0, b.0))
        };
        if ranked.len() > top_k {
            ranked.select_nth_unstable_by(top_k, order);
            ranked.truncate(top_k);
        }
        ranked.sort_unstable_by(order);
        Ok(ranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalization_validation_and_atomic_replacement() {
        let idx = SparseIndex::new();
        let vector = SparseVector::from_pairs([(u32::MAX, 1.0), (2, 1.0), (2, 2.0), (3, 0.0)]);
        let canonical = vector.canonicalized().unwrap();
        assert_eq!(
            canonical,
            SparseVector::from_pairs([(2, 3.0), (u32::MAX, 1.0)])
        );
        idx.upsert(1, &vector).unwrap();
        idx.upsert(2, &canonical).unwrap();
        assert_eq!(idx.inner.read().unwrap().postings.len(), 2);
        assert_eq!(
            idx.search_dot(&vector, 10).unwrap(),
            vec![(1, 10.0), (2, 10.0)]
        );
        assert_eq!(
            idx.search_dot_filtered_by(&vector, 1, |_| true, |a, b| b.cmp(&a))
                .unwrap()[0]
                .0,
            2
        );
        assert_eq!(
            idx.search_bm25_filtered_by(&vector, 1, 1.2, 0.75, |_| true, |a, b| b.cmp(&a))
                .unwrap()[0]
                .0,
            2
        );
        for invalid in [
            SparseVector {
                indices: vec![1],
                values: vec![],
            },
            SparseVector::from_pairs([(2, f32::NAN)]),
            SparseVector::from_pairs([(2, f32::INFINITY)]),
            SparseVector::from_pairs([(2, -1.0)]),
            SparseVector::from_pairs([(2, f32::MAX), (2, f32::MAX)]),
        ] {
            assert!(idx.upsert(1, &invalid).is_err());
            assert!(idx.search_dot(&invalid, 10).is_err());
            assert_eq!(
                idx.search_dot(&vector, 10).unwrap(),
                vec![(1, 10.0), (2, 10.0)]
            );
        }
        let huge = SparseVector::from_pairs([(7, f32::MAX)]);
        idx.upsert(3, &huge).unwrap();
        assert_eq!(idx.search_dot(&huge, 1), Err(SparseError::Overflow));
    }

    #[test]
    fn replacement_deletion_filter_and_independent_snapshot() {
        let idx = SparseIndex::new();
        idx.upsert(1, &SparseVector::from_pairs([(0, 10.0)]))
            .unwrap();
        idx.upsert(2, &SparseVector::from_pairs([(0, 2.0)]))
            .unwrap();
        let snapshot = idx.clone();
        let query = SparseVector::from_pairs([(0, 1.0)]);
        assert_eq!(
            idx.search_dot_filtered(&query, 1, |id| id == 2).unwrap(),
            vec![(2, 2.0)]
        );
        idx.upsert(1, &SparseVector::from_pairs([(5, 1.0)]))
            .unwrap();
        assert_eq!(idx.search_dot(&query, 10).unwrap(), vec![(2, 2.0)]);
        assert!(idx.delete(2));
        assert!(!idx.delete(2));
        assert!(idx.search_dot(&query, 10).unwrap().is_empty());
        assert_eq!(
            snapshot.search_dot(&query, 10).unwrap(),
            vec![(1, 10.0), (2, 2.0)]
        );
        idx.upsert(2, &query).unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.search_dot(&query, 10).unwrap(), vec![(2, 1.0)]);
        assert!(idx.delete(1));
        assert!(idx.delete(2));
        let inner = idx.inner.read().unwrap();
        assert!(inner.postings.is_empty());
        assert!(inner.length_bins.is_empty());
    }

    #[test]
    fn bm25_matches_independent_oracle_after_mutations() {
        let idx = SparseIndex::new();
        let mut corpus: HashMap<u64, [f64; 4]> = HashMap::new();
        let query = SparseVector::from_pairs([(0, 1.0), (2, 0.5), (3, 1.0)]);
        for step in 0..80u64 {
            let id = step * 7 % 11;
            if step % 4 == 3 {
                assert_eq!(idx.delete(id), corpus.remove(&id).is_some());
            } else {
                let counts = std::array::from_fn(|term| ((step + term as u64 * 3) % 6) as f64);
                corpus.insert(id, counts);
                idx.upsert(
                    id,
                    &SparseVector::from_pairs(
                        counts
                            .iter()
                            .enumerate()
                            .map(|(term, &tf)| (term as u32, tf as f32)),
                    ),
                )
                .unwrap();
            }
            let n = corpus.len() as f64;
            let avgdl = corpus.values().flatten().sum::<f64>() / n;
            let k1 = f64::from(1.2f32);
            let mut expected = Vec::new();
            for (&doc, counts) in &corpus {
                if doc % 2 != 0 {
                    continue;
                }
                let dl = counts.iter().sum::<f64>();
                let mut score = 0.0;
                for (term, q) in [(0, 1.0), (2, 0.5), (3, 1.0)] {
                    if counts[term] == 0.0 {
                        continue;
                    }
                    let df = corpus.values().filter(|counts| counts[term] > 0.0).count() as f64;
                    let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                    score += q * idf * (counts[term] * (k1 + 1.0))
                        / (counts[term] + k1 * (0.25 + 0.75 * dl / avgdl));
                }
                if score > 0.0 {
                    expected.push((doc, score as f32));
                }
            }
            expected.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            expected.truncate(3);
            let actual = idx
                .search_bm25_filtered(&query, 3, 1.2, 0.75, |id| id % 2 == 0)
                .unwrap();
            assert_eq!(
                actual.iter().map(|hit| hit.0).collect::<Vec<_>>(),
                expected.iter().map(|hit| hit.0).collect::<Vec<_>>(),
                "step {step}"
            );
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual.1 - expected.1).abs() < 1e-6,
                    "step {step}: {actual:?} vs {expected:?}"
                );
            }
        }
        assert!(idx.search_bm25(&query, 3, -1.0, 0.75).is_err());
        assert!(idx.search_bm25(&query, 3, 1.2, f32::NAN).is_err());
        assert!(idx.search_dot(&SparseVector::new(), 10).unwrap().is_empty());
        assert!(
            SparseIndex::new()
                .search_bm25(&query, 10, 1.2, 0.75)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn deleting_large_documents_preserves_small_document_lengths() {
        let index = SparseIndex::new();
        let weights = [f32::MAX, 1e16, 1.0, f32::MIN_POSITIVE];
        let query = SparseVector::from_pairs([(0, 1.0)]);
        for (id, &weight) in weights.iter().enumerate() {
            index
                .upsert(id as u64, &SparseVector::from_pairs([(0, weight)]))
                .unwrap();
        }
        // Three separated scales defeat both ordinary f64 summation and a
        // two-part compensated total when large documents are removed first.
        for removed in 0..weights.len() {
            assert!(index.delete(removed as u64));
            let rebuilt = SparseIndex::new();
            for (id, &weight) in weights.iter().enumerate().skip(removed + 1) {
                rebuilt
                    .upsert(id as u64, &SparseVector::from_pairs([(0, weight)]))
                    .unwrap();
            }
            assert_eq!(
                index.search_bm25(&query, 10, 1.2, 0.75).unwrap(),
                rebuilt.search_bm25(&query, 10, 1.2, 0.75).unwrap()
            );
        }
        assert!(index.inner.read().unwrap().length_bins.is_empty());
    }
}
