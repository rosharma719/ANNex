//! Post-retrieval helpers: hybrid fusion, MMR result diversity, group-by.
//!
//! These are the "make the top-K useful" primitives that sit between raw
//! retrieval scores and what a RAG application actually wants to render.
//! Every one is engine-agnostic — feed them any ranked list.

use std::collections::HashMap;
use std::hash::Hash;

use crate::utils::types::PointId;

// ── Reciprocal Rank Fusion ─────────────────────────────────────────────

/// Reciprocal Rank Fusion — the standard way to combine independent
/// ranked result lists (e.g. dense + sparse + multi-vector) without
/// needing to calibrate their score scales.
///
/// For each doc, the fused score is `sum over lists of 1 / (k + rank)`
/// where `rank` is 0-indexed within each list. `k=60` is the value from
/// the original TREC paper and is the sensible default; smaller values
/// weight top-ranks more heavily.
///
/// Documents not present in a list contribute 0 for that list. Missing
/// entries are treated as rank=infinity.
pub fn reciprocal_rank_fusion(
    result_sets: &[&[(PointId, f32)]],
    top_k: usize,
    k: f32,
) -> Vec<(PointId, f32)> {
    if top_k == 0 || result_sets.is_empty() {
        return Vec::new();
    }
    let mut fused: HashMap<PointId, f32> = HashMap::new();
    for results in result_sets {
        for (rank, (id, _score)) in results.iter().enumerate() {
            let contribution = 1.0 / (k + rank as f32 + 1.0);
            *fused.entry(*id).or_insert(0.0) += contribution;
        }
    }
    let mut ranked: Vec<_> = fused.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(top_k);
    ranked
}

/// Weighted score fusion — for callers that HAVE calibrated their scores
/// to a common scale (rare — normally use RRF). Each result set is scaled
/// by the matching weight and summed per doc.
pub fn weighted_fusion(
    result_sets: &[(&[(PointId, f32)], f32)],
    top_k: usize,
) -> Vec<(PointId, f32)> {
    if top_k == 0 {
        return Vec::new();
    }
    let mut fused: HashMap<PointId, f32> = HashMap::new();
    for (results, weight) in result_sets {
        for (id, score) in *results {
            *fused.entry(*id).or_insert(0.0) += weight * score;
        }
    }
    let mut ranked: Vec<_> = fused.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(top_k);
    ranked
}

// ── Maximal Marginal Relevance ─────────────────────────────────────────

/// Greedy MMR (Carbonell & Goldstein 1998). Given candidates ranked by
/// relevance to a query, iteratively pick the next candidate that maximises
/// `lambda * relevance - (1 - lambda) * max_similarity(selected)`.
///
/// - `lambda = 1.0` → pure relevance (equivalent to top-K by score)
/// - `lambda = 0.0` → pure diversity
/// - `lambda = 0.5-0.7` → sensible default for RAG top-10
///
/// Similarity is passed in as a callback so the caller controls the metric
/// (cosine on stored vectors, jaccard on payload tags, etc). The helper
/// stays independent of the vector storage layer.
pub fn mmr_select<T, F>(
    candidates: &[T],
    scores: &[f32],
    top_k: usize,
    lambda: f32,
    similarity: F,
) -> Vec<usize>
where
    F: Fn(&T, &T) -> f32,
{
    debug_assert_eq!(candidates.len(), scores.len());
    if candidates.is_empty() || top_k == 0 {
        return Vec::new();
    }
    let lambda = lambda.clamp(0.0, 1.0);
    let mut selected: Vec<usize> = Vec::with_capacity(top_k.min(candidates.len()));
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();
    // First pick is always the top relevance — no diversity penalty applies.
    let (first_pos, _) = remaining
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| scores[**a].total_cmp(&scores[**b]))
        .expect("non-empty candidates");
    selected.push(remaining.remove(first_pos));

    while selected.len() < top_k && !remaining.is_empty() {
        let (best_pos, _) = remaining
            .iter()
            .enumerate()
            .map(|(i, &cand_idx)| {
                let rel = scores[cand_idx];
                let max_sim = selected
                    .iter()
                    .map(|&s| similarity(&candidates[cand_idx], &candidates[s]))
                    .fold(f32::NEG_INFINITY, f32::max);
                let effective_sim = if max_sim.is_finite() { max_sim } else { 0.0 };
                let mmr_score = lambda * rel - (1.0 - lambda) * effective_sim;
                (i, mmr_score)
            })
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("non-empty remaining");
        selected.push(remaining.remove(best_pos));
    }
    selected
}

// ── Group-By (top-N per group, up to G groups) ─────────────────────────

/// Top-`per_group` results per unique group key, up to `total_groups` groups.
/// Preserves relative score order both within a group and across groups
/// (a group's rank is set by its best-scoring member).
///
/// Typical use: "5 results per author, top 10 authors" for search UI
/// pagination, or "top 3 chunks per document" for RAG deduplication when a
/// long document produced many nearby chunks.
pub fn group_by_top_n<K>(
    results: &[(PointId, f32, K)],
    per_group: usize,
    total_groups: usize,
) -> Vec<(PointId, f32, K)>
where
    K: Eq + Hash + Clone,
{
    if results.is_empty() || per_group == 0 || total_groups == 0 {
        return Vec::new();
    }
    // First pass: for each group, keep the first `per_group` entries in
    // the input's ranked order. Also track the first (best) score per
    // group so we can order groups by their leader.
    let mut counts: HashMap<K, usize> = HashMap::new();
    let mut group_leader_score: HashMap<K, f32> = HashMap::new();
    let mut group_first_seen: HashMap<K, usize> = HashMap::new();
    let mut kept: Vec<(PointId, f32, K)> = Vec::new();
    for (i, entry) in results.iter().enumerate() {
        let (id, score, key) = entry;
        let count = counts.entry(key.clone()).or_insert(0);
        if *count < per_group {
            group_leader_score.entry(key.clone()).or_insert(*score);
            group_first_seen.entry(key.clone()).or_insert(i);
            *count += 1;
            kept.push((*id, *score, key.clone()));
        }
    }
    // Second pass: only keep the top `total_groups` groups by leader score.
    let mut ordered_groups: Vec<_> = group_leader_score.iter().collect();
    ordered_groups.sort_by(|a, b| {
        b.1.total_cmp(a.1)
            .then_with(|| group_first_seen[a.0].cmp(&group_first_seen[b.0]))
    });
    let kept_keys: std::collections::HashSet<_> = ordered_groups
        .into_iter()
        .take(total_groups)
        .map(|(k, _)| k.clone())
        .collect();
    kept.into_iter()
        .filter(|(_, _, k)| kept_keys.contains(k))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_promotes_docs_ranked_high_in_multiple_lists() {
        let dense = vec![(1u64, 0.9f32), (2, 0.8), (3, 0.7)];
        let sparse = vec![(3u64, 5.0f32), (1, 4.0), (4, 3.0)];
        let fused = reciprocal_rank_fusion(&[&dense, &sparse], 4, 60.0);
        // Doc 1 is #1 in dense (contribution 1/61) + #2 in sparse (1/62).
        // Doc 3 is #3 in dense (1/63) + #1 in sparse (1/61).
        // Both are strong; either could lead. Doc 4 appears once.
        assert!(fused.iter().any(|(id, _)| *id == 1));
        assert!(fused.iter().any(|(id, _)| *id == 3));
        // Doc 4 must rank below 1 and 3 (single-list appearance).
        let pos = |id: u64| fused.iter().position(|(i, _)| *i == id).unwrap();
        assert!(pos(4) > pos(1));
        assert!(pos(4) > pos(3));
    }

    #[test]
    fn mmr_prefers_diverse_when_lambda_small() {
        // Two clusters: docs 0-2 similar to each other, doc 3 different.
        let candidates: Vec<usize> = vec![0, 1, 2, 3];
        let scores = vec![0.9, 0.85, 0.8, 0.7];
        let similarity = |a: &usize, b: &usize| -> f32 {
            match (a, b) {
                (0..=2, 0..=2) => 0.95,
                (3, 3) => 1.0,
                _ => 0.1,
            }
        };
        // Pure relevance (lambda=1.0) picks top 2: [0, 1].
        let pure_rel = mmr_select(&candidates, &scores, 2, 1.0, similarity);
        assert_eq!(pure_rel, vec![0, 1]);
        // Diversity-heavy (lambda=0.2) should pick 0 then 3, not 0 then 1.
        let diverse = mmr_select(&candidates, &scores, 2, 0.2, similarity);
        assert_eq!(diverse[0], 0);
        assert_eq!(diverse[1], 3);
    }

    #[test]
    fn group_by_keeps_top_n_per_group_and_orders_by_leader() {
        let results = vec![
            (1u64, 0.9f32, "A"),
            (2, 0.85, "A"),
            (3, 0.8, "B"),
            (4, 0.75, "A"), // dropped: A already has 2
            (5, 0.7, "B"),
            (6, 0.65, "C"),
            (7, 0.6, "C"),
            (8, 0.55, "C"), // dropped: C already has 2
        ];
        // 2 per group, 3 total groups → all three groups qualify, but each
        // capped at 2 members.
        let grouped = group_by_top_n(&results, 2, 3);
        assert_eq!(grouped.len(), 6);
        assert!(!grouped.iter().any(|(id, _, _)| *id == 4));
        assert!(!grouped.iter().any(|(id, _, _)| *id == 8));
        // Group A leader (0.9) beats B leader (0.8), etc. First entry
        // should be A's leader.
        assert_eq!(grouped[0].0, 1);
    }

    #[test]
    fn group_by_limits_total_groups() {
        let results = vec![
            (1u64, 0.9f32, "A"),
            (2, 0.85, "B"),
            (3, 0.8, "C"),
            (4, 0.75, "D"),
        ];
        let grouped = group_by_top_n(&results, 1, 2);
        // Only 2 groups (A and B) survive.
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].2, "A");
        assert_eq!(grouped[1].2, "B");
    }
}
