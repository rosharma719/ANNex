use crate::payload_storage::filters::Filter;
use crate::utils::payload::{Payload, PayloadValue};
use crate::utils::types::PointId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Inverted index: field_name -> field_value -> set of PointIds
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct PayloadIndex {
    index: HashMap<String, HashMap<PayloadValue, HashSet<PointId>>>,
    #[serde(skip)]
    revision: u64,
}

impl PayloadIndex {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            revision: 0,
        }
    }

    /// Indexes the payload of a given point.
    pub fn insert(&mut self, point_id: PointId, payload: &Payload) {
        for (key, value) in &payload.0 {
            if !Self::is_indexable(value) {
                continue;
            }

            self.index
                .entry(key.clone())
                .or_default()
                .entry(value.clone())
                .or_default()
                .insert(point_id);
        }
        self.revision = self.revision.wrapping_add(1);
    }

    /// Removes a point's payload from the index.
    pub fn remove(&mut self, point_id: PointId, payload: &Payload) {
        for (key, value) in &payload.0 {
            if !Self::is_indexable(value) {
                continue;
            }

            if let Some(value_map) = self.index.get_mut(key) {
                if let Some(id_set) = value_map.get_mut(value) {
                    id_set.remove(&point_id);
                    if id_set.is_empty() {
                        value_map.remove(value);
                    }
                }
                if value_map.is_empty() {
                    self.index.remove(key);
                }
            }
        }
        self.revision = self.revision.wrapping_add(1);
    }

    /// Returns a set of point IDs that match exactly this key-value pair.
    pub fn query_exact(&self, key: &str, value: &PayloadValue) -> Option<&HashSet<PointId>> {
        if !Self::is_indexable(value) {
            return None;
        }

        self.index.get(key)?.get(value)
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Builds a candidate mask for Match/And/Or filters from the payload index.
    /// For And, it intersects all indexable clauses (ignoring non-indexable ones).
    /// For Or, all clauses must be indexable to return a safe candidate mask.
    pub fn build_filter_mask(
        &self,
        filter: &Filter,
        point_to_idx: &HashMap<PointId, usize>,
        len: usize,
    ) -> Option<Vec<bool>> {
        let idx_len = point_to_idx.len();
        if idx_len == 0 {
            return Some(Vec::new());
        }
        let len = if len == idx_len { len } else { idx_len };
        if len > 10_000_000 {
            return None;
        }
        match filter {
            Filter::Match { key, value } => {
                let Some(ids) = self.query_exact(key, value) else {
                    return Some(vec![false; len]);
                };
                let mut mask = vec![false; len];
                for id in ids {
                    if let Some(&idx) = point_to_idx.get(id) {
                        mask[idx] = true;
                    }
                }
                Some(mask)
            }
            Filter::And(conds) => {
                let mut sets: Vec<&HashSet<PointId>> = Vec::new();
                let mut empty = false;
                for cond in conds {
                    self.collect_and_sets(cond, &mut sets, &mut empty);
                }
                if empty {
                    return Some(vec![false; len]);
                }
                if sets.is_empty() {
                    return None;
                }
                sets.sort_by_key(|s| s.len());
                let smallest = sets[0];
                let rest = &sets[1..];
                let mut mask = vec![false; len];
                for id in smallest {
                    if rest.iter().all(|s| s.contains(id))
                        && let Some(&idx) = point_to_idx.get(id)
                    {
                        mask[idx] = true;
                    }
                }
                Some(mask)
            }
            Filter::Or(conds) => {
                let mut mask = vec![false; len];
                for cond in conds {
                    let ids = self.query_filter_ids(cond)?;
                    for id in ids {
                        if let Some(&idx) = point_to_idx.get(&id) {
                            mask[idx] = true;
                        }
                    }
                }
                Some(mask)
            }
            Filter::Compare { .. } | Filter::Not(_) => None,
        }
    }

    fn collect_and_sets<'a>(
        &'a self,
        filter: &'a Filter,
        sets: &mut Vec<&'a HashSet<PointId>>,
        empty: &mut bool,
    ) {
        match filter {
            Filter::Match { key, value } => match self.query_exact(key, value) {
                Some(ids) => sets.push(ids),
                None => *empty = true,
            },
            Filter::And(conds) => {
                for cond in conds {
                    self.collect_and_sets(cond, sets, empty);
                }
            }
            Filter::Or(_) | Filter::Compare { .. } | Filter::Not(_) => {}
        }
    }

    /// Returns a candidate set for Match/And/Or filters based on the payload index.
    /// For And, it intersects all indexable clauses (ignoring non-indexable ones).
    /// For Or, all clauses must be indexable to return a safe candidate set.
    pub fn query_filter_ids(&self, filter: &Filter) -> Option<HashSet<PointId>> {
        match filter {
            Filter::Match { key, value } => self
                .query_exact(key, value)
                .map(|ids| ids.iter().copied().collect()),
            Filter::And(conds) => {
                let mut sets: Vec<&HashSet<PointId>> = Vec::new();
                let mut empty = false;
                for cond in conds {
                    self.collect_and_sets(cond, &mut sets, &mut empty);
                }
                if empty {
                    return Some(HashSet::new());
                }
                if sets.is_empty() {
                    return None;
                }
                sets.sort_by_key(|s| s.len());
                let smallest = sets[0];
                let rest = &sets[1..];
                let mut acc = HashSet::with_capacity(smallest.len());
                for id in smallest {
                    if rest.iter().all(|s| s.contains(id)) {
                        acc.insert(*id);
                    }
                }
                Some(acc)
            }
            Filter::Or(conds) => {
                let mut acc = HashSet::new();
                for cond in conds {
                    let set = self.query_filter_ids(cond)?;
                    acc.extend(set);
                }
                Some(acc)
            }
            Filter::Compare { .. } | Filter::Not(_) => None,
        }
    }

    fn is_indexable(value: &PayloadValue) -> bool {
        matches!(
            value,
            PayloadValue::Int(_)
                | PayloadValue::Float(_)
                | PayloadValue::Str(_)
                | PayloadValue::Bool(_)
        )
    }

    /// Optional: Returns all point IDs that have any value for the given key.
    pub fn all_for_key(&self, key: &str) -> Option<HashSet<PointId>> {
        self.index.get(key).map(|map| {
            map.values().fold(HashSet::new(), |mut acc, set| {
                acc.extend(set.iter().copied());
                acc
            })
        })
    }
}
