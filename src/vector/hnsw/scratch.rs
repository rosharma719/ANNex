use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;

use super::types::{NodeCandidate, NodeResult, NodeRoutingEntry};

const FILTER_MASK_CACHE_CAP: usize = 64;

pub(crate) struct SearchScratch {
    pub(crate) visited_epoch: Vec<u8>,
    pub(crate) epoch: u8,
    pub(crate) candidate_queue: BinaryHeap<NodeCandidate>,
    pub(crate) result_set: BinaryHeap<NodeResult>,
    pub(crate) routing_pq: BinaryHeap<NodeRoutingEntry>,
    pub(crate) results_pq: BinaryHeap<NodeResult>,
    pub(crate) seed_epoch: Vec<u32>,
    pub(crate) seed_epoch_val: u32,
    pub(crate) seed_list: Vec<usize>,
    pub(crate) temp_epoch: Vec<u32>,
    pub(crate) temp_epoch_val: u32,
    pub(crate) temp_list: Vec<usize>,
    pub(crate) extend_seen_epoch: Vec<u32>,
    pub(crate) extend_seen_val: u32,
    pub(crate) extend_base: Vec<usize>,
    pub(crate) extend_neighbors: Vec<usize>,
    pub(crate) extend_extra: Vec<NodeCandidate>,
    pub(crate) filter_mask_cache: HashMap<u64, Arc<Vec<bool>>>,
    pub(crate) filter_mask_lru: VecDeque<u64>,
    pub(crate) filter_mask_cap: usize,
}

impl Default for SearchScratch {
    fn default() -> Self {
        Self {
            visited_epoch: Vec::new(),
            epoch: 0,
            candidate_queue: BinaryHeap::new(),
            result_set: BinaryHeap::new(),
            routing_pq: BinaryHeap::new(),
            results_pq: BinaryHeap::new(),
            seed_epoch: Vec::new(),
            seed_epoch_val: 0,
            seed_list: Vec::new(),
            temp_epoch: Vec::new(),
            temp_epoch_val: 0,
            temp_list: Vec::new(),
            extend_seen_epoch: Vec::new(),
            extend_seen_val: 0,
            extend_base: Vec::new(),
            extend_neighbors: Vec::new(),
            extend_extra: Vec::new(),
            filter_mask_cache: HashMap::new(),
            filter_mask_lru: VecDeque::new(),
            filter_mask_cap: FILTER_MASK_CACHE_CAP,
        }
    }
}

impl SearchScratch {
    pub(crate) fn next_epoch(&mut self, len: usize) {
        if self.visited_epoch.len() < len {
            self.visited_epoch.resize(len, 0);
        }
        if self.epoch == u8::MAX {
            self.visited_epoch.fill(0);
            self.epoch = 1;
        } else {
            self.epoch = self.epoch.wrapping_add(1);
            if self.epoch == 0 {
                self.epoch = 1;
            }
        }
    }

    pub(crate) fn mark_visited(&mut self, idx: usize) -> bool {
        if self.visited_epoch[idx] == self.epoch {
            false
        } else {
            self.visited_epoch[idx] = self.epoch;
            true
        }
    }

    pub(crate) fn reset_seed(&mut self, len: usize) {
        if self.seed_epoch.len() < len {
            self.seed_epoch.resize(len, 0);
        }
        if self.seed_epoch_val == u32::MAX {
            self.seed_epoch.fill(0);
            self.seed_epoch_val = 1;
        } else {
            self.seed_epoch_val = self.seed_epoch_val.wrapping_add(1);
            if self.seed_epoch_val == 0 {
                self.seed_epoch_val = 1;
            }
        }
        self.seed_list.clear();
    }

    pub(crate) fn reset_temp(&mut self, len: usize) {
        if self.temp_epoch.len() < len {
            self.temp_epoch.resize(len, 0);
        }
        if self.temp_epoch_val == u32::MAX {
            self.temp_epoch.fill(0);
            self.temp_epoch_val = 1;
        } else {
            self.temp_epoch_val = self.temp_epoch_val.wrapping_add(1);
            if self.temp_epoch_val == 0 {
                self.temp_epoch_val = 1;
            }
        }
        self.temp_list.clear();
    }

    pub(crate) fn mark_seed(&mut self, idx: usize) {
        if self.seed_epoch[idx] != self.seed_epoch_val {
            self.seed_epoch[idx] = self.seed_epoch_val;
            self.seed_list.push(idx);
        }
    }

    pub(crate) fn mark_temp(&mut self, idx: usize) {
        if self.temp_epoch[idx] != self.temp_epoch_val {
            self.temp_epoch[idx] = self.temp_epoch_val;
            self.temp_list.push(idx);
        }
    }

    pub(crate) fn reset_extend_seen(&mut self, len: usize) {
        if self.extend_seen_epoch.len() < len {
            self.extend_seen_epoch.resize(len, 0);
        }
        if self.extend_seen_val == u32::MAX {
            self.extend_seen_epoch.fill(0);
            self.extend_seen_val = 1;
        } else {
            self.extend_seen_val = self.extend_seen_val.wrapping_add(1);
            if self.extend_seen_val == 0 {
                self.extend_seen_val = 1;
            }
        }
    }

    /// Returns true if `idx` was newly marked (i.e., not already seen this epoch).
    pub(crate) fn mark_extend_seen(&mut self, idx: usize) -> bool {
        if self.extend_seen_epoch[idx] == self.extend_seen_val {
            false
        } else {
            self.extend_seen_epoch[idx] = self.extend_seen_val;
            true
        }
    }

    pub(crate) fn is_seed(&self, idx: usize) -> bool {
        self.seed_epoch.get(idx).copied().unwrap_or(0) == self.seed_epoch_val
    }

    pub(crate) fn is_temp(&self, idx: usize) -> bool {
        self.temp_epoch.get(idx).copied().unwrap_or(0) == self.temp_epoch_val
    }

    pub(crate) fn get_filter_mask(&mut self, key: u64, len: usize) -> Option<Arc<Vec<bool>>> {
        let mask = self.filter_mask_cache.get(&key)?;
        if mask.len() != len {
            self.filter_mask_cache.remove(&key);
            self.filter_mask_lru.retain(|k| *k != key);
            return None;
        }
        self.filter_mask_lru.retain(|k| *k != key);
        self.filter_mask_lru.push_back(key);
        Some(Arc::clone(mask))
    }

    pub(crate) fn put_filter_mask(&mut self, key: u64, mask: Arc<Vec<bool>>) {
        if self.filter_mask_cap == 0 {
            return;
        }
        if self.filter_mask_cache.contains_key(&key) {
            self.filter_mask_lru.retain(|k| *k != key);
        } else if self.filter_mask_cache.len() >= self.filter_mask_cap
            && let Some(oldest) = self.filter_mask_lru.pop_front()
        {
            self.filter_mask_cache.remove(&oldest);
        }
        self.filter_mask_cache.insert(key, Arc::clone(&mask));
        self.filter_mask_lru.push_back(key);
    }
}

thread_local! {
    pub(crate) static SEARCH_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::default());
}

#[cfg(test)]
mod tests {
    use super::SearchScratch;

    #[test]
    fn visited_epochs_survive_wrap_growth_and_smaller_indexes() {
        let mut scratch = SearchScratch::default();
        scratch.next_epoch(8);
        assert!(scratch.mark_visited(7));
        for _ in 0..254 {
            scratch.next_epoch(2);
            assert!(scratch.mark_visited(0));
            assert!(!scratch.mark_visited(0));
        }
        // Clear the entire allocation on wrap, including a previous larger index.
        scratch.next_epoch(2);
        assert_eq!(scratch.epoch, 1);
        assert!(scratch.mark_visited(7));
        scratch.next_epoch(16);
        assert!(scratch.mark_visited(15));
        for _ in 0..512 {
            scratch.next_epoch(16);
            for idx in 0..16 {
                assert!(scratch.mark_visited(idx));
                assert!(!scratch.mark_visited(idx));
            }
        }
    }
}
