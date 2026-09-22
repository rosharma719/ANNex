use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::payload_storage::filters::{Filter, evaluate_filter};
use crate::payload_storage::stores::PayloadIndex;
use crate::utils::errors::DBError;
use crate::utils::payload::Payload;
use crate::utils::types::{DistanceMetric, PointId, Vector};

use super::HNSWIndex;
use super::config::{
    disable_early_exit, early_exit_patience, filter_entry_candidates, filter_expansion_cap,
    filter_failing_budget, filter_passing_budget, filter_search_logger, filter_seed_log_chunk,
    next_filter_search_seq,
};
use super::scratch::SEARCH_SCRATCH;
use super::stats::{FILTER_SEED_COUNT, FilterSearchLogEntry};
use super::types::{
    NodeCandidate, NodeResult, NodeRoutingEntry, ScoredPoint, SearchRuntimeOptions,
};

impl HNSWIndex {
    fn best_entry_in_mask(&self, mask: &[bool]) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (idx, &allowed) in mask.iter().enumerate() {
            if !allowed
                || idx >= self.len()
                || self.deleted.with(idx, |flag| flag.load(Ordering::Acquire))
            {
                continue;
            }
            let level = self.levels.with(idx, |level| level.load(Ordering::Relaxed)) as usize;
            if best.map_or(true, |(_, best_level)| level > best_level) {
                best = Some((idx, level));
            }
        }
        best.map(|(idx, _)| idx)
    }

    fn top_entries_in_mask(&self, mask: &[bool], cap: usize) -> Vec<usize> {
        if cap == 0 {
            return Vec::new();
        }
        let mut best: Vec<(usize, usize)> = Vec::with_capacity(cap);
        for (idx, &allowed) in mask.iter().enumerate() {
            if !allowed
                || idx >= self.len()
                || self.deleted.with(idx, |flag| flag.load(Ordering::Acquire))
            {
                continue;
            }
            let level = self.levels.with(idx, |level| level.load(Ordering::Relaxed)) as usize;
            if best.len() < cap {
                best.push((idx, level));
                if best.len() == cap {
                    best.sort_by_key(|&(_, lvl)| lvl);
                }
                continue;
            }
            if let Some(&(worst_idx, worst_level)) = best.first() {
                if level > worst_level {
                    let _ = worst_idx;
                    best[0] = (idx, level);
                    best.sort_by_key(|&(_, lvl)| lvl);
                }
            }
        }
        best.sort_by_key(|&(_, lvl)| std::cmp::Reverse(lvl));
        best.into_iter().map(|(idx, _)| idx).collect()
    }

    pub fn find_entry_point_matching_filter(
        &self,
        filter: &Filter,
        payload_index: &PayloadIndex,
        payloads: &HashMap<PointId, Payload>,
    ) -> Option<usize> {
        match filter {
            Filter::Match { key, value } => payload_index.query_exact(key, value).and_then(|ids| {
                ids.iter()
                    .filter_map(|id| self.point_to_idx.get(id).copied())
                    .filter(|&idx| !self.deleted.with(idx, |flag| flag.load(Ordering::Acquire)))
                    .max_by_key(|&idx| self.levels.with(idx, |level| level.load(Ordering::Relaxed)))
            }),
            Filter::And(conds) | Filter::Or(conds) => {
                let mut best: Option<(usize, usize)> = None;

                for cond in conds {
                    if let Some(id) =
                        self.find_entry_point_matching_filter(cond, payload_index, payloads)
                    {
                        let level =
                            self.levels.with(id, |level| level.load(Ordering::Relaxed)) as usize;
                        if best.map_or(true, |(_, l)| level > l) {
                            best = Some((id, level));
                        }
                    }
                }

                best.map(|(id, _)| id)
            }
            Filter::Not(_) | Filter::Compare { .. } => None,
        }
    }

    fn extract_match_filter(&self, orig: Option<&Filter>) -> Option<Filter> {
        match orig {
            None => None,
            Some(Filter::Match { key, value }) => Some(Filter::Match {
                key: key.clone(),
                value: value.clone(),
            }),
            Some(Filter::And(conds)) => {
                let mut keep = Vec::new();
                for c in conds {
                    if let Filter::Match { .. } = c {
                        keep.push(c.clone());
                    }
                }
                match keep.len() {
                    0 => None,
                    1 => Some(keep.into_iter().next().unwrap()),
                    _ => Some(Filter::And(keep)),
                }
            }
            _ => None,
        }
    }

    pub fn in_place_filtered_search(
        &self,
        query: &Vector,
        top_k: usize,
        opts: &SearchRuntimeOptions,
        payloads: &HashMap<PointId, Payload>,
        payload_index: &PayloadIndex,
        full_filter: Option<&Filter>,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        self.validate_dim(query)?;

        let (normalize_query, use_normalize_score) = match self.metric {
            DistanceMetric::Cosine => (true, true),
            DistanceMetric::Dot => (false, true),
            DistanceMetric::Euclidean => (false, false),
        };

        let prepared_query = if normalize_query {
            self.maybe_normalize(query)
        } else {
            query.clone()
        };

        let query_for_greedy = &prepared_query;
        let query_for_search = &prepared_query;

        let match_filter = self.extract_match_filter(full_filter);
        let mut allowed_mask: Option<Arc<Vec<bool>>> = None;

        if let Some(f) = full_filter {
            let mask_key = {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                f.hash(&mut hasher);
                payload_index.revision().hash(&mut hasher);
                self.len().hash(&mut hasher);
                hasher.finish()
            };
            if let Some(mask) = SEARCH_SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                scratch.get_filter_mask(mask_key, self.len())
            }) {
                allowed_mask = Some(mask);
            }
            if self.exact_fallback_enabled
                && let Some(candidate_ids) = payload_index.query_filter_ids(f)
            {
                if candidate_ids.is_empty() {
                    return Ok(vec![]);
                }
                if candidate_ids.len() <= self.exact_fallback_threshold {
                    let mut out = Vec::with_capacity(candidate_ids.len().min(top_k));
                    for id in candidate_ids {
                        let Some(idx) = self.idx_of(id) else {
                            continue;
                        };
                        if self.deleted.with(idx, |b| b.load(Ordering::Acquire)) {
                            continue;
                        }
                        let Some(payload) = payloads.get(&id) else {
                            continue;
                        };
                        if !evaluate_filter(f, payload)? {
                            continue;
                        }
                        let Some(vec) = self.get_vector_by_idx(idx) else {
                            continue;
                        };
                        let raw = self.fast_score(query_for_search, vec);
                        let sk = if use_normalize_score {
                            self.normalize_score(raw)
                        } else {
                            raw
                        };
                        out.push(ScoredPoint {
                            id,
                            raw_score: raw,
                            sort_key: sk,
                        });
                    }
                    out.sort_by(|a, b| a.sort_key.partial_cmp(&b.sort_key).unwrap());
                    out.truncate(top_k);
                    return Ok(out);
                }
            }
            if allowed_mask.is_none()
                && let Some(mask) =
                    payload_index.build_filter_mask(f, &self.point_to_idx, self.point_to_idx.len())
            {
                if !mask.iter().any(|v| *v) {
                    return Ok(vec![]);
                }
                let arc = Arc::new(mask);
                SEARCH_SCRATCH.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    scratch.put_filter_mask(mask_key, Arc::clone(&arc));
                });
                allowed_mask = Some(arc);
            }
        }

        let (entry_point, current_max_level) = self.entry_level();
        let mut entry = match entry_point {
            Some(idx) => {
                if let Some(mask) = allowed_mask.as_ref() {
                    if mask.get(idx).copied().unwrap_or(false)
                        && !self.deleted.with(idx, |b| b.load(Ordering::Acquire))
                    {
                        idx
                    } else {
                        match self.best_entry_in_mask(mask) {
                            Some(best) => best,
                            None => return Ok(vec![]),
                        }
                    }
                } else if let Some(f) = full_filter {
                    let entry_id = self.point_id(idx);
                    if let Some(p) = payloads.get(&entry_id) {
                        if evaluate_filter(f, p)? {
                            idx
                        } else {
                            self.find_entry_point_matching_filter(f, payload_index, payloads)
                                .unwrap_or(idx)
                        }
                    } else {
                        self.find_entry_point_matching_filter(f, payload_index, payloads)
                            .unwrap_or(idx)
                    }
                } else {
                    idx
                }
            }
            None => {
                if let Some(mask) = allowed_mask.as_ref() {
                    match self.best_entry_in_mask(mask) {
                        Some(best) => best,
                        None => return Ok(vec![]),
                    }
                } else if let Some(f) = full_filter {
                    match self.find_entry_point_matching_filter(f, payload_index, payloads) {
                        Some(id) => id,
                        None => return Ok(vec![]),
                    }
                } else {
                    return Ok(vec![]);
                }
            }
        };

        for level in (1..=current_max_level).rev() {
            entry = self.greedy_search_layer_with_filter(
                query_for_greedy,
                entry,
                level,
                payloads,
                if allowed_mask.is_some() {
                    full_filter
                } else {
                    match_filter.as_ref()
                },
            )?;
        }

        let result = SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.next_epoch(self.len());
            scratch.routing_pq.clear();
            scratch.results_pq.clear();
            let mut visited_count = 0usize;

            let log_seed = std::env::var("VECTORDB_LOG_FILTER_SEED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(false);
            let allow_early_exit = self.metric != DistanceMetric::Dot && !disable_early_exit();
            let patience_limit = if allow_early_exit {
                opts.early_exit_patience
                    .unwrap_or_else(early_exit_patience)
            } else {
                0
            };
            let mut no_improve_streak = 0usize;
            let mut early_exit = false;

            let search_start = std::time::Instant::now();
            let dist0 = self.fast_score(query_for_search, self.get_vector_by_idx(entry).unwrap());
            let sk0 = if use_normalize_score {
                self.normalize_score(dist0)
            } else {
                dist0
            };
            let first = NodeCandidate {
                idx: entry,
                raw_score: dist0,
                sort_key: sk0,
            };
            let mut filter_checked = 0usize;
            let mut filter_passed = 0usize;
            let mut seeds_popped = 0usize;
            let ef_search = opts.ef_search.unwrap_or(self.ef).max(top_k);
            let max_expansions = filter_expansion_cap().unwrap_or(ef_search);
            let result_cap = ef_search.max(top_k);
            let mut routing_popped_total = 0usize;
            let mut routing_popped_passing = 0usize;
            let mut results_inserted = 0usize;
            let passing_budget = filter_passing_budget(self.m);
            let failing_budget = filter_failing_budget(self.m);

            let entry_passes = full_filter.is_none_or(|f| {
                filter_checked += 1;
                let entry_id = self.point_id(entry);
                let ok = payloads
                    .get(&entry_id)
                    .is_some_and(|p| evaluate_filter(f, p).unwrap_or(false));
                if ok {
                    filter_passed += 1;
                }
                ok
            });
            scratch.routing_pq.push(NodeRoutingEntry {
                node: first.clone(),
                passes_filter: entry_passes,
                budget: if entry_passes { passing_budget } else { failing_budget },
            });
            if entry_passes {
                scratch.results_pq.push(NodeResult(first));
            }
            if scratch.mark_visited(entry) {
                visited_count += 1;
            }

            let mut worst = scratch.results_pq.peek().map(|rp| rp.0.sort_key).unwrap_or(f32::MAX);

            if let Some(mask) = allowed_mask.as_ref() {
                let cap = filter_entry_candidates().unwrap_or(ef_search).max(1);
                let entry_candidates = self.top_entries_in_mask(mask, cap);
                for idx in entry_candidates {
                    if idx == entry {
                        continue;
                    }
                    if !scratch.mark_visited(idx) {
                        continue;
                    }
                    visited_count += 1;
                    let Some(vec) = self.get_vector_by_idx(idx) else { continue; };
                    let d = self.fast_score(query_for_search, vec);
                    let sk = if use_normalize_score { self.normalize_score(d) } else { d };
                    let sp = NodeCandidate {
                        idx,
                        raw_score: d,
                        sort_key: sk,
                    };
                    let passes = full_filter.is_none_or(|f| {
                        filter_checked += 1;
                        let id = self.point_id(idx);
                        let ok = payloads
                            .get(&id)
                            .is_some_and(|p| evaluate_filter(f, p).unwrap_or(false));
                        if ok {
                            filter_passed += 1;
                        }
                        ok
                    });
                    let budget = if passes { passing_budget } else { failing_budget };
                    if budget > 0 {
                        scratch.routing_pq.push(NodeRoutingEntry {
                            node: sp.clone(),
                            passes_filter: passes,
                            budget,
                        });
                    }
                    if passes {
                        scratch.results_pq.push(NodeResult(sp));
                        results_inserted += 1;
                        if scratch.results_pq.len() > result_cap {
                            scratch.results_pq.pop();
                        }
                        if scratch.results_pq.len() >= top_k {
                            worst = scratch.results_pq.peek().unwrap().0.sort_key;
                        }
                    }
                }
            }

            let seed_limit = ef_search;
            let seed_len = self.len();
            fn collect_match_ids(
                filter: &Filter,
                idx: &PayloadIndex,
                scratch: &mut super::scratch::SearchScratch,
                map: &HashMap<PointId, usize>,
                len: usize,
                primary: bool,
            ) {
                match filter {
                    Filter::Match { key, value } => {
                        if let Some(ids) = idx.query_exact(key, value) {
                            for id in ids {
                                if let Some(&node_idx) = map.get(id) {
                                    if primary {
                                        scratch.mark_seed(node_idx);
                                    } else {
                                        scratch.mark_temp(node_idx);
                                    }
                                }
                            }
                        }
                    }
                    Filter::Or(parts) => {
                        for p in parts {
                            if primary {
                                scratch.reset_temp(len);
                                collect_match_ids(p, idx, scratch, map, len, false);
                                let other = scratch.temp_list.clone();
                                for id in other {
                                    scratch.mark_seed(id);
                                }
                            } else {
                                scratch.reset_seed(len);
                                collect_match_ids(p, idx, scratch, map, len, true);
                                let other = scratch.seed_list.clone();
                                for id in other {
                                    scratch.mark_temp(id);
                                }
                            }
                        }
                    }
                    Filter::And(parts) => {
                        if parts.is_empty() {
                            return;
                        }
                        if primary {
                            scratch.reset_seed(len);
                            collect_match_ids(&parts[0], idx, scratch, map, len, true);
                            for p in parts.iter().skip(1) {
                                scratch.reset_temp(len);
                                collect_match_ids(p, idx, scratch, map, len, false);
                                let retained = std::mem::take(&mut scratch.seed_list);
                                scratch.reset_seed(len);
                                for id in retained {
                                    if scratch.is_temp(id) {
                                        scratch.mark_seed(id);
                                    }
                                }
                            }
                        } else {
                            scratch.reset_temp(len);
                            collect_match_ids(&parts[0], idx, scratch, map, len, false);
                            for p in parts.iter().skip(1) {
                                scratch.reset_seed(len);
                                collect_match_ids(p, idx, scratch, map, len, true);
                                let retained = std::mem::take(&mut scratch.temp_list);
                                scratch.reset_temp(len);
                                for id in retained {
                                    if scratch.is_seed(id) {
                                        scratch.mark_temp(id);
                                    }
                                }
                            }
                        }
                    }
                    Filter::Not(_) | Filter::Compare { .. } => {}
                }
            }
            let use_seeds = std::env::var("VECTORDB_ENABLE_FILTER_SEEDS")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(false);
            if use_seeds
                && let Some(f) = full_filter
            {
                scratch.reset_seed(seed_len);
                scratch.reset_temp(seed_len);
                collect_match_ids(f, payload_index, &mut scratch, &self.point_to_idx, seed_len, true);
            }
            let seed_pool_size = scratch.seed_list.len();
            let mut seeds_added = 0usize;
            let mut seeds_accepted = 0usize;
            if use_seeds && !scratch.seed_list.is_empty() {
                let seed_ids = scratch.seed_list.clone();
                for idx in seed_ids {
                    if seeds_added >= seed_limit {
                        break;
                    }
                    if allowed_mask
                        .as_ref()
                        .is_some_and(|mask| !mask.get(idx).copied().unwrap_or(false))
                    {
                        continue;
                    }
                    if self.deleted.with(idx, |b| b.load(Ordering::Acquire))
                        || !scratch.mark_visited(idx)
                    {
                        continue;
                    }
                    visited_count += 1;
                    let Some(vec) = self.get_vector_by_idx(idx) else { continue; };
                    let d = self.fast_score(query_for_search, vec);
                    let sk = if use_normalize_score { self.normalize_score(d) } else { d };
                    let sp = NodeCandidate {
                        idx,
                        raw_score: d,
                        sort_key: sk,
                    };
                    let passes = full_filter.is_none_or(|f| {
                        filter_checked += 1;
                        let id = self.point_id(idx);
                        let ok = payloads
                            .get(&id)
                            .is_some_and(|p| evaluate_filter(f, p).unwrap_or(false));
                        if ok {
                            filter_passed += 1;
                        }
                        ok
                    });
                    let budget = if passes { passing_budget } else { failing_budget };
                    scratch.routing_pq.push(NodeRoutingEntry {
                        node: sp.clone(),
                        passes_filter: passes,
                        budget,
                    });
                    if passes {
                        scratch.results_pq.push(NodeResult(sp));
                        seeds_accepted += 1;
                    }
                    seeds_added += 1;
                }
            }

            let mut expansions = 0usize;
            let mut stop_reason = "queue_exhausted".to_string();

            while let Some(curr) = scratch.routing_pq.pop() {
                expansions += 1;
                if scratch.is_seed(curr.node.idx) {
                    seeds_popped += 1;
                }
                routing_popped_total += 1;
                if curr.passes_filter {
                    routing_popped_passing += 1;
                }
                if allow_early_exit && patience_limit > 0 && scratch.results_pq.len() >= ef_search {
                    if let Some(next) = scratch.routing_pq.peek() {
                        if next.node.sort_key > worst {
                            no_improve_streak += 1;
                        } else {
                            no_improve_streak = 0;
                        }
                    } else {
                        no_improve_streak += 1;
                    }
                    if no_improve_streak > patience_limit {
                        early_exit = true;
                        stop_reason = "early_exit_patience".to_string();
                        break;
                    }
                }

                if expansions >= max_expansions {
                    stop_reason = "max_expansions".to_string();
                    break;
                }
                if scratch.results_pq.len() >= top_k && curr.node.sort_key > worst {
                    stop_reason = "pruned_by_worst".to_string();
                    break;
                }

                if curr.budget <= 1 {
                    continue;
                }

                if let Some(neighs) = self.layer_neighbors(0, curr.node.idx) {
                    for &nb in neighs.iter() {
                        if allowed_mask
                            .as_ref()
                            .is_some_and(|mask| !mask.get(nb).copied().unwrap_or(false))
                        {
                            continue;
                        }
                        if self.deleted.with(nb, |b| b.load(Ordering::Acquire))
                            || !scratch.mark_visited(nb)
                        {
                            continue;
                        }
                        visited_count += 1;

                        let d = self.fast_score(query_for_search, self.get_vector_by_idx(nb).unwrap());
                        let sk = if use_normalize_score { self.normalize_score(d) } else { d };
                        let sp = NodeCandidate {
                            idx: nb,
                            raw_score: d,
                            sort_key: sk,
                        };

                        let passes = full_filter.is_none_or(|f| {
                            filter_checked += 1;
                            let id = self.point_id(nb);
                            let ok = payloads
                                .get(&id)
                                .is_some_and(|p| evaluate_filter(f, p).unwrap_or(false));
                            if ok {
                                filter_passed += 1;
                            }
                            ok
                        });
                        let budget = if passes { passing_budget } else { failing_budget };
                        if budget > 0 {
                            scratch.routing_pq.push(NodeRoutingEntry {
                                node: sp.clone(),
                                passes_filter: passes,
                                budget,
                            });
                        }
                        if passes {
                            scratch.results_pq.push(NodeResult(sp));
                            results_inserted += 1;
                            if scratch.results_pq.len() > result_cap {
                                scratch.results_pq.pop();
                            }
                            if scratch.results_pq.len() >= top_k {
                                worst = scratch.results_pq.peek().unwrap().0.sort_key;
                            }
                        }
                    }
                }
            }

            let mut out = std::mem::take(&mut scratch.results_pq)
                .into_sorted_vec()
                .into_iter()
                .map(|rp| {
                    let cand = rp.0;
                    ScoredPoint {
                        id: self.point_id(cand.idx),
                        raw_score: cand.raw_score,
                        sort_key: cand.sort_key,
                    }
                })
                .collect::<Vec<_>>();
            out.truncate(top_k);

            let seeds_in_results = out
                .iter()
                .filter(|sp| self.idx_of(sp.id).is_some_and(|idx| scratch.is_seed(idx)))
                .count();

            if log_seed && full_filter.is_some() {
                let chunk = filter_seed_log_chunk();
                let should_emit = FILTER_SEED_COUNT.with(|c| {
                    let mut v = c.borrow_mut();
                    *v += 1;
                    *v % chunk == 0
                });
                if should_emit {
                    let msg = format!(
                        "[filter_search_seed] seeds_pool={} seeds_added={} seeds_accepted={} seeds_in_results={} final_results={}",
                        seed_pool_size,
                        seeds_added,
                        seeds_accepted,
                        seeds_in_results,
                        out.len()
                    );
                    log::info!(target: "hnsw", "{}", msg);
                }
            }

            if let Some(log) = filter_search_logger() {
                let best_routing_dist_at_exit = scratch
                    .routing_pq
                    .peek()
                    .map(|sp| sp.node.sort_key)
                    .unwrap_or(f32::MAX);
                let results_pq_peek_dist = worst;
                let entry = FilterSearchLogEntry {
                    seq: next_filter_search_seq(),
                    ef_search,
                    top_k,
                    filter_present: full_filter.is_some(),
                    seeds_pool: seed_pool_size,
                    seeds_added,
                    seeds_accepted,
                    seeds_in_results,
                    seeds_popped,
                    filter_checked,
                    filter_passed,
                    visited: visited_count,
                    expansions,
                    max_expansions,
                    results_len: out.len(),
                    stop_reason,
                    patience_limit,
                    early_exit,
                    routing_popped_total,
                    routing_popped_passing,
                    routing_popped_failing: routing_popped_total.saturating_sub(routing_popped_passing),
                    results_inserted,
                    results_pq_peek_dist,
                    best_routing_dist_at_exit,
                    elapsed_ms: search_start.elapsed().as_secs_f64() * 1000.0,
                };
                if let Ok(mut writer) = log.lock()
                    && serde_json::to_writer(&mut *writer, &entry).is_ok()
                {
                    let _ = writer.write_all(b"\n");
                }
            }

            Ok(out)
        });

        result
    }
}
