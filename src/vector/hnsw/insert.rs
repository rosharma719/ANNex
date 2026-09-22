use std::collections::HashSet;
use std::io::Write;
use std::time::Instant;

use rand::seq::IteratorRandom;
use serde::Serialize;

use crate::payload_storage::filters::{Filter, evaluate_filter};
use crate::payload_storage::stores::PayloadIndex;
use crate::utils::errors::DBError;
use crate::utils::payload::{Payload, PayloadValue};
use crate::utils::types::{DistanceMetric, PointId, Vector};

use std::cmp::Ordering;

use super::HNSWIndex;
use super::config::{
    FILTER_EDGE_LOG_CHUNK, VERBOSE, diversity_alpha_for_level, diversity_prune_floor,
    enforce_neighbor_caps, insert_trace_logger, next_insert_trace_seq, trace_every,
};
use super::scratch::SEARCH_SCRATCH;
use super::stats::{FILTER_EDGE_STATS, FILTER_EDGE_TOTAL_KEYS, FilterEdgeAgg};
use super::types::{NodeCandidate, SearchRuntimeOptions};

#[derive(Serialize)]
struct InsertTraceEntry {
    insert_id: u64,
    point_id: PointId,
    level: usize,
    current_max_level: usize,
    layer: usize,
    entry_point: Option<PointId>,
    current_entry: PointId,
    candidates: usize,
    neighbors: usize,
    best_neighbor: Option<PointId>,
    elapsed_ms: f64,
}

fn log_insert_trace(entry: &InsertTraceEntry) {
    let Some(logger) = insert_trace_logger() else {
        return;
    };
    if let Ok(mut guard) = logger.lock()
        && serde_json::to_writer(&mut *guard, entry).is_ok()
    {
        let _ = guard.write_all(b"\n");
        let _ = guard.flush();
    }
}

impl HNSWIndex {
    pub fn insert(&mut self, point_id: PointId, vector: Vector) -> Result<(), DBError> {
        if self.point_to_idx.contains_key(&point_id) {
            if VERBOSE {
                log::debug!(target: "vector::hnsw", "[INSERT] Point {} already exists. Skipping.", point_id);
            }
            return Ok(());
        }

        self.validate_dim(&vector)?;
        let trace_id = next_insert_trace_seq();
        let trace_mod = trace_every() as u64;
        let trace_enabled = insert_trace_logger().is_some() && trace_id.is_multiple_of(trace_mod);
        let trace_start = if trace_enabled {
            Some(Instant::now())
        } else {
            None
        };

        let level = self.assign_random_level();
        let vec = self.maybe_normalize(&vector);
        let idx = self.register_node(point_id, vec, level);
        let nodes_len = self.len();
        self.ensure_level_capacity(level, nodes_len);
        self.extend_layers_for_new_node(nodes_len);

        for l in 0..=level {
            self.layers[l].with(idx, |lock| lock.write().push(idx));
        }

        if self.entry_point().is_none() {
            if VERBOSE {
                log::debug!(
                    target: "vector::hnsw",
                    "[INSERT] First point. Setting entry point to {} at level {}",
                    point_id,
                    level
                );
            }
            self.allocate_entry_point(idx, level);
            return Ok(());
        }

        let (entry_point, current_max_level) = self.entry_level();
        let mut current_entry = if let Some(ep) = entry_point {
            if self
                .deleted
                .with(ep, |b| b.load(::std::sync::atomic::Ordering::Acquire))
            {
                self.find_highest_level_entry_point().unwrap_or(idx)
            } else {
                ep
            }
        } else {
            self.find_highest_level_entry_point().unwrap_or(idx)
        };

        for l in ((level + 1)..=current_max_level).rev() {
            current_entry =
                self.greedy_search_layer_unfiltered(self.vector_slice(idx), current_entry, l);
        }

        for l in (0..=level).rev() {
            let use_norm =
                self.metric == DistanceMetric::Cosine || self.metric == DistanceMetric::Dot;
            // Build-time: don't apply candidate-pool expansion; ef_construct already controls quality.
            let opts = SearchRuntimeOptions {
                expansion_mult: Some(1),
                ..SearchRuntimeOptions::default()
            };
            let (mut candidates, _) = self.search_layer_unfiltered(
                self.vector_slice(idx),
                &[current_entry],
                l,
                self.ef_construct,
                &opts,
                use_norm,
                None,
                None,
            )?;
            Self::pre_sort_candidates(&mut candidates);

            // Extend the candidate pool with graph-neighbors of the top-m candidates
            // (HNSW Heuristic 2 "extend candidates", §4 of the original paper).
            self.extend_candidates(&mut candidates, idx, l, use_norm);

            let m_for_layer = if l == 0 { self.m0 } else { self.m };
            let neighbors: Vec<usize> =
                self.select_diverse_neighbors(&candidates, m_for_layer, use_norm, l);
            // `neighbors` is already sorted by distance-from-idx (candidates were pre-sorted
            // and select_diverse preserves order). The self-link has distance 0 so it belongs
            // at position 0. Constructing the list in sorted order avoids a redundant resort.
            {
                let mut linked = Vec::with_capacity(neighbors.len() + 1);
                linked.push(idx);
                for &n in &neighbors {
                    if n != idx {
                        linked.push(n);
                    }
                }
                self.layers[l].with(idx, |lock| *lock.write() = linked.clone());
                if l == 0 {
                    let src = self.vector_slice(idx).to_vec();
                    let dists: Vec<f32> = linked
                        .iter()
                        .map(|&n| {
                            if n == idx {
                                return 0.0;
                            }
                            self.fast_score(&src, self.vector_slice(n))
                        })
                        .collect();
                    self.edge_dists_l0.with(idx, |lock| *lock.write() = dists);
                }
            }
            if enforce_neighbor_caps() {
                self.cap_layer_neighbors(l, idx);
            }

            for &n in &neighbors {
                let Some(n_vec) = self.get_vector_by_idx(n) else {
                    continue;
                };
                let n_vec = n_vec.to_vec();
                let new_score =
                    self.normalize_score(self.fast_score(&n_vec, self.vector_slice(idx)));
                self.layers[l].with(n, |lock| {
                    let mut nb_list = lock.write();
                    // idx is freshly allocated; it can't already be in nb_list.
                    let pos = nb_list.partition_point(|&nb| {
                        self.get_vector_by_idx(nb)
                            .map(|v| self.normalize_score(self.fast_score(&n_vec, v)) <= new_score)
                            .unwrap_or(true)
                    });
                    nb_list.insert(pos, idx);
                    if l == 0 {
                        let n_dists: Vec<f32> = nb_list
                            .iter()
                            .map(|&nb| {
                                if nb == n {
                                    return 0.0;
                                }
                                self.fast_score(&n_vec, self.vector_slice(nb))
                            })
                            .collect();
                        self.edge_dists_l0.with(n, |ed| *ed.write() = n_dists);
                    }
                });
                if enforce_neighbor_caps() {
                    self.cap_layer_neighbors(l, n);
                }
            }

            if let Some(&best) = neighbors.first() {
                current_entry = best;
            }

            if trace_enabled {
                let (trace_entry, trace_max_level) = self.entry_level();
                let entry = InsertTraceEntry {
                    insert_id: trace_id,
                    point_id,
                    level,
                    current_max_level: trace_max_level,
                    layer: l,
                    entry_point: trace_entry.map(|ep| self.point_id(ep)),
                    current_entry: self.point_id(current_entry),
                    candidates: candidates.len(),
                    neighbors: neighbors.len(),
                    best_neighbor: neighbors.first().map(|&n| self.point_id(n)),
                    elapsed_ms: trace_start
                        .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0),
                };
                log_insert_trace(&entry);
            }
        }

        if level > current_max_level {
            if VERBOSE {
                log::debug!(
                    target: "vector::hnsw",
                    "[INSERT] Promoting {} to new entry point at level {}",
                    point_id,
                    level
                );
            }
            self.store_entry_level(idx, level);
        }

        Ok(())
    }

    /// Sort a batch of `(id, vector)` pairs by descending estimated Local Intrinsic
    /// Dimensionality (LID). High-LID (hub/outlier) vectors are inserted first so they
    /// propagate to upper layers, improving long-range routing and recall without changing
    /// query-time behavior.
    ///
    /// LID estimation: for each vector, sample `min(32, n-1)` other vectors from the batch,
    /// compute cosine distances, sort, and apply the Hill estimator:
    ///   LID ≈ −k / Σ_{i=1..k} log(d_i / d_k)
    /// where `d_1 ≤ d_2 ≤ … ≤ d_k` are the k-nearest distances from the sample.
    pub fn sort_by_lid(entries: &mut Vec<(u64, Vec<f32>)>) {
        let n = entries.len();
        if n < 4 {
            return;
        }
        let sample_k = 16usize.min(n - 1);
        let sample_n = 32usize.min(n - 1);

        // Pre-normalize for cosine similarity (works for all metrics as an approximation).
        let normed: Vec<Vec<f32>> = entries
            .iter()
            .map(|(_, v)| {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    v.iter().map(|x| x / norm).collect()
                } else {
                    v.clone()
                }
            })
            .collect();

        let lids: Vec<f32> = (0..n)
            .map(|i| {
                let stride = n / sample_n + 1;
                let mut dists: Vec<f32> = (0..sample_n)
                    .map(|s| {
                        let j = (i + 1 + s * stride) % n;
                        let dot: f32 = normed[i]
                            .iter()
                            .zip(normed[j].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        (1.0 - dot).max(0.0)
                    })
                    .collect();
                dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                dists.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
                let k = dists.len().min(sample_k);
                if k < 2 {
                    return 0.0;
                }
                let d_k = dists[k - 1];
                if d_k < 1e-9 {
                    return 0.0;
                }
                let sum_log: f32 = dists[..k].iter().map(|&d| (d / d_k).max(1e-9).ln()).sum();
                if sum_log.abs() < 1e-9 {
                    0.0
                } else {
                    -(k as f32) / sum_log
                }
            })
            .collect();

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            lids[b]
                .partial_cmp(&lids[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sorted = Vec::with_capacity(n);
        for i in order {
            sorted.push(entries[i].clone());
        }
        *entries = sorted;
    }

    /// Insert a batch of points using concurrent search+link.
    ///
    /// Phase 1 (sequential): allocate every node — extend vecs, register IDs, push self-link.
    /// Phase 2 (parallel):   each thread searches for its node's neighbors and writes back-edges,
    ///                        protected by the per-node `RwLock<Vec<usize>>` on each neighbor list.
    /// Phase 3 (sequential): update the entry point if any new node reached a higher level.
    ///
    /// New nodes allocated in the same batch do not see each other as candidates during Phase 2
    /// (they have no graph edges yet), which is the standard trade-off for batch HNSW builds.
    pub fn par_insert_batch(&mut self, entries: &[(PointId, Vector)]) -> Result<usize, DBError> {
        use super::config::lid_sort_enabled;
        // LID sort: build a permuted view of entries without changing the signature.
        // When enabled, high-LID vectors are allocated first so they reach upper layers.
        let sort_order: Vec<usize> = if lid_sort_enabled() && entries.len() > 3 {
            let mut owned: Vec<(u64, Vec<f32>)> =
                entries.iter().map(|(id, v)| (*id, v.clone())).collect();
            Self::sort_by_lid(&mut owned);
            let id_to_orig: std::collections::HashMap<u64, usize> = entries
                .iter()
                .enumerate()
                .map(|(i, (id, _))| (*id, i))
                .collect();
            owned.iter().map(|(id, _)| id_to_orig[id]).collect()
        } else {
            (0..entries.len()).collect()
        };

        let mut node_infos: Vec<(usize, usize)> = Vec::new(); // (idx, level)
        for i in 0..entries.len() {
            let (point_id, vector) = &entries[sort_order[i]];
            if self.point_to_idx.contains_key(point_id) {
                continue;
            }
            self.validate_dim(vector)?;
            let level = self.assign_random_level();
            let normalized = self.maybe_normalize(vector);
            let idx = self.register_node(*point_id, normalized, level);
            let nodes_len = self.len();
            self.ensure_level_capacity(level, nodes_len);
            self.extend_layers_for_new_node(nodes_len);
            for l in 0..=level {
                self.layers[l].with(idx, |lock| lock.write().push(idx));
            }
            node_infos.push((idx, level));
        }

        let n_new = node_infos.len();
        if n_new == 0 {
            return Ok(0);
        }

        // Determine the initial entry point for Phase 2 searches.
        // When the graph is empty, the first node in the batch becomes the entry. It has no
        // neighbors to connect to (only its self-link), so search_and_link is skipped for it.
        // Every other node in the batch CAN search from that entry and connect to it, so the
        // batch builds up real connectivity rather than every node being an isolated self-link.
        let (first_batch_node, skip_first) = match self.entry_point() {
            Some(ep) => (ep, false),
            None => {
                let (first_idx, first_level) = node_infos[0];
                self.store_entry_level(first_idx, first_level);
                (first_idx, true) // skip linking first node — nothing to connect to yet
            }
        };
        let initial_entry = first_batch_node;
        let link_slice = if skip_first {
            &node_infos[1..]
        } else {
            &node_infos[..]
        };

        // Phase 2: search + link in parallel.
        // Spawn exactly `parallelism` threads regardless of batch size. Each thread processes
        // its slice of link_slice sequentially, avoiding per-entry thread lifecycle costs.
        let n_link = link_slice.len();
        let first_error: std::sync::Mutex<Option<DBError>> = std::sync::Mutex::new(None);
        if n_link > 0 {
            let parallelism = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(n_link);
            let chunk_size = (n_link + parallelism - 1) / parallelism;
            let self_ref: &Self = self;
            let error_ref = &first_error;
            std::thread::scope(|s| {
                for chunk in link_slice.chunks(chunk_size) {
                    s.spawn(move || {
                        for &(idx, level) in chunk {
                            if let Err(e) = self_ref.search_and_link(idx, level, initial_entry) {
                                let mut slot = error_ref.lock().unwrap();
                                if slot.is_none() {
                                    *slot = Some(e);
                                }
                                return;
                            }
                        }
                    });
                }
            });
        }
        if let Some(e) = first_error.into_inner().unwrap() {
            return Err(e);
        }

        // Phase 3: promote entry point.
        for &(idx, level) in &node_infos {
            if level > self.current_max_level() {
                self.store_entry_level(idx, level);
            }
        }

        Ok(n_new)
    }

    /// Search for the best neighbors of node `idx` (already allocated) and write bidirectional
    /// edges. Designed to run concurrently with other nodes via per-node `RwLock` guards.
    fn search_and_link(
        &self,
        idx: usize,
        level: usize,
        initial_entry: usize,
    ) -> Result<(), DBError> {
        if !self.deleted.with(initial_entry, |b| {
            !b.load(::std::sync::atomic::Ordering::Acquire)
        }) {
            return Ok(());
        }
        let mut current_entry = initial_entry;
        let use_norm = self.metric == DistanceMetric::Cosine || self.metric == DistanceMetric::Dot;
        // Build-time: don't apply candidate-pool expansion; ef_construct already controls quality.
        let opts = SearchRuntimeOptions {
            expansion_mult: Some(1),
            ..SearchRuntimeOptions::default()
        };

        // Greedy descent above the insertion level.
        for l in ((level + 1)..=self.current_max_level()).rev() {
            current_entry =
                self.greedy_search_layer_unfiltered(self.vector_slice(idx), current_entry, l);
        }

        for l in (0..=level).rev() {
            let (mut candidates, _) = self.search_layer_unfiltered(
                self.vector_slice(idx),
                &[current_entry],
                l,
                self.ef_construct,
                &opts,
                use_norm,
                None,
                None,
            )?;
            Self::pre_sort_candidates(&mut candidates);

            // Extend candidates: top-m only (matches sequential insert path).
            self.extend_candidates(&mut candidates, idx, l, use_norm);

            let m_for_layer = if l == 0 { self.m0 } else { self.m };
            let neighbors = self.select_diverse_neighbors(&candidates, m_for_layer, use_norm, l);

            // Write new node's neighbor list, sorted with self-link at position 0.
            {
                let mut linked = Vec::with_capacity(neighbors.len() + 1);
                linked.push(idx);
                for &n in &neighbors {
                    if n != idx {
                        linked.push(n);
                    }
                }
                self.layers[l].with(idx, |lock| *lock.write() = linked.clone());
                if l == 0 {
                    let src = self.vector_slice(idx).to_vec();
                    let dists: Vec<f32> = linked
                        .iter()
                        .map(|&n| {
                            if n == idx {
                                return 0.0;
                            }
                            self.fast_score(&src, self.vector_slice(n))
                        })
                        .collect();
                    self.edge_dists_l0.with(idx, |lock| *lock.write() = dists);
                }
            }

            // Write back-edges into neighbors, sorted by distance from each neighbor.
            for &n in &neighbors {
                let Some(n_vec) = self.get_vector_by_idx(n) else {
                    continue;
                };
                let n_vec = n_vec.to_vec();
                let new_score =
                    self.normalize_score(self.fast_score(&n_vec, self.vector_slice(idx)));
                self.layers[l].with(n, |lock| {
                    let mut nb_list = lock.write();
                    // idx is freshly allocated; it can't already be in nb_list.
                    let pos = nb_list.partition_point(|&nb| {
                        self.get_vector_by_idx(nb)
                            .map(|v| self.normalize_score(self.fast_score(&n_vec, v)) <= new_score)
                            .unwrap_or(true)
                    });
                    nb_list.insert(pos, idx);
                    if l == 0 {
                        let n_dists: Vec<f32> = nb_list
                            .iter()
                            .map(|&nb| {
                                if nb == n {
                                    return 0.0;
                                }
                                self.fast_score(&n_vec, self.vector_slice(nb))
                            })
                            .collect();
                        self.edge_dists_l0.with(n, |ed| *ed.write() = n_dists);
                    }
                });
                // Apply diversity cap if enabled — only does work when caps are on.
                if enforce_neighbor_caps() {
                    let cap = self.neighbor_list_capacity(l);
                    let len = self.layers[l].with(n, |lock| lock.read().len());
                    if len > cap {
                        let nb_indices: Vec<usize> =
                            self.layers[l].with(n, |lock| lock.read().clone());
                        let mut cands: Vec<NodeCandidate> = nb_indices
                            .into_iter()
                            .map(|nb_idx| {
                                let raw = self.fast_score(&n_vec, self.vector_slice(nb_idx));
                                NodeCandidate {
                                    idx: nb_idx,
                                    raw_score: raw,
                                    sort_key: self.normalize_score(raw),
                                }
                            })
                            .collect();
                        cands.sort_by(|a, b| {
                            a.sort_key
                                .partial_cmp(&b.sort_key)
                                .unwrap_or(Ordering::Equal)
                        });
                        let selected = self.select_diverse_neighbors(&cands, cap, use_norm, l);
                        self.layers[l].with(n, |lock| *lock.write() = selected.clone());
                        // Keep edge_dists_l0 in sync with the post-cap neighbor list.
                        if l == 0 && !self.edge_dists_l0.is_empty() {
                            let dists: Vec<f32> = selected
                                .iter()
                                .map(|&nb| {
                                    if nb == n {
                                        return 0.0;
                                    }
                                    self.fast_score(&n_vec, self.vector_slice(nb))
                                })
                                .collect();
                            self.edge_dists_l0.with(n, |lock| *lock.write() = dists);
                        }
                    }
                }
            }

            if let Some(&best) = neighbors.first() {
                current_entry = best;
            }
        }

        Ok(())
    }

    pub fn build_filter_aware_edges(
        &mut self,
        point_id: PointId,
        vector: &[f32],
        payload: &Payload,
        payload_index: &PayloadIndex,
        _payloads: &std::collections::HashMap<PointId, Payload>,
        filter_keys: &[String],
    ) -> Result<(), DBError> {
        if filter_keys.is_empty() {
            return Ok(());
        }
        let query_vector = if self.metric == DistanceMetric::Cosine {
            self.maybe_normalize(vector)
        } else {
            vector.to_vec()
        };

        let mut extra_neighbors = HashSet::new();
        let m = self.m0();
        let log_edges_agg = std::env::var("VECTORDB_LOG_FILTER_EDGES_AGG")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);

        let sample_limit: usize = m.saturating_mul(2);

        for key in filter_keys {
            if let Some(value) = payload.get(key) {
                let key_start = if log_edges_agg {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let mut sample_len = 0usize;
                let mut scored_len = 0usize;
                let prev_len = extra_neighbors.len();

                if let Some(id_set) = payload_index.query_exact(key, value) {
                    let mut rng = rand::rng();
                    let candidates: Vec<_> = if id_set.len() <= sample_limit {
                        id_set
                            .iter()
                            .filter(|&&id| id != point_id && self.get_vector(&id).is_some())
                            .copied()
                            .collect()
                    } else {
                        id_set
                            .iter()
                            .filter(|&&id| id != point_id && self.get_vector(&id).is_some())
                            .copied()
                            .choose_multiple(&mut rng, sample_limit)
                    };

                    let mut scored: Vec<_> = candidates
                        .into_iter()
                        .filter_map(|id| {
                            self.get_vector(&id).map(|vec| {
                                let raw = self.fast_score(&query_vector, vec);
                                let sort_key = self.normalize_score(raw);
                                super::types::ScoredPoint {
                                    id,
                                    raw_score: raw,
                                    sort_key,
                                }
                            })
                        })
                        .collect();
                    sample_len = scored.len();

                    scored.sort_by(|a, b| {
                        if self.metric == DistanceMetric::Dot {
                            b.raw_score.partial_cmp(&a.raw_score).unwrap()
                        } else {
                            a.raw_score.partial_cmp(&b.raw_score).unwrap()
                        }
                    });
                    scored_len = scored.len();

                    for sp in scored.into_iter().take(m) {
                        extra_neighbors.insert(sp.id);
                    }

                    if extra_neighbors.len() >= m {
                        if let Some(start) = key_start {
                            let dur = start.elapsed();
                            if log_edges_agg && sample_len > 0 {
                                let bucket = match value {
                                    PayloadValue::Bool(_) => 0,
                                    PayloadValue::Str(_) => 1,
                                    PayloadValue::Int(_) => 2,
                                    PayloadValue::Float(_) => 3,
                                    _ => 3,
                                };
                                FILTER_EDGE_STATS.with(|cell| {
                                    let mut agg = cell.borrow_mut();
                                    agg.count += 1;
                                    agg.samples += sample_len;
                                    agg.scored += scored_len;
                                    agg.added += extra_neighbors.len().saturating_sub(prev_len);
                                    agg.ns_by_type[bucket] += dur.as_nanos();
                                    if agg.count % FILTER_EDGE_LOG_CHUNK == 0 {
                                        FILTER_EDGE_TOTAL_KEYS.with(|tk| *tk.borrow_mut() += agg.count);
                                        let cum = FILTER_EDGE_TOTAL_KEYS.with(|tk| *tk.borrow());
                                        let total_ns: u128 = agg.ns_by_type.iter().sum();
                                        let to_ms = |ns: u128| (ns as f64) / 1_000_000.0;
                                log::info!(
                                    target: "hnsw",
                                    "[filter_edges_agg] n={} cum_n={} samples={} scored={} added={} total_ms={:.3} bool_ms={:.3} str_ms={:.3} int_ms={:.3} float_ms={:.3}",
                                    agg.count,
                                    cum,
                                    agg.samples,
                                    agg.scored,
                                    agg.added,
                                    to_ms(total_ns),
                                    to_ms(agg.ns_by_type[0]),
                                    to_ms(agg.ns_by_type[1]),
                                    to_ms(agg.ns_by_type[2]),
                                    to_ms(agg.ns_by_type[3]),
                                );
                                        *agg = FilterEdgeAgg::default();
                                    }
                                });
                            }
                        }
                        continue;
                    }
                }

                if let Some(start) = key_start {
                    let dur = start.elapsed();
                    if log_edges_agg && sample_len > 0 {
                        let bucket = match value {
                            PayloadValue::Bool(_) => 0,
                            PayloadValue::Str(_) => 1,
                            PayloadValue::Int(_) => 2,
                            PayloadValue::Float(_) => 3,
                            _ => 3,
                        };
                        FILTER_EDGE_STATS.with(|cell| {
                            let mut agg = cell.borrow_mut();
                            agg.count += 1;
                            agg.samples += sample_len;
                            agg.scored += scored_len;
                            agg.added += extra_neighbors.len().saturating_sub(prev_len);
                            agg.ns_by_type[bucket] += dur.as_nanos();
                            if agg.count % FILTER_EDGE_LOG_CHUNK == 0 {
                                FILTER_EDGE_TOTAL_KEYS.with(|tk| *tk.borrow_mut() += agg.count);
                                let cum = FILTER_EDGE_TOTAL_KEYS.with(|tk| *tk.borrow());
                                let total_ns: u128 = agg.ns_by_type.iter().sum();
                                let to_ms = |ns: u128| (ns as f64) / 1_000_000.0;
                                log::info!(
                                    target: "hnsw",
                                    "[filter_edges_agg] n={} cum_n={} samples={} scored={} added={} total_ms={:.3} bool_ms={:.3} str_ms={:.3} int_ms={:.3} float_ms={:.3}",
                                    agg.count,
                                    cum,
                                    agg.samples,
                                    agg.scored,
                                    agg.added,
                                    to_ms(total_ns),
                                    to_ms(agg.ns_by_type[0]),
                                    to_ms(agg.ns_by_type[1]),
                                    to_ms(agg.ns_by_type[2]),
                                    to_ms(agg.ns_by_type[3]),
                                );
                                *agg = FilterEdgeAgg::default();
                            }
                        });
                    }
                }
            }
        }

        let cap = m.max(1);
        for neighbor_id in extra_neighbors
            .into_iter()
            .filter(|id| *id != point_id)
            .take(cap)
        {
            self.add_one_way_edge(0, point_id, neighbor_id);
        }

        Ok(())
    }

    pub fn add_bidirectional_edge(&mut self, level: usize, a: PointId, b: PointId) {
        let (Some(a_idx), Some(b_idx)) = (self.idx_of(a), self.idx_of(b)) else {
            return;
        };
        let nodes_len = self.len();
        self.ensure_level_capacity(level, nodes_len);
        self.extend_layers_for_new_node(nodes_len);
        self.layers[level].with(a_idx, |lock| Self::push_unique(&mut lock.write(), b_idx));
        self.layers[level].with(b_idx, |lock| Self::push_unique(&mut lock.write(), a_idx));
        self.sort_layer_neighbors(level, a_idx);
        self.sort_layer_neighbors(level, b_idx);
        if enforce_neighbor_caps() {
            self.cap_layer_neighbors(level, a_idx);
            self.cap_layer_neighbors(level, b_idx);
        }
    }

    pub fn add_one_way_edge(&mut self, level: usize, from: PointId, to: PointId) {
        let (Some(from_idx), Some(to_idx)) = (self.idx_of(from), self.idx_of(to)) else {
            return;
        };
        let nodes_len = self.len();
        self.ensure_level_capacity(level, nodes_len);
        self.extend_layers_for_new_node(nodes_len);
        self.layers[level].with(from_idx, |lock| {
            Self::push_unique(&mut lock.write(), to_idx)
        });
        self.sort_layer_neighbors(level, from_idx);
        if enforce_neighbor_caps() {
            self.cap_layer_neighbors(level, from_idx);
        }
    }

    #[inline]
    fn push_unique(vec: &mut Vec<usize>, val: usize) {
        if !vec.contains(&val) {
            vec.push(val);
        }
    }

    pub fn greedy_search_layer_unfiltered(
        &self,
        query: &[f32],
        entry: usize,
        level: usize,
    ) -> usize {
        let mut current = entry;
        let mut changed = true;
        let mut steps = 0;

        while changed && steps < 1000 {
            steps += 1;
            changed = false;
            let Some(layer) = self.layers.get(level) else {
                break;
            };
            if current >= layer.len() {
                break;
            }
            let neighbors: Vec<usize> = layer.with(current, |rw| rw.read().clone());
            for neighbor in neighbors {
                if self
                    .deleted
                    .with(neighbor, |b| b.load(::std::sync::atomic::Ordering::Acquire))
                {
                    continue;
                }

                let d_current = self.fast_score(query, self.vector_slice(current));
                let d_new = self.fast_score(query, self.vector_slice(neighbor));
                let s_current = self.normalize_score(d_current);
                let s_new = self.normalize_score(d_new);

                if s_new < s_current {
                    current = neighbor;
                    changed = true;
                    break;
                }
            }
        }

        if steps >= 1000 {
            log::warn!(target: "vector::hnsw", "[GREEDY] Reached max steps at level {}, current = {}", level, current);
        }

        current
    }

    /// HNSW Heuristic 2 "extend candidates": expand the candidate pool with one-hop neighbors
    /// of the top-m candidates. Uses thread-local scratch (epoch-based seen set, reusable Vecs)
    /// to avoid per-call allocations.
    fn extend_candidates(
        &self,
        candidates: &mut Vec<NodeCandidate>,
        idx: usize,
        level: usize,
        use_norm: bool,
    ) {
        let m_for_layer = if level == 0 { self.m0 } else { self.m };
        let nodes_len = self.levels.len();
        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.reset_extend_seen(nodes_len);
            scratch.extend_base.clear();
            scratch.extend_extra.clear();

            for c in candidates.iter() {
                scratch.mark_extend_seen(c.idx);
            }
            scratch.mark_extend_seen(idx);

            for c in candidates.iter().take(m_for_layer) {
                scratch.extend_base.push(c.idx);
            }

            let base_len = scratch.extend_base.len();
            for i in 0..base_len {
                let base_idx = scratch.extend_base[i];
                scratch.extend_neighbors.clear();
                if let Some(layer) = self.layers.get(level) {
                    if base_idx < layer.len() {
                        layer.with(base_idx, |rw| {
                            scratch.extend_neighbors.extend_from_slice(&rw.read());
                        });
                    }
                }
                let nb_count = scratch.extend_neighbors.len();
                for j in 0..nb_count {
                    let nb = scratch.extend_neighbors[j];
                    if self
                        .deleted
                        .with(nb, |b| b.load(::std::sync::atomic::Ordering::Acquire))
                    {
                        continue;
                    }
                    if !scratch.mark_extend_seen(nb) {
                        continue;
                    }
                    let raw = self.fast_score(self.vector_slice(idx), self.vector_slice(nb));
                    let sort_key = if use_norm {
                        self.normalize_score(raw)
                    } else {
                        raw
                    };
                    scratch.extend_extra.push(NodeCandidate {
                        idx: nb,
                        raw_score: raw,
                        sort_key,
                    });
                }
            }

            candidates.extend(scratch.extend_extra.drain(..));
        });
        Self::pre_sort_candidates(candidates);
    }

    /// Heuristic neighbor selector that enforces diversity (HNSW heuristic 2).
    pub(crate) fn select_diverse_neighbors(
        &self,
        candidates: &[NodeCandidate],
        m: usize,
        normalize_scores: bool,
        level: usize,
    ) -> Vec<usize> {
        let alpha = diversity_alpha_for_level(level);
        let prune_floor = diversity_prune_floor().min(m);
        let mut result = Vec::with_capacity(m);
        for cand in candidates {
            if result.len() >= m {
                break;
            }
            if result.contains(&cand.idx) {
                continue;
            }
            let Some(cand_vec) = self.get_vector_by_idx(cand.idx) else {
                continue;
            };
            let d_qc = cand.sort_key;
            if result.len() < prune_floor {
                result.push(cand.idx);
                continue;
            }
            let mut too_close = false;
            for &r_id in &result {
                let Some(r_vec) = self.get_vector_by_idx(r_id) else {
                    continue;
                };
                let d_cr_raw = self.fast_score(cand_vec, r_vec);
                let d_cr = if normalize_scores {
                    self.normalize_score(d_cr_raw)
                } else {
                    d_cr_raw
                };
                if d_cr < d_qc * alpha {
                    too_close = true;
                    break;
                }
            }
            if !too_close {
                result.push(cand.idx);
            }
        }
        if result.len() < m {
            for cand in candidates {
                if result.len() >= m {
                    break;
                }
                if !result.contains(&cand.idx) {
                    result.push(cand.idx);
                }
            }
        }
        result
    }

    fn pre_sort_candidates(candidates: &mut [NodeCandidate]) {
        candidates.sort_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap_or(Ordering::Equal)
        });
    }

    fn cap_layer_neighbors(&mut self, level: usize, node_idx: usize) {
        if level >= self.layers.len() {
            return;
        }
        let cap = self.neighbor_list_capacity(level);
        let layer = &self.layers[level];
        if node_idx >= layer.len() {
            return;
        }
        let neighbors_len = layer.with(node_idx, |rw| rw.read().len());
        if neighbors_len <= cap {
            return;
        }

        let neighbor_indices: Vec<usize> = layer.with(node_idx, |rw| rw.read().clone());
        let node_vec: Vec<f32> = self.vector_slice(node_idx).to_vec();

        let mut candidates: Vec<NodeCandidate> = neighbor_indices
            .into_iter()
            .map(|nb_idx| {
                let raw = self.fast_score(&node_vec, self.vector_slice(nb_idx));
                let sort_key = self.normalize_score(raw);
                NodeCandidate {
                    idx: nb_idx,
                    raw_score: raw,
                    sort_key,
                }
            })
            .collect();

        candidates.sort_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap_or(Ordering::Equal)
        });

        let selected = self.select_diverse_neighbors(&candidates, cap, true, level);
        self.layers[level].with(node_idx, |lock| *lock.write() = selected.clone());
        // Keep edge_dists_l0 in sync with the post-cap neighbor list.
        if level == 0 && !self.edge_dists_l0.is_empty() {
            let node_vec = self.vector_slice(node_idx).to_vec();
            let dists: Vec<f32> = selected
                .iter()
                .map(|&nb| {
                    if nb == node_idx {
                        return 0.0;
                    }
                    self.fast_score(&node_vec, self.vector_slice(nb))
                })
                .collect();
            self.edge_dists_l0
                .with(node_idx, |lock| *lock.write() = dists);
        }
    }

    fn sort_layer_neighbors(&mut self, level: usize, node_idx: usize) {
        if level >= self.layers.len() {
            return;
        }
        let layer = &self.layers[level];
        if node_idx >= layer.len() {
            return;
        }
        let neighbors_len = layer.with(node_idx, |rw| rw.read().len());
        if neighbors_len <= 1 {
            return;
        }
        let Some(node_vec) = self.get_vector_by_idx(node_idx) else {
            return;
        };
        let node_vec = node_vec.to_vec();
        let neighbors: Vec<usize> = layer.with(node_idx, |rw| rw.read().clone());
        let mut scored: Vec<(usize, f32)> = neighbors
            .into_iter()
            .filter_map(|nb| {
                self.get_vector_by_idx(nb).map(|v| {
                    let raw = self.fast_score(&node_vec, v);
                    (nb, self.normalize_score(raw))
                })
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let sorted_indices: Vec<usize> = scored.iter().map(|(idx, _)| *idx).collect();
        self.layers[level].with(node_idx, |lock| *lock.write() = sorted_indices.clone());
        // Keep edge_dists_l0 in sync with the reordered neighbor list.
        if level == 0 && !self.edge_dists_l0.is_empty() {
            let dists: Vec<f32> = sorted_indices
                .iter()
                .map(|&nb| {
                    if nb == node_idx {
                        return 0.0;
                    }
                    self.fast_score(&node_vec, self.vector_slice(nb))
                })
                .collect();
            self.edge_dists_l0
                .with(node_idx, |lock| *lock.write() = dists);
        }
    }
}

impl HNSWIndex {
    pub fn greedy_search_layer_with_filter(
        &self,
        query: &[f32],
        entry: usize,
        level: usize,
        payloads: &std::collections::HashMap<PointId, Payload>,
        filter: Option<&Filter>,
    ) -> Result<usize, DBError> {
        let mut current = entry;
        let mut changed = true;
        let mut steps = 0;

        while changed && steps < 1000 {
            steps += 1;
            changed = false;

            if let Some(neighbors) = self.layer_neighbors(level, current) {
                for &neighbor in neighbors.iter() {
                    if self
                        .deleted
                        .with(neighbor, |b| b.load(::std::sync::atomic::Ordering::Acquire))
                    {
                        continue;
                    }

                    if let Some(f) = filter {
                        let id = self.point_id(neighbor);
                        let Some(payload) = payloads.get(&id) else {
                            continue;
                        };
                        if !evaluate_filter(f, payload)? {
                            continue;
                        }
                    }

                    let Some(current_vec) = self.get_vector_by_idx(current) else {
                        break;
                    };
                    let Some(neighbor_vec) = self.get_vector_by_idx(neighbor) else {
                        continue;
                    };
                    let d_current = self.fast_score(query, current_vec);
                    let d_new = self.fast_score(query, neighbor_vec);

                    let s_current = match self.metric {
                        DistanceMetric::Dot => -d_current,
                        _ => d_current,
                    };
                    let s_new = match self.metric {
                        DistanceMetric::Dot => -d_new,
                        _ => d_new,
                    };

                    if s_new < s_current {
                        current = neighbor;
                        changed = true;
                    }
                }
            }
        }

        Ok(current)
    }
}
