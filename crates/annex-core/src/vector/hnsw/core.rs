use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};

use crate::utils::errors::DBError;
use crate::utils::types::{DistanceMetric, PointId, Vector};
use crate::vector::hnsw::arena::{ChunkedArray, VectorArena};

use super::config::{
    DEFAULT_EXACT_FALLBACK_THRESHOLD, VERBOSE, exact_fallback_enabled_override,
    exact_fallback_threshold_override, log_unfiltered_enabled,
};
use super::stats::UNFILTERED_SEARCH_AGG;

#[derive(Serialize, Deserialize, Clone)]
pub struct HnswSnapshot {
    pub layers: HashMap<usize, HashMap<PointId, Vec<PointId>>>,
    pub vectors: HashMap<PointId, Vector>,
    pub levels: HashMap<PointId, usize>,
    pub entry_point: Option<PointId>,
    pub metric: DistanceMetric,
    pub m: usize,
    #[serde(default)]
    pub m0: usize,
    /// Maximum stored neighbors at L0 (excluding self-link). Defaults to m0 when 0.
    /// Separates build-time minimum connectivity (m0) from maximum stored degree (stored_cap_l0).
    #[serde(default)]
    pub stored_cap_l0: usize,
    pub ef: usize,
    pub ef_construct: usize,
    pub max_level_cap: usize,
    pub level_scale: f64,
    pub current_max_level: usize,
    pub dim: usize,
    pub deleted: HashSet<PointId>,
    pub exact_fallback_enabled: bool,
    pub exact_fallback_threshold: usize,
}

// ── Node lifecycle states ────────────────────────────────────────────────────
// A node is transiently RESERVED between the allocation of its slots and the
// publication of its neighbor links. Readers must skip anything that is not
// LIVE so that half-initialised nodes cannot leak into search results.
pub(crate) const NODE_RESERVED: u8 = 0;
pub(crate) const NODE_LIVE: u8 = 1;
pub(crate) const NODE_DELETED: u8 = 2;

/// Packed `(current_max_level, entry_point_idx)` in one `AtomicU64`.
/// Upper 32 bits = max_level; lower 32 bits = entry_idx.
/// `NO_EP` in the lower 32 bits means no entry point has been set.
pub(crate) const NO_EP: u32 = u32::MAX;
pub(crate) const MAX_STORED_LEVEL: usize = u8::MAX as usize;

#[inline]
pub(crate) fn pack_ep(entry_idx: usize, max_level: usize) -> u64 {
    debug_assert!(entry_idx <= u32::MAX as usize);
    debug_assert!(max_level <= u32::MAX as usize);
    ((max_level as u64) << 32) | (entry_idx as u32 as u64)
}

#[inline]
pub(crate) fn unpack_ep(v: u64) -> (Option<usize>, usize) {
    let idx = (v & 0xffff_ffff) as u32;
    let level = (v >> 32) as usize;
    if idx == NO_EP {
        (None, level)
    } else {
        (Some(idx as usize), level)
    }
}

/// Mutex-guarded allocation state shared by all concurrent writers.
///
/// Holds the id ↔ idx mapping and the running node count. Every `insert`
/// takes the mutex only during ALLOC (slot reservation + id publication);
/// LINK and PUBLISH work off per-node RwLocks / atomics.
pub struct AllocState {
    pub(crate) point_to_idx: HashMap<PointId, usize>,
    pub(crate) idx_to_point: Vec<PointId>,
    pub(crate) node_count: usize,
}

pub struct HNSWIndex {
    /// Per-node neighbor lists at each level.
    /// The outer `Vec` is pre-allocated to `max_level_cap + 1` at construction
    /// and never grown; the inner `ChunkedArray` grows by node idx and keeps
    /// each per-node `RwLock` at a stable address so concurrent readers can
    /// safely hold shared references while other nodes are being appended.
    pub(crate) layers: Vec<ChunkedArray<RwLock<Vec<usize>>>>,
    /// Stable-address vector storage. Never reallocates existing data.
    pub(crate) vectors: VectorArena,
    /// Per-node assigned level (fits in u8; max_level_cap ≤ 255).
    pub(crate) levels: ChunkedArray<AtomicU8>,
    /// Packed (current_max_level, entry_point_idx) as a single atomic.
    /// Load/store with Acquire/Release for publication ordering.
    pub(crate) entry_level_ep: AtomicU64,
    pub(crate) metric: DistanceMetric,
    pub(crate) m: usize,
    pub(crate) m0: usize,
    pub(crate) stored_cap_l0: usize,
    pub(crate) ef: usize,
    pub(crate) ef_construct: usize,
    pub(crate) max_level_cap: usize,
    pub(crate) level_scale: f64,
    pub(crate) dim: usize,
    /// Per-node deletion flag.
    pub(crate) deleted: ChunkedArray<AtomicBool>,
    /// Maintained on deletion; cached count for O(1) deleted_count().
    pub(crate) deleted_count: AtomicUsize,
    /// Parallel to layers[0]: edge_dists_l0.get(idx) holds the distance from
    /// node idx to each of its L0 neighbors, co-indexed with layers[0].get(idx).
    /// Protected by the same logical lock as layers[0].get(idx): always write
    /// under layers[0].with(idx, |l| l.write()) and read under `.read()`.
    /// Empty (per-slot `Vec` empty) until [`build_edge_distances`] runs.
    pub(crate) edge_dists_l0: ChunkedArray<RwLock<Vec<f32>>>,
    /// SQ8 quantized vectors: quantized[idx * dim + d] = u8 encoding of dimension d.
    /// Empty until `quantize_all()` is called.
    pub(crate) quantized: Vec<u8>,
    /// Per-dimension minimum value used for SQ8 quantization.
    pub(crate) quant_min: Vec<f32>,
    /// Per-dimension scale: (max - min) / 255.0. Set to 1.0 for constant dimensions.
    pub(crate) quant_scale: Vec<f32>,
    /// Per-node lifecycle state (RESERVED / LIVE / DELETED). Written with
    /// Release when transitioning; readers use Acquire so a node becomes
    /// reachable only after its links are fully published.
    pub(crate) node_state: ChunkedArray<AtomicU8>,
    /// Stable-slot id storage: `ids.get(idx)` holds the `PointId` for slot
    /// `idx`. Written under the `alloc` mutex before the node transitions to
    /// LIVE, so any Acquire-load of `node_state == LIVE` sees a valid id.
    /// Read locklessly on the hot query paths (Relaxed is sufficient because
    /// the LIVE check on `node_state` provides the happens-before edge).
    pub(crate) ids: ChunkedArray<AtomicU64>,
    /// Shared allocation state guarded by a single mutex. Held only for the
    /// ALLOC phase of a concurrent insert; hot query paths never touch it.
    pub(crate) alloc: std::sync::Mutex<AllocState>,
    pub(crate) exact_fallback_enabled: bool,
    pub(crate) exact_fallback_threshold: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HnswConfigSummary {
    pub metric: DistanceMetric,
    pub m: usize,
    pub m0: usize,
    #[serde(default)]
    pub stored_cap_l0: usize,
    pub ef: usize,
    pub ef_construct: usize,
    pub max_level_cap: usize,
    pub level_scale: f64,
    pub current_max_level: usize, // derived from entry_level_ep at snapshot time
    pub dim: usize,
    pub exact_fallback_enabled: bool,
    pub exact_fallback_threshold: usize,
}

impl HNSWIndex {
    pub fn new(
        metric: DistanceMetric,
        m: usize,
        ef: usize,
        max_level_cap: usize,
        dim: usize,
    ) -> Self {
        // Touch the flag early so the enable banner shows up before long inserts.
        let _ = log_unfiltered_enabled();
        let base_level_scale = 1.0 / (m as f64).ln();
        let level_scale = Self::level_scale_from_env(base_level_scale).unwrap_or(base_level_scale);
        let max_level_cap = Self::max_level_cap_from_env(max_level_cap).min(MAX_STORED_LEVEL);
        if VERBOSE {
            log::debug!(
                target: "vector::hnsw",
                "Creating new HNSWIndex with dim {}, M {}, ef {}, max_level_cap {}",
                dim,
                m,
                ef,
                max_level_cap
            );
        }
        const CHUNK_CAP: usize = 4096;
        let layers: Vec<ChunkedArray<RwLock<Vec<usize>>>> = (0..=max_level_cap)
            .map(|_| ChunkedArray::new(CHUNK_CAP))
            .collect();
        Self {
            layers,
            vectors: VectorArena::new(dim, CHUNK_CAP),
            levels: ChunkedArray::new(CHUNK_CAP),
            entry_level_ep: AtomicU64::new(pack_ep(NO_EP as usize, 0)),
            metric,
            m,
            m0: m * 2,
            stored_cap_l0: m * 2,
            ef,
            ef_construct: ef,
            max_level_cap,
            level_scale,
            dim,
            deleted: ChunkedArray::new(CHUNK_CAP),
            deleted_count: AtomicUsize::new(0),
            edge_dists_l0: ChunkedArray::new(CHUNK_CAP),
            node_state: ChunkedArray::new(CHUNK_CAP),
            ids: ChunkedArray::new(CHUNK_CAP),
            alloc: std::sync::Mutex::new(AllocState {
                point_to_idx: HashMap::new(),
                idx_to_point: Vec::new(),
                node_count: 0,
            }),
            quantized: Vec::new(),
            quant_min: Vec::new(),
            quant_scale: Vec::new(),
            exact_fallback_enabled: exact_fallback_enabled_override().unwrap_or(false),
            exact_fallback_threshold: exact_fallback_threshold_override()
                .unwrap_or(DEFAULT_EXACT_FALLBACK_THRESHOLD),
        }
    }

    pub fn point_level(&self, point_id: PointId) -> Option<usize> {
        let idx = self.idx_of(point_id)?;
        Some(self.levels.with(idx, |l| l.load(Ordering::Relaxed)) as usize)
    }

    pub fn point_degree(&self, point_id: PointId, level: usize) -> Option<usize> {
        let idx = self.idx_of(point_id)?;
        let layer = self.layers.get(level)?;
        if idx >= layer.len() {
            return None;
        }
        Some(layer.with(idx, |rw| rw.read().len()))
    }

    // ── Entry-point / level helpers ───────────────────────────────────────────

    /// Returns the current entry point index, if any.
    #[inline]
    pub(crate) fn entry_point(&self) -> Option<usize> {
        unpack_ep(self.entry_level_ep.load(Ordering::Acquire)).0
    }

    /// Returns the current maximum level.
    #[inline]
    pub fn current_max_level(&self) -> usize {
        unpack_ep(self.entry_level_ep.load(Ordering::Acquire)).1
    }

    /// Load the entry point and maximum level from one atomic observation.
    #[inline]
    pub(crate) fn entry_level(&self) -> (Option<usize>, usize) {
        unpack_ep(self.entry_level_ep.load(Ordering::Acquire))
    }

    /// Atomically set both entry point and max level together.
    ///
    /// Uses interior mutability on the packed `AtomicU64` so it is safe to
    /// call under `&self` from concurrent inserters.
    #[inline]
    pub(crate) fn store_entry_level(&self, idx: usize, level: usize) {
        debug_assert!(idx < NO_EP as usize);
        debug_assert!(level <= MAX_STORED_LEVEL);
        self.entry_level_ep
            .store(pack_ep(idx, level), Ordering::Release);
    }

    /// Promote `idx` to entry point only if `level` strictly exceeds the
    /// currently-published `current_max_level`. Uses a CAS loop so two
    /// concurrent inserters at overlapping upper levels resolve to a single
    /// winner without dropping either promotion.
    #[inline]
    pub(crate) fn promote_entry_if_higher(&self, idx: usize, level: usize) {
        debug_assert!(idx < NO_EP as usize);
        debug_assert!(level <= MAX_STORED_LEVEL);
        let new_val = pack_ep(idx, level);
        loop {
            let old = self.entry_level_ep.load(Ordering::Acquire);
            let (_, old_max) = unpack_ep(old);
            if level <= old_max {
                return;
            }
            match self.entry_level_ep.compare_exchange_weak(
                old,
                new_val,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(_) => continue,
            }
        }
    }

    /// Acquire-load the lifecycle state for `idx`. Pairs with the Release
    /// store in `publish_live` / `mark_deleted` so all upstream writes are
    /// observed before the state is inspected.
    #[inline]
    pub(crate) fn state_of(&self, idx: usize) -> u8 {
        self.node_state.with(idx, |s| s.load(Ordering::Acquire))
    }

    /// True only if the node at `idx` is fully published and not deleted.
    /// The canonical visibility check for concurrent BFS readers.
    #[inline]
    pub(crate) fn is_live(&self, idx: usize) -> bool {
        self.state_of(idx) == NODE_LIVE
    }

    // ── vector_slice bridge ───────────────────────────────────────────────────

    /// Return a slice into the flat vector storage.
    /// For the BFS hot path, prefer `VectorArenaView::get` acquired once per search.
    #[inline]
    pub(crate) fn vector_slice(&self, idx: usize) -> &[f32] {
        self.vectors.get_direct(idx)
    }

    pub(crate) fn assign_random_level(&self) -> usize {
        let r: f64 = rand::rng().random_range(0.0..1.0);
        let l = (-r.ln() * self.level_scale).floor() as usize;
        let level = l.min(self.max_level_cap);
        level
    }

    pub fn normalize_score(&self, raw: f32) -> f32 {
        match self.metric {
            DistanceMetric::Cosine | DistanceMetric::Euclidean => raw,
            DistanceMetric::Dot => -raw,
        }
    }

    #[inline]
    pub(crate) fn fast_score(&self, query: &[f32], vec: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => {
                let dot: f32 = dot_product(query, vec);
                let sim = if dot > 1.0 {
                    1.0
                } else if dot < -1.0 {
                    -1.0
                } else {
                    dot
                };
                1.0 - sim
            }
            DistanceMetric::Dot => dot_product(query, vec),
            DistanceMetric::Euclidean => l2_squared(query, vec),
        }
    }

    pub fn mark_deleted(&self, point_id: PointId) {
        let Some(idx) = self.idx_of(point_id) else {
            return;
        };
        // Load with Acquire so the caller's subsequent reads see this deletion.
        let flag = self.deleted.with(idx, |b| b.load(Ordering::Acquire));
        if !flag {
            self.deleted.with(idx, |b| b.store(true, Ordering::Release));
            self.deleted_count.fetch_add(1, Ordering::Relaxed);
        }
        // Publish DELETED so BFS readers filtering on node_state also skip it.
        // This is a Release store so any prior writes are visible to readers
        // that later Acquire the state.
        self.node_state
            .with(idx, |s| s.store(NODE_DELETED, Ordering::Release));
        if Some(idx) == self.entry_point() {
            let new_ep = self.find_highest_level_entry_point();
            match new_ep {
                Some(ep) => {
                    let level = self.levels.with(ep, |l| l.load(Ordering::Relaxed)) as usize;
                    self.store_entry_level(ep, level);
                }
                None => {
                    self.entry_level_ep
                        .store(pack_ep(NO_EP as usize, 0), Ordering::Release);
                }
            }
        }
    }

    pub fn find_highest_level_entry_point(&self) -> Option<usize> {
        let n = self.len();
        let mut best_idx = None;
        let mut best_level = 0usize;
        for idx in 0..n {
            if self.deleted.with(idx, |b| b.load(Ordering::Acquire)) {
                continue;
            }
            let level = self.levels.with(idx, |l| l.load(Ordering::Relaxed)) as usize;
            if best_idx.is_none() || level >= best_level {
                best_idx = Some(idx);
                best_level = level;
            }
        }
        best_idx
    }

    #[inline]
    pub(crate) fn idx_of(&self, point_id: PointId) -> Option<usize> {
        self.alloc
            .lock()
            .expect("alloc mutex poisoned")
            .point_to_idx
            .get(&point_id)
            .copied()
    }

    /// Lockless read of the stable id for slot `idx`. Callers must ensure
    /// the slot was allocated (idx < len()); the load is Relaxed because the
    /// happens-before edge is established by the LIVE-state Acquire load
    /// that gates BFS neighbor traversal.
    #[inline]
    pub(crate) fn point_id(&self, idx: usize) -> PointId {
        self.ids.with(idx, |a| a.load(Ordering::Relaxed))
    }

    #[inline]
    pub(crate) fn get_vector_by_idx(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.vectors.len() || self.deleted.with(idx, |b| b.load(Ordering::Acquire)) {
            None
        } else {
            Some(self.vector_slice(idx))
        }
    }

    #[inline]
    pub(crate) fn neighbor_list_capacity(&self, level: usize) -> usize {
        if level == 0 {
            self.stored_cap_l0 + 1
        } else {
            self.m + 1
        }
    }

    /// The outer level `Vec` is pre-sized to `max_level_cap + 1` at construction,
    /// so no dynamic growth of the level axis is needed. Left in place as a
    /// bounds assertion / no-op for callers migrated from the growable layout.
    pub(crate) fn ensure_level_capacity(&self, level: usize, _nodes_len: usize) {
        debug_assert!(
            level < self.layers.len(),
            "level {} exceeds pre-allocated layers ({})",
            level,
            self.layers.len()
        );
    }

    /// Allocate one stable slot per level (and one edge-dist slot) so the newly
    /// registered node at `nodes_len - 1` has a per-level `RwLock` at a stable
    /// address. Idempotent — extra pushes are avoided by comparing to the
    /// existing chunked-array length.
    pub(crate) fn extend_layers_for_new_node(&self, nodes_len: usize) {
        for layer in self.layers.iter() {
            while layer.len() < nodes_len {
                layer.push_default();
            }
        }
        while self.edge_dists_l0.len() < nodes_len {
            self.edge_dists_l0.push_default();
        }
    }

    pub fn contains(&self, point_id: &PointId) -> bool {
        self.alloc
            .lock()
            .expect("alloc mutex poisoned")
            .point_to_idx
            .contains_key(point_id)
    }

    pub fn len(&self) -> usize {
        self.alloc.lock().expect("alloc mutex poisoned").node_count
    }

    pub fn config_summary(&self) -> HnswConfigSummary {
        HnswConfigSummary {
            metric: self.metric,
            m: self.m,
            m0: self.m0,
            stored_cap_l0: self.stored_cap_l0,
            ef: self.ef,
            ef_construct: self.ef_construct,
            max_level_cap: self.max_level_cap,
            level_scale: self.level_scale,
            current_max_level: self.current_max_level(),
            dim: self.dim,
            exact_fallback_enabled: self.exact_fallback_enabled,
            exact_fallback_threshold: self.exact_fallback_threshold,
        }
    }

    /// Return a cloned snapshot of the neighbor list at `layers[level][idx]`.
    /// Cloning avoids returning a `RwLockReadGuard` that would borrow into
    /// the `ChunkedArray`; callers that need zero-copy access should use
    /// [`with_layer_neighbors`].
    pub fn layer_neighbors(&self, level: usize, idx: usize) -> Option<Vec<usize>> {
        let layer = self.layers.get(level)?;
        if idx >= layer.len() {
            return None;
        }
        Some(layer.with(idx, |rw| rw.read().clone()))
    }

    /// Invoke `f` with a shared reference to the neighbor list at `layers[level][idx]`.
    /// The `RwLock` read-guard is held only for the duration of the callback.
    #[inline]
    pub fn with_layer_neighbors<F, R>(&self, level: usize, idx: usize, f: F) -> Option<R>
    where
        F: FnOnce(&[usize]) -> R,
    {
        let layer = self.layers.get(level)?;
        if idx >= layer.len() {
            return None;
        }
        Some(layer.with(idx, |rw| f(&rw.read())))
    }

    pub fn iter_vectors(&self) -> impl Iterator<Item = (PointId, &[f32])> {
        (0..self.len()).map(move |idx| (self.point_id(idx), self.vector_slice(idx)))
    }

    pub fn iter_active_vectors(&self) -> impl Iterator<Item = (PointId, &[f32])> {
        (0..self.len())
            .filter(move |&idx| !self.deleted.with(idx, |b| b.load(Ordering::Acquire)))
            .map(move |idx| (self.point_id(idx), self.vector_slice(idx)))
    }

    pub fn deleted_count(&self) -> usize {
        self.deleted_count.load(Ordering::Relaxed)
    }

    pub fn deleted_fraction(&self) -> f64 {
        let n = self.len();
        if n == 0 {
            return 0.0;
        }
        self.deleted_count() as f64 / n as f64
    }

    pub fn iter_active_levels(&self) -> impl Iterator<Item = usize> + '_ {
        let n = self.len();
        (0..n)
            .filter(move |&idx| !self.deleted.with(idx, |b| b.load(Ordering::Acquire)))
            .map(move |idx| self.levels.with(idx, |l| l.load(Ordering::Relaxed)) as usize)
    }

    pub fn level_histogram(&self) -> Vec<usize> {
        let max_level = self.current_max_level();
        let mut counts = vec![0usize; max_level + 1];
        for lvl in self.iter_active_levels() {
            let idx = lvl.min(max_level);
            counts[idx] += 1;
        }
        counts
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn m0(&self) -> usize {
        self.m0
    }

    pub fn ef(&self) -> usize {
        self.ef
    }

    pub fn set_ef_construct(&mut self, ef: usize) {
        self.ef_construct = ef;
    }

    pub fn set_ef_search(&mut self, ef: usize) {
        if log_unfiltered_enabled() {
            UNFILTERED_SEARCH_AGG.with(|cell| cell.borrow_mut().flush());
        }
        self.ef = ef;
    }

    pub fn set_m0(&mut self, m0: usize) {
        self.m0 = m0.max(1);
        if self.stored_cap_l0 < self.m0 {
            self.stored_cap_l0 = self.m0;
        }
    }

    pub fn stored_cap_l0(&self) -> usize {
        self.stored_cap_l0
    }

    pub fn set_stored_cap_l0(&mut self, cap: usize) {
        self.stored_cap_l0 = cap.max(self.m0).max(1);
    }

    pub fn flush_unfiltered_search_stats(&self) {
        if log_unfiltered_enabled() {
            UNFILTERED_SEARCH_AGG.with(|cell| cell.borrow_mut().flush());
        }
    }

    pub fn max_level_cap(&self) -> usize {
        self.max_level_cap
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get_vector(&self, point_id: &PointId) -> Option<&[f32]> {
        let idx = self.idx_of(*point_id)?;
        if self.deleted.with(idx, |b| b.load(Ordering::Acquire)) {
            None
        } else if idx < self.vectors.len() {
            Some(self.vector_slice(idx))
        } else {
            None
        }
    }

    pub fn get_entry_point(&self) -> Option<u64> {
        self.entry_point().map(|idx| self.point_id(idx))
    }

    // entry_point() and current_max_level() are defined as helpers above.

    pub fn set_entry_point(&mut self, point_id: PointId) {
        if let Some(idx) = self.idx_of(point_id) {
            let (_, current_max_level) = self.entry_level();
            self.store_entry_level(idx, current_max_level);
        }
    }

    pub fn set_current_max_level(&mut self, level: usize) {
        let level = level.min(self.max_level_cap).min(MAX_STORED_LEVEL);
        let ep = self.entry_level_ep.load(Ordering::Acquire);
        let (ep_idx, _) = unpack_ep(ep);
        let new_ep = match ep_idx {
            Some(idx) => pack_ep(idx, level),
            None => pack_ep(NO_EP as usize, level),
        };
        self.entry_level_ep.store(new_ep, Ordering::Release);
    }

    pub fn set_exact_fallback_enabled(&mut self, enabled: bool) {
        self.exact_fallback_enabled = enabled;
    }

    pub fn set_exact_fallback_threshold(&mut self, threshold: usize) {
        self.exact_fallback_threshold = threshold;
    }

    fn level_scale_from_env(base: f64) -> Option<f64> {
        if let Ok(raw) = env::var("VECTORDB_LEVEL_SCALE") {
            if let Ok(value) = raw.replace('_', "").parse::<f64>() {
                if value.is_finite() && value > 0.0 {
                    return Some(value);
                }
            }
        }
        if let Ok(raw) = env::var("VECTORDB_LEVEL_SCALE_MULT") {
            if let Ok(mult) = raw.replace('_', "").parse::<f64>() {
                if mult.is_finite() && mult > 0.0 {
                    return Some(base * mult);
                }
            }
        }
        None
    }

    fn max_level_cap_from_env(default_cap: usize) -> usize {
        if let Ok(raw) = env::var("VECTORDB_MAX_LEVEL_CAP") {
            if let Ok(value) = raw.replace('_', "").parse::<usize>() {
                if value > 0 {
                    return value;
                }
            }
        }
        if let Ok(raw) = env::var("VECTORDB_MAX_LEVEL") {
            if let Ok(value) = raw.replace('_', "").parse::<usize>() {
                if value > 0 {
                    return value;
                }
            }
        }
        default_cap
    }

    pub fn maybe_normalize(&self, vec: &[f32]) -> Vector {
        match self.metric {
            DistanceMetric::Cosine => {
                let norm = vec
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt();
                if norm == 0.0 {
                    vec.to_vec()
                } else {
                    let inv = 1.0 / norm;
                    vec.iter().map(|x| (*x as f64 * inv) as f32).collect()
                }
            }
            _ => vec.to_vec(),
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product(query: &[f32], vec: &[f32]) -> f32 {
    unsafe { dot_neon(query, vec) }
}

#[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
#[inline]
fn dot_product(query: &[f32], vec: &[f32]) -> f32 {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe { dot_avx2_fma(query, vec) }
    } else if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { dot_avx2(query, vec) }
    } else {
        dot_scalar(query, vec)
    }
}

#[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
#[inline]
fn dot_product(query: &[f32], vec: &[f32]) -> f32 {
    dot_scalar(query, vec)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn l2_squared(query: &[f32], vec: &[f32]) -> f32 {
    unsafe { l2_neon(query, vec) }
}

#[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
#[inline]
fn l2_squared(query: &[f32], vec: &[f32]) -> f32 {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe { l2_avx2_fma(query, vec) }
    } else if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { l2_avx2(query, vec) }
    } else {
        l2_scalar(query, vec)
    }
}

#[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
#[inline]
fn l2_squared(query: &[f32], vec: &[f32]) -> f32 {
    l2_scalar(query, vec)
}

#[allow(dead_code)]
#[inline]
fn dot_scalar(query: &[f32], vec: &[f32]) -> f32 {
    query.iter().zip(vec.iter()).map(|(x, y)| x * y).sum()
}

#[allow(dead_code)]
#[inline]
fn l2_scalar(query: &[f32], vec: &[f32]) -> f32 {
    query
        .iter()
        .zip(vec.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn dot_avx2(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(q, v));
        i += 8;
    }
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        acc += query.get_unchecked(i) * vec.get_unchecked(i);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx2_fma(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // 4 accumulators × 8 floats/register = 32 floats/iteration.
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 32 <= len {
        let q0 = _mm256_loadu_ps(query.as_ptr().add(i));
        let q1 = _mm256_loadu_ps(query.as_ptr().add(i + 8));
        let q2 = _mm256_loadu_ps(query.as_ptr().add(i + 16));
        let q3 = _mm256_loadu_ps(query.as_ptr().add(i + 24));
        let v0 = _mm256_loadu_ps(vec.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(vec.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(vec.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(vec.as_ptr().add(i + 24));
        s0 = _mm256_fmadd_ps(q0, v0, s0);
        s1 = _mm256_fmadd_ps(q1, v1, s1);
        s2 = _mm256_fmadd_ps(q2, v2, s2);
        s3 = _mm256_fmadd_ps(q3, v3, s3);
        i += 32;
    }
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        s0 = _mm256_fmadd_ps(q, v, s0);
        i += 8;
    }
    s0 = _mm256_add_ps(s0, s1);
    s2 = _mm256_add_ps(s2, s3);
    s0 = _mm256_add_ps(s0, s2);
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        acc += query.get_unchecked(i) * vec.get_unchecked(i);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn l2_avx2(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        let diff = _mm256_sub_ps(q, v);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        i += 8;
    }
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        let diff = query.get_unchecked(i) - vec.get_unchecked(i);
        acc += diff * diff;
        i += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn l2_avx2_fma(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 32 <= len {
        let q0 = _mm256_loadu_ps(query.as_ptr().add(i));
        let q1 = _mm256_loadu_ps(query.as_ptr().add(i + 8));
        let q2 = _mm256_loadu_ps(query.as_ptr().add(i + 16));
        let q3 = _mm256_loadu_ps(query.as_ptr().add(i + 24));
        let v0 = _mm256_loadu_ps(vec.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(vec.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(vec.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(vec.as_ptr().add(i + 24));
        let d0 = _mm256_sub_ps(q0, v0);
        let d1 = _mm256_sub_ps(q1, v1);
        let d2 = _mm256_sub_ps(q2, v2);
        let d3 = _mm256_sub_ps(q3, v3);
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
        i += 32;
    }
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        let d = _mm256_sub_ps(q, v);
        s0 = _mm256_fmadd_ps(d, d, s0);
        i += 8;
    }
    s0 = _mm256_add_ps(s0, s1);
    s2 = _mm256_add_ps(s2, s3);
    s0 = _mm256_add_ps(s0, s2);
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        let diff = query.get_unchecked(i) - vec.get_unchecked(i);
        acc += diff * diff;
        i += 1;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn dot_neon(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    // 4 independent accumulators break the FMA latency chain.
    // At 4 floats/register × 4 accumulators = 16 floats/iteration.
    let mut s0 = vdupq_n_f32(0.0);
    let mut s1 = vdupq_n_f32(0.0);
    let mut s2 = vdupq_n_f32(0.0);
    let mut s3 = vdupq_n_f32(0.0);
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 16 <= len {
        let q0 = vld1q_f32(query.as_ptr().add(i));
        let q1 = vld1q_f32(query.as_ptr().add(i + 4));
        let q2 = vld1q_f32(query.as_ptr().add(i + 8));
        let q3 = vld1q_f32(query.as_ptr().add(i + 12));
        let v0 = vld1q_f32(vec.as_ptr().add(i));
        let v1 = vld1q_f32(vec.as_ptr().add(i + 4));
        let v2 = vld1q_f32(vec.as_ptr().add(i + 8));
        let v3 = vld1q_f32(vec.as_ptr().add(i + 12));
        s0 = vmlaq_f32(s0, q0, v0);
        s1 = vmlaq_f32(s1, q1, v1);
        s2 = vmlaq_f32(s2, q2, v2);
        s3 = vmlaq_f32(s3, q3, v3);
        i += 16;
    }
    // Drain remaining full NEON registers.
    while i + 4 <= len {
        let q = vld1q_f32(query.as_ptr().add(i));
        let v = vld1q_f32(vec.as_ptr().add(i));
        s0 = vmlaq_f32(s0, q, v);
        i += 4;
    }
    // Reduce four accumulators to one.
    s0 = vaddq_f32(s0, s1);
    s2 = vaddq_f32(s2, s3);
    s0 = vaddq_f32(s0, s2);
    let mut acc = vaddvq_f32(s0);
    while i < len {
        acc += query.get_unchecked(i) * vec.get_unchecked(i);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn l2_neon(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut s0 = vdupq_n_f32(0.0);
    let mut s1 = vdupq_n_f32(0.0);
    let mut s2 = vdupq_n_f32(0.0);
    let mut s3 = vdupq_n_f32(0.0);
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 16 <= len {
        let q0 = vld1q_f32(query.as_ptr().add(i));
        let q1 = vld1q_f32(query.as_ptr().add(i + 4));
        let q2 = vld1q_f32(query.as_ptr().add(i + 8));
        let q3 = vld1q_f32(query.as_ptr().add(i + 12));
        let v0 = vld1q_f32(vec.as_ptr().add(i));
        let v1 = vld1q_f32(vec.as_ptr().add(i + 4));
        let v2 = vld1q_f32(vec.as_ptr().add(i + 8));
        let v3 = vld1q_f32(vec.as_ptr().add(i + 12));
        let d0 = vsubq_f32(q0, v0);
        let d1 = vsubq_f32(q1, v1);
        let d2 = vsubq_f32(q2, v2);
        let d3 = vsubq_f32(q3, v3);
        s0 = vmlaq_f32(s0, d0, d0);
        s1 = vmlaq_f32(s1, d1, d1);
        s2 = vmlaq_f32(s2, d2, d2);
        s3 = vmlaq_f32(s3, d3, d3);
        i += 16;
    }
    while i + 4 <= len {
        let q = vld1q_f32(query.as_ptr().add(i));
        let v = vld1q_f32(vec.as_ptr().add(i));
        let d = vsubq_f32(q, v);
        s0 = vmlaq_f32(s0, d, d);
        i += 4;
    }
    s0 = vaddq_f32(s0, s1);
    s2 = vaddq_f32(s2, s3);
    s0 = vaddq_f32(s0, s2);
    let mut acc = vaddvq_f32(s0);
    while i < len {
        let diff = query.get_unchecked(i) - vec.get_unchecked(i);
        acc += diff * diff;
        i += 1;
    }
    acc
}

impl HNSWIndex {
    /// ALLOC phase for a single node.
    ///
    /// Serialised on `alloc.lock()`: publishes fresh slots in every stable
    /// storage (`vectors`, `levels`, `deleted`, `edge_dists_l0`, `node_state`,
    /// `ids`) and every pre-allocated per-level `layers` array, then registers
    /// the id ↔ idx mapping. Returns the newly-assigned `idx`, still in
    /// `NODE_RESERVED` state.
    ///
    /// The mutex holds throughout so `idx` is monotonic and every array's
    /// `push` returns the same index.
    pub(crate) fn alloc_node(
        &self,
        point_id: PointId,
        vector: Vector,
        level: usize,
    ) -> Option<usize> {
        let stored_level = u8::try_from(level).expect("HNSW level exceeds u8 storage");
        let mut alloc = self.alloc.lock().expect("alloc mutex poisoned");
        if alloc.point_to_idx.contains_key(&point_id) {
            return None;
        }
        let idx = alloc.node_count;
        let vector_idx = self.vectors.push(&vector);
        debug_assert_eq!(vector_idx, idx);
        let lvl_idx = self.levels.push_default();
        debug_assert_eq!(lvl_idx, idx);
        self.levels
            .with(idx, |l| l.store(stored_level, Ordering::Relaxed));
        let del_idx = self.deleted.push_default();
        debug_assert_eq!(del_idx, idx);
        let ed_idx = self.edge_dists_l0.push_default();
        debug_assert_eq!(ed_idx, idx);
        let st_idx = self.node_state.push_default();
        debug_assert_eq!(st_idx, idx);
        self.node_state
            .with(idx, |s| s.store(NODE_RESERVED, Ordering::Relaxed));
        let id_idx = self.ids.push_default();
        debug_assert_eq!(id_idx, idx);
        self.ids.with(idx, |a| a.store(point_id, Ordering::Relaxed));
        for layer in self.layers.iter() {
            let li = layer.push_default();
            debug_assert_eq!(li, idx);
        }
        alloc.idx_to_point.push(point_id);
        alloc.point_to_idx.insert(point_id, idx);
        alloc.node_count = idx + 1;
        Some(idx)
    }

    /// Publish a node as reachable to queries. Must be called after every
    /// per-node array is populated and every back-edge into `idx` is written.
    #[inline]
    pub(crate) fn publish_live(&self, idx: usize) {
        // Release so that all prior neighbor/edge/id writes are observed by
        // readers that Acquire-load `node_state == NODE_LIVE`.
        self.node_state
            .with(idx, |s| s.store(NODE_LIVE, Ordering::Release));
    }

    /// Legacy synchronous variant retained for the offline (`&mut self`) build
    /// paths that still assume single-threaded insertion. Reservers under the
    /// same alloc mutex, publishes LIVE immediately, and returns the idx.
    pub(crate) fn register_node(
        &mut self,
        point_id: PointId,
        vector: Vector,
        level: usize,
    ) -> usize {
        let idx = self.alloc_node(point_id, vector, level).unwrap_or_else(|| {
            self.alloc
                .lock()
                .expect("alloc mutex poisoned")
                .point_to_idx[&point_id]
        });
        self.publish_live(idx);
        idx
    }
}

impl HNSWIndex {
    pub(crate) fn allocate_entry_point(&self, idx: usize, level: usize) {
        self.store_entry_level(idx, level);
    }
}

impl HNSWIndex {
    /// Compute and store L0 edge distances, enabling triangle-inequality neighbor
    /// skipping during search (`VECTORDB_TI_SKIP=true` / `use_ti_skip: Some(true)`).
    ///
    /// This is NOT called automatically on snapshot load — doing so caused a ~50%
    /// throughput regression from heap fragmentation (290k scattered Vec allocations
    /// competing with the flat vector array for cache). Call this explicitly after
    /// loading or building an index when TI skip is actually needed.
    pub fn build_edge_distances(&mut self) {
        let n = self.len();
        // Ensure a stable slot per node in the chunked array. All slots are
        // default-initialised (empty `Vec`) at chunk creation.
        while self.edge_dists_l0.len() < n {
            self.edge_dists_l0.push_default();
        }
        if let Some(l0) = self.layers.first() {
            let l0_view = l0.view();
            let ed_view = self.edge_dists_l0.view();
            for idx in 0..l0_view.len() {
                let nb: Vec<usize> = l0_view.get(idx).read().clone();
                if nb.is_empty() {
                    continue;
                }
                let src: Vec<f32> = self.vector_slice(idx).to_vec();
                let dists: Vec<f32> = nb
                    .iter()
                    .map(|&nbi| {
                        if nbi == idx {
                            return 0.0;
                        }
                        self.fast_score(&src, self.vector_slice(nbi))
                    })
                    .collect();
                *ed_view.get(idx).write() = dists;
            }
        }
    }
}

impl HNSWIndex {
    /// Convenience optimizer: applies the winning search-layout stack in one call.
    ///
    /// Performs in order:
    ///   1. `reorder_rcm()` — permute nodes for cache locality (−5–9% latency)
    ///   2. `quantize_all()` — build SQ8 codes for neighbor screening
    ///
    /// After this call, set `sq8_screen: Some(true)` in `SearchRuntimeOptions` to
    /// activate neighbor screening (−43–50% latency at equivalent recall).
    ///
    /// Benchmark results on NYT-256-Angular, Apple M2, recall@20:
    ///   ef=128: 1.104ms (pre-optimization) → 0.545ms (post), +1.3pp recall
    ///   ef=256: 1.960ms → 1.008ms, +0.8pp recall
    ///
    /// Note: clears any previously built SQ8 state before reordering, then rebuilds.
    pub fn enable_sq8_screening(&mut self) {
        self.reorder_rcm(); // clears quantized state as a side effect
        self.quantize_all(); // rebuild after reorder
    }
}

impl HNSWIndex {
    /// Permute node indices using Reverse Cuthill-McKee so that graph-adjacent
    /// nodes at L0 become memory-adjacent. Reduces cache miss rate during BFS.
    /// All node-indexed arrays (vectors, layers, idx_to_point, deleted,
    /// point_to_idx, edge_dists_l0) are permuted consistently.
    /// Note: clears any SQ8 quantized state (`quantized`/`quant_min`/`quant_scale`).
    /// Call `quantize_all()` again after reordering if SQ8 search is needed.
    pub fn reorder_rcm(&mut self) {
        // Quantized codes are indexed by node idx; they are invalidated by reordering.
        // Clear them so callers know to re-run quantize_all() after reorder.
        self.quantized.clear();
        self.quant_min.clear();
        self.quant_scale.clear();

        let n = self.len();
        if n == 0 {
            return;
        }

        // Build undirected adjacency list from L0.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        if let Some(l0) = self.layers.first() {
            let view = l0.view();
            for u in 0..view.len() {
                let nb_lock = view.get(u);
                for &v in nb_lock.read().iter() {
                    if v != u {
                        if !adj[u].contains(&v) {
                            adj[u].push(v);
                        }
                        if !adj[v].contains(&u) {
                            adj[v].push(u);
                        }
                    }
                }
            }
        }

        // RCM: BFS starting from the node with minimum degree.
        let start = (0..n).min_by_key(|&i| adj[i].len()).unwrap_or(0);
        let mut perm = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(u) = queue.pop_front() {
            perm.push(u);
            let mut nbrs: Vec<usize> = adj[u].iter().copied().filter(|&v| !visited[v]).collect();
            nbrs.sort_by_key(|&v| adj[v].len()); // ascending degree
            for v in nbrs {
                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }
        // Handle disconnected nodes.
        for i in 0..n {
            if !visited[i] {
                perm.push(i);
            }
        }
        perm.reverse(); // Reverse Cuthill-McKee

        // Build inverse permutation: inv_perm[old_idx] = new_idx
        let mut inv_perm = vec![0usize; n];
        for (new_idx, &old_idx) in perm.iter().enumerate() {
            inv_perm[old_idx] = new_idx;
        }

        // Permute vectors into a fresh VectorArena.
        const CHUNK_CAP: usize = 4096;
        let dim = self.dim;
        let new_vectors = VectorArena::new(dim, CHUNK_CAP);
        for &old_idx in &perm {
            new_vectors.push(self.vectors.get_direct(old_idx));
        }
        self.vectors = new_vectors;

        // Permute idx_to_point, deleted, levels, node_state, and ids.
        let old_itp: Vec<PointId> = {
            let mut alloc = self.alloc.lock().expect("alloc mutex poisoned");
            std::mem::take(&mut alloc.idx_to_point)
        };
        let new_del: ChunkedArray<AtomicBool> = ChunkedArray::new(CHUNK_CAP);
        let new_lvl: ChunkedArray<AtomicU8> = ChunkedArray::new(CHUNK_CAP);
        let new_state: ChunkedArray<AtomicU8> = ChunkedArray::new(CHUNK_CAP);
        let new_ids: ChunkedArray<AtomicU64> = ChunkedArray::new(CHUNK_CAP);
        for &old_idx in &perm {
            let di = new_del.push_default();
            let was_deleted = self.deleted.with(old_idx, |b| b.load(Ordering::Relaxed));
            new_del.with(di, |b| b.store(was_deleted, Ordering::Relaxed));
            let li = new_lvl.push_default();
            let lvl = self.levels.with(old_idx, |l| l.load(Ordering::Relaxed));
            new_lvl.with(li, |l| l.store(lvl, Ordering::Relaxed));
            let si = new_state.push_default();
            let st = self.node_state.with(old_idx, |s| s.load(Ordering::Relaxed));
            new_state.with(si, |s| s.store(st, Ordering::Relaxed));
            let ii = new_ids.push_default();
            let id_val = self.ids.with(old_idx, |a| a.load(Ordering::Relaxed));
            new_ids.with(ii, |a| a.store(id_val, Ordering::Relaxed));
        }
        self.deleted = new_del;
        self.levels = new_lvl;
        self.node_state = new_state;
        self.ids = new_ids;

        // Rebuild alloc state from the permuted idx_to_point.
        {
            let mut alloc = self.alloc.lock().expect("alloc mutex poisoned");
            let new_itp: Vec<PointId> = perm.iter().map(|&o| old_itp[o]).collect();
            alloc.point_to_idx.clear();
            for (new_idx, &id) in new_itp.iter().enumerate() {
                alloc.point_to_idx.insert(id, new_idx);
            }
            alloc.node_count = new_itp.len();
            alloc.idx_to_point = new_itp;
        }

        // Update entry point, preserving the packed current_max_level as-is
        // (it may legitimately differ from the entry node's own level via set_current_max_level).
        let (old_ep, preserved_max_level) = self.entry_level();
        if let Some(ep) = old_ep {
            let new_ep = inv_perm[ep];
            self.store_entry_level(new_ep, preserved_max_level);
        }

        // Permute all layers: remap neighbor indices through inv_perm.
        for layer in self.layers.iter_mut() {
            let new_layer: ChunkedArray<RwLock<Vec<usize>>> = ChunkedArray::new(CHUNK_CAP);
            for _ in 0..n {
                new_layer.push_default();
            }
            let old_view = layer.view();
            let new_view = new_layer.view();
            for old_idx in 0..old_view.len() {
                let nb_lock = old_view.get(old_idx);
                let new_nbs: Vec<usize> = nb_lock.read().iter().map(|&nb| inv_perm[nb]).collect();
                *new_view.get(inv_perm[old_idx]).write() = new_nbs;
            }
            *layer = new_layer;
        }

        // Permute edge_dists_l0 if present.
        if !self.edge_dists_l0.is_empty() {
            let new_ed: ChunkedArray<RwLock<Vec<f32>>> = ChunkedArray::new(CHUNK_CAP);
            for _ in 0..n {
                new_ed.push_default();
            }
            let old_view = self.edge_dists_l0.view();
            let new_view = new_ed.view();
            for old_idx in 0..old_view.len() {
                let dists = old_view.get(old_idx).read().clone();
                *new_view.get(inv_perm[old_idx]).write() = dists;
            }
            self.edge_dists_l0 = new_ed;
        }
    }
}

impl HNSWIndex {
    pub(crate) fn validate_dim(&self, vec: &[f32]) -> Result<(), DBError> {
        if vec.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: vec.len(),
            });
        }
        Ok(())
    }
}

impl HNSWIndex {
    /// Build SQ8 quantization tables and encode all stored vectors.
    ///
    /// For Cosine: uses zero-centered encoding so the SQ8 dot product is an unbiased
    /// approximation of the true dot product. Stored code = round(v[d]*127.5 + 128),
    /// mapping [-1,1] → [0.5,255.5]. `quant_min` and `quant_scale` are unused for
    /// cosine and are left empty.
    ///
    /// For Euclidean/Dot: uses per-dimension min-max encoding. `quant_min[d]` and
    /// `quant_scale[d]` = (max-min)/255 are stored for query quantization.
    ///
    /// Not persisted in snapshots; call again after loading from disk.
    pub fn quantize_all(&mut self) {
        let n = self.len();
        let dim = self.dim;
        if n == 0 || dim == 0 {
            return;
        }
        let mut quantized = vec![0u8; n * dim];
        if self.metric == DistanceMetric::Cosine {
            // Zero-centered: v[d] ∈ [-1,1] → code ∈ [0,255]. Eliminates the per-vector
            // mean bias that destroys ranking quality for per-dim min-max on unit vectors.
            for idx in 0..n {
                let v = self.vector_slice(idx);
                for d in 0..dim {
                    quantized[idx * dim + d] =
                        (v[d] * 127.5 + 128.0).clamp(0.0, 255.0).round() as u8;
                }
            }
            self.quant_min = Vec::new();
            self.quant_scale = Vec::new();
        } else {
            let mut min_d = vec![f32::MAX; dim];
            let mut max_d = vec![f32::MIN; dim];
            for idx in 0..n {
                if self.deleted.with(idx, |b| b.load(Ordering::Acquire)) {
                    continue;
                }
                let v = self.vector_slice(idx);
                for d in 0..dim {
                    min_d[d] = min_d[d].min(v[d]);
                    max_d[d] = max_d[d].max(v[d]);
                }
            }
            let mut scale = vec![0.0f32; dim];
            for d in 0..dim {
                let range = max_d[d] - min_d[d];
                scale[d] = if range > 0.0 { range / 255.0 } else { 1.0 };
            }
            for idx in 0..n {
                let v = self.vector_slice(idx);
                for d in 0..dim {
                    quantized[idx * dim + d] =
                        ((v[d] - min_d[d]) / scale[d]).clamp(0.0, 255.0).round() as u8;
                }
            }
            self.quant_min = min_d;
            self.quant_scale = scale;
        }
        self.quantized = quantized;
    }

    /// Quantize a query vector into signed i16 codes for `sq8_approx_dot`.
    ///
    /// Cosine: `query_q[d] = round(q[d] * 127.5)` — centered, so dot with stored codes
    /// (which are offset by +128) gives an unbiased similarity estimate.
    /// Non-cosine: per-dimension min-max, consistent with `quantize_all`.
    pub(crate) fn quantize_query(&self, query: &[f32]) -> Vec<i16> {
        if self.metric == DistanceMetric::Cosine {
            query
                .iter()
                .map(|&v| (v * 127.5).clamp(-128.0, 127.0).round() as i16)
                .collect()
        } else {
            query
                .iter()
                .enumerate()
                .map(|(d, &v)| {
                    let min = self.quant_min.get(d).copied().unwrap_or(0.0);
                    let scale = self.quant_scale.get(d).copied().unwrap_or(1.0);
                    ((v - min) / scale).clamp(0.0, 255.0).round() as i16
                })
                .collect()
        }
    }

    /// Approximate SQ8 dot product. Result is monotonically related to true similarity
    /// (higher = closer). The f32 rerank pass overwrites raw scores before returning.
    ///
    /// Cosine: `sum_d (code[d] - 128) * query_q[d]` — the -128 removes the storage
    /// offset, leaving an unbiased approximation of 127.5^2 * true_dot(v, q).
    /// Non-cosine: `sum_d code[d] * query_q[d]` (no offset needed).
    #[inline]
    pub(crate) fn sq8_approx_dot(&self, query_q: &[i16], idx: usize) -> i32 {
        let v = &self.quantized[idx * self.dim..(idx + 1) * self.dim];
        if self.metric == DistanceMetric::Cosine {
            v.iter()
                .zip(query_q.iter())
                .map(|(&b, &q)| (b as i32 - 128) * (q as i32))
                .sum()
        } else {
            v.iter()
                .zip(query_q.iter())
                .map(|(&b, &q)| (b as i32) * (q as i32))
                .sum()
        }
    }

    /// Slice accessor for quantized codes. Panics if `quantized` is empty.
    #[inline]
    pub(crate) fn quantized_slice(&self, idx: usize) -> &[u8] {
        &self.quantized[idx * self.dim..(idx + 1) * self.dim]
    }

    /// Quantize a cosine query into i8 for screen_dot. Caller must ensure metric==Cosine.
    /// Same centering as quantize_all: q[d]*127.5 rounded to [-128, 127].
    pub(crate) fn quantize_query_i8(&self, query: &[f32]) -> Vec<i8> {
        query
            .iter()
            .map(|&v| (v * 127.5).clamp(-128.0, 127.0).round() as i8)
            .collect()
    }

    /// Fast dot product for SQ8 screening: stored u8 codes (centered at 128) × i8 query.
    /// Dispatches to NEON sdot on aarch64 (4 cache lines vs 16 for f32), scalar fallback
    /// elsewhere. Monotone with true cosine similarity — higher = closer.
    #[inline]
    pub(crate) fn screen_dot(query_i8: &[i8], stored: &[u8]) -> i32 {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { screen_dot_neon_sdot(query_i8, stored) };
        }
        #[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { screen_dot_avx2(query_i8, stored) };
        }
        screen_dot_scalar(query_i8, stored)
    }
}

fn screen_dot_scalar(query_i8: &[i8], stored: &[u8]) -> i32 {
    let n = query_i8.len().min(stored.len());
    let mut a = [0i32; 4];
    let mut i = 0;
    while i + 4 <= n {
        a[0] += (query_i8[i] as i32) * (stored[i] as i32 - 128);
        a[1] += (query_i8[i + 1] as i32) * (stored[i + 1] as i32 - 128);
        a[2] += (query_i8[i + 2] as i32) * (stored[i + 2] as i32 - 128);
        a[3] += (query_i8[i + 3] as i32) * (stored[i + 3] as i32 - 128);
        i += 4;
    }
    let mut acc = a[0] + a[1] + a[2] + a[3];
    while i < n {
        acc += (query_i8[i] as i32) * (stored[i] as i32 - 128);
        i += 1;
    }
    acc
}

/// NEON sdot: vdotq_s32 processes 4 groups of 4 i8 products per instruction.
/// For 256-dim: 256/16 = 16 sdot calls with 4 accumulators = 4 iterations of 64 values.
/// Memory: 256 B/vector (4 cache lines) vs 1024 B for f32 (16 cache lines).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn screen_dot_neon_sdot(query_i8: &[i8], stored: &[u8]) -> i32 {
    use std::arch::aarch64::*;
    let n = query_i8.len().min(stored.len());
    let sub128 = vdupq_n_u8(128);
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);
    let mut i = 0;
    while i + 64 <= n {
        let q0 = vld1q_s8(query_i8.as_ptr().add(i));
        let q1 = vld1q_s8(query_i8.as_ptr().add(i + 16));
        let q2 = vld1q_s8(query_i8.as_ptr().add(i + 32));
        let q3 = vld1q_s8(query_i8.as_ptr().add(i + 48));
        // Subtract 128 from u8: u8-128 wraps to the correct signed i8 bit pattern.
        let s0 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i)), sub128));
        let s1 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i + 16)), sub128));
        let s2 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i + 32)), sub128));
        let s3 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i + 48)), sub128));
        acc0 = vdotq_s32(acc0, q0, s0);
        acc1 = vdotq_s32(acc1, q1, s1);
        acc2 = vdotq_s32(acc2, q2, s2);
        acc3 = vdotq_s32(acc3, q3, s3);
        i += 64;
    }
    while i + 16 <= n {
        let q = vld1q_s8(query_i8.as_ptr().add(i));
        let s = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(stored.as_ptr().add(i)), sub128));
        acc0 = vdotq_s32(acc0, q, s);
        i += 16;
    }
    acc0 = vaddq_s32(acc0, acc1);
    acc2 = vaddq_s32(acc2, acc3);
    acc0 = vaddq_s32(acc0, acc2);
    let mut sum = vaddvq_s32(acc0);
    while i < n {
        sum += (*query_i8.get_unchecked(i) as i32) * (*stored.get_unchecked(i) as i32 - 128);
        i += 1;
    }
    sum
}

/// AVX2 path using maddubs + madd pattern.
#[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn screen_dot_avx2(query_i8: &[i8], stored: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    // maddubs(u8, i8): multiplies pairs of unsigned×signed bytes, adds adjacent pairs → i16.
    // We pass stored (u8) as first arg and query (i8 cast to u8 by adding 128) as second,
    // then correct the bias with a separate accumulator.
    let n = query_i8.len().min(stored.len());
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 32 <= n {
        let s = _mm256_loadu_si256(stored.as_ptr().add(i) as *const __m256i);
        // treat query i8 as u8 offset by 128: q_u8[d] = q_i8[d] + 128
        let q_raw = _mm256_loadu_si256(query_i8.as_ptr().add(i) as *const __m256i);
        let offset128 = _mm256_set1_epi8(-128i8); // = 128 as u8
        let q_u8 = _mm256_add_epi8(q_raw, offset128);
        // maddubs(s[u8], q_u8[u8]): s×q_u8, adjacent pairs summed → i16
        // But q_u8 is interpreted as i8 by maddubs... actually:
        // _mm256_maddubs_epi16(a: u8, b: i8) computes a*b not b*a. Treat s as u8, q_u8 as i8.
        // Since q_u8 = q_i8 + 128, the range is [0,255] interpreted as i8 wraps, but
        // this gives wrong products. Use widening multiply instead.
        // Widen s (u8) and q_raw (i8) to i16, multiply, reduce.
        let s_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s));
        let s_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s, 1));
        let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q_raw));
        let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q_raw, 1));
        // Subtract 128 from s as i16 to center it
        let bias = _mm256_set1_epi16(128);
        let s_lo_c = _mm256_sub_epi16(s_lo, bias);
        let s_hi_c = _mm256_sub_epi16(s_hi, bias);
        // Multiply i16 × i16 → keep low 16 bits, then use madd to accumulate into i32
        let prod_lo = _mm256_madd_epi16(_mm256_mullo_epi16(s_lo_c, q_lo), ones);
        let prod_hi = _mm256_madd_epi16(_mm256_mullo_epi16(s_hi_c, q_hi), ones);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(prod_lo, prod_hi));
        i += 32;
    }
    // Reduce acc (8 × i32) to scalar
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256(acc, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b_01_00_11_10));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 1));
    let mut result = _mm_cvtsi128_si32(sum32);
    while i < n {
        result += (*query_i8.get_unchecked(i) as i32) * (*stored.get_unchecked(i) as i32 - 128);
        i += 1;
    }
    result
}

impl HNSWIndex {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::time::Instant;

    fn parse_metric() -> DistanceMetric {
        match env::var("VECTORDB_KERNEL_METRIC")
            .unwrap_or_else(|_| "cosine".to_string())
            .to_lowercase()
            .as_str()
        {
            "dot" => DistanceMetric::Dot,
            "euclidean" | "l2" => DistanceMetric::Euclidean,
            _ => DistanceMetric::Cosine,
        }
    }

    fn gen_vec(seed: u32, dim: usize) -> Vector {
        let mut v = Vec::with_capacity(dim);
        let mut x = seed as u64 + 1;
        for _ in 0..dim {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((x >> 8) as u32) as f32 / (u32::MAX as f32);
            v.push(f * 2.0 - 1.0);
        }
        v
    }

    #[test]
    fn simd_kernels_match_scalar_within_tolerance() {
        // Validate that arch-specific SIMD kernels (AVX2/NEON) stay numerically close to the
        // scalar reference implementation. This catches accidental UB or logic drift.
        let dim = 256;
        let trials = 200;

        for seed in 0..trials {
            let query = gen_vec(seed as u32, dim);
            let vec = gen_vec((seed as u32).wrapping_add(10_000), dim);

            let dot_ref = dot_scalar(&query, &vec);
            let dot_fast = dot_product(&query, &vec);
            let dot_err = (dot_ref - dot_fast).abs();
            assert!(
                dot_err <= 1e-3,
                "dot mismatch seed={seed}: ref={dot_ref} fast={dot_fast} err={dot_err}"
            );

            let l2_ref = l2_scalar(&query, &vec);
            let l2_fast = l2_squared(&query, &vec);
            let l2_err = (l2_ref - l2_fast).abs();
            assert!(
                l2_err <= 1e-3,
                "l2 mismatch seed={seed}: ref={l2_ref} fast={l2_fast} err={l2_err}"
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_fast_score_kernel() {
        let dim = env::var("VECTORDB_KERNEL_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1536);
        let vecs = env::var("VECTORDB_KERNEL_VECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);
        let iters = env::var("VECTORDB_KERNEL_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let metric = parse_metric();

        let hnsw = HNSWIndex::new(metric, 16, 64, 16, dim);
        let mut data: Vec<Vector> = (0..vecs as u32).map(|i| gen_vec(i, dim)).collect();
        let mut query = gen_vec(999_983, dim);
        if metric == DistanceMetric::Cosine {
            query = hnsw.maybe_normalize(&query);
            for v in &mut data {
                *v = hnsw.maybe_normalize(v);
            }
        }

        let start = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..iters {
            for v in &data {
                acc += hnsw.fast_score(&query, v);
            }
        }
        let elapsed = start.elapsed();
        let total = (iters as u64) * (vecs as u64);
        let ns_per = elapsed.as_secs_f64() * 1e9 / total as f64;
        let scores_per_sec = total as f64 / elapsed.as_secs_f64();
        println!(
            "kernel metric={:?} dim={} vecs={} iters={} total={} scores/s={:.2} ns/score={:.2} acc={:.4}",
            metric, dim, vecs, iters, total, scores_per_sec, ns_per, acc
        );
    }

    #[test]
    #[ignore]
    fn bench_screen_dot_kernel() {
        let dim = env::var("VECTORDB_KERNEL_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let vecs = env::var("VECTORDB_KERNEL_VECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);
        let iters = env::var("VECTORDB_KERNEL_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(500);

        let query_f32: Vector = gen_vec(42, dim)
            .into_iter()
            .map(|v| v * 0.5 + 0.5)
            .collect();
        let query_norm = {
            let n: f32 = query_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
            query_f32.iter().map(|x| x / n).collect::<Vec<_>>()
        };
        let query_i8: Vec<i8> = query_norm
            .iter()
            .map(|&v| (v * 127.5).clamp(-128.0, 127.0).round() as i8)
            .collect();

        let stored_vecs: Vec<Vec<u8>> = (0..vecs as u32)
            .map(|seed| {
                let v = gen_vec(seed, dim);
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter()
                    .map(|&x| (x / n * 127.5 + 128.0).clamp(0.0, 255.0).round() as u8)
                    .collect()
            })
            .collect();

        // Correctness check: screen_dot vs scalar
        for s in &stored_vecs {
            let fast = HNSWIndex::screen_dot(&query_i8, s);
            let scalar = screen_dot_scalar(&query_i8, s);
            assert_eq!(fast, scalar, "screen_dot mismatch");
        }

        let mut acc: i64 = 0;
        let start = Instant::now();
        for _ in 0..iters {
            for s in &stored_vecs {
                acc += HNSWIndex::screen_dot(&query_i8, s) as i64;
            }
        }
        let elapsed = start.elapsed();
        let total = (iters as u64) * (vecs as u64);
        let ns_per = elapsed.as_secs_f64() * 1e9 / total as f64;
        println!(
            "screen_dot dim={} vecs={} iters={} total={} ns/call={:.2} acc={}",
            dim, vecs, iters, total, ns_per, acc
        );
    }
}
