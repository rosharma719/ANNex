use parking_lot::{RwLock, RwLockReadGuard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;

use crate::utils::errors::DBError;
use crate::utils::types::{DistanceMetric, PointId, Vector};

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

pub struct HNSWIndex {
    /// Per-node neighbor lists at each level.
    /// `layers[level][node]` is protected by a `RwLock` so concurrent
    /// inserts can modify different nodes simultaneously while searches
    /// hold shared read locks.
    pub(crate) layers: Vec<Vec<RwLock<Vec<usize>>>>,
    pub(crate) vectors: Vec<f32>,
    pub(crate) levels: Vec<usize>,
    pub(crate) entry_point: Option<usize>,
    pub(crate) metric: DistanceMetric,
    pub(crate) m: usize,
    pub(crate) m0: usize,
    pub(crate) stored_cap_l0: usize,
    pub(crate) ef: usize,
    pub(crate) ef_construct: usize,
    pub(crate) max_level_cap: usize,
    pub(crate) level_scale: f64,
    pub(crate) current_max_level: usize,
    pub(crate) dim: usize,
    pub(crate) deleted: Vec<bool>,
    // Maintained on deletion and recomputed on snapshot load; keeps query setup O(1).
    pub(crate) deleted_count: usize,
    /// Parallel to layers[0]: edge_dists_l0[idx] holds the distance from node idx to
    /// each of its L0 neighbors, co-indexed with layers[0][idx].
    /// Protected by the same logical lock as layers[0][idx]: always write under
    /// layers[0][idx].write() and read under layers[0][idx].read().
    pub(crate) edge_dists_l0: Vec<parking_lot::RwLock<Vec<f32>>>,
    /// SQ8 quantized vectors: quantized[idx * dim + d] = u8 encoding of dimension d.
    /// Empty until `quantize_all()` is called.
    pub(crate) quantized: Vec<u8>,
    /// Per-dimension minimum value used for SQ8 quantization.
    pub(crate) quant_min: Vec<f32>,
    /// Per-dimension scale: (max - min) / 255.0. Set to 1.0 for constant dimensions.
    pub(crate) quant_scale: Vec<f32>,
    pub(crate) point_to_idx: HashMap<PointId, usize>,
    pub(crate) idx_to_point: Vec<PointId>,
    pub(crate) exact_fallback_enabled: bool,
    pub(crate) exact_fallback_threshold: usize,
    /// Global read-write lock for concurrent insert coordination.
    /// Write: held exclusively during node allocation and entry-point promotion.
    /// Read: held by concurrent insert threads during their search phase so
    ///        the vectors/levels/layers vecs cannot reallocate under them.
    #[allow(dead_code)]
    pub(crate) alloc_lock: RwLock<()>,
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
    pub current_max_level: usize,
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
        let max_level_cap = Self::max_level_cap_from_env(max_level_cap);
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
        Self {
            layers: Vec::new(),
            vectors: Vec::with_capacity(dim * 64),
            levels: Vec::new(),
            entry_point: None,
            metric,
            m,
            m0: m * 2,
            stored_cap_l0: m * 2,
            ef,
            ef_construct: ef,
            max_level_cap,
            level_scale,
            current_max_level: 0,
            dim,
            deleted: Vec::new(),
            deleted_count: 0,
            edge_dists_l0: Vec::new(),
            quantized: Vec::new(),
            quant_min: Vec::new(),
            quant_scale: Vec::new(),
            point_to_idx: HashMap::new(),
            idx_to_point: Vec::new(),
            exact_fallback_enabled: exact_fallback_enabled_override().unwrap_or(false),
            exact_fallback_threshold: exact_fallback_threshold_override()
                .unwrap_or(DEFAULT_EXACT_FALLBACK_THRESHOLD),
            alloc_lock: RwLock::new(()),
        }
    }

    pub fn point_level(&self, point_id: PointId) -> Option<usize> {
        self.point_to_idx
            .get(&point_id)
            .and_then(|&idx| self.levels.get(idx).copied())
    }

    pub fn point_degree(&self, point_id: PointId, level: usize) -> Option<usize> {
        let idx = *self.point_to_idx.get(&point_id)?;
        self.layers
            .get(level)
            .and_then(|layer| layer.get(idx))
            .map(|rw| rw.read().len())
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
    pub(crate) fn vector_slice(&self, idx: usize) -> &[f32] {
        &self.vectors[idx * self.dim..(idx + 1) * self.dim]
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

    pub fn mark_deleted(&mut self, point_id: PointId) {
        let Some(idx) = self.idx_of(point_id) else {
            return;
        };
        if let Some(flag) = self.deleted.get_mut(idx)
            && !*flag
        {
            *flag = true;
            self.deleted_count += 1;
        }
        if Some(idx) == self.entry_point {
            self.entry_point = self.find_highest_level_entry_point();
        }
    }

    pub fn find_highest_level_entry_point(&self) -> Option<usize> {
        self.levels
            .iter()
            .enumerate()
            .filter(|(idx, _)| !self.deleted.get(*idx).copied().unwrap_or(false))
            .max_by_key(|(_, level)| *level)
            .map(|(idx, _)| idx)
    }

    #[inline]
    pub(crate) fn idx_of(&self, point_id: PointId) -> Option<usize> {
        self.point_to_idx.get(&point_id).copied()
    }

    #[inline]
    pub(crate) fn point_id(&self, idx: usize) -> PointId {
        self.idx_to_point[idx]
    }

    #[inline]
    pub(crate) fn get_vector_by_idx(&self, idx: usize) -> Option<&[f32]> {
        if self.deleted.get(idx).copied().unwrap_or(false) {
            None
        } else if (idx + 1) * self.dim <= self.vectors.len() {
            Some(self.vector_slice(idx))
        } else {
            None
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

    pub(crate) fn ensure_level_capacity(&mut self, level: usize, nodes_len: usize) {
        if self.layers.len() <= level {
            let start = self.layers.len();
            for level_idx in start..=level {
                let mut layer = Vec::with_capacity(nodes_len);
                for _ in 0..nodes_len {
                    layer.push(RwLock::new(Vec::with_capacity(
                        self.neighbor_list_capacity(level_idx),
                    )));
                }
                self.layers.push(layer);
            }
        }
    }

    pub(crate) fn extend_layers_for_new_node(&mut self, nodes_len: usize) {
        for level in 0..self.layers.len() {
            let cap = self.neighbor_list_capacity(level);
            if self.layers[level].len() < nodes_len {
                self.layers[level].push(RwLock::new(Vec::with_capacity(cap)));
            }
        }
        while self.edge_dists_l0.len() < nodes_len {
            self.edge_dists_l0
                .push(parking_lot::RwLock::new(Vec::new()));
        }
    }

    pub fn contains(&self, point_id: &PointId) -> bool {
        self.point_to_idx.contains_key(point_id)
    }

    pub fn len(&self) -> usize {
        self.idx_to_point.len()
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
            current_max_level: self.current_max_level,
            dim: self.dim,
            exact_fallback_enabled: self.exact_fallback_enabled,
            exact_fallback_threshold: self.exact_fallback_threshold,
        }
    }

    pub fn layer_neighbors(
        &self,
        level: usize,
        idx: usize,
    ) -> Option<RwLockReadGuard<'_, Vec<usize>>> {
        Some(self.layers.get(level)?.get(idx)?.read())
    }

    pub fn iter_vectors(&self) -> impl Iterator<Item = (&PointId, &[f32])> {
        let dim = self.dim;
        (0..self.len()).map(move |idx| {
            let vec = &self.vectors[idx * dim..(idx + 1) * dim];
            (&self.idx_to_point[idx], vec)
        })
    }

    pub fn iter_active_vectors(&self) -> impl Iterator<Item = (&PointId, &[f32])> {
        let dim = self.dim;
        (0..self.len())
            .filter(move |&idx| !self.deleted.get(idx).copied().unwrap_or(false))
            .map(move |idx| {
                let vec = &self.vectors[idx * dim..(idx + 1) * dim];
                (&self.idx_to_point[idx], vec)
            })
    }

    pub fn deleted_count(&self) -> usize {
        self.deleted_count
    }

    pub fn deleted_fraction(&self) -> f64 {
        let n = self.len();
        if n == 0 {
            return 0.0;
        }
        self.deleted_count() as f64 / n as f64
    }

    pub fn iter_active_levels(&self) -> impl Iterator<Item = usize> + '_ {
        self.levels
            .iter()
            .enumerate()
            .filter(|(idx, _)| !self.deleted.get(*idx).copied().unwrap_or(false))
            .map(|(_, lvl)| *lvl)
    }

    pub fn level_histogram(&self) -> Vec<usize> {
        let max_level = self.current_max_level;
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
        if self.deleted.get(idx).copied().unwrap_or(false) {
            None
        } else if (idx + 1) * self.dim <= self.vectors.len() {
            Some(self.vector_slice(idx))
        } else {
            None
        }
    }

    pub fn get_entry_point(&self) -> Option<u64> {
        self.entry_point.map(|idx| self.point_id(idx))
    }

    pub fn current_max_level(&self) -> usize {
        self.current_max_level
    }

    pub fn set_entry_point(&mut self, point_id: PointId) {
        if let Some(idx) = self.idx_of(point_id) {
            self.entry_point = Some(idx);
        }
    }

    pub fn set_current_max_level(&mut self, level: usize) {
        self.current_max_level = level;
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
    pub(crate) fn register_node(
        &mut self,
        point_id: PointId,
        vector: Vector,
        level: usize,
    ) -> usize {
        let idx = self.idx_to_point.len();
        self.vectors.extend_from_slice(&vector);
        self.levels.push(level);
        self.deleted.push(false);
        self.edge_dists_l0
            .push(parking_lot::RwLock::new(Vec::new()));
        self.idx_to_point.push(point_id);
        self.point_to_idx.insert(point_id, idx);
        idx
    }
}

impl HNSWIndex {
    pub(crate) fn allocate_entry_point(&mut self, idx: usize, level: usize) {
        self.entry_point = Some(idx);
        self.current_max_level = level;
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
        let dim = self.dim;
        // Allocate the outer Vec once; inner Vecs are populated below.
        if self.edge_dists_l0.is_empty() {
            self.edge_dists_l0 = (0..n)
                .map(|_| parking_lot::RwLock::new(Vec::new()))
                .collect();
        }
        if let Some(l0) = self.layers.first() {
            for (idx, nb_lock) in l0.iter().enumerate() {
                let nb: Vec<usize> = nb_lock.read().clone();
                if nb.is_empty() {
                    continue;
                }
                let src: Vec<f32> = self.vectors[idx * dim..(idx + 1) * dim].to_vec();
                let dists: Vec<f32> = nb
                    .iter()
                    .map(|&n| {
                        if n == idx {
                            return 0.0;
                        }
                        self.fast_score(&src, self.vector_slice(n))
                    })
                    .collect();
                *self.edge_dists_l0[idx].write() = dists;
            }
        }
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
            for (u, nb_lock) in l0.iter().enumerate() {
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

        // Permute vectors (flat slice: each node occupies `dim` consecutive f32s).
        let dim = self.dim;
        let mut new_vectors = vec![0.0f32; n * dim];
        for (new_idx, &old_idx) in perm.iter().enumerate() {
            let src = &self.vectors[old_idx * dim..(old_idx + 1) * dim];
            new_vectors[new_idx * dim..(new_idx + 1) * dim].copy_from_slice(src);
        }
        self.vectors = new_vectors;

        // Permute idx_to_point, deleted, and levels.
        let old_itp = std::mem::take(&mut self.idx_to_point);
        let old_del = std::mem::take(&mut self.deleted);
        let old_lvl = std::mem::take(&mut self.levels);
        self.idx_to_point = perm.iter().map(|&o| old_itp[o]).collect();
        self.deleted = perm.iter().map(|&o| old_del[o]).collect();
        self.levels = perm.iter().map(|&o| old_lvl[o]).collect();

        // Rebuild point_to_idx from new idx_to_point.
        self.point_to_idx.clear();
        for (new_idx, &id) in self.idx_to_point.iter().enumerate() {
            self.point_to_idx.insert(id, new_idx);
        }

        // Update entry point.
        if let Some(ep) = self.entry_point {
            self.entry_point = Some(inv_perm[ep]);
        }

        // Permute all layers: remap neighbor indices through inv_perm.
        for layer in self.layers.iter_mut() {
            let new_layer: Vec<parking_lot::RwLock<Vec<usize>>> = (0..n)
                .map(|_| parking_lot::RwLock::new(Vec::new()))
                .collect();
            for (old_idx, nb_lock) in layer.iter().enumerate() {
                let new_nbs: Vec<usize> = nb_lock.read().iter().map(|&nb| inv_perm[nb]).collect();
                *new_layer[inv_perm[old_idx]].write() = new_nbs;
            }
            *layer = new_layer;
        }

        // Permute edge_dists_l0 if present.
        if !self.edge_dists_l0.is_empty() {
            let new_ed: Vec<parking_lot::RwLock<Vec<f32>>> = (0..n)
                .map(|_| parking_lot::RwLock::new(Vec::new()))
                .collect();
            for (old_idx, ed_lock) in self.edge_dists_l0.iter().enumerate() {
                let dists = ed_lock.read().clone();
                *new_ed[inv_perm[old_idx]].write() = dists;
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
                if self.deleted.get(idx).copied().unwrap_or(false) {
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
}

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
}
