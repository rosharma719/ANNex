use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread_local;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use serde::{Deserialize, Serialize};

use crate::payload_storage::filters::Filter;
use crate::payload_storage::stores::PayloadIndex;
use crate::segment::wal::{WalConfig, WalReader, WalRecord, WalWriter};
use crate::utils::errors::DBError;
use crate::utils::io::{adler32, write_atomic_with_checksum};
use crate::utils::payload::{Payload, PayloadValue};
use crate::utils::types::{PointId, Vector};
use crate::vector::hnsw::{
    HNSWIndex, HnswConfigSummary, HnswSnapshot, ScoredPoint, SearchRuntimeOptions, SearchStats,
};

/// A segment is the core unit that wraps vector storage, indexing, payloads, and deletion.
pub struct Segment {
    hnsw: HNSWIndex,
    payload_index: PayloadIndex,
    payloads: HashMap<PointId, Payload>,
    // This set is maintained in parallel with the HNSW deletion set.
    deleted: HashSet<PointId>,
    next_id: PointId,
    op_count: u64,
    wal: Option<WalWriter>,
    rebuilding: AtomicBool,
    memory_frozen: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::types::DistanceMetric;

    #[test]
    fn searches_fail_when_rebuilding_flag_set() {
        let mut seg = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));
        for i in 0..5u64 {
            seg.insert_with_id(i, vec![i as f32, 0.0], None).unwrap();
        }
        seg.rebuilding.store(true, Ordering::SeqCst);
        let res = seg.search(&vec![1.0, 0.0], 1);
        assert!(matches!(res, Err(DBError::SearchError(_))));
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SegmentSnapshot {
    hnsw: HnswSnapshot,
    payload_index: PayloadIndex,
    payloads: HashMap<PointId, Payload>,
    deleted: HashSet<PointId>,
    next_id: PointId,
}

const SEGMENT_SNAPSHOT_MAGIC: [u8; 4] = *b"VDBS";
const SEGMENT_SNAPSHOT_VERSION: u32 = 3;
const SEGMENT_SNAPSHOT_FOOTER: [u8; 4] = *b"VDBF";
const SEGMENT_SNAPSHOT_META_MAGIC: [u8; 4] = *b"VDBM";
const SEGMENT_SNAPSHOT_META_VERSION: u32 = 1;

static MAX_RSS_BYTES: OnceLock<Option<u64>> = OnceLock::new();
static OOM_SNAPSHOT_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SnapshotMetadata {
    pub created_at_ms: u64,
    pub hnsw: HnswConfigSummary,
    pub points: usize,
    pub payloads: usize,
}

#[derive(Serialize, Deserialize)]
struct HnswSnapshotV1 {
    layers: HashMap<usize, HashMap<PointId, Vec<PointId>>>,
    vectors: HashMap<PointId, Vector>,
    levels: HashMap<PointId, usize>,
    entry_point: Option<PointId>,
    metric: crate::utils::types::DistanceMetric,
    m: usize,
    ef: usize,
    ef_construct: usize,
    max_level_cap: usize,
    level_scale: f64,
    current_max_level: usize,
    dim: usize,
    deleted: HashSet<PointId>,
    exact_fallback_enabled: bool,
    exact_fallback_threshold: usize,
}

impl From<HnswSnapshotV1> for HnswSnapshot {
    fn from(snapshot: HnswSnapshotV1) -> Self {
        let m0 = snapshot.m * 2;
        Self {
            layers: snapshot.layers,
            vectors: snapshot.vectors,
            levels: snapshot.levels,
            entry_point: snapshot.entry_point,
            metric: snapshot.metric,
            m: snapshot.m,
            m0,
            stored_cap_l0: m0,
            ef: snapshot.ef,
            ef_construct: snapshot.ef_construct,
            max_level_cap: snapshot.max_level_cap,
            level_scale: snapshot.level_scale,
            current_max_level: snapshot.current_max_level,
            dim: snapshot.dim,
            deleted: snapshot.deleted,
            exact_fallback_enabled: snapshot.exact_fallback_enabled,
            exact_fallback_threshold: snapshot.exact_fallback_threshold,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SegmentSnapshotV1 {
    hnsw: HnswSnapshotV1,
    payload_index: PayloadIndex,
    payloads: HashMap<PointId, Payload>,
    deleted: HashSet<PointId>,
    next_id: PointId,
}

impl From<SegmentSnapshotV1> for SegmentSnapshot {
    fn from(snapshot: SegmentSnapshotV1) -> Self {
        Self {
            hnsw: snapshot.hnsw.into(),
            payload_index: snapshot.payload_index,
            payloads: snapshot.payloads,
            deleted: snapshot.deleted,
            next_id: snapshot.next_id,
        }
    }
}

#[derive(Default)]
struct InsertTiming {
    count: usize,
    hnsw: Duration,
    payload_idx: Duration,
    filter_edges: Duration,
    total: Duration,
}

thread_local! {
    static INSERT_TIMINGS: RefCell<InsertTiming> = RefCell::new(InsertTiming::default());
    static INSERT_TOTAL: RefCell<usize> = RefCell::new(0);
}

impl Segment {
    /// Construct a `Segment` from a pre-built HNSW index. Prefer
    /// [`Segment::with_config`] for typical usage.
    #[doc(hidden)]
    pub fn new(hnsw: HNSWIndex) -> Self {
        let mut segment = Self {
            hnsw,
            payload_index: PayloadIndex::new(),
            payloads: HashMap::new(),
            deleted: HashSet::new(),
            next_id: 1,
            op_count: 0,
            wal: None,
            rebuilding: AtomicBool::new(false),
            memory_frozen: false,
        };
        if let Err(err) = segment.enable_wal_from_env_default_dir(None) {
            log::warn!(target: "segment::wal", "failed to enable WAL from env: {}", err);
        }
        segment
    }

    /// Build a fresh `Segment` with an empty HNSW index configured with the
    /// given parameters.
    ///
    /// * `metric`       — distance metric.
    /// * `m`            — target neighbours per node at levels > 0.
    /// * `ef`           — default search candidate list size.
    /// * `max_level_cap`— maximum HNSW level.
    /// * `dim`          — vector dimensionality.
    pub fn with_config(
        metric: crate::utils::types::DistanceMetric,
        m: usize,
        ef: usize,
        max_level_cap: usize,
        dim: usize,
    ) -> Self {
        Self::new(HNSWIndex::new(metric, m, ef, max_level_cap, dim))
    }

    /// Insert a new vector and optional payload. Auto-generates ID.
    pub fn insert(&mut self, vector: Vector, payload: Option<Payload>) -> Result<PointId, DBError> {
        let point_id = self.next_id;
        self.insert_with_id(point_id, vector, payload)
    }

    /// Insert a vector with a caller-provided ID. Fails if the ID already exists or was deleted.
    pub fn insert_with_id(
        &mut self,
        point_id: PointId,
        vector: Vector,
        payload: Option<Payload>,
    ) -> Result<PointId, DBError> {
        self.insert_with_id_internal(point_id, vector, payload, true)
    }

    fn insert_with_id_internal(
        &mut self,
        point_id: PointId,
        vector: Vector,
        payload: Option<Payload>,
        write_wal: bool,
    ) -> Result<PointId, DBError> {
        if write_wal {
            self.reject_if_frozen()?;
            self.enforce_memory_cap()?;
        }
        let log_timing = Self::log_insert_timing();
        let total_start = if log_timing {
            Some(Instant::now())
        } else {
            None
        };
        let mut last = total_start;
        let mut chunk_start = total_start;
        let mut hnsw_dur = Duration::from_millis(0);
        let mut payload_idx_dur = Duration::from_millis(0);
        let mut filter_edges_dur = Duration::from_millis(0);

        if self.hnsw.contains(&point_id)
            || self.payloads.contains_key(&point_id)
            || self.deleted.contains(&point_id)
        {
            return Err(DBError::DuplicatePointId(point_id));
        }

        if write_wal {
            self.append_wal(WalRecord::Insert {
                point_id,
                vector: vector.clone(),
                payload: payload.clone(),
            })?;
        }

        self.hnsw.insert(point_id, vector.clone())?;
        if let Some(t) = last.as_mut() {
            hnsw_dur = t.elapsed();
            *t = Instant::now();
        }

        if let Some(p) = payload {
            self.payload_index.insert(point_id, &p);
            self.payloads.insert(point_id, p.clone());
            if let Some(t) = last.as_mut() {
                payload_idx_dur = t.elapsed();
                *t = Instant::now();
            }

            let filter_keys = Self::filter_keys_for_payload(&p);

            if !filter_keys.is_empty() && Self::filter_edges_enabled() {
                self.hnsw.build_filter_aware_edges(
                    point_id,
                    &vector,
                    &p,
                    &self.payload_index,
                    &self.payloads,
                    &filter_keys,
                )?;
                if let Some(t) = last.as_mut() {
                    filter_edges_dur = t.elapsed();
                    *t = Instant::now();
                }
            }
        }

        if point_id >= self.next_id {
            self.next_id = point_id.saturating_add(1);
        }

        if let Some(start) = total_start {
            let total = start.elapsed();
            // Accumulate and log every CHUNK inserts to avoid log spam.
            const CHUNK: usize = 5000;
            INSERT_TIMINGS.with(|cell| {
                INSERT_TOTAL.with(|tc| {
                    let mut s = cell.borrow_mut();
                    let mut total_count = tc.borrow_mut();
                    s.count += 1;
                    s.hnsw += hnsw_dur;
                    s.payload_idx += payload_idx_dur;
                    s.filter_edges += filter_edges_dur;
                    s.total += total;
                    if s.count % CHUNK == 0 {
                        let c = s.count as u32;
                        *total_count += s.count;
                        let chunk_elapsed = chunk_start.map(|cs| cs.elapsed()).unwrap_or_default();
                        let msg = format!(
                            "[insert_timing_chunk] n={} cum_n={} avg_hnsw={:?} avg_payload_idx={:?} avg_filter_edges={:?} avg_total={:?} chunk_elapsed={:?}",
                            s.count,
                            *total_count,
                            s.hnsw / c,
                            s.payload_idx / c,
                            s.filter_edges / c,
                            s.total / c,
                            chunk_elapsed
                        );
                        log::info!(target: "segment", "{}", msg);
                        // Reset for next chunk
                        *s = InsertTiming::default();
                        chunk_start = Some(Instant::now());
                    }
                });
            });
        }

        self.op_count = self.op_count.saturating_add(1);
        Ok(point_id)
    }

    /// Parallel bulk insert. Internally chunks to `parallelism * 4` entries per
    /// `par_insert_batch` call so nearby vectors in the same outer chunk can see
    /// each other's back-edges. No WAL, no payload support — bulk-load only.
    pub fn bulk_load(&mut self, entries: &[(PointId, Vector)]) -> Result<usize, DBError> {
        let parallelism = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let inner = (parallelism * 4).max(1);
        let mut total = 0usize;
        for chunk in entries.chunks(inner) {
            let n = self.hnsw.par_insert_batch(chunk)?;
            total += n;
            for &(point_id, _) in chunk {
                if point_id >= self.next_id {
                    self.next_id = point_id.saturating_add(1);
                }
            }
            self.op_count = self.op_count.saturating_add(n as u64);
        }
        Ok(total)
    }

    /// Get the vector for a given point ID, if it exists and is not deleted.
    pub fn get_vector(&self, point_id: PointId) -> Option<&[f32]> {
        if self.deleted.contains(&point_id) {
            return None;
        }
        self.hnsw.get_vector(&point_id)
    }

    pub fn delete(&mut self, point_id: PointId) -> Result<(), DBError> {
        self.delete_internal(point_id, true)
    }

    fn delete_internal(&mut self, point_id: PointId, write_wal: bool) -> Result<(), DBError> {
        if write_wal {
            self.reject_if_frozen()?;
        }
        // If the point is already marked as deleted OR is no longer in the index,
        // treat it as already deleted.
        if self.deleted.contains(&point_id) || !self.hnsw.contains(&point_id) {
            return Ok(());
        }

        if write_wal {
            self.append_wal(WalRecord::Delete { point_id })?;
        }

        if let Some(p) = self.payloads.get(&point_id) {
            self.payload_index.remove(point_id, p);
        }
        self.payloads.remove(&point_id);

        self.deleted.insert(point_id);
        self.hnsw.mark_deleted(point_id);
        self.op_count = self.op_count.saturating_add(1);

        let deleted_count = self.deleted.len();
        let total_count = self.hnsw.len();

        const MIN_DELETIONS_BEFORE_PURGE: usize = 100;
        const MAX_DELETION_RATIO: f32 = 0.25;

        if Self::purge_on_delete_enabled()
            && deleted_count >= MIN_DELETIONS_BEFORE_PURGE
            && (deleted_count as f32 / total_count as f32) >= MAX_DELETION_RATIO
        {
            log::info!(
                target: "segment",
                "[DELETE] Triggering purge: {}/{} ({:.2}%) deleted",
                deleted_count,
                total_count,
                100.0 * deleted_count as f32 / total_count as f32
            );
            self.purge()?;
        }

        Ok(())
    }

    /// Update payload for an existing point ID.
    pub fn update_payload(&mut self, point_id: PointId, payload: Payload) -> Result<(), DBError> {
        self.update_payload_internal(point_id, payload, true)
    }

    fn update_payload_internal(
        &mut self,
        point_id: PointId,
        payload: Payload,
        write_wal: bool,
    ) -> Result<(), DBError> {
        if write_wal {
            self.reject_if_frozen()?;
        }
        if self.deleted.contains(&point_id) || !self.hnsw.contains(&point_id) {
            return Err(DBError::NotFound(point_id));
        }

        if write_wal {
            self.append_wal(WalRecord::UpdatePayload {
                point_id,
                payload: payload.clone(),
            })?;
        }

        if let Some(old) = self.payloads.get(&point_id) {
            self.payload_index.remove(point_id, old);
        }
        self.payload_index.insert(point_id, &payload);
        self.payloads.insert(point_id, payload);
        self.op_count = self.op_count.saturating_add(1);
        Ok(())
    }

    pub fn search(&self, query: &Vector, top_k: usize) -> Result<Vec<ScoredPoint>, DBError> {
        self.search_with_options(query, top_k, &SearchRuntimeOptions::default())
    }

    pub fn search_with_options(
        &self,
        query: &Vector,
        top_k: usize,
        opts: &SearchRuntimeOptions,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        self.ensure_not_rebuilding()?;
        let total_non_deleted = self.hnsw.len().saturating_sub(self.deleted.len());
        if total_non_deleted == 0 {
            return Err(DBError::SearchError(
                "No active points available to search.".into(),
            ));
        }

        // HNSWIndex now internally skips deleted points.
        let candidates = self.hnsw.search_with_options(query, top_k, opts)?;
        // (The following filter is kept as extra safety.)
        let filtered = candidates
            .into_iter()
            .filter(|sp| !self.deleted.contains(&sp.id))
            .take(top_k)
            .collect();

        Ok(filtered)
    }

    pub fn search_with_stats(
        &self,
        query: &Vector,
        top_k: usize,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), DBError> {
        self.search_with_stats_with_options(query, top_k, &SearchRuntimeOptions::default())
    }

    pub fn search_with_stats_with_options(
        &self,
        query: &Vector,
        top_k: usize,
        opts: &SearchRuntimeOptions,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), DBError> {
        self.ensure_not_rebuilding()?;
        let total_non_deleted = self.hnsw.len().saturating_sub(self.deleted.len());
        if total_non_deleted == 0 {
            return Err(DBError::SearchError(
                "No active points available to search.".into(),
            ));
        }

        let (candidates, stats) = self
            .hnsw
            .search_with_stats_with_options(query, top_k, opts)?;
        let filtered = candidates
            .into_iter()
            .filter(|sp| !self.deleted.contains(&sp.id))
            .take(top_k)
            .collect();

        Ok((filtered, stats))
    }

    pub fn search_with_filter(
        &self,
        query: &Vector,
        top_k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        self.search_with_filter_with_options(query, top_k, filter, &SearchRuntimeOptions::default())
    }

    pub fn search_with_filter_with_options(
        &self,
        query: &Vector,
        top_k: usize,
        filter: Option<&Filter>,
        opts: &SearchRuntimeOptions,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        self.ensure_not_rebuilding()?;
        let total_non_deleted = self.hnsw.len().saturating_sub(self.deleted.len());
        if total_non_deleted == 0 {
            return Err(DBError::SearchError(
                "No active points available to search.".into(),
            ));
        }

        let results = self.hnsw.in_place_filtered_search(
            query,
            top_k * 4,
            opts,
            &self.payloads,
            &self.payload_index,
            filter,
        )?;

        let filtered: Vec<_> = results
            .into_iter()
            .filter(|sp| !self.deleted.contains(&sp.id))
            .take(top_k)
            .collect();

        Ok(filtered)
    }

    /// Internal unfiltered search (used for diagnostics or filtered versions).
    pub fn search_unfiltered(
        &self,
        query: &Vector,
        top_k: usize,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        self.ensure_not_rebuilding()?;
        self.hnsw.search(query, top_k)
    }

    /// Get payload metadata for a point.
    pub fn get_payload(&self, point_id: PointId) -> Option<&Payload> {
        self.payloads.get(&point_id)
    }

    /// Check if a point is deleted.
    pub fn is_deleted(&self, point_id: PointId) -> bool {
        self.deleted.contains(&point_id)
    }

    pub fn purge(&mut self) -> Result<(), DBError> {
        if self.rebuilding.swap(true, Ordering::SeqCst) {
            return Err(DBError::SearchError(
                "Segment rebuild already in progress.".into(),
            ));
        }
        struct RebuildGuard<'a> {
            flag: &'a AtomicBool,
        }
        impl<'a> Drop for RebuildGuard<'a> {
            fn drop(&mut self) {
                self.flag.store(false, Ordering::SeqCst);
            }
        }
        let _guard = RebuildGuard {
            flag: &self.rebuilding,
        };
        let mut new_hnsw = HNSWIndex::new(
            self.hnsw.metric(),
            self.hnsw.m(),
            self.hnsw.ef(),
            self.hnsw.max_level_cap(),
            self.hnsw.dim(),
        );

        let mut new_payload_index = PayloadIndex::new();
        let mut new_payloads = HashMap::new();

        for (&id, vector) in self.hnsw.iter_vectors() {
            if self.deleted.contains(&id) {
                continue;
            }

            // Reinsert into HNSW
            new_hnsw.insert(id, vector.to_vec())?;

            if let Some(p) = self.payloads.get(&id) {
                // Reinsert into payload structures
                new_payload_index.insert(id, p);
                new_payloads.insert(id, p.clone());

                // Rebuild filter-aware edges
                let filter_keys = Self::filter_keys_for_payload(p);

                if Self::filter_edges_enabled() {
                    new_hnsw.build_filter_aware_edges(
                        id,
                        vector,
                        p,
                        &new_payload_index,
                        &new_payloads,
                        &filter_keys,
                    )?;
                }
            }
        }

        // Swap in the rebuilt structures
        self.hnsw = new_hnsw;
        self.payload_index = new_payload_index;
        self.payloads = new_payloads;

        self.deleted.clear();

        Ok(())
    }

    /// Immutable reference to underlying HNSW index.
    #[doc(hidden)]
    pub fn hnsw(&self) -> &HNSWIndex {
        &self.hnsw
    }

    /// Immutable reference to point payloads.
    #[doc(hidden)]
    pub fn payloads(&self) -> &HashMap<PointId, Payload> {
        &self.payloads
    }

    /// Mutable reference to underlying HNSW index (for tuning ef_construct in benches/tests).
    #[doc(hidden)]
    pub fn hnsw_mut(&mut self) -> &mut HNSWIndex {
        &mut self.hnsw
    }

    #[doc(hidden)]
    pub fn payload_index(&self) -> &PayloadIndex {
        &self.payload_index
    }

    fn reject_if_frozen(&self) -> Result<(), DBError> {
        if self.memory_frozen {
            return Err(DBError::MemoryCapExceeded(
                "segment unloaded after memory cap exceeded".into(),
            ));
        }
        Ok(())
    }

    fn enforce_memory_cap(&mut self) -> Result<(), DBError> {
        let Some(cap_bytes) = max_rss_bytes() else {
            return Ok(());
        };
        let Some(rss_bytes) = current_rss_bytes() else {
            return Ok(());
        };
        if rss_bytes <= cap_bytes {
            return Ok(());
        }
        let snapshot_path = oom_snapshot_path();
        if let Some(path) = snapshot_path.as_ref() {
            let snapshot_result = if self.wal.is_some() {
                self.save_to_path_and_checkpoint(path)
            } else {
                self.save_to_path(path)
            };
            if let Err(err) = snapshot_result {
                log::warn!(
                    target: "segment::memory",
                    "memory cap snapshot failed ({}): {}",
                    path.display(),
                    err
                );
            }
        } else {
            log::warn!(
                target: "segment::memory",
                "memory cap exceeded but VECTORDB_OOM_SNAPSHOT_PATH is not set"
            );
        }
        self.unload_after_memory_cap();
        Err(DBError::MemoryCapExceeded(format!(
            "rss_bytes={} cap_bytes={}",
            rss_bytes, cap_bytes
        )))
    }

    fn unload_after_memory_cap(&mut self) {
        let cfg = self.hnsw.config_summary();
        let mut hnsw = HNSWIndex::new(cfg.metric, cfg.m, cfg.ef, cfg.max_level_cap, cfg.dim);
        hnsw.set_m0(cfg.m0);
        if cfg.stored_cap_l0 > 0 {
            hnsw.set_stored_cap_l0(cfg.stored_cap_l0);
        }
        hnsw.set_ef_construct(cfg.ef_construct);
        hnsw.set_exact_fallback_enabled(cfg.exact_fallback_enabled);
        hnsw.set_exact_fallback_threshold(cfg.exact_fallback_threshold);
        self.hnsw = hnsw;
        self.payload_index = PayloadIndex::new();
        self.payloads.clear();
        self.deleted.clear();
        self.op_count = 0;
        self.wal = None;
        self.memory_frozen = true;
    }

    /// Toggle filter-aware edge building via env var. Defaults to false.
    fn filter_edges_enabled() -> bool {
        env::var("VECTORDB_FILTER_EDGES")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    }

    /// Toggle per-insert timing logs via env var. Defaults to false.
    fn log_insert_timing() -> bool {
        env::var("VECTORDB_LOG_INSERT_TIMING")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    }

    fn log_snapshot_load_timing() -> bool {
        env::var("VECTORDB_LOG_SNAPSHOT_LOAD_TIMING")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    }

    /// Toggle purge-on-delete behavior. Defaults to false (keep tombstones).
    fn purge_on_delete_enabled() -> bool {
        env::var("VECTORDB_PURGE_DELETIONS")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    }

    /// Persist the entire segment (graph, vectors, payloads, and inverted index) to disk.
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), DBError> {
        let snapshot = self.build_snapshot();
        let metadata = self.snapshot_metadata();
        Self::persist_snapshot(&snapshot, Some(&metadata), path)
    }

    /// Restore a segment that was previously persisted with `save_to_path`.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, DBError> {
        let (segment, _metadata) = Self::load_from_path_with_wal_and_metadata(path, None)?;
        Ok(segment)
    }

    /// Restore a segment and optionally replay a WAL (defaults to `<snapshot>.wal` if present).
    pub fn load_from_path_with_wal<P: AsRef<Path>>(
        path: P,
        wal_path: Option<PathBuf>,
    ) -> Result<Self, DBError> {
        let (segment, _metadata) = Self::load_from_path_with_wal_and_metadata(path, wal_path)?;
        Ok(segment)
    }

    /// Restore a segment and return snapshot metadata when available.
    pub fn load_from_path_with_metadata<P: AsRef<Path>>(
        path: P,
    ) -> Result<(Self, Option<SnapshotMetadata>), DBError> {
        Self::load_from_path_with_wal_and_metadata(path, None)
    }

    /// Restore a segment and optionally replay a WAL, returning snapshot metadata when available.
    pub fn load_from_path_with_wal_and_metadata<P: AsRef<Path>>(
        path: P,
        wal_path: Option<PathBuf>,
    ) -> Result<(Self, Option<SnapshotMetadata>), DBError> {
        let timing_enabled = Self::log_snapshot_load_timing();
        let total_start = Instant::now();

        let t_read = Instant::now();
        let bytes = std::fs::read(&path)?;
        let read_ms = t_read.elapsed().as_millis();
        let (payload, checksum) = if bytes.len() >= 8
            && bytes[bytes.len() - 8..bytes.len() - 4] == SEGMENT_SNAPSHOT_FOOTER
        {
            let checksum = u32::from_le_bytes([
                bytes[bytes.len() - 4],
                bytes[bytes.len() - 3],
                bytes[bytes.len() - 2],
                bytes[bytes.len() - 1],
            ]);
            (&bytes[..bytes.len() - 8], Some(checksum))
        } else {
            (bytes.as_slice(), None)
        };

        let t_checksum = Instant::now();
        if let Some(expected) = checksum {
            let actual = adler32(payload);
            if actual != expected {
                return Err(DBError::SerializationError(anyhow!(
                    "segment snapshot checksum mismatch"
                )));
            }
        }
        let checksum_ms = t_checksum.elapsed().as_millis();

        let t_deser = Instant::now();
        let (snapshot, metadata): (SegmentSnapshot, Option<SnapshotMetadata>) =
            if payload.len() >= 8 && payload[..4] == SEGMENT_SNAPSHOT_MAGIC {
                let version = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
                let body = &payload[8..];
                match version {
                    2 => (
                        bincode::deserialize(body)
                            .map_err(|e| DBError::SerializationError(anyhow!(e)))?,
                        None,
                    ),
                    3 => {
                        let (snapshot_bytes, metadata) = Self::parse_snapshot_metadata(body)?;
                        (
                            bincode::deserialize(snapshot_bytes)
                                .map_err(|e| DBError::SerializationError(anyhow!(e)))?,
                            metadata,
                        )
                    }
                    _ => {
                        return Err(DBError::SerializationError(anyhow!(
                            "unsupported segment snapshot version {}",
                            version
                        )));
                    }
                }
            } else if let Ok(snapshot) = bincode::deserialize::<SegmentSnapshot>(payload) {
                (snapshot, None)
            } else {
                let legacy: SegmentSnapshotV1 = bincode::deserialize(payload)
                    .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
                (legacy.into(), None)
            };
        let deser_ms = t_deser.elapsed().as_millis();

        let t_build = Instant::now();
        let mut segment = Self {
            hnsw: HNSWIndex::from_snapshot(snapshot.hnsw),
            payload_index: snapshot.payload_index,
            payloads: snapshot.payloads,
            deleted: snapshot.deleted,
            next_id: snapshot.next_id,
            op_count: 0,
            wal: None,
            rebuilding: AtomicBool::new(false),
            memory_frozen: false,
        };
        let build_ms = t_build.elapsed().as_millis();

        let wal_path = wal_path.unwrap_or_else(|| Self::default_wal_path(path.as_ref()));
        let auto_replay = std::env::var("VECTORDB_WAL_AUTO_REPLAY")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);
        let mut wal_replay_ms = 0u128;
        if auto_replay && wal_path.exists() {
            let t_wal = Instant::now();
            let reader = WalReader::new(&wal_path);
            reader.replay(|record| segment.apply_wal_record(record))?;
            segment.wal = Some(WalWriter::open(WalConfig::from_env(wal_path))?);
            wal_replay_ms = t_wal.elapsed().as_millis();
        }

        if timing_enabled {
            let msg = format!(
                "snapshot_load path={:?} bytes={} read_ms={} checksum_ms={} deserialize_ms={} build_ms={} wal_replay_ms={} total_ms={}",
                path.as_ref(),
                bytes.len(),
                read_ms,
                checksum_ms,
                deser_ms,
                build_ms,
                wal_replay_ms,
                total_start.elapsed().as_millis(),
            );
            eprintln!("{msg}");
            log::info!(target: "snapshot", "{msg}");
        }

        Ok((segment, metadata))
    }

    pub fn enable_wal<P: AsRef<Path>>(&mut self, path: P) -> Result<(), DBError> {
        let writer = WalWriter::open(WalConfig::from_env(path.as_ref().to_path_buf()))?;
        self.wal = Some(writer);
        Ok(())
    }

    pub fn enable_wal_from_env_default_dir(
        &mut self,
        default_dir: Option<PathBuf>,
    ) -> Result<(), DBError> {
        if let Ok(path) = env::var("VECTORDB_WAL_PATH")
            && !path.is_empty()
        {
            return self.enable_wal(path);
        }
        if let Ok(dir) = env::var("VECTORDB_WAL_DIR")
            && !dir.is_empty()
        {
            let wal_path = Path::new(&dir).join("segment.wal");
            return self.enable_wal(wal_path);
        }
        if let Some(dir) = default_dir {
            let wal_path = dir.join("segment.wal");
            return self.enable_wal(wal_path);
        }
        Ok(())
    }

    pub fn save_to_path_and_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<(), DBError> {
        let snapshot = self.build_snapshot();
        let metadata = self.snapshot_metadata();
        Self::persist_snapshot(&snapshot, Some(&metadata), &path)?;
        self.checkpoint_wal()?;
        Ok(())
    }

    pub fn wal_path_for_snapshot<P: AsRef<Path>>(path: P) -> PathBuf {
        Self::default_wal_path(path.as_ref())
    }

    fn default_wal_path(path: &Path) -> PathBuf {
        path.with_extension("wal")
    }

    pub(crate) fn build_snapshot(&self) -> SegmentSnapshot {
        SegmentSnapshot {
            hnsw: self.hnsw.to_snapshot(),
            payload_index: self.payload_index.clone(),
            payloads: self.payloads.clone(),
            deleted: self.deleted.clone(),
            next_id: self.next_id,
        }
    }

    pub fn snapshot_metadata(&self) -> SnapshotMetadata {
        let cfg = self.hnsw.config_summary();
        let created_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        SnapshotMetadata {
            created_at_ms,
            hnsw: cfg,
            points: self.hnsw.len(),
            payloads: self.payloads.len(),
        }
    }

    pub fn snapshot_name_with_config(&self, prefix: &str) -> String {
        let cfg = self.hnsw.config_summary();
        let metric = match cfg.metric {
            crate::utils::types::DistanceMetric::Cosine => "cosine",
            crate::utils::types::DistanceMetric::Dot => "dot",
            crate::utils::types::DistanceMetric::Euclidean => "euclidean",
        };
        format!(
            "{}_m{}_m0_{}_efc{}_dim{}_{}.bin",
            prefix, cfg.m, cfg.m0, cfg.ef_construct, cfg.dim, metric
        )
    }

    pub fn snapshot_path_with_config<P: AsRef<Path>>(&self, dir: P, prefix: &str) -> PathBuf {
        dir.as_ref().join(self.snapshot_name_with_config(prefix))
    }

    pub(crate) fn persist_snapshot<P: AsRef<Path>>(
        snapshot: &SegmentSnapshot,
        metadata: Option<&SnapshotMetadata>,
        path: P,
    ) -> Result<(), DBError> {
        write_atomic_with_checksum(path, SEGMENT_SNAPSHOT_FOOTER, |writer| {
            writer.write_all(&SEGMENT_SNAPSHOT_MAGIC)?;
            writer.write_all(&SEGMENT_SNAPSHOT_VERSION.to_le_bytes())?;
            bincode::serialize_into(&mut *writer, snapshot)
                .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
            if let Some(meta) = metadata {
                let meta_bytes = bincode::serialize(meta)
                    .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
                let meta_len = meta_bytes.len() as u32;
                writer.write_all(&meta_bytes)?;
                writer.write_all(&meta_len.to_le_bytes())?;
                writer.write_all(&SEGMENT_SNAPSHOT_META_VERSION.to_le_bytes())?;
                writer.write_all(&SEGMENT_SNAPSHOT_META_MAGIC)?;
            }
            Ok(())
        })
    }

    fn parse_snapshot_metadata(
        payload: &[u8],
    ) -> Result<(&[u8], Option<SnapshotMetadata>), DBError> {
        if payload.len() < 12 || payload[payload.len() - 4..] != SEGMENT_SNAPSHOT_META_MAGIC {
            return Ok((payload, None));
        }
        let version_start = payload.len() - 8;
        let len_start = payload.len() - 12;
        let version = u32::from_le_bytes([
            payload[version_start],
            payload[version_start + 1],
            payload[version_start + 2],
            payload[version_start + 3],
        ]);
        if version != SEGMENT_SNAPSHOT_META_VERSION {
            return Ok((payload, None));
        }
        let meta_len = u32::from_le_bytes([
            payload[len_start],
            payload[len_start + 1],
            payload[len_start + 2],
            payload[len_start + 3],
        ]) as usize;
        if payload.len() < 12 + meta_len {
            return Ok((payload, None));
        }
        let meta_start = payload.len() - 12 - meta_len;
        let meta_bytes = &payload[meta_start..meta_start + meta_len];
        let snapshot_bytes = &payload[..meta_start];
        let metadata = match bincode::deserialize(meta_bytes) {
            Ok(meta) => meta,
            Err(_) => return Ok((payload, None)),
        };
        Ok((snapshot_bytes, Some(metadata)))
    }

    pub(crate) fn op_count(&self) -> u64 {
        self.op_count
    }

    fn ensure_not_rebuilding(&self) -> Result<(), DBError> {
        if self.rebuilding.load(Ordering::SeqCst) {
            return Err(DBError::SearchError(
                "Segment rebuild in progress. Retry later.".into(),
            ));
        }
        Ok(())
    }

    pub fn checkpoint_wal(&mut self) -> Result<(), DBError> {
        if let Some(wal) = &mut self.wal {
            wal.truncate()?;
        }
        Ok(())
    }

    fn append_wal(&mut self, record: WalRecord) -> Result<(), DBError> {
        if let Some(wal) = &mut self.wal {
            wal.append(&record)?;
        }
        Ok(())
    }

    fn apply_wal_record(&mut self, record: WalRecord) -> Result<(), DBError> {
        match record {
            WalRecord::Insert {
                point_id,
                vector,
                payload,
            } => match self.insert_with_id_internal(point_id, vector, payload, false) {
                Ok(_) => Ok(()),
                Err(DBError::DuplicatePointId(_)) => Ok(()),
                Err(err) => Err(err),
            },
            WalRecord::Delete { point_id } => self.delete_internal(point_id, false),
            WalRecord::UpdatePayload { point_id, payload } => {
                self.update_payload_internal(point_id, payload, false)
            }
        }
    }

    /// Build the list of payload keys to use for filter-aware edges, honoring an optional allowlist,
    /// a max count, and a type preference (Bool -> Str -> Int -> Float).
    fn filter_keys_for_payload(payload: &Payload) -> Vec<String> {
        let config = Self::filter_key_config();
        let allow = config.allow.as_ref();
        let max_keys = config.max_keys;

        fn type_rank(v: &PayloadValue) -> usize {
            match v {
                PayloadValue::Bool(_) => 0,
                PayloadValue::Str(_) => 1,
                PayloadValue::Int(_) => 2,
                PayloadValue::Float(_) => 3,
                _ => 4,
            }
        }

        let mut keys_with_rank: Vec<(usize, String)> = payload
            .0
            .iter()
            .filter_map(|(k, v)| {
                if matches!(
                    v,
                    PayloadValue::Int(_)
                        | PayloadValue::Float(_)
                        | PayloadValue::Str(_)
                        | PayloadValue::Bool(_)
                ) && allow.is_none_or(|set| set.contains(k))
                {
                    Some((type_rank(v), k.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Prefer cheaper/more selective types first, then deterministic key order.
        keys_with_rank.sort();
        if let Some(cap) = max_keys {
            keys_with_rank.truncate(cap);
        }
        keys_with_rank.into_iter().map(|(_, k)| k).collect()
    }

    fn filter_key_config() -> &'static FilterKeyConfig {
        static FILTER_KEY_CONFIG: OnceLock<FilterKeyConfig> = OnceLock::new();
        FILTER_KEY_CONFIG.get_or_init(|| FilterKeyConfig {
            allow: env::var("VECTORDB_FILTER_KEYS").ok().map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }),
            max_keys: env::var("VECTORDB_FILTER_MAX_KEYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v| *v > 0),
        })
    }
}

struct FilterKeyConfig {
    allow: Option<HashSet<String>>,
    max_keys: Option<usize>,
}

fn max_rss_bytes() -> Option<u64> {
    *MAX_RSS_BYTES.get_or_init(|| {
        env::var("VECTORDB_MAX_RSS_MB")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<u64>().ok())
            .filter(|v| *v > 0)
            .map(|mb| mb.saturating_mul(1024).saturating_mul(1024))
    })
}

fn oom_snapshot_path() -> Option<PathBuf> {
    OOM_SNAPSHOT_PATH
        .get_or_init(|| {
            env::var("VECTORDB_OOM_SNAPSHOT_PATH")
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .clone()
}

#[cfg(unix)]
fn current_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let res = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if res != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let raw = usage.ru_maxrss as u64;
    #[cfg(target_os = "macos")]
    {
        Some(raw)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Some(raw.saturating_mul(1024))
    }
}

#[cfg(not(unix))]
fn current_rss_bytes() -> Option<u64> {
    None
}
