use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering;

use anyhow::anyhow;
use serde::{Deserialize, Serialize};

use crate::utils::errors::DBError;
use crate::utils::io::{adler32, write_atomic_with_checksum};
use crate::utils::types::{DistanceMetric, PointId, Vector};
use crate::vector::hnsw::arena::{ChunkedArray, VectorArena};

use super::config::{exact_fallback_enabled_override, exact_fallback_threshold_override};
use super::core::{MAX_STORED_LEVEL, NO_EP, pack_ep};
use super::{HNSWIndex, HnswSnapshot};

const HNSW_SNAPSHOT_MAGIC: [u8; 4] = *b"VDBH";
const HNSW_SNAPSHOT_VERSION: u32 = 2;
const HNSW_SNAPSHOT_FOOTER: [u8; 4] = *b"VDBF";

#[derive(Serialize, Deserialize)]
struct HnswSnapshotV1 {
    layers: HashMap<usize, HashMap<PointId, Vec<PointId>>>,
    vectors: HashMap<PointId, Vector>,
    levels: HashMap<PointId, usize>,
    entry_point: Option<PointId>,
    metric: DistanceMetric,
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

impl HNSWIndex {
    pub fn to_snapshot(&self) -> HnswSnapshot {
        let n = self.len();
        let (entry_idx, current_max_level) = self.entry_level();
        let mut vectors = HashMap::with_capacity(n);
        let mut levels = HashMap::with_capacity(n);
        let mut deleted = HashSet::new();
        for idx in 0..n {
            let id = self.point_id(idx);
            vectors.insert(id, self.vector_slice(idx).to_vec());
            levels.insert(
                id,
                self.levels.with(idx, |l| l.load(Ordering::Relaxed)) as usize,
            );
            if self.deleted.with(idx, |b| b.load(Ordering::Relaxed)) {
                deleted.insert(id);
            }
        }

        let mut layers = HashMap::new();
        for (level, layer) in self.layers.iter().enumerate() {
            let mut level_map: HashMap<PointId, Vec<PointId>> = HashMap::new();
            let view = layer.view();
            for idx in 0..view.len() {
                let neighbors = view.get(idx).read();
                if neighbors.is_empty() {
                    continue;
                }
                let id = self.point_id(idx);
                let mapped = neighbors
                    .iter()
                    .map(|&n| self.point_id(n))
                    .collect::<Vec<_>>();
                level_map.insert(id, mapped);
            }
            if !level_map.is_empty() {
                layers.insert(level, level_map);
            }
        }

        HnswSnapshot {
            layers,
            vectors,
            levels,
            entry_point: entry_idx.map(|idx| self.point_id(idx)),
            metric: self.metric,
            m: self.m,
            m0: self.m0,
            stored_cap_l0: self.stored_cap_l0,
            ef: self.ef,
            ef_construct: self.ef_construct,
            max_level_cap: self.max_level_cap,
            level_scale: self.level_scale,
            current_max_level,
            dim: self.dim,
            deleted,
            exact_fallback_enabled: self.exact_fallback_enabled,
            exact_fallback_threshold: self.exact_fallback_threshold,
        }
    }

    pub fn from_snapshot(snapshot: HnswSnapshot) -> Self {
        let m0 = if snapshot.m0 == 0 {
            snapshot.m * 2
        } else {
            snapshot.m0
        };
        let stored_cap_l0 = if snapshot.stored_cap_l0 == 0 {
            m0
        } else {
            snapshot.stored_cap_l0
        };
        let mut ids: Vec<PointId> = snapshot.vectors.keys().copied().collect();
        ids.sort_unstable();
        let mut point_to_idx = HashMap::with_capacity(ids.len());
        for (idx, id) in ids.iter().copied().enumerate() {
            point_to_idx.insert(id, idx);
        }

        const CHUNK_CAP: usize = 4096;
        let vectors = VectorArena::new(snapshot.dim, CHUNK_CAP);
        let levels_arr: ChunkedArray<std::sync::atomic::AtomicU8> = ChunkedArray::new(CHUNK_CAP);
        let deleted_arr: ChunkedArray<std::sync::atomic::AtomicBool> = ChunkedArray::new(CHUNK_CAP);
        let mut deleted_count = 0usize;

        for (idx, id) in ids.iter().copied().enumerate() {
            if let Some(vec) = snapshot.vectors.get(&id) {
                let pushed = vectors.push(vec);
                debug_assert_eq!(pushed, idx);
            } else {
                let zeros = vec![0.0f32; snapshot.dim];
                let pushed = vectors.push(&zeros);
                debug_assert_eq!(pushed, idx);
            }
            let li = levels_arr.push_default();
            debug_assert_eq!(li, idx);
            let level = snapshot
                .levels
                .get(&id)
                .copied()
                .unwrap_or(0)
                .min(MAX_STORED_LEVEL) as u8;
            levels_arr.with(idx, |l| l.store(level, Ordering::Relaxed));
            let di = deleted_arr.push_default();
            debug_assert_eq!(di, idx);
            if snapshot.deleted.contains(&id) {
                deleted_arr.with(idx, |b| b.store(true, Ordering::Relaxed));
                deleted_count += 1;
            }
        }

        // Preserve the pre-allocated level-slot layout that `HNSWIndex::new`
        // establishes so query-side code (and reorder/quantize) never observe
        // level-axis growth during operation. Cover both the snapshot's own
        // max level and the index's configured cap.
        let max_level_cap = snapshot.max_level_cap.min(MAX_STORED_LEVEL);
        let snapshot_max_level = snapshot
            .layers
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            .max(snapshot.current_max_level);
        let num_levels = max_level_cap.max(snapshot_max_level) + 1;
        let mut layers: Vec<ChunkedArray<parking_lot::RwLock<Vec<usize>>>> =
            Vec::with_capacity(num_levels);
        for _ in 0..num_levels {
            let arr: ChunkedArray<parking_lot::RwLock<Vec<usize>>> = ChunkedArray::new(CHUNK_CAP);
            for _ in 0..ids.len() {
                arr.push_default();
            }
            layers.push(arr);
        }
        for (level, layer_map) in snapshot.layers.iter() {
            if *level >= layers.len() {
                continue;
            }
            let view = layers[*level].view();
            for (id, neighbors) in layer_map {
                let Some(&idx) = point_to_idx.get(id) else {
                    continue;
                };
                let mapped = neighbors
                    .iter()
                    .filter_map(|n| point_to_idx.get(n).copied())
                    .collect::<Vec<_>>();
                *view.get(idx).write() = mapped;
            }
        }

        // Edge distances are NOT backfilled on load — lazy via build_edge_distances().
        // Allocate a stable slot per node so that build_edge_distances() and
        // read-side code can safely index by node idx.
        let edge_dists_l0: ChunkedArray<parking_lot::RwLock<Vec<f32>>> =
            ChunkedArray::new(CHUNK_CAP);
        for _ in 0..ids.len() {
            edge_dists_l0.push_default();
        }

        let stored_max_level = snapshot.current_max_level.min(MAX_STORED_LEVEL);
        let entry_level_ep = {
            use std::sync::atomic::AtomicU64;
            let ep_idx = snapshot
                .entry_point
                .and_then(|id| point_to_idx.get(&id).copied());
            AtomicU64::new(match ep_idx {
                Some(idx) => pack_ep(idx, stored_max_level),
                None => pack_ep(NO_EP as usize, stored_max_level),
            })
        };

        Self {
            layers,
            vectors,
            levels: levels_arr,
            entry_level_ep,
            metric: snapshot.metric,
            m: snapshot.m,
            m0,
            stored_cap_l0,
            ef: snapshot.ef,
            ef_construct: snapshot.ef_construct,
            max_level_cap,
            level_scale: snapshot.level_scale,
            dim: snapshot.dim,
            deleted_count: std::sync::atomic::AtomicUsize::new(deleted_count),
            deleted: deleted_arr,
            edge_dists_l0,
            // SQ8 quantization is not persisted; rebuilt lazily via quantize_all().
            quantized: Vec::new(),
            quant_min: Vec::new(),
            quant_scale: Vec::new(),
            point_to_idx,
            idx_to_point: ids,
            exact_fallback_enabled: exact_fallback_enabled_override().unwrap_or(false),
            exact_fallback_threshold: exact_fallback_threshold_override()
                .unwrap_or(snapshot.exact_fallback_threshold),
        }
    }

    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), DBError> {
        write_atomic_with_checksum(path, HNSW_SNAPSHOT_FOOTER, |writer| {
            writer.write_all(&HNSW_SNAPSHOT_MAGIC)?;
            writer.write_all(&HNSW_SNAPSHOT_VERSION.to_le_bytes())?;
            bincode::serialize_into(writer, &self.to_snapshot())
                .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
            Ok(())
        })
    }

    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, DBError> {
        let bytes = std::fs::read(path)?;
        let (payload, checksum) = if bytes.len() >= 8
            && bytes[bytes.len() - 8..bytes.len() - 4] == HNSW_SNAPSHOT_FOOTER
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

        if let Some(expected) = checksum {
            let actual = adler32(payload);
            if actual != expected {
                return Err(DBError::SerializationError(anyhow!(
                    "HNSW snapshot checksum mismatch"
                )));
            }
        }

        if payload.len() >= 8 && payload[..4] == HNSW_SNAPSHOT_MAGIC {
            let version = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
            return match version {
                2 => {
                    let snapshot: HnswSnapshot = bincode::deserialize(&payload[8..])
                        .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
                    Ok(Self::from_snapshot(snapshot))
                }
                _ => Err(DBError::SerializationError(anyhow!(
                    "unsupported HNSW snapshot version {}",
                    version
                ))),
            };
        }

        if let Ok(snapshot) = bincode::deserialize::<HnswSnapshot>(payload) {
            return Ok(Self::from_snapshot(snapshot));
        }

        let snapshot_v1: HnswSnapshotV1 =
            bincode::deserialize(payload).map_err(|e| DBError::SerializationError(anyhow!(e)))?;
        Ok(Self::from_snapshot(snapshot_v1.into()))
    }
}
