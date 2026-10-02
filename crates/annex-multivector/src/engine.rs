#[path = "calibration.rs"]
mod calibration;
#[path = "planner.rs"]
mod planner;
#[path = "policy.rs"]
mod policy;
#[path = "retrieval.rs"]
mod retrieval;
use crate::{
    analyzer::{Analyzer, TextAnalyzer},
    fde::{MaxSimQuery, Vector, dot, maxsim_flat, normalize},
    muvera::FdeEncoder,
    storage::{
        CompressedVectorStore, FixedVectorStore, ObjectLocation, atomic_write, commit_boundary,
        record_bytes, verify_record,
    },
};
use annex::{
    Filter, Payload,
    payload_storage::stores::PayloadIndex,
    utils::types::DistanceMetric,
    vector::hnsw::{HNSWIndex, SearchRuntimeOptions},
};
pub use calibration::{CalibrationEntry, CalibrationKey, CalibrationSnapshot, CalibrationTarget};
pub use planner::{
    ContextOperator, ContextPlan, FieldStats, FilterStats, FilterStrategy, FusionOperator,
    LogicalChannel, LogicalChannelKind, LogicalFusion, LogicalPlan, PhysicalOperator, PlanEstimate,
    PlanReason, PlanStage, PlannedChannel, PlannerStats, QualityPreference, RepresentationKind,
    RerankPlan, RetrievalObjective, RetrievalPlan,
};
pub use policy::{PlanningMode, PolicyPlan, QueryIntent, QueryRepresentations};
use rayon::prelude::*;
pub use retrieval::{
    AdaptiveRerank, Channel, Chunk, ContextHit, ContextOptions, Fusion, Predicate, RankingSignals,
    Representation, Rerank, RetrievalDocument, RetrievalResponse, RetrievalTrace, RetrieveRequest,
};
use retrieval::{DocSet, FieldSchema, Fields, RetrievalState};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
    sync::{Mutex, RwLock},
};
use thiserror::Error;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IndexConfig {
    pub dimension: usize,
    #[serde(default = "default_centroids")]
    pub centroids: usize,
    #[serde(default = "default_bits")]
    pub residual_bits: u8,
    #[serde(default = "default_probes")]
    pub probes: usize,
    #[serde(default = "default_fde_repetitions")]
    pub fde_repetitions: usize,
    #[serde(default = "default_fde_ksim")]
    pub fde_ksim: usize,
    #[serde(default = "default_fde_projected")]
    pub fde_projected: usize,
    /// Text analysis policy for the lexical field. Persisted with the
    /// collection; reopening with a different policy is a configuration error.
    #[serde(default)]
    pub analyzer: TextAnalyzer,
}
fn default_centroids() -> usize {
    64
}
fn default_bits() -> u8 {
    2
}
fn default_probes() -> usize {
    4
}
fn default_fde_repetitions() -> usize {
    20
}
fn default_fde_ksim() -> usize {
    4
}
fn default_fde_projected() -> usize {
    8
}
impl IndexConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            centroids: 64,
            residual_bits: 2,
            probes: 4,
            fde_repetitions: 20,
            fde_ksim: 4,
            fde_projected: 8,
            analyzer: TextAnalyzer::plain(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DocumentRecord {
    centroid_ids: Vec<u32>,
    unique_centroids: Vec<u32>,
    location: ObjectLocation,
    fde_location: ObjectLocation,
    metadata: Value,
    tokens: usize,
    compressed_bytes: u64,
    #[serde(default)]
    fields: Arc<Fields>,
    #[serde(default)]
    storage_id: u64,
}
/// Persisted manifest header. `format_version` lets us evolve the on-disk
/// layout later without silently accepting mismatched files. `generation`
/// is bumped on every mutation and lets any derived structure (HNSW-over-
/// FDE in particular) prove it was built against the current document set.
///
/// Format changes: bump FORMAT_VERSION and add a From<oldManifest> path.
const FORMAT_VERSION: u32 = 3;

/// Fsync is the default: acknowledge only after segment data, the manifest,
/// and its directory entry are synced. Buffered retains atomic visibility,
/// but does not promise power-loss durability. There is no background flusher.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Durability {
    #[default]
    Fsync,
    Buffered,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
struct SegmentBoundaries {
    objects: u64,
    fde: u64,
}

/// Digest and exact manifest bytes are replaced in ONE rename, eliminating
/// the torn sidecar/manifest pair in format 1. The digest is BLAKE3, not SHA256.
#[derive(Deserialize, Serialize)]
struct ManifestEnvelope {
    format_version: u32,
    checksum_blake3: String,
    manifest: String,
}

#[derive(Deserialize, Serialize)]
struct Manifest {
    #[serde(default = "current_format_version")]
    format_version: u32,
    #[serde(default)]
    generation: u64,
    #[serde(default = "legacy_fde_encoding_version")]
    fde_encoding_version: u32,
    config: IndexConfig,
    codebook: Vec<Vector>,
    residual_codebook: Vec<f32>,
    documents: HashMap<String, DocumentRecord>,
    /// Immutable kind/dimension contract for every named representation ever
    /// committed in this collection. It outlives the last document using a field.
    #[serde(default)]
    representation_schema: BTreeMap<String, FieldSchema>,
    #[serde(default)]
    segments: Option<SegmentBoundaries>,
    #[serde(default)]
    storage_generation: Option<u64>,
    #[serde(default)]
    sealed: Vec<(u64, SegmentBoundaries)>,
}

fn legacy_fde_encoding_version() -> u32 {
    1
}

fn current_format_version() -> u32 {
    1 // legacy manifests omitted this field
}

/// Verifies the legacy BLAKE3 sidecar (historically misnamed sha256).
/// The sidecar is optional (older indexes were written without one), so we
/// only enforce when the file exists. New commits use a self-contained envelope.
fn verify_manifest_checksum(manifest_bytes: &[u8], sidecar_path: &Path) -> Result<(), IndexError> {
    if !sidecar_path.exists() {
        return Ok(());
    }
    let expected = fs::read_to_string(sidecar_path)?;
    let expected = expected.trim();
    let actual = blake3::hash(manifest_bytes);
    if actual.to_hex().as_str() != expected {
        return Err(IndexError::Invalid(format!(
            "manifest.sha256 does not match manifest.json — index may be corrupt or torn: expected={expected}, actual={}",
            actual.to_hex(),
        )));
    }
    Ok(())
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Hit {
    pub id: String,
    pub score: f32,
    /// FDE score this hit had before the compressed-MaxSim rescore. Exposed
    /// so callers can compute per-query FDE-vs-MaxSim rank disagreement —
    /// a signal for the confidence-output / adaptive-escalation primitive.
    /// Skipped from JSON for centroid-only probing, which has no FDE stage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fde_score: Option<f32>,
    pub metadata: Value,
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CandidateHit {
    pub id: String,
    pub score: f32,
}
#[derive(Clone, Debug)]
pub struct UpsertDocument {
    pub id: String,
    pub vectors: Vec<Vector>,
    pub metadata: Value,
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct IndexStats {
    pub documents: usize,
    pub generation: u64,
    pub token_vectors: usize,
    pub compressed_bytes: u64,
    pub centroids: usize,
    pub residual_bits: u8,
    pub trained: bool,
    pub fde_dimension: usize,
    pub fde_ann_nodes: usize,
    pub fde_ann_base_nodes: usize,
    pub fde_ann_delta_documents: usize,
    pub fde_ann_tombstones: usize,
    pub fde_encoding_version: u32,
    pub storage_segments: usize,
    pub dense_ann_fields: HashMap<String, usize>,
}
#[derive(Debug, Error)]
pub enum IndexError {
    #[error("{0}")]
    Invalid(String),
    #[error("index configuration is {actual:?}, requested {requested:?}")]
    Config {
        actual: Box<IndexConfig>,
        requested: Box<IndexConfig>,
    },
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(
        "generation published, but durable commit is uncertain; reopen or retry idempotently: {0}"
    )]
    CommitUncertain(io::Error),
}
/// Immutable graph and dense external-ID mapping, shared by staged generations.
struct FdeAnnBase {
    index: HNSWIndex,
    ids: Vec<String>,
    by_id: HashMap<String, u64>,
    field: Option<String>,
    payloads: HashMap<u64, Payload>,
    payload_index: PayloadIndex,
}

#[derive(Clone)]
struct FdeAnn {
    base: Arc<FdeAnnBase>,
    // Updated IDs are scanned exactly from the append-only FDE store; a
    // rebuild folds them into a new graph without duplicating vectors in RAM.
    delta: HashSet<String>,
    tombstones: HashSet<u64>,
    generation: u64,
}

#[derive(Clone)]
struct State {
    generation: u64,
    codebook: Vec<Vector>,
    residual_codebook: Vec<f32>,
    documents: HashMap<String, DocumentRecord>,
    postings: Vec<HashSet<String>>,
    fde_ann: Option<FdeAnn>,
    named_ann: HashMap<String, FdeAnn>,
    stores: Arc<SegmentStores>,
    retrieval: Arc<RetrievalState>,
    planner_stats: planner::CachedPlannerStats,
    objects_map: Option<Arc<memmap2::Mmap>>,
    fde_map: Option<Arc<memmap2::Mmap>>,
    sealed: HashMap<u64, Arc<SegmentSnapshot>>,
}
struct SegmentSnapshot {
    stores: Arc<SegmentStores>,
    objects: Option<Arc<memmap2::Mmap>>,
    fde: Option<Arc<memmap2::Mmap>>,
    bounds: SegmentBoundaries,
}
impl SegmentSnapshot {
    fn open(root: &Path, id: u64, bounds: SegmentBoundaries) -> Result<Self, IndexError> {
        let path = if id == 0 {
            root.to_owned()
        } else {
            root.join("segments").join(id.to_string())
        };
        if !path.is_dir() {
            return Err(IndexError::Invalid("missing sealed segment".into()));
        }
        let stores = Arc::new(SegmentStores {
            objects: CompressedVectorStore::new(path.join("objects"))?,
            fde: FixedVectorStore::new(path.join("fde"))?,
            root: path,
            id: if id == 0 { None } else { Some(id) },
            retired: std::sync::atomic::AtomicBool::new(false),
        });
        if stores.objects.len()? < bounds.objects || stores.fde.len()? < bounds.fde {
            return Err(IndexError::Invalid(
                "sealed segment shorter than committed boundary".into(),
            ));
        }
        Ok(Self {
            objects: if bounds.objects > 0 {
                Some(Arc::new(stores.objects.map()?))
            } else {
                None
            },
            fde: if bounds.fde > 0 {
                Some(Arc::new(stores.fde.map()?))
            } else {
                None
            },
            stores,
            bounds,
        })
    }
}
impl State {
    fn record_fde(&self, record: &DocumentRecord) -> &[u8] {
        if record.storage_id == self.stores.id.unwrap_or(0) {
            self.fde_bytes()
        } else {
            self.sealed[&record.storage_id]
                .fde
                .as_deref()
                .map(|m| &m[..self.sealed[&record.storage_id].bounds.fde as usize])
                .unwrap_or(&[])
        }
    }
    fn record_objects(&self, record: &DocumentRecord) -> &[u8] {
        if record.storage_id == self.stores.id.unwrap_or(0) {
            self.object_bytes()
        } else {
            self.sealed[&record.storage_id]
                .objects
                .as_deref()
                .map(|m| &m[..self.sealed[&record.storage_id].bounds.objects as usize])
                .unwrap_or(&[])
        }
    }
    fn object_bytes(&self) -> &[u8] {
        self.objects_map.as_deref().map(|m| &m[..]).unwrap_or(&[])
    }
    fn fde_bytes(&self) -> &[u8] {
        self.fde_map.as_deref().map(|m| &m[..]).unwrap_or(&[])
    }
}

struct SegmentStores {
    objects: CompressedVectorStore,
    fde: FixedVectorStore,
    root: PathBuf,
    id: Option<u64>,
    retired: std::sync::atomic::AtomicBool,
}

impl Drop for SegmentStores {
    fn drop(&mut self) {
        if self.retired.load(std::sync::atomic::Ordering::Relaxed) {
            if self.id.is_some() {
                let _ = fs::remove_dir_all(&self.root);
            } else {
                let _ = fs::remove_dir_all(self.root.join("objects"));
                let _ = fs::remove_dir_all(self.root.join("fde"));
            }
        }
    }
}

struct DirectoryLock(File);

impl Drop for DirectoryLock {
    fn drop(&mut self) {
        // Closing one descriptor does not release flock while a fork/dup copy
        // survives. Release logical ownership explicitly before closing ours.
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

pub struct MultiVectorIndex {
    root: PathBuf,
    config: IndexConfig,
    fde: FdeEncoder,
    state: RwLock<Arc<State>>,
    writer: Mutex<()>,
    calibration: Mutex<calibration::CalibrationStats>,
    durability: Durability,
    // One process/handle owns append offsets and manifest publication at a time.
    _directory_lock: DirectoryLock,
}

impl MultiVectorIndex {
    pub(crate) fn initialize(&self) -> Result<(), IndexError> {
        let _writer = self.writer.lock().unwrap();
        self.persist(&self.snapshot())
    }
    pub fn open_existing(
        path: impl AsRef<Path>,
        durability: Durability,
    ) -> Result<Self, IndexError> {
        let bytes = fs::read(path.as_ref().join("manifest.json"))?;
        let value: Value = serde_json::from_slice(&bytes)?;
        let manifest: Manifest = if value.get("manifest").is_some() {
            let envelope: ManifestEnvelope = serde_json::from_value(value)?;
            if blake3::hash(envelope.manifest.as_bytes()).to_hex().as_str()
                != envelope.checksum_blake3
            {
                return Err(IndexError::Invalid("manifest checksum mismatch".into()));
            }
            serde_json::from_str(&envelope.manifest)?
        } else {
            serde_json::from_slice(&bytes)?
        };
        Self::open_with_durability(path, manifest.config, durability)
    }
    pub fn open(path: impl AsRef<Path>, config: IndexConfig) -> Result<Self, IndexError> {
        Self::open_with_durability(path, config, Durability::Fsync)
    }

    pub fn open_with_durability(
        path: impl AsRef<Path>,
        config: IndexConfig,
        durability: Durability,
    ) -> Result<Self, IndexError> {
        if config.dimension == 0
            || config.centroids == 0
            || config.probes == 0
            || !(1..=8).contains(&config.residual_bits)
            || config.fde_repetitions == 0
            || config.fde_ksim == 0
            || config.fde_ksim > 12
            || config.fde_projected == 0
        {
            return Err(IndexError::Invalid(
                "dimension, centroids, probes, and residual_bits (1..=8) must be valid".into(),
            ));
        }
        validate_config_size(&config)?;
        config.analyzer.validate().map_err(IndexError::Invalid)?;
        let root = path.as_ref().to_owned();
        let mut created_parents = Vec::new();
        let mut missing = root.as_path();
        while !missing.exists() {
            let parent = missing
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            created_parents.push(parent.to_owned());
            missing = parent;
        }
        fs::create_dir_all(&root)?;
        let directory_lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join("index.lock"))?;
        fs2::FileExt::try_lock_exclusive(&directory_lock).map_err(|e| {
            IndexError::Invalid(format!("index already open or cannot lock directory: {e}"))
        })?;
        let directory_lock = DirectoryLock(directory_lock);
        let manifest_path = root.join("manifest.json");
        let (
            generation,
            fde_encoding_version,
            codebook,
            residual_codebook,
            mut documents,
            representation_schema,
            segments,
            storage_generation,
            sealed_bounds,
            checksummed,
        ) = if manifest_path.exists() {
            let bytes = fs::read(&manifest_path)?;
            let header: Value = serde_json::from_slice(&bytes)?;
            let (m, checksummed): (Manifest, bool) = if header.get("manifest").is_some() {
                let envelope: ManifestEnvelope = serde_json::from_slice(&bytes)?;
                if !matches!(envelope.format_version, 2 | FORMAT_VERSION)
                    || blake3::hash(envelope.manifest.as_bytes()).to_hex().as_str()
                        != envelope.checksum_blake3
                {
                    return Err(IndexError::Invalid(
                        "manifest checksum or envelope version mismatch".into(),
                    ));
                }
                (serde_json::from_str(&envelope.manifest)?, true)
            } else {
                verify_manifest_checksum(&bytes, &root.join("manifest.sha256"))?;
                (serde_json::from_slice(&bytes)?, false)
            };
            if !(if checksummed {
                matches!(m.format_version, 2 | FORMAT_VERSION)
            } else {
                m.format_version == 1
            }) || (checksummed && m.segments.is_none())
            {
                return Err(IndexError::Invalid(
                    "unsupported manifest version or missing committed boundaries".into(),
                ));
            }
            if m.config != config {
                return Err(IndexError::Config {
                    actual: Box::new(m.config),
                    requested: Box::new(config),
                });
            }
            (
                m.generation,
                m.fde_encoding_version,
                m.codebook,
                m.residual_codebook,
                m.documents,
                m.representation_schema,
                m.segments,
                m.storage_generation,
                m.sealed,
                checksummed,
            )
        } else {
            // Without a committed manifest all segment bytes are uncommitted.
            (
                0,
                2,
                vec![],
                vec![],
                HashMap::new(),
                BTreeMap::new(),
                Some(SegmentBoundaries { objects: 0, fde: 0 }),
                None,
                Vec::new(),
                false,
            )
        };
        if !matches!(fde_encoding_version, 1 | 2) {
            return Err(IndexError::Invalid(format!(
                "unsupported FDE encoding version {fde_encoding_version}"
            )));
        }
        validate_codebooks(
            &config,
            &codebook,
            &residual_codebook,
            documents.values().any(|d| d.tokens > 0),
        )?;
        let store_root = storage_generation
            .map(|id| root.join("segments").join(id.to_string()))
            .unwrap_or_else(|| root.clone());
        if storage_generation.is_some() && !store_root.is_dir() {
            return Err(IndexError::Invalid(
                "missing committed segment directory".into(),
            ));
        }
        let objects = CompressedVectorStore::new(store_root.join("objects"))?;
        let fde_store = FixedVectorStore::new(store_root.join("fde"))?;
        if !manifest_path.exists()
            && (objects.len()? != 0
                || fde_store.len()? != 0
                || root
                    .join("segments")
                    .read_dir()
                    .is_ok_and(|mut entries| entries.next().is_some()))
        {
            return Err(IndexError::Invalid(
                "missing manifest for non-empty segments".into(),
            ));
        }
        let boundaries = segments.unwrap_or(SegmentBoundaries {
            objects: objects.len()?,
            fde: fde_store.len()?,
        });
        if objects.len()? < boundaries.objects || fde_store.len()? < boundaries.fde {
            return Err(IndexError::Invalid(
                "segment shorter than committed boundary".into(),
            ));
        }
        let mut sealed = HashMap::new();
        for (id, bounds) in sealed_bounds {
            if id == storage_generation.unwrap_or(0) || sealed.contains_key(&id) {
                return Err(IndexError::Invalid("duplicate storage segment".into()));
            }
            sealed.insert(id, Arc::new(SegmentSnapshot::open(&root, id, bounds)?));
        }
        if !documents.is_empty() {
            let objects_map = if objects.len()? > 0 {
                Some(objects.map()?)
            } else {
                None
            };
            let fde_map = if fde_store.len()? > 0 {
                Some(fde_store.map()?)
            } else {
                None
            };
            let object_bytes = objects_map
                .as_deref()
                .unwrap_or(&[])
                .get(
                    ..usize::try_from(boundaries.objects)
                        .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?,
                )
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
            let fde_bytes = fde_map
                .as_deref()
                .unwrap_or(&[])
                .get(
                    ..usize::try_from(boundaries.fde)
                        .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?,
                )
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
            let fde_dimension =
                (1usize << config.fde_ksim) * config.fde_projected * config.fde_repetitions;
            for d in documents.values_mut() {
                let (object_bytes, fde_bytes) = if d.storage_id == storage_generation.unwrap_or(0) {
                    (object_bytes, fde_bytes)
                } else {
                    let segment = sealed.get(&d.storage_id).ok_or_else(|| {
                        IndexError::Invalid("document references missing segment".into())
                    })?;
                    (
                        segment
                            .objects
                            .as_deref()
                            .map(|m| &m[..segment.bounds.objects as usize])
                            .unwrap_or(&[]),
                        segment
                            .fde
                            .as_deref()
                            .map(|m| &m[..segment.bounds.fde as usize])
                            .unwrap_or(&[]),
                    )
                };
                d.fields.verify(fde_bytes)?;
                if d.tokens == 0 {
                    if !d.centroid_ids.is_empty()
                        || !d.unique_centroids.is_empty()
                        || d.location.length != 0
                        || d.fde_location.length != 0
                    {
                        return Err(IndexError::Invalid("invalid empty multivector".into()));
                    }
                    continue;
                }
                verify_record(object_bytes, d.location, checksummed)?;
                verify_record(fde_bytes, d.fde_location, checksummed)?;
                let decoded = CompressedVectorStore::decode(
                    object_bytes,
                    d.location,
                    &codebook,
                    &residual_codebook,
                )?;
                if decoded.dimension != config.dimension
                    || d.tokens == 0
                    || decoded.values.len() / config.dimension != d.tokens
                    || d.centroid_ids.len() != d.tokens
                    || d.compressed_bytes != d.location.length
                    || d.centroid_ids.iter().any(|&c| c as usize >= codebook.len())
                    || decoded.values.iter().any(|v| !v.is_finite())
                {
                    return Err(IndexError::Invalid(
                        "invalid document shape or centroid IDs".into(),
                    ));
                }
                let bytes = record_bytes(object_bytes, d.location)?;
                let stored_ids = bytes[16..16 + d.tokens * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_le_bytes(*v));
                if !stored_ids.eq(d.centroid_ids.iter().copied()) {
                    return Err(IndexError::Invalid(
                        "manifest/record centroid mismatch".into(),
                    ));
                }
                let mut unique = d.centroid_ids.clone();
                unique.sort_unstable();
                unique.dedup();
                if unique != d.unique_centroids {
                    return Err(IndexError::Invalid("invalid document posting list".into()));
                }
                if FixedVectorStore::get(fde_bytes, d.fde_location, fde_dimension)?
                    .iter()
                    .any(|v| !v.is_finite())
                {
                    return Err(IndexError::Invalid("non-finite FDE record".into()));
                }
                // Upgrade legacy records in memory; the next commit writes their digests.
                d.location.checksum =
                    Some(*blake3::hash(record_bytes(object_bytes, d.location)?).as_bytes());
                d.fde_location.checksum =
                    Some(*blake3::hash(record_bytes(fde_bytes, d.fde_location)?).as_bytes());
            }
        }
        // No mappings survive this point: discard only the uncommitted tail.
        for segment in sealed.values() {
            segment.stores.objects.recover(segment.bounds.objects)?;
            segment.stores.fde.recover(segment.bounds.fde)?;
        }
        objects.recover(boundaries.objects)?;
        fde_store.recover(boundaries.fde)?;
        if durability == Durability::Fsync {
            File::open(store_root.join("objects"))?.sync_all()?;
            File::open(store_root.join("fde"))?.sync_all()?;
            File::open(&root)?.sync_all()?;
            for parent in created_parents {
                File::open(parent)?.sync_all()?;
            }
        }
        if manifest_path.exists() {
            if let Ok(entries) = fs::read_dir(root.join("segments")) {
                for entry in entries {
                    let entry = entry?;
                    if !entry.file_type()?.is_dir() {
                        continue;
                    }
                    if let Some(id) = entry
                        .file_name()
                        .to_str()
                        .and_then(|s| s.parse::<u64>().ok())
                        && Some(id) != storage_generation
                        && !sealed.contains_key(&id)
                    {
                        fs::remove_dir_all(entry.path())?;
                    }
                }
            }
            if storage_generation.is_some() && !sealed.contains_key(&0) {
                for name in ["objects", "fde"] {
                    let path = root.join(name);
                    if path.exists() {
                        fs::remove_dir_all(path)?;
                    }
                }
            }
        }
        let mut postings = vec![HashSet::new(); codebook.len()];
        for (id, d) in &documents {
            for &c in &d.centroid_ids {
                postings[c as usize].insert(id.clone());
            }
        }
        // New manifests preserve the collection's immutable named-field
        // contract even when no live document still uses a field. Legacy
        // manifests default to an empty map and infer it while replaying docs;
        // the next mutation persists the inferred schema.
        let mut retrieval = RetrievalState::from_schema(
            representation_schema,
            Analyzer::new(config.analyzer.clone()),
        )?;
        let mut ordered: Vec<_> = documents.iter().collect();
        ordered.sort_by(|a, b| a.0.cmp(b.0));
        for (id, d) in ordered {
            retrieval.insert(id, &d.fields, &d.metadata)?;
        }
        let planner_stats =
            planner::CachedPlannerStats::from_documents(&documents, retrieval.schema());
        Ok(Self {
            fde: if fde_encoding_version == 2 {
                FdeEncoder::new(
                    config.dimension,
                    config.fde_ksim,
                    config.fde_projected,
                    config.fde_repetitions,
                    0x4d55_5645_5241,
                )
            } else {
                FdeEncoder::with_version(
                    config.dimension,
                    config.fde_ksim,
                    config.fde_projected,
                    config.fde_repetitions,
                    0x4d55_5645_5241,
                    fde_encoding_version,
                )
            },
            writer: Mutex::new(()),
            calibration: Mutex::new(calibration::CalibrationStats::default()),
            durability,
            _directory_lock: directory_lock,
            state: RwLock::new(Arc::new(State {
                generation,
                codebook,
                residual_codebook,
                documents,
                postings,
                fde_ann: None,
                named_ann: HashMap::new(),
                objects_map: if objects.len()? > 0 {
                    Some(Arc::new(objects.map()?))
                } else {
                    None
                },
                fde_map: if fde_store.len()? > 0 {
                    Some(Arc::new(fde_store.map()?))
                } else {
                    None
                },
                stores: Arc::new(SegmentStores {
                    objects,
                    fde: fde_store,
                    root: store_root,
                    id: storage_generation,
                    retired: std::sync::atomic::AtomicBool::new(false),
                }),
                retrieval: Arc::new(retrieval),
                planner_stats,
                sealed,
            })),
            root,
            config,
        })
    }
    fn snapshot(&self) -> Arc<State> {
        Arc::clone(&self.state.read().unwrap())
    }

    pub fn calibration_snapshot(&self) -> CalibrationSnapshot {
        self.calibration.lock().unwrap().snapshot()
    }

    fn validate(&self, v: &[Vector]) -> Result<(), IndexError> {
        if v.is_empty()
            || v.iter()
                .any(|x| x.len() != self.config.dimension || x.iter().any(|n| !n.is_finite()))
        {
            Err(IndexError::Invalid(format!(
                "vectors must be a non-empty matrix of {} finite values",
                self.config.dimension
            )))
        } else {
            Ok(())
        }
    }
    fn persist(&self, s: &State) -> Result<(), IndexError> {
        if self.durability == Durability::Fsync {
            s.stores.objects.sync()?;
            commit_boundary("objects_synced")?;
            s.stores.fde.sync()?;
            commit_boundary("fde_synced")?;
        }
        let manifest = serde_json::to_string(&Manifest {
            format_version: FORMAT_VERSION,
            generation: s.generation,
            fde_encoding_version: self.fde.encoding_version(),
            config: self.config.clone(),
            codebook: s.codebook.clone(),
            residual_codebook: s.residual_codebook.clone(),
            documents: s.documents.clone(),
            representation_schema: s.retrieval.schema().clone(),
            storage_generation: s.stores.id,
            sealed: s
                .sealed
                .iter()
                .map(|(&id, segment)| (id, segment.bounds))
                .collect(),
            segments: Some(SegmentBoundaries {
                objects: s.stores.objects.len()?,
                fde: s.stores.fde.len()?,
            }),
        })?;
        let envelope = ManifestEnvelope {
            format_version: FORMAT_VERSION,
            checksum_blake3: blake3::hash(manifest.as_bytes()).to_hex().to_string(),
            manifest,
        };
        let bytes = serde_json::to_vec(&envelope)?;
        atomic_write(
            &self.root.join("manifest.json"),
            &bytes,
            self.durability == Durability::Fsync,
        )
        .map_err(|e| {
            if e.published {
                IndexError::CommitUncertain(e.source)
            } else {
                IndexError::Io(e.source)
            }
        })
    }

    fn commit(&self, current: &State, mut next: State) -> Result<(), IndexError> {
        next.generation = current
            .generation
            .checked_add(1)
            .ok_or_else(|| IndexError::Invalid("generation exhausted".into()))?;
        // Every derived change is staged before persistence, then published
        // together with its documents, including a post-rename uncertain commit.
        if let Some(ann) = next.fde_ann.as_mut() {
            ann.generation = next.generation;
        }
        for ann in next.named_ann.values_mut() {
            ann.generation = next.generation;
        }
        if next.stores.objects.len()? as usize != next.object_bytes().len() {
            next.objects_map = Some(Arc::new(next.stores.objects.map()?));
        }
        if next.stores.fde.len()? as usize != next.fde_bytes().len() {
            next.fde_map = Some(Arc::new(next.stores.fde.map()?));
        }
        let result = self.persist(&next);
        if result.is_ok() || matches!(result, Err(IndexError::CommitUncertain(_))) {
            next.stores
                .retired
                .store(false, std::sync::atomic::Ordering::Relaxed);
            *self.state.write().unwrap() = Arc::new(next);
        }
        result
    }
    fn invalidate_fde_ann(next: &mut State) {
        next.fde_ann = None;
    }
    /// Train PLAID's coarse k-means codebook. Must happen before ingestion.
    pub fn train(&self, samples: &[Vector], iterations: usize) -> Result<(), IndexError> {
        self.validate(samples)?;
        if samples.len() < self.config.centroids {
            return Err(IndexError::Invalid(
                "training samples must be >= centroid count".into(),
            ));
        }
        let _writer = self.writer.lock().unwrap();
        let s = self.snapshot();
        if s.documents.values().any(|d| d.tokens > 0) {
            return Err(IndexError::Invalid(
                "cannot retrain an index containing token vectors".into(),
            ));
        }
        let samples: Vec<_> = samples.iter().map(|sample| normalize(sample)).collect();
        let mut centers: Vec<_> = (0..self.config.centroids)
            .map(|i| samples[i * samples.len() / self.config.centroids].clone())
            .collect();
        for _ in 0..iterations.max(1) {
            let mut sums = vec![vec![0.; self.config.dimension]; centers.len()];
            let mut counts = vec![0usize; centers.len()];
            for v in &samples {
                let c = nearest(v, &centers);
                counts[c] += 1;
                for (i, x) in v.iter().enumerate() {
                    sums[c][i] += x;
                }
            }
            for c in 0..centers.len() {
                if counts[c] > 0 {
                    for x in &mut sums[c] {
                        *x /= counts[c] as f32;
                    }
                    centers[c] = normalize(&sums[c]);
                }
            }
        }
        let mut next = (*s).clone();
        next.codebook = centers;
        let residuals: Vec<f32> = samples
            .iter()
            .flat_map(|vector| {
                let center = &next.codebook[nearest(vector, &next.codebook)];
                vector
                    .iter()
                    .zip(center)
                    .map(|(value, centroid)| value - centroid)
                    .collect::<Vec<_>>()
            })
            .collect();
        next.residual_codebook =
            train_scalar_codebook(&residuals, 1usize << self.config.residual_bits, 12);
        next.postings = vec![HashSet::new(); next.codebook.len()];
        // Retraining an empty collection resets its derived graph/overlay.
        Self::invalidate_fde_ann(&mut next);
        self.commit(&s, next)
    }
    pub fn upsert(
        &self,
        id: impl Into<String>,
        vectors: Vec<Vector>,
        metadata: Value,
    ) -> Result<(), IndexError> {
        self.upsert_batch(vec![UpsertDocument {
            id: id.into(),
            vectors,
            metadata,
        }])
    }
    /// Atomically replace the whole batch. Duplicate IDs use the last value.
    /// Empty batches are no-ops. On a pre-publication error no document,
    /// posting, generation, or existing ANN changes; unreachable bytes may
    /// remain in the append-only segments until recovery/compaction.
    pub fn upsert_batch(&self, batch: Vec<UpsertDocument>) -> Result<(), IndexError> {
        self.upsert_records(
            batch
                .into_iter()
                .map(|d| RetrievalDocument {
                    id: d.id,
                    vectors: d.vectors,
                    metadata: d.metadata,
                    ..RetrievalDocument::default()
                })
                .collect(),
        )
    }
    pub fn upsert_records(&self, batch: Vec<RetrievalDocument>) -> Result<(), IndexError> {
        if batch.is_empty() {
            return Ok(());
        }
        for document in &batch {
            if !document.vectors.is_empty() {
                self.validate(&document.vectors)?;
            }
        }
        let _writer = self.writer.lock().unwrap();
        let s = self.snapshot();
        if s.codebook.is_empty() && batch.iter().any(|d| !d.vectors.is_empty()) {
            return Err(IndexError::Invalid(
                "index is untrained; call train first".into(),
            ));
        }
        let mut next = (*s).clone();
        if next.stores.objects.len()? + next.stores.fde.len()? >= 64 * 1024 * 1024 {
            self.rotate_segment(&mut next)?;
        }
        for document in batch {
            let fields = Arc::new(Fields::prepare(&document, &next.stores)?);
            let id = document.id;
            Arc::make_mut(&mut next.retrieval).insert(&id, &fields, &document.metadata)?;
            if let Some(old) = next.documents.get(&id) {
                next.planner_stats.remove(old, next.retrieval.schema());
            }
            if let Some(old_ids) = next
                .documents
                .get(&id)
                .map(|old| old.unique_centroids.clone())
            {
                for c in old_ids {
                    next.postings[c as usize].remove(&id);
                }
            }
            let vectors: Vec<_> = document
                .vectors
                .iter()
                .map(|vector| normalize(vector))
                .collect();
            let ids: Vec<u32> = vectors
                .iter()
                .map(|v| nearest(v, &next.codebook) as u32)
                .collect();
            let mut unique_centroids = ids.clone();
            unique_centroids.sort_unstable();
            unique_centroids.dedup();
            let (location, size) = if vectors.is_empty() {
                (
                    ObjectLocation {
                        offset: 0,
                        length: 0,
                        checksum: None,
                    },
                    0,
                )
            } else {
                next.stores.objects.put(
                    &vectors,
                    &ids,
                    &next.codebook,
                    &next.residual_codebook,
                    self.config.residual_bits,
                )?
            };
            commit_boundary("object_appended")?;
            let fde_location = if vectors.is_empty() {
                ObjectLocation {
                    offset: 0,
                    length: 0,
                    checksum: None,
                }
            } else {
                next.stores.fde.put(&self.fde.encode_document(&vectors))?
            };
            commit_boundary("fde_appended")?;
            for &c in &unique_centroids {
                next.postings[c as usize].insert(id.clone());
            }
            if let Some(ann) = next.fde_ann.as_mut() {
                if let Some(&point_id) = ann.base.by_id.get(&id) {
                    ann.tombstones.insert(point_id);
                }
                if vectors.is_empty() {
                    ann.delta.remove(&id);
                } else {
                    ann.delta.insert(id.clone());
                }
            }
            for (field, ann) in &mut next.named_ann {
                if let Some(&point) = ann.base.by_id.get(&id) {
                    ann.tombstones.insert(point);
                }
                if fields.has_dense(field) {
                    ann.delta.insert(id.clone());
                } else {
                    ann.delta.remove(&id);
                }
            }
            let record = DocumentRecord {
                centroid_ids: ids,
                unique_centroids,
                location,
                fde_location,
                metadata: document.metadata,
                tokens: vectors.len(),
                compressed_bytes: size,
                fields,
                storage_id: next.stores.id.unwrap_or(0),
            };
            next.planner_stats.add(&record, next.retrieval.schema());
            next.documents.insert(id, record);
        }
        self.commit(&s, next)
    }
    pub fn delete(&self, id: &str) -> Result<bool, IndexError> {
        let _writer = self.writer.lock().unwrap();
        let s = self.snapshot();
        if !s.documents.contains_key(id) {
            return Ok(false);
        }
        let mut next = (*s).clone();
        let d = next.documents.remove(id).unwrap();
        Arc::make_mut(&mut next.retrieval).remove(id);
        next.planner_stats.remove(&d, next.retrieval.schema());
        for c in d.unique_centroids {
            next.postings[c as usize].remove(id);
        }
        if let Some(ann) = next.fde_ann.as_mut() {
            if let Some(&point_id) = ann.base.by_id.get(id) {
                ann.tombstones.insert(point_id);
            }
            ann.delta.remove(id);
        }
        for ann in next.named_ann.values_mut() {
            if let Some(&point) = ann.base.by_id.get(id) {
                ann.tombstones.insert(point);
            }
            ann.delta.remove(id);
        }
        self.commit(&s, next)?;
        Ok(true)
    }
    /// Exact MUVERA candidates followed by compressed MaxSim rescoring.
    pub fn query(
        &self,
        vectors: &[Vector],
        top_k: usize,
        candidates: Option<usize>,
    ) -> Result<Vec<Hit>, IndexError> {
        self.validate(vectors)?;
        if top_k == 0 {
            return Err(IndexError::Invalid("top_k must be positive".into()));
        }
        let normalized: Vec<_> = vectors.iter().map(|vector| normalize(vector)).collect();
        // Rescore only ever reads the top `count` — mirror the same cap here
        // so exact_fde_scores can partial-sort instead of fully sorting the
        // 10K+ pool it just scored.
        let cap = candidates.unwrap_or(top_k.saturating_mul(8)).max(top_k);
        let s = self.snapshot();
        let approximate = self.exact_fde_scores_capped(&s, &normalized, Some(cap))?;
        self.rescore(&s, &normalized, approximate, top_k, candidates)
    }
    /// Whether the base plus mutable overlay covers the current generation.
    pub fn hnsw_ready(&self) -> bool {
        let s = self.snapshot();
        s.fde_ann
            .as_ref()
            .is_some_and(|ann| ann.generation == s.generation)
    }

    /// Use a fresh FDE-HNSW base plus exact delta when available; otherwise
    /// use exact FDE. Selection and rescoring share one generation read guard.
    pub fn query_auto(
        &self,
        vectors: &[Vector],
        top_k: usize,
        candidates: Option<usize>,
        ef_search: usize,
    ) -> Result<Vec<Hit>, IndexError> {
        self.query_auto_with_backend(vectors, top_k, candidates, ef_search)
            .map(|(hits, _)| hits)
    }

    /// Return the actual backend used under the same snapshot as the results.
    pub fn query_auto_with_backend(
        &self,
        vectors: &[Vector],
        top_k: usize,
        candidates: Option<usize>,
        ef_search: usize,
    ) -> Result<(Vec<Hit>, &'static str), IndexError> {
        self.validate(vectors)?;
        if top_k == 0 || ef_search == 0 {
            return Err(IndexError::Invalid(
                "top_k and ef_search must be positive".into(),
            ));
        }
        let normalized: Vec<_> = vectors.iter().map(|v| normalize(v)).collect();
        let count = candidates.unwrap_or(top_k.saturating_mul(8)).max(top_k);
        let s = self.snapshot();
        let (approximate, backend) = if s
            .fde_ann
            .as_ref()
            .is_some_and(|ann| ann.generation == s.generation)
        {
            (
                self.ann_fde_scores(&s, &self.fde.encode_query(&normalized), count, ef_search)?,
                "hnsw",
            )
        } else {
            (
                self.exact_fde_scores_capped(&s, &normalized, Some(count))?,
                "muvera",
            )
        };
        Ok((
            self.rescore(&s, &normalized, approximate, top_k, candidates)?,
            backend,
        ))
    }
    /// Generate broad FDE candidates, prune with centroid-only MaxSim, then
    /// decode residuals only for the surviving documents.
    pub fn query_with_centroid_pruning(
        &self,
        vectors: &[Vector],
        top_k: usize,
        candidates: usize,
        rerank_candidates: usize,
    ) -> Result<Vec<Hit>, IndexError> {
        self.validate(vectors)?;
        check_pruning_shape(top_k, candidates, rerank_candidates)?;
        let normalized: Vec<_> = vectors.iter().map(|vector| normalize(vector)).collect();
        let s = self.snapshot();
        let approximate = self.exact_fde_scores_capped(&s, &normalized, Some(candidates))?;
        self.prune_and_rescore(
            &s,
            &normalized,
            approximate,
            top_k,
            candidates,
            rerank_candidates,
        )
    }
    /// Same pipeline as `query_with_centroid_pruning`, but pulls the broad
    /// candidate set from the FDE HNSW graph instead of the exact FDE scan.
    pub fn query_with_fde_ann_and_pruning(
        &self,
        vectors: &[Vector],
        top_k: usize,
        candidates: usize,
        rerank_candidates: usize,
        ef_search: usize,
    ) -> Result<Vec<Hit>, IndexError> {
        self.validate(vectors)?;
        if ef_search == 0 {
            return Err(IndexError::Invalid("ef_search must be positive".into()));
        }
        check_pruning_shape(top_k, candidates, rerank_candidates)?;
        let normalized: Vec<_> = vectors.iter().map(|vector| normalize(vector)).collect();
        let query_fde = self.fde.encode_query(&normalized);
        let s = self.snapshot();
        let approximate = self.ann_fde_scores(&s, &query_fde, candidates, ef_search)?;
        self.prune_and_rescore(
            &s,
            &normalized,
            approximate,
            top_k,
            candidates,
            rerank_candidates,
        )
    }
    fn prune_and_rescore(
        &self,
        s: &State,
        normalized: &[Vector],
        approximate: Vec<(String, f32)>,
        top_k: usize,
        candidates: usize,
        rerank_candidates: usize,
    ) -> Result<Vec<Hit>, IndexError> {
        let broad: Vec<_> = approximate.into_iter().take(candidates).collect();
        let pruned = centroid_prune(s, normalized, broad, rerank_candidates);
        self.rescore(s, normalized, pruned, top_k, Some(rerank_candidates))
    }
    fn exact_fde_scores(&self, normalized: &[Vector]) -> Result<Vec<(String, f32)>, IndexError> {
        let s = self.snapshot();
        self.exact_fde_scores_capped(&s, normalized, None)
    }

    /// FDE exhaustive scan, optionally returning only the top `cap` results.
    /// When capped, partitions the top-cap with `select_nth_unstable_by`
    /// (O(n) expected) then fully sorts only the surviving prefix
    /// (O(cap log cap)), instead of paying O(n log n) to sort a 10K+
    /// candidate pool whose tail is discarded by rescore anyway.
    fn exact_fde_scores_capped(
        &self,
        s: &State,
        normalized: &[Vector],
        cap: Option<usize>,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        self.exact_fde_scores_filtered(s, normalized, cap, None)
    }
    fn exact_fde_scores_filtered(
        &self,
        s: &State,
        normalized: &[Vector],
        cap: Option<usize>,
        eligible: Option<&retrieval::DocSet>,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        if !s.documents.values().any(|d| d.tokens > 0) {
            return Ok(Vec::new());
        }
        let query_fde = self.fde.encode_query(normalized);
        let fde_dimension = self.fde.output_dimension();
        let score = |id: &str, record: &DocumentRecord| {
            (record.tokens > 0).then(|| {
                Ok((
                    id.to_owned(),
                    dot(
                        &query_fde,
                        FixedVectorStore::get(
                            s.record_fde(record),
                            record.fde_location,
                            fde_dimension,
                        )?,
                    ),
                ))
            })
        };
        let approximate_results: Result<Vec<_>, io::Error> = match eligible {
            Some(eligible) => eligible
                .iter()
                .filter_map(|number| s.retrieval.external_id(number))
                .collect::<Vec<_>>()
                .into_par_iter()
                .filter_map(|id| s.documents.get(id).and_then(|record| score(id, record)))
                .collect(),
            None => s
                .documents
                .par_iter()
                .filter_map(|(id, record)| score(id, record))
                .collect(),
        };
        let mut approximate = approximate_results?;
        let n = approximate.len();
        let by_desc =
            |a: &(String, f32), b: &(String, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
        match cap {
            Some(k) if k < n => {
                approximate.select_nth_unstable_by(k, by_desc);
                approximate.truncate(k);
                approximate.par_sort_unstable_by(by_desc);
            }
            _ => {
                approximate.par_sort_unstable_by(by_desc);
            }
        }
        Ok(approximate)
    }
    /// Return exact FDE candidates before compressed MaxSim reranking.
    pub fn exact_fde_candidates(
        &self,
        vectors: &[Vector],
        count: usize,
    ) -> Result<Vec<CandidateHit>, IndexError> {
        self.validate(vectors)?;
        if count == 0 {
            return Err(IndexError::Invalid(
                "candidate count must be positive".into(),
            ));
        }
        let normalized: Vec<_> = vectors.iter().map(|vector| normalize(vector)).collect();
        Ok(self
            .exact_fde_scores(&normalized)?
            .into_iter()
            .take(count)
            .map(|(id, score)| CandidateHit { id, score })
            .collect())
    }
    /// Build an HNSW index over persisted FDEs. Exact FDE scan remains available as an oracle.
    pub fn build_fde_ann(&self, m: usize, ef_construct: usize) -> Result<usize, IndexError> {
        self.build_ann(None, m, ef_construct)
    }
    pub fn build_dense_ann(
        &self,
        field: &str,
        m: usize,
        ef_construct: usize,
    ) -> Result<usize, IndexError> {
        self.build_ann(Some(field), m, ef_construct)
    }
    fn build_ann(
        &self,
        field: Option<&str>,
        m: usize,
        ef_construct: usize,
    ) -> Result<usize, IndexError> {
        if !(1..=128).contains(&m) || !(1..=65_536).contains(&ef_construct) {
            return Err(IndexError::Invalid(
                "HNSW m must be in 1..=128 and ef_construct in 1..=65536".into(),
            ));
        }
        let s = self.snapshot();
        let dimension = if let Some(field) = field {
            s.retrieval.dense_dimension(field)?
        } else {
            self.fde.output_dimension()
        };
        let mut ids: Vec<_> = s
            .documents
            .iter()
            .filter(|(_, d)| field.map_or(d.tokens > 0, |f| d.fields.has_dense(f)))
            .map(|(id, _)| id.clone())
            .collect();
        ids.sort();
        let mut hnsw = HNSWIndex::new(DistanceMetric::Dot, m, ef_construct, 16, dimension);
        // Build in chunks so peak memory stays bounded regardless of corpus
        // size (each chunk holds only its own decoded vectors). Each chunk is
        // handed to par_insert_batch, which uses annex-core's concurrent
        // &self insert path to parallelise linking across threads.
        const CHUNK: usize = 4096;
        for (chunk_idx, chunk_ids) in ids.chunks(CHUNK).enumerate() {
            let base = chunk_idx * CHUNK;
            let entries: Vec<(u64, Vec<f32>)> = chunk_ids
                .iter()
                .enumerate()
                .map(|(offset, id)| {
                    let vector = if let Some(field) = field {
                        self.dense_vector(&s, id, field)?
                    } else {
                        FixedVectorStore::get(
                            s.record_fde(&s.documents[id]),
                            s.documents[id].fde_location,
                            dimension,
                        )?
                    }
                    .to_vec();
                    Ok::<_, IndexError>(((base + offset) as u64, vector))
                })
                .collect::<Result<Vec<_>, _>>()?;
            hnsw.par_insert_batch(&entries).map_err(|error| {
                IndexError::Invalid(format!("HNSW par_insert_batch failed: {error}"))
            })?;
        }
        // Reorder only. SQ8 screening currently supports Cosine, whereas
        // FDE uses Dot; allocating codes here would add cost without benefit.
        if !ids.is_empty() {
            hnsw.reorder_rcm();
        }
        let built_generation = s.generation;
        drop(s);
        commit_boundary("ann_built_before_publish")?;
        #[cfg(test)]
        transaction_tests::before_ann_publish();
        let _writer = self.writer.lock().unwrap();
        let s = self.snapshot();
        if s.generation != built_generation {
            return Err(IndexError::Invalid(format!(
                "index generation moved during HNSW build ({} -> {}); retry",
                built_generation, s.generation,
            )));
        }
        let count = ids.len();
        let by_id = ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.clone(), idx as u64))
            .collect();
        let mut payloads = HashMap::with_capacity(ids.len());
        let mut payload_index = PayloadIndex::new();
        for (point, id) in ids.iter().enumerate() {
            let payload = retrieval::metadata_ann_payload(&s.documents[id].metadata);
            payload_index.insert(point as u64, &payload);
            payloads.insert(point as u64, payload);
        }
        let mut next = (*s).clone();
        let ann = FdeAnn {
            base: Arc::new(FdeAnnBase {
                index: hnsw,
                ids,
                by_id,
                field: field.map(str::to_owned),
                payloads,
                payload_index,
            }),
            delta: HashSet::new(),
            tombstones: HashSet::new(),
            generation: built_generation,
        };
        if let Some(field) = field {
            next.named_ann.insert(field.to_owned(), ann);
        } else {
            next.fde_ann = Some(ann);
        }
        *self.state.write().unwrap() = Arc::new(next);
        Ok(count)
    }
    pub fn query_with_fde_ann(
        &self,
        vectors: &[Vector],
        top_k: usize,
        candidates: Option<usize>,
        ef_search: usize,
    ) -> Result<Vec<Hit>, IndexError> {
        self.validate(vectors)?;
        if top_k == 0 || ef_search == 0 {
            return Err(IndexError::Invalid(
                "top_k and ef_search must be positive".into(),
            ));
        }
        let normalized: Vec<_> = vectors.iter().map(|v| normalize(v)).collect();
        let query_fde = self.fde.encode_query(&normalized);
        let s = self.snapshot();
        let count = candidates.unwrap_or(top_k.saturating_mul(8)).max(top_k);
        let approximate = self.ann_fde_scores(&s, &query_fde, count, ef_search)?;
        self.rescore(&s, &normalized, approximate, top_k, candidates)
    }
    fn ann_fde_scores(
        &self,
        s: &State,
        query_fde: &Vector,
        count: usize,
        ef_search: usize,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        self.ann_fde_scores_filtered(s, query_fde, count, ef_search, None, None)
    }

    fn ann_fde_scores_filtered(
        &self,
        s: &State,
        query_fde: &Vector,
        count: usize,
        ef_search: usize,
        filter: Option<&Filter>,
        eligible: Option<&retrieval::DocSet>,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        let ann = s.fde_ann.as_ref().ok_or_else(|| {
            IndexError::Invalid("FDE ANN is not built; call /v1/fde/index".into())
        })?;
        self.ann_scores_filtered(s, ann, query_fde, count, ef_search, filter, eligible)
    }
    fn ann_scores_filtered(
        &self,
        s: &State,
        ann: &FdeAnn,
        query_fde: &Vector,
        count: usize,
        ef_search: usize,
        filter: Option<&Filter>,
        eligible: Option<&retrieval::DocSet>,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        if ann.generation != s.generation {
            return Err(IndexError::Invalid(
                "FDE ANN generation is stale; rebuild".into(),
            ));
        }
        let mut scores = Vec::new();
        if ann.base.ids.len() > ann.tombstones.len() {
            // Account for masked base hits before asking the graph for candidates.
            let base_count = count
                .saturating_add(ann.tombstones.len())
                .min(ann.base.ids.len());
            let options = SearchRuntimeOptions {
                ef_search: Some(ef_search.max(base_count)),
                ..SearchRuntimeOptions::default()
            };
            let points = match filter {
                Some(filter) => ann.base.index.in_place_filtered_search(
                    query_fde,
                    base_count,
                    &options,
                    &ann.base.payloads,
                    &ann.base.payload_index,
                    Some(filter),
                ),
                None => ann
                    .base
                    .index
                    .search_with_options(query_fde, base_count, &options),
            }
            .map_err(|error| IndexError::Invalid(format!("HNSW search failed: {error}")))?;
            for point in points {
                if ann.tombstones.contains(&point.id) {
                    continue;
                }
                let id = ann
                    .base
                    .ids
                    .get(point.id as usize)
                    .ok_or_else(|| IndexError::Invalid("invalid HNSW point id".into()))?;
                if !s.documents.contains_key(id) || ann.delta.contains(id) {
                    return Err(IndexError::Invalid(
                        "FDE ANN overlay is inconsistent; rebuild".into(),
                    ));
                }
                scores.push((id.clone(), -point.sort_key));
            }
        }
        if !ann.delta.is_empty() {
            for id in &ann.delta {
                if eligible
                    .is_some_and(|eligible| !s.retrieval.contains_external(eligible, id.as_str()))
                {
                    continue;
                }
                let record = s
                    .documents
                    .get(id)
                    .ok_or_else(|| IndexError::Invalid("missing delta document".into()))?;
                let vector = if let Some(field) = &ann.base.field {
                    self.dense_vector(s, id, field)?
                } else {
                    FixedVectorStore::get(
                        s.record_fde(record),
                        record.fde_location,
                        self.fde.output_dimension(),
                    )?
                };
                scores.push((id.clone(), dot(query_fde, vector)));
            }
        }
        let by_score =
            |a: &(String, f32), b: &(String, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
        if scores.len() > count {
            scores.select_nth_unstable_by(count, by_score);
            scores.truncate(count);
        }
        scores.sort_unstable_by(by_score);
        Ok(scores)
    }
    /// Return HNSW FDE candidates before compressed MaxSim reranking.
    pub fn ann_fde_candidates(
        &self,
        vectors: &[Vector],
        count: usize,
        ef_search: usize,
    ) -> Result<Vec<CandidateHit>, IndexError> {
        self.validate(vectors)?;
        if count == 0 || ef_search == 0 {
            return Err(IndexError::Invalid(
                "candidate count and ef_search must be positive".into(),
            ));
        }
        let normalized: Vec<_> = vectors.iter().map(|vector| normalize(vector)).collect();
        let query_fde = self.fde.encode_query(&normalized);
        let s = self.snapshot();
        Ok(self
            .ann_fde_scores(&s, &query_fde, count, ef_search)?
            .into_iter()
            .map(|(id, score)| CandidateHit { id, score })
            .collect())
    }
    pub fn query_with_probes(
        &self,
        vectors: &[Vector],
        top_k: usize,
        candidates: Option<usize>,
        probes: usize,
    ) -> Result<Vec<Hit>, IndexError> {
        self.validate(vectors)?;
        if top_k == 0 {
            return Err(IndexError::Invalid("top_k must be positive".into()));
        }
        let normalized: Vec<_> = vectors.iter().map(|vector| normalize(vector)).collect();
        let s = self.snapshot();
        if s.codebook.is_empty() {
            return Err(IndexError::Invalid("index is untrained".into()));
        }
        // Compute Q x C once; all later centroid interaction is a table lookup.
        let interaction: Vec<Vec<f32>> = normalized
            .iter()
            .map(|q| s.codebook.iter().map(|c| dot(q, c)).collect())
            .collect();
        let mut selected = HashSet::new();
        for row in &interaction {
            let mut scored: Vec<_> = row.iter().copied().enumerate().collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            for &(c, _) in scored.iter().take(probes.max(1)) {
                selected.insert(c);
            }
        }
        let mut candidate_ids = HashSet::new();
        for c in selected {
            candidate_ids.extend(s.postings[c].iter().cloned());
        }
        let mut approx: Vec<_> = candidate_ids
            .into_iter()
            .map(|id| {
                let d = &s.documents[&id];
                let score = interaction
                    .iter()
                    .map(|row| {
                        d.unique_centroids
                            .iter()
                            .map(|&c| row[c as usize])
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .sum::<f32>();
                (id, score)
            })
            .collect();
        approx.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut hits = self.rescore(&s, &normalized, approx, top_k, candidates)?;
        for hit in &mut hits {
            hit.fde_score = None;
        }
        Ok(hits)
    }
    fn rescore(
        &self,
        s: &State,
        normalized: &[Vector],
        approximate: Vec<(String, f32)>,
        top_k: usize,
        candidates: Option<usize>,
    ) -> Result<Vec<Hit>, IndexError> {
        if approximate.is_empty() {
            return Ok(Vec::new());
        }
        let count = candidates.unwrap_or(top_k.saturating_mul(8)).max(top_k);
        // Per-stage timing accumulators, gated by MULTIVECTOR_TIMING. Cached
        // in a OnceLock so a live server pays the env::var HashMap lookup
        // exactly once, not per rescoring call.
        use std::sync::OnceLock;
        static TIMING: OnceLock<bool> = OnceLock::new();
        let timing = *TIMING.get_or_init(|| std::env::var("MULTIVECTOR_TIMING").is_ok());
        let decode_ns = std::sync::atomic::AtomicU64::new(0);
        let maxsim_ns = std::sync::atomic::AtomicU64::new(0);
        // Per-worker scratch buffer for compressed decode. Reused across every
        // candidate this thread scores in this rescoring call — turns
        // (candidates x per-doc) Vec::with_capacity(count*dim) allocations
        // into one grow-once-per-thread. Concretely on a FiQA-shaped query
        // with 250 candidates x 200 tokens x 128 dims that removes about
        // 25 MB of scratch f32 allocs per query.
        thread_local! {
            static DECODE_SCRATCH: std::cell::RefCell<Vec<f32>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        // Two-phase: (1) score every candidate producing only (idx, score),
        // then (2) pick the top_k and materialise Hit structs only for the
        // survivors. This avoids cloning id + metadata for the (count -
        // top_k) candidates that get thrown away after the sort — on a
        // typical query with candidates=500 and top_k=100 that saves 400
        // String + Value clones per query.
        // Pack the query once for every candidate (x86 AVX2/AVX-512).
        let prepared = MaxSimQuery::new(normalized, self.config.dimension);
        let approximate_slice: &[(String, f32)] = approximate.as_slice();
        let scored: Vec<(usize, f32)> = approximate_slice
            .par_iter()
            .take(count)
            .enumerate()
            .map(|(idx, (id, _))| -> Result<(usize, f32), io::Error> {
                let record = &s.documents[id];
                let t0 = if timing {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let (score, t1) = DECODE_SCRATCH.with(
                    |cell| -> Result<(f32, Option<std::time::Instant>), io::Error> {
                        let mut scratch = cell.borrow_mut();
                        let dim = CompressedVectorStore::decode_into(
                            s.record_objects(record),
                            record.location,
                            &s.codebook,
                            &s.residual_codebook,
                            &mut scratch,
                        )?;
                        let t1 = if timing {
                            Some(std::time::Instant::now())
                        } else {
                            None
                        };
                        Ok((prepared.score(&scratch, dim), t1))
                    },
                )?;
                let t2 = if timing {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
                    decode_ns.fetch_add(
                        t1.duration_since(t0).as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    maxsim_ns.fetch_add(
                        t2.duration_since(t1).as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                Ok((idx, score))
            })
            .collect::<Result<Vec<_>, io::Error>>()?;
        if timing {
            let d = decode_ns.load(std::sync::atomic::Ordering::Relaxed);
            let m = maxsim_ns.load(std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[timing] candidates={count} decode_us={} maxsim_us={} (per-thread aggregate)",
                d / 1000,
                m / 1000
            );
        }
        // Partition + sort only the top_k survivors, then materialise the
        // Hit structs — avoids id/metadata clones for candidates outside
        // top_k. select_nth_unstable_by would be O(n) but we still need a
        // sorted top_k, so partition then sort the small prefix.
        let mut scored = scored;
        let n = scored.len();
        let by_score_desc = |a: &(usize, f32), b: &(usize, f32)| {
            b.1.total_cmp(&a.1)
                .then_with(|| approximate_slice[a.0].0.cmp(&approximate_slice[b.0].0))
        };
        if top_k < n {
            scored.select_nth_unstable_by(top_k, by_score_desc);
            scored.truncate(top_k);
        }
        scored.sort_unstable_by(by_score_desc);
        let hits: Vec<Hit> = scored
            .into_iter()
            .map(|(idx, score)| {
                let (id, fde) = &approximate_slice[idx];
                let record = &s.documents[id];
                Hit {
                    id: id.clone(),
                    score,
                    fde_score: Some(*fde),
                    metadata: record.metadata.clone(),
                }
            })
            .collect();
        Ok(hits)
    }
    fn new_segment(&self, generation: u64) -> Result<Arc<SegmentStores>, IndexError> {
        let mut id = generation
            .checked_add(1)
            .ok_or_else(|| IndexError::Invalid("generation exhausted".into()))?;
        let parent = self.root.join("segments");
        fs::create_dir_all(&parent)?;
        let root = loop {
            let path = parent.join(id.to_string());
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    id = id
                        .checked_add(1)
                        .ok_or_else(|| IndexError::Invalid("segment IDs exhausted".into()))?;
                }
                Err(e) => return Err(e.into()),
            }
        };
        let stores = Arc::new(SegmentStores {
            objects: CompressedVectorStore::new(root.join("objects"))?,
            fde: FixedVectorStore::new(root.join("fde"))?,
            root,
            id: Some(id),
            retired: std::sync::atomic::AtomicBool::new(true),
        });
        Ok(stores)
    }
    fn rotate_segment(&self, next: &mut State) -> Result<(), IndexError> {
        let stores = self.new_segment(next.generation)?;
        // A failed append can leave a tail beyond the published mappings.
        // Sealing preserves the committed prefix, never the physical file size.
        let bounds = SegmentBoundaries {
            objects: next.object_bytes().len() as u64,
            fde: next.fde_bytes().len() as u64,
        };
        next.sealed.insert(
            next.stores.id.unwrap_or(0),
            Arc::new(SegmentSnapshot {
                stores: Arc::clone(&next.stores),
                objects: next.objects_map.take(),
                fde: next.fde_map.take(),
                bounds,
            }),
        );
        if self.durability == Durability::Fsync {
            for path in [
                stores.root.join("objects"),
                stores.root.join("fde"),
                stores.root.clone(),
                self.root.join("segments"),
            ] {
                File::open(path)?.sync_all()?;
            }
            File::open(&self.root)?.sync_all()?;
        }
        next.stores = stores;
        Ok(())
    }
    pub fn seal(&self) -> Result<(), IndexError> {
        let _writer = self.writer.lock().unwrap();
        let s = self.snapshot();
        let mut next = (*s).clone();
        self.rotate_segment(&mut next)?;
        self.commit(&s, next)
    }
    /// Copy live records into a new append segment and atomically publish its
    /// locations. Existing readers retain their old files until they finish.
    pub fn compact(&self) -> Result<serde_json::Value, IndexError> {
        let s = self.snapshot();
        let before = s.stores.objects.len()?
            + s.stores.fde.len()?
            + s.sealed
                .values()
                .map(|v| Ok::<_, io::Error>(v.stores.objects.len()? + v.stores.fde.len()?))
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .sum::<u64>();
        let stores = self.new_segment(s.generation)?;
        let mut next = (*s).clone();
        next.stores = Arc::clone(&stores);
        next.objects_map = None;
        next.fde_map = None;
        next.sealed.clear();
        for document in next.documents.values_mut() {
            if document.tokens > 0 {
                document.location = stores
                    .objects
                    .copy_record(s.record_objects(document), document.location)?;
                document.fde_location = stores
                    .fde
                    .copy_record(s.record_fde(document), document.fde_location)?;
            }
            let source = s.record_fde(document);
            Arc::make_mut(&mut document.fields).relocate(source, &stores.fde)?;
            document.storage_id = stores.id.unwrap();
        }
        if self.durability == Durability::Fsync {
            for path in [
                stores.root.join("objects"),
                stores.root.join("fde"),
                stores.root.clone(),
                self.root.join("segments"),
            ] {
                File::open(path)?.sync_all()?;
            }
            File::open(&self.root)?.sync_all()?;
        }
        commit_boundary("compaction_copied")?;
        let _writer = self.writer.lock().unwrap();
        if self.snapshot().generation != s.generation {
            return Err(IndexError::Invalid(
                "generation changed during compaction; retry".into(),
            ));
        }
        let result = self.commit(&s, next);
        if result.is_ok() || matches!(result, Err(IndexError::CommitUncertain(_))) {
            stores
                .retired
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        if result.is_ok() {
            s.stores
                .retired
                .store(true, std::sync::atomic::Ordering::Relaxed);
            for old in s.sealed.values() {
                old.stores
                    .retired
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        result?;
        let after = stores.objects.len()? + stores.fde.len()?;
        Ok(
            serde_json::json!({"generation":s.generation+1,"bytes_before":before,"bytes_after":after,"bytes_reclaimed":before.saturating_sub(after)}),
        )
    }

    pub fn stats(&self) -> IndexStats {
        let s = self.snapshot();
        IndexStats {
            documents: s.documents.len(),
            generation: s.generation,
            token_vectors: s.documents.values().map(|d| d.tokens).sum(),
            compressed_bytes: s.documents.values().map(|d| d.compressed_bytes).sum(),
            centroids: s.codebook.len(),
            residual_bits: self.config.residual_bits,
            trained: !s.codebook.is_empty(),
            fde_dimension: self.fde.output_dimension(),
            fde_ann_nodes: s.fde_ann.as_ref().map_or(0, |ann| {
                ann.base.ids.len() - ann.tombstones.len() + ann.delta.len()
            }),
            fde_ann_base_nodes: s.fde_ann.as_ref().map_or(0, |ann| ann.base.ids.len()),
            fde_ann_delta_documents: s.fde_ann.as_ref().map_or(0, |ann| ann.delta.len()),
            fde_ann_tombstones: s.fde_ann.as_ref().map_or(0, |ann| ann.tombstones.len()),
            fde_encoding_version: self.fde.encoding_version(),
            storage_segments: s.sealed.len() + 1,
            dense_ann_fields: s
                .named_ann
                .iter()
                .map(|(field, ann)| {
                    (
                        field.clone(),
                        ann.base.ids.len() - ann.tombstones.len() + ann.delta.len(),
                    )
                })
                .collect(),
        }
    }
    /// Diagnostic score over caller-provided vectors; used to verify scorer parity.
    pub fn score_uncompressed(
        &self,
        query: &[Vector],
        document: &[Vector],
    ) -> Result<f32, IndexError> {
        self.validate(query)?;
        self.validate(document)?;
        let query: Vec<_> = query.iter().map(|v| normalize(v)).collect();
        let document: Vec<_> = document.iter().map(|v| normalize(v)).collect();
        let flat: Vec<_> = document.into_iter().flatten().collect();
        Ok(maxsim_flat(&query, &flat, self.config.dimension))
    }
    /// Diagnostic score over the actual compressed bytes stored for a document.
    pub fn score_compressed(&self, query: &[Vector], id: &str) -> Result<f32, IndexError> {
        self.validate(query)?;
        let query: Vec<_> = query.iter().map(|v| normalize(v)).collect();
        let s = self.snapshot();
        let record = s
            .documents
            .get(id)
            .ok_or_else(|| IndexError::Invalid(format!("unknown document: {id}")))?;
        let document = CompressedVectorStore::decode(
            s.record_objects(record),
            record.location,
            &s.codebook,
            &s.residual_codebook,
        )?;
        Ok(maxsim_flat(&query, &document.values, document.dimension))
    }
}

fn validate_config_size(c: &IndexConfig) -> Result<(), IndexError> {
    // Upper bound on any individual encoder/codebook workspace (64 MiB f32).
    const MAX_VALUES: usize = 16 * 1024 * 1024;
    let bounded = |n: Option<usize>| n.is_some_and(|n| n <= MAX_VALUES);
    let buckets = 1usize << c.fde_ksim;
    if c.centroids > u32::MAX as usize
        || !bounded(c.dimension.checked_mul(c.centroids))
        || !bounded(
            buckets
                .checked_mul(c.fde_projected)
                .and_then(|n| n.checked_mul(c.fde_repetitions)),
        )
        || !bounded(
            c.fde_projected
                .checked_add(c.fde_ksim)
                .and_then(|n| n.checked_mul(c.dimension))
                .and_then(|n| n.checked_mul(c.fde_repetitions)),
        )
        || !bounded(buckets.checked_mul(c.dimension))
    {
        return Err(IndexError::Invalid(
            "configuration exceeds checked 64 MiB workspace limit".into(),
        ));
    }
    Ok(())
}

fn validate_codebooks(
    c: &IndexConfig,
    centers: &[Vector],
    residuals: &[f32],
    has_docs: bool,
) -> Result<(), IndexError> {
    if centers.is_empty() && residuals.is_empty() && !has_docs {
        return Ok(());
    }
    if centers.len() != c.centroids
        || residuals.len() != 1usize << c.residual_bits
        || centers
            .iter()
            .any(|v| v.len() != c.dimension || v.iter().any(|x| !x.is_finite()))
        || residuals.iter().any(|x| !x.is_finite())
    {
        return Err(IndexError::Invalid("invalid persisted codebook".into()));
    }
    Ok(())
}

fn check_pruning_shape(
    top_k: usize,
    candidates: usize,
    rerank_candidates: usize,
) -> Result<(), IndexError> {
    if top_k == 0 || rerank_candidates < top_k || candidates < rerank_candidates {
        return Err(IndexError::Invalid(
            "require top_k > 0 and candidates >= rerank_candidates >= top_k".into(),
        ));
    }
    Ok(())
}
fn centroid_prune(
    s: &State,
    query: &[Vector],
    candidates: Vec<(String, f32)>,
    survivors: usize,
) -> Vec<(String, f32)> {
    let interaction: Vec<Vec<f32>> = query
        .iter()
        .map(|q| s.codebook.iter().map(|centroid| dot(q, centroid)).collect())
        .collect();
    let mut scored: Vec<_> = candidates
        .into_par_iter()
        .map(|(id, fde_score)| {
            let document = &s.documents[&id];
            let score = interaction
                .iter()
                .map(|row| {
                    document
                        .unique_centroids
                        .iter()
                        .map(|&centroid| row[centroid as usize])
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .sum::<f32>();
            (id, fde_score, score)
        })
        .collect();
    scored.par_sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(survivors);
    scored
        .into_iter()
        .map(|(id, fde_score, _)| (id, fde_score))
        .collect()
}
fn nearest(vector: &Vector, centroids: &[Vector]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| distance(vector, a).total_cmp(&distance(vector, b)))
        .unwrap()
        .0
}
fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}
fn train_scalar_codebook(values: &[f32], levels: usize, iterations: usize) -> Vec<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let mut centers: Vec<_> = (0..levels)
        .map(|i| sorted[((2 * i + 1) * sorted.len() / (2 * levels)).min(sorted.len() - 1)])
        .collect();
    for _ in 0..iterations {
        let mut sums = vec![0.; levels];
        let mut counts = vec![0usize; levels];
        for &value in values {
            let index = centers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (value - **a).abs().total_cmp(&(value - **b).abs()))
                .unwrap()
                .0;
            sums[index] += value;
            counts[index] += 1;
        }
        for i in 0..levels {
            if counts[i] > 0 {
                centers[i] = sums[i] / counts[i] as f32;
            }
        }
    }
    centers.sort_unstable_by(|a, b| a.total_cmp(b));
    centers
}

#[cfg(test)]
#[path = "transaction_tests.rs"]
mod transaction_tests;
