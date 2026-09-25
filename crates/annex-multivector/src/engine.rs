use crate::{
    fde::{Vector, dot, maxsim_flat, normalize},
    muvera::FdeEncoder,
    storage::{
        CompressedVectorStore, FixedVectorStore, ObjectLocation, atomic_write, commit_boundary,
        record_bytes, verify_record,
    },
};
use annex::{
    utils::types::DistanceMetric,
    vector::hnsw::{HNSWIndex, SearchRuntimeOptions},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
    sync::RwLock,
};
use thiserror::Error;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
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
}
/// Persisted manifest header. `format_version` lets us evolve the on-disk
/// layout later without silently accepting mismatched files. `generation`
/// is bumped on every mutation and lets any derived structure (HNSW-over-
/// FDE in particular) prove it was built against the current document set.
///
/// Format changes: bump FORMAT_VERSION and add a From<oldManifest> path.
const FORMAT_VERSION: u32 = 2;

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
    #[serde(default)]
    segments: Option<SegmentBoundaries>,
}

fn legacy_fde_encoding_version() -> u32 { 1 }

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
    /// Skipped from JSON when the underlying approximate list didn't carry
    /// FDE scores (e.g. HNSW-backend candidate gen returned raw distances).
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
}
#[derive(Debug, Error)]
pub enum IndexError {
    #[error("{0}")]
    Invalid(String),
    #[error("index configuration is {actual:?}, requested {requested:?}")]
    Config {
        actual: IndexConfig,
        requested: IndexConfig,
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
}
pub struct MultiVectorIndex {
    root: PathBuf,
    config: IndexConfig,
    objects: CompressedVectorStore,
    fde: FdeEncoder,
    fde_store: FixedVectorStore,
    state: RwLock<State>,
    durability: Durability,
    // One process/handle owns append offsets and manifest publication at a time.
    _directory_lock: File,
}

impl MultiVectorIndex {
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
        let manifest_path = root.join("manifest.json");
        let (generation, fde_encoding_version, codebook, residual_codebook, mut documents, segments, checksummed) =
            if manifest_path.exists() {
                let bytes = fs::read(&manifest_path)?;
                let header: Value = serde_json::from_slice(&bytes)?;
                let (m, checksummed): (Manifest, bool) = if header.get("manifest").is_some() {
                    let envelope: ManifestEnvelope = serde_json::from_slice(&bytes)?;
                    if envelope.format_version != FORMAT_VERSION
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
                if m.format_version != if checksummed { FORMAT_VERSION } else { 1 }
                    || (checksummed && m.segments.is_none())
                {
                    return Err(IndexError::Invalid(
                        "unsupported manifest version or missing committed boundaries".into(),
                    ));
                }
                if m.config != config {
                    return Err(IndexError::Config {
                        actual: m.config,
                        requested: config,
                    });
                }
                (
                    m.generation,
                    m.fde_encoding_version,
                    m.codebook,
                    m.residual_codebook,
                    m.documents,
                    m.segments,
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
                    Some(SegmentBoundaries { objects: 0, fde: 0 }),
                    false,
                )
            };
        if !matches!(fde_encoding_version, 1 | 2) {
            return Err(IndexError::Invalid(format!("unsupported FDE encoding version {fde_encoding_version}")));
        }
        validate_codebooks(
            &config,
            &codebook,
            &residual_codebook,
            !documents.is_empty(),
        )?;
        let objects = CompressedVectorStore::new(root.join("objects"))?;
        let fde_store = FixedVectorStore::new(root.join("fde"))?;
        if !manifest_path.exists() && (objects.len()? != 0 || fde_store.len()? != 0) {
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
        if !documents.is_empty() {
            let objects_map = objects.map()?;
            let fde_map = fde_store.map()?;
            let object_bytes = objects_map
                .get(
                    ..usize::try_from(boundaries.objects)
                        .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?,
                )
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
            let fde_bytes = fde_map
                .get(
                    ..usize::try_from(boundaries.fde)
                        .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?,
                )
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
            let fde_dimension =
                (1usize << config.fde_ksim) * config.fde_projected * config.fde_repetitions;
            for d in documents.values_mut() {
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
                    .chunks_exact(4)
                    .map(|v| u32::from_le_bytes(v.try_into().unwrap()));
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
        objects.recover(boundaries.objects)?;
        fde_store.recover(boundaries.fde)?;
        if durability == Durability::Fsync {
            File::open(root.join("objects"))?.sync_all()?;
            File::open(root.join("fde"))?.sync_all()?;
            File::open(&root)?.sync_all()?;
            for parent in created_parents {
                File::open(parent)?.sync_all()?;
            }
        }
        let mut postings = vec![HashSet::new(); codebook.len()];
        for (id, d) in &documents {
            for &c in &d.centroid_ids {
                postings[c as usize].insert(id.clone());
            }
        }
        Ok(Self {
            fde: if fde_encoding_version == 2 {
                FdeEncoder::new(
                    config.dimension, config.fde_ksim, config.fde_projected,
                    config.fde_repetitions, 0x4d55_5645_5241,
                )
            } else {
                FdeEncoder::with_version(
                    config.dimension, config.fde_ksim, config.fde_projected,
                    config.fde_repetitions, 0x4d55_5645_5241, fde_encoding_version,
                )
            },
            fde_store,
            objects,
            durability,
            _directory_lock: directory_lock,
            state: RwLock::new(State {
                generation,
                codebook,
                residual_codebook,
                documents,
                postings,
                fde_ann: None,
            }),
            root,
            config,
        })
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
            self.objects.sync()?;
            commit_boundary("objects_synced")?;
            self.fde_store.sync()?;
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
            segments: Some(SegmentBoundaries {
                objects: self.objects.len()?,
                fde: self.fde_store.len()?,
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

    fn commit(&self, current: &mut State, mut next: State) -> Result<(), IndexError> {
        next.generation = current
            .generation
            .checked_add(1)
            .ok_or_else(|| IndexError::Invalid("generation exhausted".into()))?;
        // Every derived change is staged before persistence, then published
        // together with its documents, including a post-rename uncertain commit.
        if let Some(ann) = next.fde_ann.as_mut() {
            ann.generation = next.generation;
        }
        let result = self.persist(&next);
        if result.is_ok() || matches!(result, Err(IndexError::CommitUncertain(_))) {
            *current = next;
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
        let mut s = self.state.write().unwrap();
        if !s.documents.is_empty() {
            return Err(IndexError::Invalid(
                "cannot retrain a non-empty index".into(),
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
        let mut next = s.clone();
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
        self.commit(&mut s, next)
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
        if batch.is_empty() {
            return Ok(());
        }
        for document in &batch {
            self.validate(&document.vectors)?;
        }
        let mut s = self.state.write().unwrap();
        if s.codebook.is_empty() {
            return Err(IndexError::Invalid(
                "index is untrained; call train first".into(),
            ));
        }
        let mut next = s.clone();
        for document in batch {
            let id = document.id;
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
            let (location, size) = self.objects.put(
                &vectors,
                &ids,
                &next.codebook,
                &next.residual_codebook,
                self.config.residual_bits,
            )?;
            commit_boundary("object_appended")?;
            let fde = self.fde.encode_document(&vectors);
            let fde_location = self.fde_store.put(&fde)?;
            commit_boundary("fde_appended")?;
            for &c in &unique_centroids {
                next.postings[c as usize].insert(id.clone());
            }
            if let Some(ann) = next.fde_ann.as_mut() {
                if let Some(&point_id) = ann.base.by_id.get(&id) {
                    ann.tombstones.insert(point_id);
                }
                ann.delta.insert(id.clone());
            }
            next.documents.insert(
                id,
                DocumentRecord {
                    centroid_ids: ids,
                    unique_centroids,
                    location,
                    fde_location,
                    metadata: document.metadata,
                    tokens: vectors.len(),
                    compressed_bytes: size,
                },
            );
        }
        self.commit(&mut s, next)
    }
    pub fn delete(&self, id: &str) -> Result<bool, IndexError> {
        let mut s = self.state.write().unwrap();
        if !s.documents.contains_key(id) {
            return Ok(false);
        }
        let mut next = s.clone();
        let d = next.documents.remove(id).unwrap();
        for c in d.unique_centroids {
            next.postings[c as usize].remove(id);
        }
        if let Some(ann) = next.fde_ann.as_mut() {
            if let Some(&point_id) = ann.base.by_id.get(id) {
                ann.tombstones.insert(point_id);
            }
            ann.delta.remove(id);
        }
        self.commit(&mut s, next)?;
        Ok(true)
    }
    /// PLAID centroid interaction -> inverted-list candidate generation -> residual MaxSim.
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
        let s = self.state.read().unwrap();
        let approximate = self.exact_fde_scores_capped(&s, &normalized, Some(cap))?;
        self.rescore(&s, &normalized, approximate, top_k, candidates)
    }
    /// Whether the base plus mutable overlay covers the current generation.
    pub fn hnsw_ready(&self) -> bool {
        let s = self.state.read().unwrap();
        s.fde_ann.as_ref().is_some_and(|ann| ann.generation == s.generation)
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
        self.query_auto_with_backend(vectors, top_k, candidates, ef_search).map(|(hits, _)| hits)
    }

    /// Return the actual backend used under the same snapshot as the results.
    pub fn query_auto_with_backend(
        &self, vectors: &[Vector], top_k: usize, candidates: Option<usize>, ef_search: usize,
    ) -> Result<(Vec<Hit>, &'static str), IndexError> {
        self.validate(vectors)?;
        if top_k == 0 || ef_search == 0 {
            return Err(IndexError::Invalid("top_k and ef_search must be positive".into()));
        }
        let normalized: Vec<_> = vectors.iter().map(|v| normalize(v)).collect();
        let count = candidates.unwrap_or(top_k.saturating_mul(8)).max(top_k);
        let s = self.state.read().unwrap();
        let (approximate, backend) = if s.fde_ann.as_ref().is_some_and(|ann| ann.generation == s.generation) {
            (self.ann_fde_scores(&s, &self.fde.encode_query(&normalized), count, ef_search)?, "hnsw")
        } else {
            (self.exact_fde_scores_capped(&s, &normalized, Some(count))?, "muvera")
        };
        Ok((self.rescore(&s, &normalized, approximate, top_k, candidates)?, backend))
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
        let s = self.state.read().unwrap();
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
        let s = self.state.read().unwrap();
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
        let s = self.state.read().unwrap();
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
        if s.documents.is_empty() {
            return Ok(Vec::new());
        }
        let query_fde = self.fde.encode_query(normalized);
        let mapped_fdes = self.fde_store.map()?;
        let fde_dimension = self.fde.output_dimension();
        let approximate_results: Result<Vec<_>, io::Error> = s
            .documents
            .par_iter()
            .map(|(id, record)| {
                Ok((
                    id.clone(),
                    dot(
                        &query_fde,
                        FixedVectorStore::get(&mapped_fdes, record.fde_location, fde_dimension)?,
                    ),
                ))
            })
            .collect();
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
        if m == 0 || ef_construct == 0 {
            return Err(IndexError::Invalid(
                "HNSW m and ef_construct must be positive".into(),
            ));
        }
        let s = self.state.read().unwrap();
        let dimension = self.fde.output_dimension();
        let mapped = if s.documents.is_empty() {
            None
        } else {
            Some(self.fde_store.map()?)
        };
        let mut ids: Vec<_> = s.documents.keys().cloned().collect();
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
                    let vector = FixedVectorStore::get(
                        mapped.as_deref().unwrap(),
                        s.documents[id].fde_location,
                        dimension,
                    )?
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
        let mut s = self.state.write().unwrap();
        if s.generation != built_generation {
            return Err(IndexError::Invalid(format!(
                "index generation moved during HNSW build ({} -> {}); retry",
                built_generation, s.generation,
            )));
        }
        let count = ids.len();
        let by_id = ids.iter().enumerate().map(|(idx, id)| (id.clone(), idx as u64)).collect();
        s.fde_ann = Some(FdeAnn {
            base: Arc::new(FdeAnnBase { index: hnsw, ids, by_id }),
            delta: HashSet::new(),
            tombstones: HashSet::new(),
            generation: built_generation,
        });
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
        let s = self.state.read().unwrap();
        let count = candidates.unwrap_or(top_k.saturating_mul(8)).max(top_k);
        let approximate = self.ann_fde_scores(&s, &query_fde, count, ef_search)?;
        self.rescore(&s, &normalized, approximate, top_k, candidates)
    }
    fn ann_fde_scores(
        &self, s: &State, query_fde: &Vector, count: usize, ef_search: usize,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        let ann = s.fde_ann.as_ref().ok_or_else(|| {
            IndexError::Invalid("FDE ANN is not built; call /v1/fde/index".into())
        })?;
        if ann.generation != s.generation {
            return Err(IndexError::Invalid("FDE ANN generation is stale; rebuild".into()));
        }
        let mut scores = Vec::new();
        if ann.base.ids.len() > ann.tombstones.len() {
            // Account for masked base hits before asking the graph for candidates.
            let base_count = count.saturating_add(ann.tombstones.len()).min(ann.base.ids.len());
            let options = SearchRuntimeOptions {
                ef_search: Some(ef_search.max(base_count)),
                ..SearchRuntimeOptions::default()
            };
            let points = ann.base.index.search_with_options(query_fde, base_count, &options)
                .map_err(|error| IndexError::Invalid(format!("HNSW search failed: {error}")))?;
            for point in points {
                if ann.tombstones.contains(&point.id) { continue; }
                let id = ann.base.ids.get(point.id as usize)
                    .ok_or_else(|| IndexError::Invalid("invalid HNSW point id".into()))?;
                if !s.documents.contains_key(id) || ann.delta.contains(id) {
                    return Err(IndexError::Invalid("FDE ANN overlay is inconsistent; rebuild".into()));
                }
                scores.push((id.clone(), -point.sort_key));
            }
        }
        if !ann.delta.is_empty() {
            let mapped = self.fde_store.map()?;
            for id in &ann.delta {
                let record = s.documents.get(id).ok_or_else(|| IndexError::Invalid("missing delta document".into()))?;
                let vector = FixedVectorStore::get(&mapped, record.fde_location, self.fde.output_dimension())?;
                scores.push((id.clone(), dot(query_fde, vector)));
            }
        }
        let by_score = |a: &(String, f32), b: &(String, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
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
        let s = self.state.read().unwrap();
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
        let s = self.state.read().unwrap();
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
        self.rescore(&s, &normalized, approx, top_k, candidates)
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
        let mapped = self.objects.map()?;
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
                            &mapped,
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
                        Ok((maxsim_flat(normalized, &scratch, dim), t1))
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
    pub fn stats(&self) -> IndexStats {
        let s = self.state.read().unwrap();
        IndexStats {
            documents: s.documents.len(),
            generation: s.generation,
            token_vectors: s.documents.values().map(|d| d.tokens).sum(),
            compressed_bytes: s.documents.values().map(|d| d.compressed_bytes).sum(),
            centroids: s.codebook.len(),
            residual_bits: self.config.residual_bits,
            trained: !s.codebook.is_empty(),
            fde_dimension: self.fde.output_dimension(),
            fde_ann_nodes: s.fde_ann.as_ref().map_or(0, |ann| ann.base.ids.len() - ann.tombstones.len() + ann.delta.len()),
            fde_ann_base_nodes: s.fde_ann.as_ref().map_or(0, |ann| ann.base.ids.len()),
            fde_ann_delta_documents: s.fde_ann.as_ref().map_or(0, |ann| ann.delta.len()),
            fde_ann_tombstones: s.fde_ann.as_ref().map_or(0, |ann| ann.tombstones.len()),
            fde_encoding_version: self.fde.encoding_version(),
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
        let s = self.state.read().unwrap();
        let record = s
            .documents
            .get(id)
            .ok_or_else(|| IndexError::Invalid(format!("unknown document: {id}")))?;
        let mapped = self.objects.map()?;
        let document = CompressedVectorStore::decode(
            &mapped,
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
    scored.into_iter().map(|(id, fde_score, _)| (id, fde_score)).collect()
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
