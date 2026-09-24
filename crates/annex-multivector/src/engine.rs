use crate::{
    fde::{Vector, dot, maxsim_flat, normalize},
    muvera::FdeEncoder,
    storage::{CompressedVectorStore, FixedVectorStore, ObjectLocation, atomic_write},
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
    fs, io,
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
const FORMAT_VERSION: u32 = 1;

#[derive(Deserialize, Serialize)]
struct Manifest {
    #[serde(default = "current_format_version")]
    format_version: u32,
    #[serde(default)]
    generation: u64,
    config: IndexConfig,
    codebook: Vec<Vector>,
    residual_codebook: Vec<f32>,
    documents: HashMap<String, DocumentRecord>,
}

fn current_format_version() -> u32 {
    FORMAT_VERSION
}

/// Verifies the sidecar sha256 against the manifest bytes we just read.
/// The sidecar is optional (older indexes were written without one), so we
/// only enforce when the file exists — new writes always produce it, so the
/// enforcement window strengthens over time.
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
    pub token_vectors: usize,
    pub compressed_bytes: u64,
    pub centroids: usize,
    pub residual_bits: u8,
    pub trained: bool,
    pub fde_dimension: usize,
    pub fde_ann_nodes: usize,
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
}
struct State {
    /// Monotonic mutation counter. Bumped by any code path that changes
    /// documents, codebook, or residual codebook. Persisted in the
    /// manifest so a restart resumes with the right value. Any
    /// derived structure (fde_ann_generation) records the value at
    /// build time and refuses to be trusted after `generation` moves.
    generation: u64,
    codebook: Vec<Vector>,
    residual_codebook: Vec<f32>,
    documents: HashMap<String, DocumentRecord>,
    postings: Vec<HashSet<String>>,
    fde_ann: Option<Arc<HNSWIndex>>,
    fde_ann_ids: Vec<String>,
    /// Which `generation` the current `fde_ann` was built at. Query paths
    /// that dispatch to the HNSW backend refuse if this doesn't match
    /// `generation` — protects against a caller who forgot to rebuild
    /// after a batch of writes.
    fde_ann_generation: u64,
}
pub struct MultiVectorIndex {
    root: PathBuf,
    config: IndexConfig,
    objects: CompressedVectorStore,
    fde: FdeEncoder,
    fde_store: FixedVectorStore,
    state: RwLock<State>,
}

impl MultiVectorIndex {
    pub fn open(path: impl AsRef<Path>, config: IndexConfig) -> Result<Self, IndexError> {
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
        let root = path.as_ref().to_owned();
        fs::create_dir_all(&root)?;
        let path = root.join("manifest.json");
        let (generation, codebook, residual_codebook, documents) = if path.exists() {
            let bytes = fs::read(&path)?;
            verify_manifest_checksum(&bytes, &root.join("manifest.sha256"))?;
            let m: Manifest = serde_json::from_slice(&bytes)?;
            if m.format_version != FORMAT_VERSION {
                return Err(IndexError::Invalid(format!(
                    "manifest format_version={} unsupported (this build expects {})",
                    m.format_version, FORMAT_VERSION,
                )));
            }
            if m.config != config {
                return Err(IndexError::Config {
                    actual: m.config,
                    requested: config,
                });
            }
            (m.generation, m.codebook, m.residual_codebook, m.documents)
        } else {
            (0, vec![], vec![], HashMap::new())
        };
        let mut postings = vec![HashSet::new(); codebook.len()];
        for (id, d) in &documents {
            for &c in &d.centroid_ids {
                postings[c as usize].insert(id.clone());
            }
        }
        Ok(Self {
            fde: FdeEncoder::new(
                config.dimension,
                config.fde_ksim,
                config.fde_projected,
                config.fde_repetitions,
                0x4d55_5645_5241,
            ),
            fde_store: FixedVectorStore::new(root.join("fde"))?,
            objects: CompressedVectorStore::new(root.join("objects"))?,
            state: RwLock::new(State {
                generation,
                codebook,
                residual_codebook,
                documents,
                postings,
                fde_ann: None,
                fde_ann_ids: Vec::new(),
                fde_ann_generation: 0,
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
        let bytes = serde_json::to_vec(&Manifest {
            format_version: FORMAT_VERSION,
            generation: s.generation,
            config: self.config.clone(),
            codebook: s.codebook.clone(),
            residual_codebook: s.residual_codebook.clone(),
            documents: s.documents.clone(),
        })?;
        // Sidecar sha256 file so torn / partial reads or bit-flipped storage
        // are caught on next open() rather than silently propagating stale
        // documents into query results. atomic_write below already ensures
        // the primary file is either the old or the new version (never
        // half of each); the checksum guards against corruption at rest.
        let sum = blake3::hash(&bytes);
        atomic_write(&self.root.join("manifest.sha256"), sum.to_hex().as_bytes())?;
        atomic_write(&self.root.join("manifest.json"), &bytes)?;
        Ok(())
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
        s.codebook = centers;
        let residuals: Vec<f32> = samples
            .iter()
            .flat_map(|vector| {
                let center = &s.codebook[nearest(vector, &s.codebook)];
                vector
                    .iter()
                    .zip(center)
                    .map(|(value, centroid)| value - centroid)
                    .collect::<Vec<_>>()
            })
            .collect();
        s.residual_codebook =
            train_scalar_codebook(&residuals, 1usize << self.config.residual_bits, 12);
        s.postings = vec![HashSet::new(); s.codebook.len()];
        s.generation = s.generation.wrapping_add(1);
        self.persist(&s)
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
    /// Ingest a batch while writing the persistent manifest only once.
    pub fn upsert_batch(&self, batch: Vec<UpsertDocument>) -> Result<(), IndexError> {
        for document in &batch {
            self.validate(&document.vectors)?;
        }
        let mut s = self.state.write().unwrap();
        s.fde_ann = None;
        s.fde_ann_ids.clear();
        if s.codebook.is_empty() {
            return Err(IndexError::Invalid(
                "index is untrained; call train first".into(),
            ));
        }
        for document in batch {
            let id = document.id;
            if let Some(old_ids) = s.documents.get(&id).map(|old| old.unique_centroids.clone()) {
                for c in old_ids {
                    s.postings[c as usize].remove(&id);
                }
            }
            let vectors: Vec<_> = document
                .vectors
                .iter()
                .map(|vector| normalize(vector))
                .collect();
            let ids: Vec<u32> = vectors
                .iter()
                .map(|v| nearest(v, &s.codebook) as u32)
                .collect();
            let mut unique_centroids = ids.clone();
            unique_centroids.sort_unstable();
            unique_centroids.dedup();
            let (location, size) = self.objects.put(
                &vectors,
                &ids,
                &s.codebook,
                &s.residual_codebook,
                self.config.residual_bits,
            )?;
            let fde = self.fde.encode_document(&vectors);
            let fde_location = self.fde_store.put(&fde)?;
            for &c in &unique_centroids {
                s.postings[c as usize].insert(id.clone());
            }
            s.documents.insert(
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
        s.generation = s.generation.wrapping_add(1);
        self.persist(&s)
    }
    pub fn delete(&self, id: &str) -> Result<bool, IndexError> {
        let mut s = self.state.write().unwrap();
        // Any prior FDE-over-HNSW index still contains this doc's node.
        // Without invalidation, query_with_fde_ann would keep returning
        // the deleted doc through the stale ANN graph. Match upsert_batch
        // and drop the derived structure — callers rebuild on next need.
        s.fde_ann = None;
        s.fde_ann_ids.clear();
        s.fde_ann_generation = 0;
        if let Some(d) = s.documents.remove(id) {
            for c in d.unique_centroids {
                s.postings[c as usize].remove(id);
            }
            s.generation = s.generation.wrapping_add(1);
            self.persist(&s)?;
            Ok(true)
        } else {
            Ok(false)
        }
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
        let approximate = self.exact_fde_scores_capped(&normalized, Some(cap))?;
        let s = self.state.read().unwrap();
        self.rescore(&s, &normalized, approximate, top_k, candidates)
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
        let approximate = self.exact_fde_scores_capped(&normalized, Some(candidates))?;
        let s = self.state.read().unwrap();
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
        self.exact_fde_scores_capped(normalized, None)
    }

    /// FDE exhaustive scan, optionally returning only the top `cap` results.
    /// When capped, partitions the top-cap with `select_nth_unstable_by`
    /// (O(n) expected) then fully sorts only the surviving prefix
    /// (O(cap log cap)), instead of paying O(n log n) to sort a 10K+
    /// candidate pool whose tail is discarded by rescore anyway.
    fn exact_fde_scores_capped(
        &self,
        normalized: &[Vector],
        cap: Option<usize>,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        let query_fde = self.fde.encode_query(normalized);
        let s = self.state.read().unwrap();
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
        let mapped = self.fde_store.map()?;
        let mut ids: Vec<_> = s.documents.keys().cloned().collect();
        ids.sort();
        let hnsw = HNSWIndex::new(DistanceMetric::Dot, m, ef_construct, 16, dimension);
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
                    let vector =
                        FixedVectorStore::get(&mapped, s.documents[id].fde_location, dimension)?
                            .to_vec();
                    Ok::<_, IndexError>(((base + offset) as u64, vector))
                })
                .collect::<Result<Vec<_>, _>>()?;
            hnsw.par_insert_batch(&entries).map_err(|error| {
                IndexError::Invalid(format!("HNSW par_insert_batch failed: {error}"))
            })?;
        }
        let built_generation = s.generation;
        drop(s);
        let mut s = self.state.write().unwrap();
        // Race check: if a writer bumped `generation` after we started
        // building, the HNSW is already stale. Discard the work and let
        // the caller retry rather than serve stale results silently.
        if s.generation != built_generation {
            return Err(IndexError::Invalid(format!(
                "index generation moved during HNSW build ({} -> {}); retry",
                built_generation, s.generation,
            )));
        }
        s.fde_ann = Some(Arc::new(hnsw));
        s.fde_ann_ids = ids;
        s.fde_ann_generation = built_generation;
        Ok(s.fde_ann_ids.len())
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
        &self,
        s: &State,
        query_fde: &Vector,
        count: usize,
        ef_search: usize,
    ) -> Result<Vec<(String, f32)>, IndexError> {
        let ann = s.fde_ann.as_ref().ok_or_else(|| {
            IndexError::Invalid("FDE ANN is not built; call /v1/fde/index".into())
        })?;
        // Generation guard: any mutation since build() invalidated fde_ann,
        // so this check should only fail if some code path forgot to clear
        // it. Making the check explicit here means we fail loudly instead
        // of returning stale / deleted docs through the ANN path.
        if s.fde_ann_generation != s.generation {
            return Err(IndexError::Invalid(format!(
                "FDE ANN is stale: built at generation {}, current {}; rebuild",
                s.fde_ann_generation, s.generation,
            )));
        }
        let options = SearchRuntimeOptions {
            ef_search: Some(ef_search.max(count)),
            ..SearchRuntimeOptions::default()
        };
        let points = ann
            .search_with_options(query_fde, count, &options)
            .map_err(|error| IndexError::Invalid(format!("HNSW search failed: {error}")))?;
        points
            .into_iter()
            .map(|point| {
                let id = s
                    .fde_ann_ids
                    .get(point.id as usize)
                    .ok_or_else(|| IndexError::Invalid("invalid HNSW point id".into()))?;
                Ok((id.clone(), -point.sort_key))
            })
            .collect::<Result<Vec<_>, IndexError>>()
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
            token_vectors: s.documents.values().map(|d| d.tokens).sum(),
            compressed_bytes: s.documents.values().map(|d| d.compressed_bytes).sum(),
            centroids: s.codebook.len(),
            residual_bits: self.config.residual_bits,
            trained: !s.codebook.is_empty(),
            fde_dimension: self.fde.output_dimension(),
            fde_ann_nodes: s.fde_ann_ids.len(),
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
        .map(|(id, _)| {
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
            (id, score)
        })
        .collect();
    scored.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(survivors);
    scored
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
