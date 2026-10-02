/// A persistent, single-field vector index for direct embedding-space queries.
///
/// Simpler than the full multivector planner — suitable for training-loop
/// hard-negative mining, nearest-neighbour lookups, and any caller that wants
/// to query the embedding space without going through the retrieval planner.
///
/// # Persistence
///
/// `VectorIndex::build` writes to `<path>/index.ann` and `<path>/index.ids`
/// (atomic renames). `VectorIndex::open` loads them. The HNSW file format is
/// the same as `HNSWIndex::save_to_path` / `load_from_path`.
///
/// # Example
///
/// ```no_run
/// use annex::vector::index::VectorIndex;
///
/// let entries = vec![
///     ("doc-1".into(), vec![1.0_f32, 0.0, 0.0]),
///     ("doc-2".into(), vec![0.0, 1.0, 0.0]),
/// ];
/// VectorIndex::build("my_index", entries, 16, 200).unwrap();
///
/// let idx = VectorIndex::open("my_index").unwrap();
/// for (id, score) in idx.search(&[1.0, 0.0, 0.0], 5) {
///     println!("{id}: {score:.4}");
/// }
/// ```
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::hnsw::{HNSWIndex, SearchRuntimeOptions};
use crate::utils::errors::DBError;
use crate::utils::types::DistanceMetric;

/// Persistent vector index backed by HNSW. Open with [`VectorIndex::open`] or
/// create with [`VectorIndex::build`].
pub struct VectorIndex {
    inner: HNSWIndex,
    ids: Vec<String>,
    by_id: HashMap<String, u64>,
    path: PathBuf,
}

impl VectorIndex {
    /// Build a new index from `entries`, write it to `path`, and return it.
    ///
    /// `m` (graph connectivity) and `ef_construct` (build-time beam width)
    /// control quality/speed. Typical values: `m=16`, `ef_construct=200`.
    pub fn build(
        path: impl AsRef<Path>,
        entries: Vec<(String, Vec<f32>)>,
        m: usize,
        ef_construct: usize,
    ) -> Result<Self, DBError> {
        let path = path.as_ref().to_owned();
        let dim = entries.first().map_or(0, |(_, v)| v.len());
        let mut hnsw = HNSWIndex::new(DistanceMetric::Dot, m, ef_construct, 16, dim);
        let ids: Vec<String> = entries.iter().map(|(id, _)| id.clone()).collect();
        let indexed: Vec<(u64, Vec<f32>)> = entries
            .into_iter()
            .enumerate()
            .map(|(i, (_, v))| (i as u64, v))
            .collect();
        hnsw.par_insert_batch(&indexed)?;
        if !ids.is_empty() {
            hnsw.reorder_rcm();
        }
        let by_id = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i as u64))
            .collect();
        let index = Self {
            inner: hnsw,
            ids,
            by_id,
            path,
        };
        index.save()?;
        Ok(index)
    }

    /// Open a previously built index from `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DBError> {
        let path = path.as_ref().to_owned();
        let hnsw = HNSWIndex::load_from_path(Self::ann_path(&path))?;
        let bytes = fs::read(Self::ids_path(&path))?;
        let ids: Vec<String> = bincode::deserialize(&bytes)
            .map_err(|e| DBError::SerializationError(anyhow::anyhow!("id map: {e}")))?;
        let by_id = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i as u64))
            .collect();
        Ok(Self {
            inner: hnsw,
            ids,
            by_id,
            path,
        })
    }

    /// Write this index to disk, overwriting any existing files atomically.
    pub fn save(&self) -> Result<(), DBError> {
        fs::create_dir_all(&self.path)?;
        self.inner.save_to_path(Self::ann_path(&self.path))?;
        let tmp = Self::ids_path(&self.path).with_extension("ids.tmp");
        let bytes = bincode::serialize(&self.ids)
            .map_err(|e| DBError::SerializationError(anyhow::anyhow!(e)))?;
        fs::write(&tmp, &bytes)?;
        fs::rename(&tmp, Self::ids_path(&self.path))?;
        Ok(())
    }

    /// Search for the `top_k` nearest neighbours to `query`.
    ///
    /// Returns `(id, score)` pairs sorted by descending score (higher = closer).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        self.search_with_ef(query, top_k, top_k.max(64))
    }

    /// Like [`search`] but with an explicit `ef_search` beam width.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Vec<(String, f32)> {
        let opts = SearchRuntimeOptions {
            ef_search: Some(ef_search.max(top_k)),
            ..SearchRuntimeOptions::default()
        };
        let Ok(points) = self
            .inner
            .search_with_options(&query.to_vec(), top_k, &opts)
        else {
            return Vec::new();
        };
        points
            .into_iter()
            .filter_map(|p| {
                let id = self.ids.get(p.id as usize)?;
                Some((id.clone(), -p.sort_key))
            })
            .collect()
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Return the ID of the vector at internal index position `i`, if any.
    pub fn id_at(&self, i: usize) -> Option<&str> {
        self.ids.get(i).map(String::as_str)
    }

    /// Return the internal index position for a document ID, if present.
    pub fn position_of(&self, id: &str) -> Option<u64> {
        self.by_id.get(id).copied()
    }

    fn ann_path(root: &Path) -> PathBuf {
        root.join("index.ann")
    }

    fn ids_path(root: &Path) -> PathBuf {
        root.join("index.ids")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_search_persist_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let entries = vec![
            ("a".into(), vec![1.0_f32, 0.0, 0.0]),
            ("b".into(), vec![0.0, 1.0, 0.0]),
            ("c".into(), vec![0.0, 0.0, 1.0]),
            ("d".into(), vec![0.707, 0.707, 0.0]),
        ];
        let idx = VectorIndex::build(dir.path(), entries, 4, 50).unwrap();
        assert_eq!(idx.len(), 4);
        assert_eq!(idx.position_of("a"), Some(0));

        let results = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");

        // Reload and verify identical results.
        let idx2 = VectorIndex::open(dir.path()).unwrap();
        assert_eq!(idx2.len(), 4);
        let results2 = idx2.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results, results2, "results must be identical after reload");
    }
}
