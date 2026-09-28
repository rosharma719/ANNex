//! Collection namespaces share the engine contract, not document IDs or indexes.
use crate::{Durability, IndexConfig, IndexError, MultiVectorIndex};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub struct Collections {
    root: PathBuf,
    durability: Durability,
    indexes: Mutex<BTreeMap<String, Arc<MultiVectorIndex>>>,
}
impl Collections {
    pub fn open(root: impl AsRef<Path>, durability: Durability) -> Result<Self, IndexError> {
        let root = root.as_ref().to_owned();
        fs::create_dir_all(&root)?;
        let mut indexes = BTreeMap::new();
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| IndexError::Invalid("invalid collection name".into()))?;
            validate_name(&name)?;
            if entry.path().join("manifest.json").exists() {
                indexes.insert(
                    name,
                    Arc::new(MultiVectorIndex::open_existing(entry.path(), durability)?),
                );
            }
        }
        Ok(Self {
            root,
            durability,
            indexes: Mutex::new(indexes),
        })
    }
    pub fn create(
        &self,
        name: &str,
        config: IndexConfig,
    ) -> Result<Arc<MultiVectorIndex>, IndexError> {
        validate_name(name)?;
        let mut indexes = self.indexes.lock().unwrap();
        if indexes.contains_key(name) {
            return Err(IndexError::Invalid("collection already exists".into()));
        }
        let index = Arc::new(MultiVectorIndex::open_with_durability(
            self.root.join(name),
            config,
            self.durability,
        )?);
        index.initialize()?;
        indexes.insert(name.into(), Arc::clone(&index));
        Ok(index)
    }
    pub fn get(&self, name: &str) -> Option<Arc<MultiVectorIndex>> {
        self.indexes.lock().unwrap().get(name).cloned()
    }
    pub fn names(&self) -> Vec<String> {
        self.indexes.lock().unwrap().keys().cloned().collect()
    }
}
fn validate_name(name: &str) -> Result<(), IndexError> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
    {
        return Err(IndexError::Invalid(
            "collection names require 1..=64 ASCII letters, digits, '-' or '_'".into(),
        ));
    }
    Ok(())
}
