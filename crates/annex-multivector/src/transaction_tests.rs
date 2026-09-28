use super::*;
use crate::storage::FAIL_COMMIT;
use serde_json::json;
use std::process::Command;

thread_local! {
    static ANN_BUILD_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

pub(super) fn before_ann_publish() {
    let hook = ANN_BUILD_HOOK.with(|hook| hook.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

fn config() -> IndexConfig {
    IndexConfig {
        dimension: 3,
        centroids: 3,
        residual_bits: 2,
        probes: 3,
        fde_repetitions: 2,
        fde_ksim: 2,
        fde_projected: 2,
    }
}
fn train(index: &MultiVectorIndex) {
    index
        .train(
            &[
                vec![1., 0., 0.],
                vec![0., 1., 0.],
                vec![0., 0., 1.],
                vec![0.8, 0.2, 0.],
                vec![0., 0.8, 0.2],
                vec![0.2, 0., 0.8],
            ],
            3,
        )
        .unwrap();
}
fn doc(id: &str, version: u64) -> UpsertDocument {
    UpsertDocument {
        id: id.into(),
        vectors: vec![vec![1., version as f32 + 1., 0.5]],
        metadata: json!(version),
    }
}
fn baseline(path: &Path) -> MultiVectorIndex {
    let index = MultiVectorIndex::open(path, config()).unwrap();
    train(&index);
    index.upsert_batch(vec![doc("a", 0), doc("b", 0)]).unwrap();
    index
}
fn versions(index: &MultiVectorIndex) -> Vec<(String, Value)> {
    let mut hits: Vec<_> = index
        .query(&[vec![1., 0., 0.]], 100, Some(100))
        .unwrap()
        .into_iter()
        .map(|h| (h.id, h.metadata))
        .collect();
    hits.sort_by(|a, b| a.0.cmp(&b.0));
    hits
}

#[test]
fn every_precommit_failure_preserves_memory_disk_and_ann() {
    for stage in [
        "object_partial_write",
        "fde_partial_write",
        "object_appended",
        "fde_appended",
        "manifest_partial_write",
        "objects_synced",
        "fde_synced",
        "manifest_written",
        "manifest_synced",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let index = baseline(dir.path());
        index.build_fde_ann(4, 16).unwrap();
        let before = index.stats();
        let manifest = fs::read(dir.path().join("manifest.json")).unwrap();
        let boundaries = (
            index.snapshot().stores.objects.len().unwrap(),
            index.snapshot().stores.fde.len().unwrap(),
        );
        // Append faults happen on the second document: the first was fully staged.
        FAIL_COMMIT.with(|f| {
            f.set(Some((
                stage,
                if stage.ends_with("appended") { 2 } else { 1 },
            )))
        });
        assert!(
            index
                .upsert_batch(vec![doc("a", 1), doc("new", 1)])
                .is_err(),
            "{stage}"
        );
        assert_eq!(index.stats(), before, "{stage}");
        assert_eq!(
            versions(&index),
            vec![("a".into(), json!(0)), ("b".into(), json!(0))]
        );
        assert_eq!(
            fs::read(dir.path().join("manifest.json")).unwrap(),
            manifest
        );
        assert_eq!(
            index
                .ann_fde_candidates(&[vec![1., 0., 0.]], 10, 16)
                .unwrap()
                .len(),
            2
        );
        drop(index);
        let restored = MultiVectorIndex::open(dir.path(), config()).unwrap();
        assert_eq!(restored.stats().generation, before.generation);
        assert_eq!(
            (
                restored.snapshot().stores.objects.len().unwrap(),
                restored.snapshot().stores.fde.len().unwrap()
            ),
            boundaries
        );
        restored
            .upsert_batch(vec![doc("a", 2), doc("new", 2)])
            .unwrap();
        assert_eq!(restored.stats().generation, before.generation + 1);
    }
}

#[test]
fn failed_train_and_delete_do_not_change_live_state() {
    let dir = tempfile::tempdir().unwrap();
    let index = MultiVectorIndex::open(dir.path(), config()).unwrap();
    FAIL_COMMIT.with(|f| f.set(Some(("manifest_written", 1))));
    assert!(
        index
            .train(&[vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.]], 1)
            .is_err()
    );
    assert!(!index.stats().trained);
    assert_eq!(index.stats().generation, 0);
    train(&index);
    index.upsert_batch(vec![doc("a", 0)]).unwrap();
    index.build_fde_ann(4, 16).unwrap();
    let before = index.stats();
    FAIL_COMMIT.with(|f| f.set(Some(("manifest_synced", 1))));
    assert!(index.delete("a").is_err());
    assert_eq!(index.stats(), before);
    assert_eq!(versions(&index), vec![("a".into(), json!(0))]);
    assert!(!index.delete("missing").unwrap());
    index.upsert_batch(Vec::new()).unwrap();
    assert_eq!(index.stats(), before);
}

#[test]
fn postrename_error_publishes_generation_and_reports_uncertainty() {
    for stage in ["manifest_renamed", "directory_synced"] {
        let dir = tempfile::tempdir().unwrap();
        let index = baseline(dir.path());
        index.build_fde_ann(4, 16).unwrap();
        let before = index.stats().generation;
        FAIL_COMMIT.with(|f| f.set(Some((stage, 1))));
        assert!(matches!(
            index.upsert_batch(vec![doc("a", 1), doc("c", 1)]),
            Err(IndexError::CommitUncertain(_))
        ));
        assert_eq!(index.stats().generation, before + 1);
        assert!(index.hnsw_ready());
        assert_eq!(index.stats().fde_ann_delta_documents, 2);
        assert_eq!(index.stats().fde_ann_tombstones, 1);
        let expected = versions(&index);
        let mut ann_versions: Vec<_> = index
            .query_auto(&[vec![1., 0., 0.]], 10, Some(10), 16)
            .unwrap()
            .into_iter()
            .map(|hit| (hit.id, hit.metadata))
            .collect();
        ann_versions.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(ann_versions, expected);
        drop(index);
        let restored = MultiVectorIndex::open(dir.path(), config()).unwrap();
        assert_eq!(versions(&restored), expected);
        assert_eq!(restored.stats().generation, before + 1);
    }
}

#[test]
fn postrename_delete_updates_the_ann_overlay() {
    for stage in ["manifest_renamed", "directory_synced"] {
        let dir = tempfile::tempdir().unwrap();
        let index = baseline(dir.path());
        index.build_fde_ann(4, 16).unwrap();
        // Cover deletion of a base document and an overwritten delta document.
        index.upsert_batch(vec![doc("b", 1)]).unwrap();
        for id in ["a", "b"] {
            FAIL_COMMIT.with(|fault| fault.set(Some((stage, 1))));
            assert!(matches!(
                index.delete(id),
                Err(IndexError::CommitUncertain(_))
            ));
            assert!(index.hnsw_ready());
            let hits = index
                .query_auto(&[vec![1., 0., 0.]], 10, Some(10), 16)
                .unwrap();
            assert_eq!(hits.len(), index.stats().documents);
            assert!(hits.iter().all(|hit| hit.id != id));
        }
        drop(index);
        let restored = MultiVectorIndex::open(dir.path(), config()).unwrap();
        assert!(versions(&restored).is_empty());
    }
}

#[test]
fn failed_or_racing_ann_build_preserves_the_current_overlay() {
    let dir = tempfile::tempdir().unwrap();
    let index = Arc::new(baseline(dir.path()));
    index.build_fde_ann(4, 16).unwrap();
    let base = Arc::clone(&index.state.read().unwrap().fde_ann.as_ref().unwrap().base);
    FAIL_COMMIT.with(|fault| fault.set(Some(("ann_built_before_publish", 1))));
    assert!(index.build_fde_ann(4, 16).is_err());
    let writer = index.clone();
    ANN_BUILD_HOOK.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move || {
            writer.delete("a").unwrap();
            writer.upsert_batch(vec![doc("b", 1), doc("c", 1)]).unwrap();
        }))
    });
    assert!(
        matches!(index.build_fde_ann(4, 16), Err(IndexError::Invalid(message))
        if message.contains("generation moved"))
    );
    assert!(index.hnsw_ready());
    assert!(Arc::ptr_eq(
        &base,
        &index.state.read().unwrap().fde_ann.as_ref().unwrap().base
    ));
    let hits = index
        .query_auto(&[vec![1., 0., 0.]], 10, Some(10), 16)
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert!(
        hits.iter()
            .all(|hit| hit.id != "a" && hit.metadata == json!(1))
    );
    index.build_fde_ann(4, 16).unwrap();
    assert_eq!(index.stats().fde_ann_delta_documents, 0);
    assert_eq!(index.stats().fde_ann_tombstones, 0);
}

#[test]
fn fde_encoding_version_survives_legacy_reopen_and_future_writes() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = MultiVectorIndex::open(dir.path(), config()).unwrap();
    assert_eq!(index.stats().fde_encoding_version, 2);
    index.fde = FdeEncoder::with_version(3, 2, 2, 2, 0x4d55_5645_5241, 1);
    train(&index);
    let tokens = vec![vec![1., 2., 0.5]; 3];
    index.upsert("legacy", tokens.clone(), json!(0)).unwrap();
    let query = vec![vec![1., 0., 0.]];
    let expected = index.exact_fde_candidates(&query, 1).unwrap()[0].score;
    let normalized: Vec<_> = tokens.iter().map(|v| normalize(v)).collect();
    let legacy_bytes = index.fde.encode_document(&normalized);
    let current = FdeEncoder::new(3, 2, 2, 2, 0x4d55_5645_5241);
    assert_ne!(legacy_bytes, current.encode_document(&normalized));
    drop(index);
    // Real pre-versioned manifests had no FDE version field.
    let path = dir.path().join("manifest.json");
    let mut envelope: ManifestEnvelope = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let mut payload: Value = serde_json::from_str(&envelope.manifest).unwrap();
    payload
        .as_object_mut()
        .unwrap()
        .remove("fde_encoding_version");
    envelope.manifest = serde_json::to_string(&payload).unwrap();
    envelope.checksum_blake3 = blake3::hash(envelope.manifest.as_bytes())
        .to_hex()
        .to_string();
    fs::write(&path, serde_json::to_vec(&envelope).unwrap()).unwrap();

    let restored = MultiVectorIndex::open(dir.path(), config()).unwrap();
    assert_eq!(restored.stats().fde_encoding_version, 1);
    restored.upsert("new", tokens, json!(1)).unwrap();
    let hits = restored.exact_fde_candidates(&query, 2).unwrap();
    assert!(hits.iter().all(|hit| hit.score == expected));
    let stored = restored.snapshot().stores.fde.map().unwrap();
    let state = restored.state.read().unwrap();
    for record in state.documents.values() {
        assert_eq!(
            FixedVectorStore::get(&stored, record.fde_location, legacy_bytes.len()).unwrap(),
            legacy_bytes
        );
    }
    drop(state);
    drop(stored);
    drop(restored);
    let restored = MultiVectorIndex::open(dir.path(), config()).unwrap();
    assert_eq!(restored.stats().fde_encoding_version, 1);
    assert_eq!(restored.exact_fde_candidates(&query, 2).unwrap(), hits);
}

#[test]
fn ann_candidates_preserve_raw_inner_product_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let index = baseline(dir.path());
    // Valid FDE records with different norms: q.a=2 > q.b=1, but
    // cosine(q,a)=1/sqrt(2) < cosine(q,b)=1. Rescoring only the cosine
    // winner cannot recover the lost raw-inner-product winner.
    let mut a = vec![0.; index.fde.output_dimension()];
    a[0] = 2.;
    a[1] = 2.;
    let mut b = vec![0.; a.len()];
    b[0] = 1.;
    {
        let _writer = index.writer.lock().unwrap();
        let state = index.snapshot();
        let mut next = (*state).clone();
        next.documents.get_mut("a").unwrap().fde_location =
            index.snapshot().stores.fde.put(&a).unwrap();
        next.documents.get_mut("b").unwrap().fde_location =
            index.snapshot().stores.fde.put(&b).unwrap();
        index.commit(&state, next).unwrap();
    }
    index.build_fde_ann(4, 16).unwrap();
    let state = index.state.read().unwrap();
    let hits = index.ann_fde_scores(&state, &b, 1, 16).unwrap();
    assert_eq!(hits, vec![("a".into(), 2.)]);
}

// Invoked only as a subprocess by crash_at_every_commit_boundary. The environment
// switch is compiled into this unit-test binary, never the server/library build.
#[test]
fn crash_child() {
    let Ok(path) = std::env::var("ANNEX_TEST_CRASH_ROOT") else {
        return;
    };
    let index = MultiVectorIndex::open(path, config()).unwrap();
    index
        .upsert_batch(vec![doc("a", 1), doc("b", 1), doc("c", 1)])
        .unwrap();
    panic!("child failed to reach requested crash boundary");
}

#[test]
fn crash_at_every_commit_boundary_recovers_one_whole_generation() {
    for (stage, published) in [
        ("object_partial_write", false),
        ("fde_partial_write", false),
        ("object_appended", false),
        ("fde_appended", false),
        ("manifest_partial_write", false),
        ("objects_synced", false),
        ("fde_synced", false),
        ("manifest_written", false),
        ("manifest_synced", false),
        ("manifest_renamed", true),
        ("directory_synced", true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let index = baseline(dir.path());
        let generation = index.stats().generation;
        drop(index);
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "engine::transaction_tests::crash_child",
                "--nocapture",
            ])
            .env("ANNEX_TEST_CRASH_ROOT", dir.path())
            .env("ANNEX_TEST_CRASH_STAGE", stage)
            .output()
            .unwrap();
        assert_eq!(
            child.status.code(),
            Some(86),
            "{stage}: {}",
            String::from_utf8_lossy(&child.stderr)
        );
        let restored = MultiVectorIndex::open(dir.path(), config()).unwrap();
        let expected = if published {
            vec![
                ("a".into(), json!(1)),
                ("b".into(), json!(1)),
                ("c".into(), json!(1)),
            ]
        } else {
            vec![("a".into(), json!(0)), ("b".into(), json!(0))]
        };
        assert_eq!(versions(&restored), expected, "{stage}");
        assert_eq!(
            restored.stats().generation,
            generation + u64::from(published)
        );
        // The recovered append position must remain usable.
        restored.upsert_batch(vec![doc("after", 2)]).unwrap();
        drop(restored);
        assert_eq!(
            MultiVectorIndex::open(dir.path(), config())
                .unwrap()
                .stats()
                .documents,
            expected.len() + 1
        );
    }
}

#[test]
fn version_one_manifest_migrates_with_legacy_checksum() {
    for sidecar in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let index = baseline(dir.path());
        let expected = versions(&index);
        drop(index);
        let path = dir.path().join("manifest.json");
        let envelope: ManifestEnvelope = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let mut old: Value = serde_json::from_str(&envelope.manifest).unwrap();
        old.as_object_mut().unwrap().remove("format_version");
        old.as_object_mut().unwrap().remove("segments");
        for (_, doc) in old["documents"].as_object_mut().unwrap() {
            doc["location"].as_object_mut().unwrap().remove("checksum");
            doc["fde_location"]
                .as_object_mut()
                .unwrap()
                .remove("checksum");
        }
        let bytes = serde_json::to_vec(&old).unwrap();
        fs::write(&path, &bytes).unwrap();
        if sidecar {
            fs::write(
                dir.path().join("manifest.sha256"),
                blake3::hash(&bytes).to_hex().as_bytes(),
            )
            .unwrap();
        }
        let restored = MultiVectorIndex::open(dir.path(), config()).unwrap();
        assert_eq!(versions(&restored), expected);
        restored.upsert_batch(vec![doc("c", 1)]).unwrap();
        drop(restored);
        // Obsolete sidecars cannot invalidate the newly self-contained manifest.
        assert_eq!(
            MultiVectorIndex::open(dir.path(), config())
                .unwrap()
                .stats()
                .documents,
            3
        );
    }
}

#[test]
fn exclusive_directory_lock_and_buffered_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let index =
        MultiVectorIndex::open_with_durability(dir.path(), config(), Durability::Buffered).unwrap();
    assert!(MultiVectorIndex::open(dir.path(), config()).is_err());
    train(&index);
    index.upsert_batch(vec![doc("a", 0)]).unwrap();
    // Models a descriptor inherited by an unrelated subprocess before exec.
    let inherited = index._directory_lock.0.try_clone().unwrap();
    drop(index);
    assert_eq!(
        MultiVectorIndex::open(dir.path(), config())
            .unwrap()
            .stats()
            .documents,
        1
    );
    assert!(inherited.metadata().is_ok());
}

#[test]
fn checksums_truncation_versions_and_hostile_counts_are_rejected() {
    for target in ["manifest.json", "objects/vectors.plaid", "fde/fde.bin"] {
        for truncate in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            drop(baseline(dir.path()));
            let path = dir.path().join(target);
            let mut bytes = fs::read(&path).unwrap();
            if truncate {
                bytes.truncate(bytes.len() / 2);
            } else {
                let n = bytes.len();
                bytes[n / 2] ^= 0x40;
            }
            fs::write(path, bytes).unwrap();
            assert!(
                MultiVectorIndex::open(dir.path(), config()).is_err(),
                "{target}, truncate={truncate}"
            );
        }
    }
    for case in 0..6 {
        let dir = tempfile::tempdir().unwrap();
        drop(baseline(dir.path()));
        let path = dir.path().join("manifest.json");
        let mut envelope: ManifestEnvelope =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let mut payload: Value = serde_json::from_str(&envelope.manifest).unwrap();
        match case {
            0 => payload["format_version"] = json!(999),
            1 => payload["documents"]["a"]["location"]["offset"] = json!(u64::MAX),
            2 => payload["documents"]["a"]["centroid_ids"] = json!([u32::MAX]),
            3 => payload["codebook"][0] = json!([1.]),
            5 => payload["fde_encoding_version"] = json!(999),
            _ => payload["documents"]["a"]["fde_location"]
                .as_object_mut()
                .unwrap()
                .remove("checksum")
                .map(|_| ())
                .unwrap(),
        }
        envelope.manifest = serde_json::to_string(&payload).unwrap();
        envelope.checksum_blake3 = blake3::hash(envelope.manifest.as_bytes())
            .to_hex()
            .to_string();
        fs::write(path, serde_json::to_vec(&envelope).unwrap()).unwrap();
        assert!(
            MultiVectorIndex::open(dir.path(), config()).is_err(),
            "case={case}"
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let mut bad = config();
    bad.fde_repetitions = usize::MAX;
    assert!(MultiVectorIndex::open(dir.path(), bad).is_err());
    let mut bad = config();
    bad.dimension = usize::MAX;
    assert!(MultiVectorIndex::open(dir.path(), bad).is_err());
}

#[test]
fn concurrent_queries_observe_whole_batches_and_survive_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let index = Arc::new(baseline(dir.path()));
    std::thread::scope(|scope| {
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let b = barrier.clone();
        let writer = index.clone();
        scope.spawn(move || {
            b.wait();
            for version in 1..40 {
                writer
                    .upsert_batch(vec![doc("a", version), doc("b", version)])
                    .unwrap();
            }
        });
        for _ in 0..3 {
            let b = barrier.clone();
            let index = index.clone();
            scope.spawn(move || {
                b.wait();
                for _ in 0..60 {
                    let hits = index.query(&[vec![1., 0., 0.]], 2, Some(2)).unwrap();
                    assert_eq!(hits.len(), 2);
                    assert_eq!(hits[0].metadata, hits[1].metadata);
                    let hits = index
                        .query_with_centroid_pruning(&[vec![1., 0., 0.]], 2, 2, 2)
                        .unwrap();
                    assert_eq!(hits.len(), 2);
                    assert_eq!(hits[0].metadata, hits[1].metadata);
                }
            });
        }
    });
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..40 {
                index.delete("a").unwrap();
                index.upsert_batch(vec![doc("a", 99)]).unwrap();
            }
        });
        scope.spawn(|| {
            for _ in 0..80 {
                assert!(
                    index
                        .query(&[vec![1., 0., 0.]], 10, Some(10))
                        .unwrap()
                        .iter()
                        .all(|h| h.score.is_finite())
                );
            }
        });
    });
}

// An intentionally boring in-memory reference: normalize and quantize one scalar
// at a time, retain reconstructed tokens, then exhaustively score every live doc.
// It neither reads engine postings/candidates nor decodes the engine's disk bytes.
fn reference_compress(vectors: &[Vector], centers: &[Vector], residuals: &[f32]) -> Vec<Vector> {
    vectors
        .iter()
        .map(|v| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let v: Vec<_> = v
                .iter()
                .map(|x| if norm == 0. { 0. } else { x / norm })
                .collect();
            let center = centers
                .iter()
                // Coarse assignment uses squared L2, including its first-center
                // tie rule. Dot-product equivalence is not exact in FP32.
                .min_by(|a, b| {
                    let distance = |c: &&Vector| {
                        v.iter()
                            .zip(c.iter())
                            .map(|(x, y)| (x - y).powi(2))
                            .sum::<f32>()
                    };
                    distance(a).total_cmp(&distance(b))
                })
                .unwrap();
            v.iter()
                .zip(center)
                .map(|(&x, &c)| {
                    let r = residuals
                        .iter()
                        .min_by(|a, b| (x - c - **a).abs().total_cmp(&(x - c - **b).abs()))
                        .unwrap();
                    c + r
                })
                .collect()
        })
        .collect()
}
fn reference_score(query: &[Vector], document: &[Vector]) -> f32 {
    query
        .iter()
        .map(|q| {
            let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            document
                .iter()
                .map(|d| q.iter().zip(d).map(|(x, y)| x / norm * y).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[test]
fn randomized_mutation_restart_state_machine_matches_scalar_oracle() {
    let seeds = std::env::var("ANNEX_ORACLE_SEEDS")
        .ok()
        .map(|v| v.parse::<u64>().expect("oracle seed count"))
        .unwrap_or(8);
    let steps = std::env::var("ANNEX_ORACLE_STEPS")
        .ok()
        .map(|v| v.parse::<usize>().expect("oracle step count"))
        .unwrap_or(160);
    assert!(seeds > 0 && steps > 0);
    for seed in 1..=seeds {
        let dir = tempfile::tempdir().unwrap();
        let mut index = MultiVectorIndex::open(dir.path(), config()).unwrap();
        train(&index);
        let (centers, residuals) = {
            let s = index.state.read().unwrap();
            (s.codebook.clone(), s.residual_codebook.clone())
        };
        let mut model: HashMap<String, (Vec<Vector>, Value)> = HashMap::new();
        let mut generation = 1;
        let mut rng = seed;
        let mut random = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for step in 0..steps {
            let id = format!("doc-{}", random() % 12);
            match random() % 7 {
                0..=2 => {
                    let mut batch = Vec::new();
                    for _ in 0..1 + random() % 4 {
                        let id = format!("doc-{}", random() % 12);
                        let vectors = vec![
                            (0..3).map(|_| (1 + random() % 31) as f32).collect(),
                            (0..3).map(|_| (1 + random() % 31) as f32).collect(),
                        ];
                        let metadata = json!({"step":step,"seed":seed});
                        model.insert(
                            id.clone(),
                            (
                                reference_compress(&vectors, &centers, &residuals),
                                metadata.clone(),
                            ),
                        );
                        batch.push(UpsertDocument {
                            id,
                            vectors,
                            metadata,
                        });
                    }
                    index.upsert_batch(batch).unwrap();
                    generation += 1;
                }
                3 => {
                    let existed = model.remove(&id).is_some();
                    assert_eq!(index.delete(&id).unwrap(), existed);
                    generation += u64::from(existed);
                }
                4 => {
                    drop(index);
                    index = MultiVectorIndex::open(dir.path(), config()).unwrap();
                }
                5 => {
                    let before = index.stats();
                    assert!(
                        index
                            .upsert_batch(vec![
                                doc("valid", 0),
                                UpsertDocument {
                                    id,
                                    vectors: vec![vec![f32::NAN; 3]],
                                    metadata: Value::Null
                                }
                            ])
                            .is_err()
                    );
                    assert_eq!(index.stats(), before);
                }
                _ => {
                    index.build_fde_ann(4, 32).unwrap();
                }
            }
            assert_eq!(
                index.stats().generation,
                generation,
                "seed={seed} step={step}"
            );
            assert_eq!(index.stats().documents, model.len());
            let query = vec![(0..3).map(|_| (1 + random() % 17) as f32).collect()];
            let actual = index.query(&query, 20, Some(20)).unwrap();
            assert_eq!(actual.len(), model.len());
            let ids: HashSet<_> = actual.iter().map(|h| h.id.clone()).collect();
            assert_eq!(ids, model.keys().cloned().collect());
            for hit in &actual {
                let (vectors, metadata) = &model[&hit.id];
                let expected = reference_score(&query, vectors);
                assert!(
                    (hit.score - expected).abs() < 1e-5,
                    "seed={seed} step={step} id={} score={} expected={expected}",
                    hit.id,
                    hit.score
                );
                assert_eq!(&hit.metadata, metadata);
            }
            for pair in actual.windows(2) {
                assert!(pair[0].score >= pair[1].score);
            }
            if index.stats().fde_ann_nodes > 0 {
                let ann = index.query_with_fde_ann(&query, 20, Some(20), 64).unwrap();
                // ANN recall is approximate even with a broad search budget.
                // Correctness requires live IDs, unique results, current metadata,
                // and exact rescoring; recall belongs in the quality benchmark.
                assert!(!ann.is_empty());
                assert_eq!(
                    ann.iter().map(|h| &h.id).collect::<HashSet<_>>().len(),
                    ann.len()
                );
                for hit in ann {
                    let (vectors, metadata) = model.get(&hit.id).expect("ANN returned a stale ID");
                    assert_eq!(&hit.metadata, metadata);
                    assert!(
                        (hit.score - reference_score(&query, vectors)).abs() < 1e-5,
                        "ANN score: seed={seed} step={step}"
                    );
                }
            }
        }
    }
}

#[test]
fn failed_partial_append_can_retry_without_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let index = baseline(dir.path());
    for stage in ["object_partial_write", "fde_partial_write"] {
        FAIL_COMMIT.with(|f| f.set(Some((stage, 1))));
        assert!(index.upsert_batch(vec![doc("new", 1)]).is_err());
        index.upsert_batch(vec![doc("new", 2)]).unwrap();
        assert_eq!(versions(&index).last().unwrap().1, json!(2));
    }
    let expected = versions(&index);
    drop(index);
    assert_eq!(
        versions(&MultiVectorIndex::open(dir.path(), config()).unwrap()),
        expected
    );
}

#[test]
fn generation_never_wraps_and_duplicate_batch_ids_are_last_wins() {
    let dir = tempfile::tempdir().unwrap();
    let index = baseline(dir.path());
    let generation = index.stats().generation;
    index.upsert_batch(vec![doc("a", 1), doc("a", 2)]).unwrap();
    assert_eq!(index.stats().generation, generation + 1);
    assert_eq!(versions(&index)[0].1, json!(2));
    Arc::make_mut(&mut index.state.write().unwrap()).generation = u64::MAX;
    assert!(index.upsert_batch(vec![doc("new", 3)]).is_err());
    assert_eq!(index.stats().generation, u64::MAX);
    assert_eq!(index.stats().documents, 2);
}

#[test]
fn finite_extreme_vectors_do_not_normalize_to_nan_or_infinity() {
    let v = normalize(&[f32::MAX, f32::MAX, 0.]);
    assert!(v.iter().all(|x| x.is_finite()));
    assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.).abs() < 1e-6);
}

#[test]
fn missing_manifest_does_not_destroy_existing_segments() {
    let dir = tempfile::tempdir().unwrap();
    drop(baseline(dir.path()));
    let object_path = dir.path().join("objects/vectors.plaid");
    let before = fs::read(&object_path).unwrap();
    fs::remove_file(dir.path().join("manifest.json")).unwrap();
    assert!(MultiVectorIndex::open(dir.path(), config()).is_err());
    assert_eq!(fs::read(object_path).unwrap(), before);
}

#[test]
fn sealing_after_failed_append_preserves_committed_map_boundaries() {
    for stage in [
        "object_partial_write",
        "fde_partial_write",
        "manifest_written",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let index = baseline(directory.path());
        index.build_fde_ann(4, 16).unwrap();
        let before = index.snapshot();
        let expected = versions(&index);
        let expected_fde = index.exact_fde_candidates(&[vec![1., 0., 0.]], 10).unwrap();
        FAIL_COMMIT.with(|fault| fault.set(Some((stage, 1))));
        assert!(
            index
                .upsert_batch(vec![doc("a", 1), doc("new", 1)])
                .is_err(),
            "{stage}"
        );
        index.seal().unwrap();
        let sealed = index.snapshot();
        let segment = &sealed.sealed[&before.stores.id.unwrap_or(0)];
        assert!(
            segment.bounds.objects as usize <= segment.objects.as_deref().map_or(0, |m| m.len()),
            "object map shorter than sealed boundary after {stage}"
        );
        assert!(
            segment.bounds.fde as usize <= segment.fde.as_deref().map_or(0, |m| m.len()),
            "FDE map shorter than sealed boundary after {stage}"
        );
        assert_eq!(versions(&index), expected, "{stage}");
        assert_eq!(
            index.exact_fde_candidates(&[vec![1., 0., 0.]], 10).unwrap(),
            expected_fde
        );
        assert!(index.hnsw_ready());
        let envelope: ManifestEnvelope =
            serde_json::from_slice(&fs::read(directory.path().join("manifest.json")).unwrap())
                .unwrap();
        let manifest: Manifest = serde_json::from_str(&envelope.manifest).unwrap();
        let persisted = &manifest
            .sealed
            .iter()
            .find(|(id, _)| *id == before.stores.id.unwrap_or(0))
            .unwrap()
            .1;
        assert_eq!(persisted.objects, segment.bounds.objects);
        assert_eq!(persisted.fde, segment.bounds.fde);
        drop(sealed);
        drop(before);
        drop(index);
        let restored = MultiVectorIndex::open(directory.path(), config()).unwrap();
        assert_eq!(versions(&restored), expected);
        restored.upsert_batch(vec![doc("after-seal", 2)]).unwrap();
        assert_eq!(restored.stats().documents, 3);
    }
}

#[test]
fn sealed_active_legacy_segment_alias_is_rejected_before_recovery() {
    let directory = tempfile::tempdir().unwrap();
    drop(baseline(directory.path()));
    let object_path = directory.path().join("objects/vectors.plaid");
    let fde_path = directory.path().join("fde/fde.bin");
    let objects = fs::read(&object_path).unwrap();
    let fdes = fs::read(&fde_path).unwrap();
    let path = directory.path().join("manifest.json");
    let mut envelope: ManifestEnvelope = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let mut manifest: Manifest = serde_json::from_str(&envelope.manifest).unwrap();
    assert!(manifest.storage_generation.is_none());
    manifest
        .sealed
        .push((0, SegmentBoundaries { objects: 0, fde: 0 }));
    envelope.manifest = serde_json::to_string(&manifest).unwrap();
    envelope.checksum_blake3 = blake3::hash(envelope.manifest.as_bytes())
        .to_hex()
        .to_string();
    fs::write(path, serde_json::to_vec(&envelope).unwrap()).unwrap();
    assert!(MultiVectorIndex::open(directory.path(), config()).is_err());
    assert_eq!(
        fs::read(&object_path).unwrap(),
        objects,
        "invalid manifest truncated committed objects"
    );
    assert_eq!(
        fs::read(&fde_path).unwrap(),
        fdes,
        "invalid manifest truncated committed FDEs"
    );
}

#[test]
fn compaction_aborts_when_a_writer_publishes_after_its_snapshot() {
    use std::time::{Duration, Instant};

    let directory = tempfile::tempdir().unwrap();
    let index = baseline(directory.path());
    index.build_fde_ann(4, 16).unwrap();
    // Holding the writer mutex makes publication order deterministic while the
    // compactor is free to snapshot and copy. Creating its target directory
    // proves it has already captured the previous document generation.
    let writer = index.writer.lock().unwrap();
    std::thread::scope(|scope| {
        let compaction = scope.spawn(|| index.compact());
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let copying = fs::read_dir(directory.path().join("segments"))
                .is_ok_and(|mut entries| entries.next().is_some());
            if copying {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "compaction did not begin copying outside the writer mutex"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let current = index.snapshot();
        let mut next = (*current).clone();
        // Publish a metadata-only document update through the same commit
        // protocol used by normal writers, while we already hold their mutex.
        next.documents.get_mut("a").unwrap().metadata = json!(7);
        index.commit(&current, next).unwrap();
        drop(writer);
        let result = compaction.join().unwrap();
        assert!(
            matches!(result, Err(IndexError::Invalid(message)) if message.contains("generation changed during compaction"))
        );
    });
    assert_eq!(
        versions(&index),
        vec![("a".into(), json!(7)), ("b".into(), json!(0))]
    );
    assert!(index.hnsw_ready());
    assert!(
        fs::read_dir(directory.path().join("segments"))
            .unwrap()
            .next()
            .is_none(),
        "aborted compaction leaked its target segment"
    );
    drop(index);
    assert_eq!(
        versions(&MultiVectorIndex::open(directory.path(), config()).unwrap()),
        vec![("a".into(), json!(7)), ("b".into(), json!(0))]
    );
}

#[test]
fn segment_maintenance_crash_child() {
    let Ok(path) = std::env::var("ANNEX_TEST_MAINTENANCE_ROOT") else {
        return;
    };
    let index = MultiVectorIndex::open(path, config()).unwrap();
    match std::env::var("ANNEX_TEST_MAINTENANCE_KIND")
        .unwrap()
        .as_str()
    {
        "compact" => {
            index.compact().unwrap();
        }
        "seal" => index.seal().unwrap(),
        kind => panic!("unexpected maintenance kind {kind}"),
    }
    panic!("child failed to reach requested maintenance crash boundary");
}

#[test]
fn segment_maintenance_crashes_recover_one_complete_generation() {
    for kind in ["seal", "compact"] {
        for (stage, published) in [
            ("compact_partial_write", false),
            ("compaction_copied", false),
            ("objects_synced", false),
            ("fde_synced", false),
            ("manifest_partial_write", false),
            ("manifest_written", false),
            ("manifest_synced", false),
            ("manifest_renamed", true),
            ("directory_synced", true),
        ] {
            if kind == "seal" && matches!(stage, "compact_partial_write" | "compaction_copied") {
                continue;
            }
            let directory = tempfile::tempdir().unwrap();
            let index = baseline(directory.path());
            index.seal().unwrap();
            index.upsert_batch(vec![doc("a", 1), doc("c", 1)]).unwrap();
            let before = index.stats();
            let expected = versions(&index);
            let query = [vec![1., 0., 0.]];
            let fde = index.exact_fde_candidates(&query, 10).unwrap();
            let scores: Vec<_> = expected
                .iter()
                .map(|(id, _)| (id.clone(), index.score_compressed(&query, id).unwrap()))
                .collect();
            drop(index);
            let child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "engine::transaction_tests::segment_maintenance_crash_child",
                    "--nocapture",
                ])
                .env("ANNEX_TEST_MAINTENANCE_ROOT", directory.path())
                .env("ANNEX_TEST_MAINTENANCE_KIND", kind)
                .env("ANNEX_TEST_CRASH_STAGE", stage)
                .output()
                .unwrap();
            assert_eq!(
                child.status.code(),
                Some(86),
                "{kind}/{stage}: {}",
                String::from_utf8_lossy(&child.stderr)
            );
            let restored = MultiVectorIndex::open(directory.path(), config()).unwrap();
            assert_eq!(
                restored.stats().generation,
                before.generation + u64::from(published),
                "{kind}/{stage}"
            );
            assert_eq!(versions(&restored), expected, "{kind}/{stage}");
            assert_eq!(
                restored.exact_fde_candidates(&query, 10).unwrap(),
                fde,
                "{kind}/{stage}"
            );
            for (id, score) in &scores {
                assert_eq!(
                    restored.score_compressed(&query, id).unwrap(),
                    *score,
                    "{kind}/{stage}/{id}"
                );
            }
            let expected_segments = if !published {
                before.storage_segments
            } else if kind == "compact" {
                1
            } else {
                before.storage_segments + 1
            };
            assert_eq!(
                restored.stats().storage_segments,
                expected_segments,
                "{kind}/{stage}"
            );
            // Both the recovered append target and another compaction must be usable.
            restored
                .upsert_batch(vec![doc("after-maintenance", 3)])
                .unwrap();
            restored.compact().unwrap();
            drop(restored);
            let reopened = MultiVectorIndex::open(directory.path(), config()).unwrap();
            assert_eq!(reopened.stats().documents, expected.len() + 1);
            assert_eq!(reopened.stats().storage_segments, 1);
        }
    }
}

#[test]
fn compaction_record_copy_rejects_corrupt_or_unchecked_sources_before_appending() {
    let directory = tempfile::tempdir().unwrap();
    let index = baseline(directory.path());
    let state = index.snapshot();
    let record = &state.documents["a"];
    let target = tempfile::tempdir().unwrap();
    let objects = CompressedVectorStore::new(target.path().join("objects")).unwrap();
    let fdes = FixedVectorStore::new(target.path().join("fde")).unwrap();
    for (source, location, object_record) in [
        (state.record_objects(record), record.location, true),
        (state.record_fde(record), record.fde_location, false),
    ] {
        let copy = |source: &[u8], location| {
            if object_record {
                objects.copy_record(source, location)
            } else {
                fdes.copy_record(source, location)
            }
        };
        let length = || {
            if object_record {
                objects.len().unwrap()
            } else {
                fdes.len().unwrap()
            }
        };
        // Copy owned test bytes so this never mutates a live memory mapping.
        let mut corrupt = source.to_vec();
        corrupt[location.offset as usize + location.length as usize - 1] ^= 1;
        let error = copy(&corrupt, location).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(length(), 0, "corrupt bytes were appended before validation");
        let mut unchecked = location;
        unchecked.checksum = None;
        assert_eq!(
            copy(source, unchecked).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            length(),
            0,
            "unchecked bytes were appended before validation"
        );
        let copied = copy(source, location).unwrap();
        assert_eq!(copied.checksum, location.checksum);
        assert_eq!(copied.length, location.length);
        assert_eq!(length(), location.length);
    }
}
