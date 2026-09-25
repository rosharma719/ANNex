use super::*;
use crate::storage::FAIL_COMMIT;
use serde_json::json;
use std::process::Command;

/// Hook point for tests that need to inject failures between the ANN build
/// completing and its state being published under the write lock. No-op in
/// normal test runs; override at call site when needed.
pub(super) fn before_ann_publish() {}

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
        let boundaries = (index.objects.len().unwrap(), index.fde_store.len().unwrap());
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
                restored.objects.len().unwrap(),
                restored.fde_store.len().unwrap()
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
        let before = index.stats().generation;
        FAIL_COMMIT.with(|f| f.set(Some((stage, 1))));
        assert!(matches!(
            index.upsert_batch(vec![doc("a", 1), doc("c", 1)]),
            Err(IndexError::CommitUncertain(_))
        ));
        assert_eq!(index.stats().generation, before + 1);
        let expected = versions(&index);
        drop(index);
        let restored = MultiVectorIndex::open(dir.path(), config()).unwrap();
        assert_eq!(versions(&restored), expected);
        assert_eq!(restored.stats().generation, before + 1);
    }
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
    drop(index);
    assert_eq!(
        MultiVectorIndex::open(dir.path(), config())
            .unwrap()
            .stats()
            .documents,
        1
    );
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
    for case in 0..5 {
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
                .max_by(|a, b| {
                    let score =
                        |c: &&Vector| v.iter().zip(c.iter()).map(|(x, y)| x * y).sum::<f32>();
                    score(a).total_cmp(&score(b))
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
                assert!(
                    (hit.score - reference_score(&query, vectors)).abs() < 1e-5,
                    "seed={seed} step={step} id={}",
                    hit.id
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
    index.state.write().unwrap().generation = u64::MAX;
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
