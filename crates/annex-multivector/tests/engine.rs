use multivector::{IndexConfig, MultiVectorIndex, UpsertDocument};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

#[test]
fn late_interaction_persists_and_ranks() {
    let directory = tempfile::tempdir().unwrap();
    let config = IndexConfig {
        dimension: 3,
        centroids: 2,
        residual_bits: 2,
        probes: 2,
        fde_repetitions: 8,
        fde_ksim: 2,
        fde_projected: 2,
    };
    let index = MultiVectorIndex::open(directory.path(), config.clone()).unwrap();
    index
        .train(
            &[
                vec![1., 0., 0.],
                vec![0., 1., 0.],
                vec![0., 0., 1.],
                vec![0., 0.1, 0.9],
            ],
            5,
        )
        .unwrap();
    index
        .upsert(
            "code",
            vec![vec![1., 0., 0.], vec![0., 1., 0.]],
            json!({"kind":"code"}),
        )
        .unwrap();
    index
        .upsert(
            "prose",
            vec![vec![0., 0., 1.], vec![0., 0.1, 0.9]],
            json!({"kind":"text"}),
        )
        .unwrap();
    let hits = index
        .query(&[vec![1., 0., 0.], vec![0., 1., 0.]], 2, None)
        .unwrap();
    assert_eq!(hits[0].id, "code");
    assert!(hits[0].score > hits[1].score);
    assert_eq!(
        index
            .query_with_centroid_pruning(&[vec![1., 0., 0.], vec![0., 1., 0.]], 1, 2, 1,)
            .unwrap()[0]
            .id,
        "code"
    );
    index.build_fde_ann(4, 16).unwrap();
    assert_eq!(index.stats().fde_ann_nodes, 2);
    let exact_candidates = index
        .exact_fde_candidates(&[vec![1., 0., 0.], vec![0., 1., 0.]], 2)
        .unwrap();
    let ann_candidates = index
        .ann_fde_candidates(&[vec![1., 0., 0.], vec![0., 1., 0.]], 2, 16)
        .unwrap();
    assert_eq!(exact_candidates[0].id, "code");
    assert!(!ann_candidates.is_empty());
    for candidate in &ann_candidates {
        let exact = exact_candidates
            .iter()
            .find(|exact| exact.id == candidate.id)
            .unwrap();
        assert_close(candidate.score, exact.score);
    }
    assert_eq!(
        index
            .query_with_fde_ann(&[vec![1., 0., 0.], vec![0., 1., 0.]], 1, Some(2), 16)
            .unwrap()[0]
            .id,
        "code"
    );
    assert_eq!(
        index
            .query_with_fde_ann_and_pruning(&[vec![1., 0., 0.], vec![0., 1., 0.]], 1, 2, 1, 16,)
            .unwrap()[0]
            .id,
        "code"
    );
    drop(index);
    let restored = MultiVectorIndex::open(directory.path(), config).unwrap();
    assert_eq!(
        restored.query(&[vec![0., 0., 1.]], 1, None).unwrap()[0].id,
        "prose"
    );
    assert_eq!(restored.stats().documents, 2);
}

#[test]
fn overwrites_deletes_and_validates() {
    let directory = tempfile::tempdir().unwrap();
    let index = MultiVectorIndex::open(directory.path(), IndexConfig::new(2)).unwrap();
    let samples: Vec<_> = (0..64)
        .map(|i| vec![i as f32 / 64., 1. - i as f32 / 64.])
        .collect();
    index.train(&samples, 2).unwrap();
    index.upsert("a", vec![vec![1., 0.]], Value::Null).unwrap();
    index
        .upsert("a", vec![vec![0., 1.]], json!({"version":2}))
        .unwrap();
    assert_eq!(index.stats().documents, 1);
    assert_eq!(
        index.query(&[vec![0., 1.]], 1, None).unwrap()[0].metadata["version"],
        2
    );
    assert!(index.delete("a").unwrap());
    assert!(!index.delete("a").unwrap());
    assert!(
        index
            .upsert("bad", vec![vec![1., 2., 3.]], Value::Null)
            .is_err()
    );
}

/// Deletes keep the ANN ready while excluding tombstoned base documents.
#[test]
fn delete_after_hnsw_never_returns_stale_docs() {
    let directory = tempfile::tempdir().unwrap();
    let mut config = IndexConfig::new(4);
    config.centroids = 4;
    config.fde_repetitions = 4;
    config.fde_ksim = 2;
    config.fde_projected = 2;
    let index = MultiVectorIndex::open(directory.path(), config).unwrap();

    let samples: Vec<_> = (0..32)
        .map(|i| {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0 + (i as f32) * 0.01;
            v
        })
        .collect();
    index.train(&samples, 3).unwrap();

    for id in ["a", "b", "c", "d", "e"] {
        let tok = match id {
            "a" => vec![1.0f32, 0.0, 0.0, 0.0],
            "b" => vec![0.0, 1.0, 0.0, 0.0],
            "c" => vec![0.0, 0.0, 1.0, 0.0],
            "d" => vec![0.0, 0.0, 0.0, 1.0],
            _ => vec![0.5, 0.5, 0.0, 0.0],
        };
        index.upsert(id, vec![tok], Value::Null).unwrap();
    }

    // Build the FDE-over-HNSW index at the current generation.
    index.build_fde_ann(8, 32).unwrap();

    // Baseline: HNSW-backed query returns something reasonable.
    let baseline = index
        .query_with_fde_ann(&[vec![1.0, 0.0, 0.0, 0.0]], 5, None, 32)
        .unwrap();
    assert!(baseline.iter().any(|h| h.id == "a"));

    // Delete a base document without rebuilding the graph.
    assert!(index.delete("a").unwrap());

    assert!(index.hnsw_ready());
    let after_delete = index
        .query_with_fde_ann(&[vec![1.0, 0.0, 0.0, 0.0]], 5, None, 32)
        .unwrap();
    assert!(after_delete.iter().all(|hit| hit.id != "a"));
    let candidates = index
        .ann_fde_candidates(&[vec![1.0, 0.0, 0.0, 0.0]], 5, 32)
        .unwrap();
    assert!(candidates.iter().all(|candidate| candidate.id != "a"));

    // After rebuild, HNSW query must succeed AND must not surface 'a'.
    index.build_fde_ann(8, 32).unwrap();
    let after_rebuild = index
        .query_with_fde_ann(&[vec![1.0, 0.0, 0.0, 0.0]], 5, None, 32)
        .unwrap();
    assert!(
        !after_rebuild.iter().any(|h| h.id == "a"),
        "post-rebuild HNSW query still returned deleted doc"
    );
}

/// Corrupt committed manifest envelopes must be rejected rather than
/// silently returning stale/torn state. Legacy sidecars are tested separately.
#[test]
fn manifest_checksum_rejects_corruption() {
    let directory = tempfile::tempdir().unwrap();
    let mut config = IndexConfig::new(2);
    config.centroids = 2;
    config.fde_repetitions = 2;
    config.fde_ksim = 2;
    config.fde_projected = 2;
    {
        let index = MultiVectorIndex::open(directory.path(), config.clone()).unwrap();
        index
            .train(
                &[vec![1., 0.], vec![0., 1.], vec![0.5, 0.5], vec![0.7, 0.3]],
                3,
            )
            .unwrap();
        index.upsert("x", vec![vec![1., 0.]], Value::Null).unwrap();
    }
    // Corrupt the manifest bytes (flip a byte in the middle).
    let manifest = directory.path().join("manifest.json");
    let mut bytes = std::fs::read(&manifest).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&manifest, &bytes).unwrap();

    let err = match MultiVectorIndex::open(directory.path(), config) {
        Ok(_) => panic!("open() accepted a corrupted manifest"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        multivector::IndexError::Invalid(_) | multivector::IndexError::Json(_)
    ));
}

// Auto uses exact before build and keeps ANN available across writes.
#[test]
fn query_auto_dispatches_to_hnsw_and_keeps_it_ready_after_writes() {
    let directory = tempfile::tempdir().unwrap();
    let config = IndexConfig {
        dimension: 3,
        centroids: 2,
        residual_bits: 2,
        probes: 2,
        fde_repetitions: 8,
        fde_ksim: 2,
        fde_projected: 2,
    };
    let index = MultiVectorIndex::open(directory.path(), config).unwrap();
    index
        .train(
            &[
                vec![1., 0., 0.],
                vec![0., 1., 0.],
                vec![0., 0., 1.],
                vec![0., 0.1, 0.9],
            ],
            5,
        )
        .unwrap();
    index
        .upsert("code", vec![vec![1., 0., 0.]], json!({}))
        .unwrap();
    index
        .upsert("prose", vec![vec![0., 0., 1.]], json!({}))
        .unwrap();

    assert!(!index.hnsw_ready());
    let hits = index.query_auto(&[vec![1., 0., 0.]], 1, None, 16).unwrap();
    assert_eq!(hits[0].id, "code");

    index.build_fde_ann(4, 16).unwrap();
    assert!(index.hnsw_ready());
    let hits = index.query_auto(&[vec![1., 0., 0.]], 1, None, 16).unwrap();
    assert_eq!(hits[0].id, "code");

    // A subsequent upsert is searchable through the exact delta overlay.
    index
        .upsert("more", vec![vec![0., 1., 0.]], json!({}))
        .unwrap();
    assert!(index.hnsw_ready());
    let hits = index.query_auto(&[vec![0., 1., 0.]], 1, None, 16).unwrap();
    assert_eq!(hits[0].id, "more");
}

fn mutable_ann_config() -> IndexConfig {
    IndexConfig {
        dimension: 3,
        centroids: 3,
        residual_bits: 2,
        probes: 3,
        fde_repetitions: 8,
        fde_ksim: 2,
        fde_projected: 2,
    }
}

fn mutable_ann_samples() -> Vec<Vec<f32>> {
    vec![
        vec![1., 0., 0.],
        vec![0.8, 0.2, 0.],
        vec![0., 1., 0.],
        vec![0., 0.8, 0.2],
        vec![0., 0., 1.],
        vec![0.2, 0., 0.8],
    ]
}

fn document(id: &str, vector: Vec<f32>, version: usize) -> UpsertDocument {
    UpsertDocument {
        id: id.into(),
        vectors: vec![vector],
        metadata: json!({"version": version}),
    }
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1e-4 * expected.abs().max(1.),
        "actual score {actual} differs from oracle {expected}"
    );
}

/// The base graph is approximate, so only require exhaustive coverage of the
/// exact delta. Every returned candidate must still identify the current
/// document and carry its current FDE and compressed-MaxSim scores.
fn assert_mutable_ann_matches_current_documents(
    index: &MultiVectorIndex,
    expected: &[(&str, usize)],
    required_delta_ids: &[&str],
) {
    assert!(index.hnsw_ready());
    assert_eq!(index.stats().fde_ann_nodes, expected.len());
    let query = [vec![1., 0., 0.], vec![0., 1., 0.]];
    let count = expected.len().max(1);
    let versions: HashMap<_, _> = expected.iter().copied().collect();
    let exact = index.exact_fde_candidates(&query, count).unwrap();
    assert_eq!(exact.len(), expected.len());
    let exact_scores: HashMap<_, _> = exact
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate.score))
        .collect();
    assert_eq!(
        exact_scores.keys().copied().collect::<HashSet<_>>(),
        versions.keys().copied().collect()
    );

    let candidates = index.ann_fde_candidates(&query, count, 32).unwrap();
    let candidate_ids: HashSet<_> = candidates.iter().map(|hit| hit.id.as_str()).collect();
    assert_eq!(candidate_ids.len(), candidates.len(), "duplicate ANN IDs");
    assert!(candidates.len() <= expected.len());
    for candidate in &candidates {
        assert_close(candidate.score, exact_scores[candidate.id.as_str()]);
    }
    for id in required_delta_ids {
        assert!(candidate_ids.contains(id), "missing delta document {id}");
    }
    for cap in [1, 2] {
        let capped = index.ann_fde_candidates(&query, cap, 32).unwrap();
        assert!(capped.len() <= cap);
        let ids: HashSet<_> = capped.iter().map(|hit| hit.id.as_str()).collect();
        assert_eq!(ids.len(), capped.len());
        for candidate in &capped {
            assert_close(candidate.score, exact_scores[candidate.id.as_str()]);
        }
        assert!(capped.windows(2).all(|pair| {
            pair[0].score > pair[1].score
                || (pair[0].score == pair[1].score && pair[0].id < pair[1].id)
        }));
    }

    let exact_hits = index.query(&query, count, Some(count)).unwrap();
    assert_eq!(exact_hits.len(), expected.len());
    let routes = [
        exact_hits,
        index
            .query_with_fde_ann(&query, count, Some(count), 32)
            .unwrap(),
        index.query_auto(&query, count, Some(count), 32).unwrap(),
        index
            .query_with_fde_ann_and_pruning(&query, count, count, count, 32)
            .unwrap(),
    ];
    for hits in routes {
        let ids: HashSet<_> = hits.iter().map(|hit| hit.id.as_str()).collect();
        assert_eq!(ids.len(), hits.len(), "duplicate rescored IDs");
        assert!(hits.len() <= expected.len());
        for id in required_delta_ids {
            assert!(ids.contains(id), "missing rescored delta document {id}");
        }
        for hit in hits {
            assert_eq!(hit.metadata["version"], versions[hit.id.as_str()]);
            assert_close(hit.score, index.score_compressed(&query, &hit.id).unwrap());
            assert_close(hit.fde_score.unwrap(), exact_scores[hit.id.as_str()]);
        }
    }
}

#[test]
fn mutable_ann_overwrites_and_duplicate_batches_use_last_committed_values() {
    let directory = tempfile::tempdir().unwrap();
    let index = MultiVectorIndex::open(directory.path(), mutable_ann_config()).unwrap();
    index.train(&mutable_ann_samples(), 3).unwrap();
    index
        .upsert_batch(vec![
            document("a", vec![1., 0., 0.], 0),
            document("b", vec![0., 1., 0.], 0),
            document("unchanged", vec![0., 0., 1.], 0),
        ])
        .unwrap();
    index.build_fde_ann(8, 32).unwrap();
    index
        .upsert_batch(vec![
            document("a", vec![0., 1., 0.], 1),
            document("new", vec![1., 0., 0.], 1),
            document("a", vec![0., 0., 1.], 2),
            document("new", vec![0., 1., 0.], 2),
        ])
        .unwrap();
    assert_mutable_ann_matches_current_documents(
        &index,
        &[("a", 2), ("b", 0), ("unchanged", 0), ("new", 2)],
        &["a", "new"],
    );
    assert_eq!(index.stats().fde_ann_base_nodes, 3);
    assert_eq!(index.stats().fde_ann_delta_documents, 2);
    assert_eq!(index.stats().fde_ann_tombstones, 1);

    // Replacing an existing delta entry must neither keep its old score nor
    // append a duplicate. Replacing another base entry adds one tombstone.
    index
        .upsert_batch(vec![
            document("a", vec![1., 0., 0.], 3),
            document("b", vec![0., 0., 1.], 3),
        ])
        .unwrap();
    assert_mutable_ann_matches_current_documents(
        &index,
        &[("a", 3), ("b", 3), ("unchanged", 0), ("new", 2)],
        &["a", "b", "new"],
    );
    assert_eq!(index.stats().fde_ann_base_nodes, 3);
    assert_eq!(index.stats().fde_ann_delta_documents, 3);
    assert_eq!(index.stats().fde_ann_tombstones, 2);

    // Validation must preserve both the documents and the derived overlay.
    let stats = index.stats();
    assert!(
        index
            .upsert_batch(vec![
                document("a", vec![0., 1., 0.], 4),
                document("invalid", vec![1., 0.], 4),
            ])
            .is_err()
    );
    index.upsert_batch(vec![]).unwrap();
    assert_eq!(index.stats(), stats);
    assert_mutable_ann_matches_current_documents(
        &index,
        &[("a", 3), ("b", 3), ("unchanged", 0), ("new", 2)],
        &["a", "b", "new"],
    );
}

#[test]
fn mutable_ann_delete_all_reinsert_rebuild_and_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let config = mutable_ann_config();
    let index = MultiVectorIndex::open(directory.path(), config.clone()).unwrap();
    index.train(&mutable_ann_samples(), 3).unwrap();
    index
        .upsert_batch(vec![
            document("a", vec![1., 0., 0.], 0),
            document("b", vec![0., 1., 0.], 0),
            document("c", vec![0., 0., 1.], 0),
        ])
        .unwrap();
    index.build_fde_ann(8, 32).unwrap();
    index
        .upsert_batch(vec![
            document("a", vec![0., 1., 0.], 1),
            document("delta_only", vec![1., 0., 0.], 1),
        ])
        .unwrap();
    assert!(index.delete("b").unwrap());
    assert!(index.delete("a").unwrap());
    assert!(index.delete("delta_only").unwrap());
    let stats = index.stats();
    assert!(!index.delete("missing").unwrap());
    assert_eq!(index.stats(), stats);
    assert_mutable_ann_matches_current_documents(&index, &[("c", 0)], &[]);

    assert!(index.delete("c").unwrap());
    assert_mutable_ann_matches_current_documents(&index, &[], &[]);
    assert_eq!(index.stats().fde_ann_base_nodes, 3);
    assert_eq!(index.stats().fde_ann_tombstones, 3);
    assert_eq!(index.stats().fde_ann_delta_documents, 0);

    // Both a tombstoned base ID and a deleted delta-only ID can be reused.
    index
        .upsert_batch(vec![
            document("b", vec![1., 0., 0.], 2),
            document("delta_only", vec![0., 1., 0.], 2),
        ])
        .unwrap();
    assert_mutable_ann_matches_current_documents(
        &index,
        &[("b", 2), ("delta_only", 2)],
        &["b", "delta_only"],
    );
    assert_eq!(index.build_fde_ann(8, 32).unwrap(), 2);
    assert_eq!(index.stats().fde_ann_base_nodes, 2);
    assert_eq!(index.stats().fde_ann_tombstones, 0);
    assert_eq!(index.stats().fde_ann_delta_documents, 0);
    assert_mutable_ann_matches_current_documents(&index, &[("b", 2), ("delta_only", 2)], &[]);

    // Writes after compaction begin a new overlay against the rebuilt base.
    index
        .upsert_batch(vec![document("b", vec![0., 0., 1.], 3)])
        .unwrap();
    assert!(index.delete("delta_only").unwrap());
    assert_mutable_ann_matches_current_documents(&index, &[("b", 3)], &["b"]);
    drop(index);

    let reopened = MultiVectorIndex::open(directory.path(), config).unwrap();
    assert!(!reopened.hnsw_ready());
    let query = [vec![1., 0., 0.]];
    assert!(reopened.ann_fde_candidates(&query, 1, 16).is_err());
    let exact = reopened.query(&query, 1, None).unwrap();
    let auto = reopened.query_auto(&query, 1, None, 16).unwrap();
    assert_eq!(auto.len(), 1);
    assert_eq!(auto[0].id, "b");
    assert_eq!(auto[0].metadata["version"], 3);
    assert_eq!(auto[0].score, exact[0].score);
    assert_eq!(reopened.build_fde_ann(8, 32).unwrap(), 1);
    assert_mutable_ann_matches_current_documents(&reopened, &[("b", 3)], &[]);

    // Building an empty base must still accept and search later delta writes.
    assert!(reopened.delete("b").unwrap());
    assert_eq!(reopened.build_fde_ann(8, 32).unwrap(), 0);
    assert_eq!(reopened.stats().fde_ann_base_nodes, 0);
    assert_eq!(reopened.stats().fde_ann_tombstones, 0);
    assert_mutable_ann_matches_current_documents(&reopened, &[], &[]);
    reopened
        .upsert_batch(vec![document("b", vec![1., 0., 0.], 4)])
        .unwrap();
    assert_mutable_ann_matches_current_documents(&reopened, &[("b", 4)], &["b"]);
}

#[test]
fn query_auto_observes_one_committed_batch_during_concurrent_writes() {
    use std::sync::{Barrier, mpsc};

    let directory = tempfile::tempdir().unwrap();
    let index = MultiVectorIndex::open(directory.path(), mutable_ann_config()).unwrap();
    index.train(&mutable_ann_samples(), 3).unwrap();
    let batch = |version: usize| {
        (0..4)
            .map(|i| {
                let mut vector = vec![0.; 3];
                vector[(i + version) % 3] = 1.;
                document(&format!("doc-{i}"), vector, version)
            })
            .collect()
    };
    let query = [vec![1., 0., 0.]];
    let mut expected_scores = HashMap::new();
    // Save scalar compressed scores for each vector/ID before introducing
    // concurrency; looking them up from the mutable index would itself race.
    for version in 0..3 {
        index.upsert_batch(batch(version)).unwrap();
        for i in 0..4 {
            let id = format!("doc-{i}");
            expected_scores.insert(
                (version, id.clone()),
                index.score_compressed(&query, &id).unwrap(),
            );
        }
    }
    index.upsert_batch(batch(0)).unwrap();
    index.build_fde_ann(8, 32).unwrap();
    let start = Barrier::new(2);
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let done_tx = done_tx;
            start.wait();
            for version in 1..=32 {
                index.upsert_batch(batch(version)).unwrap();
                assert!(index.hnsw_ready());
                if version % 8 == 0 {
                    index.build_fde_ann(8, 32).unwrap();
                }
            }
            done_tx.send(()).unwrap();
        });
        start.wait();
        let mut reads = 0;
        loop {
            let hits = index.query_auto(&query, 4, Some(4), 32).unwrap();
            assert!(!hits.is_empty());
            let version = hits[0].metadata["version"].as_u64().unwrap() as usize;
            let mut ids = HashSet::new();
            for hit in hits {
                assert!(
                    ids.insert(hit.id.clone()),
                    "duplicate ID in a query snapshot"
                );
                assert_eq!(hit.metadata["version"], version, "mixed batch revisions");
                assert_close(hit.score, expected_scores[&(version % 3, hit.id)]);
            }
            reads += 1;
            if reads >= 64 {
                match done_rx.try_recv() {
                    Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }
            std::thread::yield_now();
        }
    });
    let expected: Vec<_> = (0..4).map(|i| (format!("doc-{i}"), 32)).collect();
    let expected: Vec<_> = expected
        .iter()
        .map(|(id, version)| (id.as_str(), *version))
        .collect();
    assert_mutable_ann_matches_current_documents(&index, &expected, &[]);
}

#[test]
fn query_auto_handles_concurrent_empty_index_ann_invalidation() {
    use std::sync::{Barrier, mpsc};

    let directory = tempfile::tempdir().unwrap();
    let index = MultiVectorIndex::open(directory.path(), mutable_ann_config()).unwrap();
    let samples = mutable_ann_samples();
    index.train(&samples, 1).unwrap();
    index.build_fde_ann(8, 32).unwrap();
    let start = Barrier::new(2);
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let done_tx = done_tx;
            start.wait();
            for _ in 0..32 {
                // Retraining an empty index intentionally invalidates ANN.
                // Auto must choose and execute its backend under one guard.
                index.train(&samples, 1).unwrap();
                index.build_fde_ann(8, 32).unwrap();
            }
            done_tx.send(()).unwrap();
        });
        start.wait();
        let mut reads = 0;
        loop {
            assert!(
                index
                    .query_auto(&[vec![1., 0., 0.]], 1, None, 16)
                    .unwrap()
                    .is_empty()
            );
            reads += 1;
            if reads >= 64 {
                match done_rx.try_recv() {
                    Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }
            std::thread::yield_now();
        }
    });
}
