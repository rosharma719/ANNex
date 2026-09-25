use multivector::{IndexConfig, MultiVectorIndex};
use serde_json::json;

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
    assert_eq!(
        exact_candidates
            .iter()
            .map(|candidate| &candidate.id)
            .collect::<std::collections::HashSet<_>>(),
        ann_candidates
            .iter()
            .map(|candidate| &candidate.id)
            .collect()
    );
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

use serde_json::Value;

/// Delete-after-FDE-HNSW correctness. Documents the fix from Block A of the
/// correctness pass: after building the FDE-over-HNSW index, deleting a doc
/// MUST invalidate the derived structure so subsequent ANN queries either
/// (a) fail loudly with a "stale" error, or (b) after rebuild, return
/// only currently-live docs. Never return the deleted doc through the
/// stale ANN graph.
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

    // Delete a doc. This MUST invalidate the derived HNSW.
    assert!(index.delete("a").unwrap());

    // Now the HNSW-backed path should refuse (stale) OR return non-deleted
    // results only. Refusal is the current behaviour — it must not return
    // the deleted doc.
    let after_delete = index.query_with_fde_ann(&[vec![1.0, 0.0, 0.0, 0.0]], 5, None, 32);
    match after_delete {
        Err(_) => { /* good — refused because generation moved */ }
        Ok(hits) => {
            assert!(
                !hits.iter().any(|h| h.id == "a"),
                "HNSW query after delete returned the deleted doc 'a' — stale ANN leak"
            );
        }
    }

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

// Auto backend picks HNSW after it's built + stays fresh, falls through to
// exact before build and after a mutation invalidates fde_ann_generation.
#[test]
fn query_auto_dispatches_to_hnsw_when_fresh_and_falls_back_otherwise() {
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

    // A subsequent upsert bumps `generation`, invalidating fde_ann. Auto
    // must silently fall back to exact — the query still succeeds.
    index
        .upsert("more", vec![vec![0., 1., 0.]], json!({}))
        .unwrap();
    assert!(!index.hnsw_ready());
    let hits = index.query_auto(&[vec![1., 0., 0.]], 1, None, 16).unwrap();
    assert_eq!(hits[0].id, "code");
}
