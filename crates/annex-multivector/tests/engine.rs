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
            .query_with_fde_ann_and_pruning(
                &[vec![1., 0., 0.], vec![0., 1., 0.]],
                1,
                2,
                1,
                16,
            )
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
