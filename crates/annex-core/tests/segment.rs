use annex::payload_storage::filters::Filter;
use annex::segment::segment::Segment;
use annex::utils::errors::DBError;
use annex::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;

const DIM: usize = 32;
const METRICS: [DistanceMetric; 3] = [
    DistanceMetric::Euclidean,
    DistanceMetric::Cosine,
    DistanceMetric::Dot,
];

fn vecf_dim(seed: usize, dim: usize) -> Vector {
    (0..dim).map(|d| ((seed + d) as f32).sin()).collect()
}

#[test]
fn insert_and_search_all_metrics() {
    for metric in METRICS {
        let mut segment = Segment::new(HNSWIndex::new(metric, 16, 50, 16, DIM));
        segment
            .hnsw_mut()
            .set_exact_fallback_enabled(metric == DistanceMetric::Dot);
        segment
            .hnsw_mut()
            .set_exact_fallback_threshold(if metric == DistanceMetric::Dot {
                1000
            } else {
                0
            });
        let vectors: Vec<_> = (0..256)
            .map(|i| {
                let vector = vecf_dim(i, DIM);
                let id = segment.insert(vector.clone(), None).unwrap();
                (id, vector)
            })
            .collect();
        for (expected_id, query) in vectors.iter().take(10) {
            let query: Vec<_> = query
                .iter()
                .enumerate()
                .map(|(i, x)| x + 0.001 * (i % 5) as f32)
                .collect();
            let results = segment.search(&query, 5).unwrap();
            assert!(!results.is_empty(), "metric={metric:?}");
            if metric == DistanceMetric::Dot {
                let dot =
                    |vector: &Vector| vector.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>();
                let best = vectors
                    .iter()
                    .map(|(_, v)| dot(v))
                    .max_by(f32::total_cmp)
                    .unwrap();
                let top = vectors.iter().find(|(id, _)| *id == results[0].id).unwrap();
                assert!(
                    dot(&top.1) + 1e-3 >= best,
                    "top={:?}, expected_score={best}",
                    results[0]
                );
            } else {
                assert!(
                    results.iter().any(|r| r.id == *expected_id),
                    "metric={metric:?}, expected={expected_id}"
                );
            }
        }
    }
}

#[test]
fn filtered_queries_all_metrics() {
    for metric in METRICS {
        let hnsw = HNSWIndex::new(metric, 16, 50, 16, DIM);
        let mut segment = Segment::new(hnsw);

        for i in 0..512 {
            let mut payload = Payload::default();
            let animal = match i % 4 {
                0 => "dog",
                1 => "cat",
                2 => "bird",
                _ => "fish",
            };
            payload.set("animal", PayloadValue::Str(animal.to_string()));
            payload.set("age", PayloadValue::Int((i % 8 + 1) as i64));
            payload.set(
                "score",
                PayloadValue::Float((60.0 + (i % 40) as f64).into()),
            );

            let vec = vecf_dim(i, DIM);
            segment.insert(vec, Some(payload)).unwrap();
        }

        let filter = Filter::And(vec![
            Filter::Match {
                key: "animal".into(),
                value: PayloadValue::Str("dog".into()),
            },
            Filter::Compare {
                key: "age".into(),
                op: ScalarComparisonOp::Gte,
                value: PayloadValue::Int(5),
            },
            Filter::Compare {
                key: "score".into(),
                op: ScalarComparisonOp::Lt,
                value: PayloadValue::Float(90.0.into()),
            },
        ]);

        let query = vecf_dim(10_000, DIM);
        let results = segment
            .search_with_filter(&query, 15, Some(&filter))
            .unwrap();

        assert!(!results.is_empty(), "filter must have matching documents");
        for r in &results {
            let p = segment.get_payload(r.id).unwrap();
            assert_eq!(p.get("animal").unwrap(), &PayloadValue::Str("dog".into()));
            assert!(matches!(p.get("age").unwrap(), PayloadValue::Int(n) if *n >= 5));
            assert!(matches!(p.get("score").unwrap(), PayloadValue::Float(f) if *f < 90.0.into()));
        }
    }
}

#[test]
fn test_list_filters_with_larger_pool_all_metrics() {
    for metric in METRICS {
        let hnsw = HNSWIndex::new(metric, 16, 50, 16, DIM);
        let mut segment = Segment::new(hnsw);

        for i in 0..512 {
            let mut payload = Payload::default();
            let tags = if i % 2 == 0 {
                vec!["cheap".to_string(), "small".to_string()]
            } else {
                vec!["expensive".to_string(), "large".to_string()]
            };
            let active = i % 3 == 0;
            payload.set("tags", PayloadValue::ListStr(tags));
            payload.set("active", PayloadValue::Bool(active));
            let vec = vecf_dim(i, DIM);
            segment.insert(vec, Some(payload)).unwrap();
        }

        let filter = Filter::Compare {
            key: "tags".into(),
            op: ScalarComparisonOp::Eq,
            value: PayloadValue::Str("cheap".into()),
        };

        let query = vecf_dim(20_000, DIM);
        let results = segment
            .search_with_filter(&query, 10, Some(&filter))
            .unwrap();

        assert!(results.iter().all(|r| {
            let p = segment.get_payload(r.id).unwrap();
            match p.get("tags") {
                Some(PayloadValue::ListStr(tags)) => tags.contains(&"cheap".to_string()),
                _ => false,
            }
        }));
        assert!(results.len() >= 1);
    }
}

#[test]
fn test_deletion_and_purge_with_large_set_all_metrics() {
    for metric in METRICS {
        let hnsw = HNSWIndex::new(metric, 16, 50, 16, DIM);
        let mut segment = Segment::new(hnsw);

        let mut ids = Vec::new();
        for i in 0..512 {
            let mut payload = Payload::default();
            payload.set("idx", PayloadValue::Int(i as i64));
            let vec = vecf_dim(i, DIM);
            let id = segment.insert(vec, Some(payload)).unwrap();
            ids.push(id);
        }

        // Delete every 7th point.
        for i in (0..200).step_by(7) {
            segment.delete(ids[i]).unwrap();
        }

        let results = segment.search(&vecf_dim(30_000, DIM), 30).unwrap();
        for r in &results {
            assert!(!segment.is_deleted(r.id));
        }

        // Delete all points. (After the previous purge, some points may already be gone;
        // our delete method now returns Ok in that case.)
        for id in ids.iter() {
            segment.delete(*id).unwrap();
        }

        for id in &ids {
            assert!(segment.get_vector(*id).is_none(), "NOT purged: id = {}", id);
        }
    }
}

#[test]
fn test_insert_with_custom_ids_and_auto_ids() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, DIM);
    let mut segment = Segment::new(hnsw);

    let custom_id = 42;
    let vec_custom = vecf_dim(1, DIM);
    let returned_id = segment
        .insert_with_id(custom_id, vec_custom.clone(), None)
        .unwrap();
    assert_eq!(returned_id, custom_id);

    // Auto IDs should advance past the highest custom ID.
    let auto_id = segment.insert(vecf_dim(2, DIM), None).unwrap();
    assert!(
        auto_id > custom_id,
        "auto-generated ID should advance past custom IDs"
    );

    // Duplicate custom ID should error.
    let dup = segment.insert_with_id(custom_id, vecf_dim(3, DIM), None);
    assert!(matches!(dup, Err(DBError::DuplicatePointId(id)) if id == custom_id));

    // Search should return the custom ID for its vector.
    let res = segment.search(&vec_custom, 1).unwrap();
    assert_eq!(res[0].id, custom_id);
}
