use annex::payload_storage::filters::Filter;
use annex::segment::segment::Segment;
use annex::utils::errors::DBError;
use annex::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;
use serde::Serialize;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;

fn vecf_dim(seed: usize, dim: usize) -> Vector {
    // Deterministic high-dim vector generator for tests
    (0..dim).map(|d| ((seed + d) as f32).sin()).collect()
}

const DIM: usize = 1536;

#[derive(Serialize)]
struct DotDiagEntry {
    query_idx: usize,
    expected_id: u64,
    expected_dot: f32,
    top_id: u64,
    top_dot: f32,
}

#[test]
fn test_large_scale_insert_and_search_all_metrics() {
    use std::time::Instant;

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::Dot,
    ] {
        println!("\n\n===========================");
        println!("STARTING TEST FOR {:?}", metric);
        println!("===========================\n");

        let hnsw = HNSWIndex::new(metric, 16, 50, 16, DIM);
        let mut segment = Segment::new(hnsw);
        if metric == DistanceMetric::Dot {
            segment.hnsw_mut().set_exact_fallback_enabled(true);
            segment.hnsw_mut().set_exact_fallback_threshold(10_000);
        } else {
            segment.hnsw_mut().set_exact_fallback_enabled(false);
            segment.hnsw_mut().set_exact_fallback_threshold(0);
        }

        let mut ids = Vec::new();
        let mut vectors = Vec::new();

        println!("--- INSERTING VECTORS ---\n");
        let insert_start = Instant::now();
        for i in 0..1_000 {
            let vec = vecf_dim(i, DIM);
            let mut payload = Payload::default();
            payload.set("index", PayloadValue::Int(i as i64));

            let id = segment.insert(vec.clone(), Some(payload)).unwrap();
            ids.push(id);
            vectors.push((id, vec));

            if i % 1000 == 0 {
                println!("Inserted {} vectors... (+{:?})", i, insert_start.elapsed());
            }
        }
        let insert_elapsed = insert_start.elapsed();
        println!(
            "[{:?}] Inserted 1000 vectors in {:?} (~{:.3} ms/insert)",
            metric,
            insert_elapsed,
            insert_elapsed.as_secs_f64() * 1e3 / 1_000.0
        );

        println!("\n--- SEARCHING QUERIES ---\n");
        let search_start = Instant::now();
        let dot_diag_path = env::var("VECTORDB_DOT_DIAG_LOG").ok();
        for (query_idx, (expected_id, query)) in vectors.iter().take(10).enumerate() {
            let noisy_query: Vec<f32> = query
                .iter()
                .enumerate()
                .map(|(idx, x)| x + 0.001 * ((idx % 5) as f32))
                .collect();

            let now = Instant::now();
            let results = segment.search(&noisy_query, 5).unwrap();
            let duration = now.elapsed();

            println!(
                "[{:?}] Search complete in {:?}. Top result: ID {:?}",
                metric, duration, results[0].id
            );

            if metric == DistanceMetric::Dot {
                // For Dot, compute highest dot product manually.
                let mut best_id = None;
                let mut best_dot = f32::NEG_INFINITY;
                for (id, vec) in &vectors {
                    let dot: f32 = vec.iter().zip(&noisy_query).map(|(a, b)| a * b).sum();
                    if dot > best_dot {
                        best_dot = dot;
                        best_id = Some(*id);
                    }
                }
                let expected_dot_id = best_id.expect("At least one vector must exist");
                let top_id = results[0].id;
                let top_dot = vectors
                    .iter()
                    .find(|(id, _)| *id == top_id)
                    .map(|(_, vec)| vec.iter().zip(&noisy_query).map(|(a, b)| a * b).sum())
                    .unwrap_or(0.0);
                if let Some(path) = dot_diag_path.as_ref() {
                    let entry = DotDiagEntry {
                        query_idx,
                        expected_id: expected_dot_id,
                        expected_dot: best_dot,
                        top_id,
                        top_dot,
                    };
                    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                        let _ = serde_json::to_writer(&mut file, &entry);
                        let _ = file.write_all(b"\n");
                    }
                }
                let eps = 1e-3;
                assert!(
                    top_dot + eps >= best_dot,
                    "For Dot metric, top-1 dot {:.6} is below best {:.6} by more than eps (expected id {:?}, got {:?})",
                    top_dot,
                    best_dot,
                    expected_dot_id,
                    top_id
                );
            } else {
                let found = results.iter().any(|r| r.id == *expected_id);
                assert!(
                    found,
                    "[{:?}] Expected ID {:?} not in top 5 results for query {:?}",
                    metric, expected_id, noisy_query
                );
            }
        }
        let search_elapsed = search_start.elapsed();
        println!(
            "[{:?}] Completed 10 searches in {:?} (~{:.3} ms/query)",
            metric,
            search_elapsed,
            search_elapsed.as_secs_f64() * 1e3 / 10.0
        );

        println!("\n✅ Completed tests for {:?}\n", metric);
    }
}

#[test]
fn test_large_scale_filtered_queries_all_metrics() {
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::Dot,
    ] {
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
                value: PayloadValue::Int(6),
            },
            Filter::Compare {
                key: "score".into(),
                op: ScalarComparisonOp::Lt,
                value: PayloadValue::Float(90.0.into()),
            },
        ]);

        // <- replaced post_filter with search_with_filter here
        let query = vecf_dim(10_000, DIM);
        let results = segment
            .search_with_filter(&query, 15, Some(&filter))
            .unwrap();

        for r in &results {
            let p = segment.get_payload(r.id).unwrap();
            assert_eq!(p.get("animal").unwrap(), &PayloadValue::Str("dog".into()));
            assert!(matches!(p.get("age").unwrap(), PayloadValue::Int(n) if *n >= 6));
            assert!(matches!(p.get("score").unwrap(), PayloadValue::Float(f) if *f < 90.0.into()));
        }
    }
}

#[test]
fn test_list_filters_with_larger_pool_all_metrics() {
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::Dot,
    ] {
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

        // <- replaced post_filter with search_with_filter here
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
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::Dot,
    ] {
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
            assert!(
                segment.get_vector(*id).is_none(),
                "❌ NOT purged: id = {}",
                id
            );
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
