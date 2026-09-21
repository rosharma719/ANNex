use annex::utils::errors::DBError;
use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;
use annex::vector::metric::score;
use rand::Rng;

fn vecf(v: &[f32]) -> Vector {
    println!("Creating vector: {:?}", v);
    v.to_vec()
}

fn generate_points(n: usize, dim: usize, spread: f32) -> Vec<Vector> {
    println!(
        "Generating {} points with dimension {} and spread {}",
        n, dim, spread
    );
    let mut rng = rand::rng();
    let points = (0..n)
        .map(|i| {
            let point = (0..dim)
                .map(|_| rng.random_range(-spread..spread))
                .collect::<Vec<f32>>();
            println!("Generated point {}: {:?}", i, point);
            point
        })
        .collect();
    println!("Point generation complete");
    points
}

#[test]
fn test_all_metrics_consistency() {
    println!("Starting test_all_metrics_consistency");

    for &metric in &[
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::Dot,
    ] {
        println!("Testing with metric: {:?}", metric);
        let mut hnsw = HNSWIndex::new(metric, 16, 64, 16, 4);
        println!("Created HNSW index with dimension 4");
        let points = generate_points(50, 4, 10.0);

        println!("Inserting {} points into index", points.len());
        for (i, vec) in points.iter().enumerate() {
            hnsw.insert(i as u64, vec.clone()).unwrap();
        }
        println!("All points inserted");

        let query = vecf(&points[0]);
        println!("Searching for query vector: {:?}", query);
        let results = hnsw.search(&query, 10).unwrap();
        println!("Search returned {} results", results.len());

        assert!(!results.is_empty(), "Results should not be empty");

        let best = &results[0];
        println!("Best match ID: {}, raw_score: {}", best.id, best.raw_score);

        match metric {
            DistanceMetric::Dot => {
                let all_dots: Vec<_> = points
                    .iter()
                    .map(|v| score(&query, v, DistanceMetric::Dot)) // dot similarity
                    .collect();

                let max_dot = all_dots.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    (best.raw_score - max_dot).abs() < 1e-4,
                    "Dot: Top result had raw_score {}, but max dot was {}",
                    best.raw_score,
                    max_dot
                );
            }
            _ => {
                let actual_dist = score(&query, &points[best.id as usize], metric);
                assert!(
                    actual_dist <= 1e-4,
                    "{:?}: Expected best match to be original query, distance was {}",
                    metric,
                    actual_dist
                );
            }
        }
    }

    println!("Completed test_all_metrics_consistency");
}

#[test]
fn test_hnsw_robust_score_metrics() {
    // Create 5 vectors in a line with easily predictable order.
    let vectors: Vec<_> = (1..=1000).map(|i| vecf(&[i as f32, 0.0, 0.0])).collect();

    // Euclidean HNSW
    let mut hnsw_euclidean = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 3);
    for (id, vec) in vectors.iter().enumerate() {
        hnsw_euclidean.insert(id as u64, vec.clone()).unwrap();
    }
    let results_euclidean = hnsw_euclidean.search(&vecf(&[1.1, 0.0, 0.0]), 1).unwrap();
    assert_eq!(results_euclidean[0].id, 0);

    // Cosine HNSW
    let mut hnsw_cosine = HNSWIndex::new(DistanceMetric::Cosine, 16, 50, 16, 3);
    for (id, vec) in vectors.iter().enumerate() {
        hnsw_cosine.insert(id as u64, vec.clone()).unwrap();
    }
    let results_cosine = hnsw_cosine.search(&vecf(&[4.9, 0.0, 0.0]), 1).unwrap();
    assert!(
        [0, 1, 2, 3, 4].contains(&results_cosine[0].id),
        "Cosine similarity returned unexpected ID: {}",
        results_cosine[0].id
    );
}

#[test]
fn test_large_insertion_and_ranking_accuracy() {
    println!("Starting test_large_insertion_and_ranking_accuracy");
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 32, 64, 16, 3);
    println!("Created HNSW index with Euclidean metric and dimension 3");

    let mut vectors = Vec::new();
    println!("Inserting 1000 sequential vectors");
    for i in 0..1000 {
        let vec = vecf(&[i as f32, i as f32, i as f32]);
        println!("Inserting vector with ID {}: {:?}", i, vec);
        hnsw.insert(i, vec.clone()).unwrap();
        vectors.push((i, vec));
    }
    println!("Insertion complete");

    let query = vecf(&[0.0, 0.0, 0.0]);
    println!("Searching for query vector: {:?}", query);
    let results = hnsw.search(&query, 5).unwrap();
    println!("Search returned {} results", results.len());

    for (i, result) in results.iter().enumerate() {
        println!("Result {}: ID={}, score={}", i, result.id, result.raw_score);
    }

    let ids: Vec<_> = results.iter().map(|r| r.id).collect();
    println!("Result IDs in order: {:?}", ids);

    // Check that the nearest inserted point (id=0) is first
    assert_eq!(ids[0], 0, "First result should be ID 0");
    assert_eq!(results.len(), 5, "Should return exactly 5 results");

    // Check sorted by distance
    for pair in results.windows(2) {
        println!(
            "Comparing scores: {} <= {}",
            pair[0].raw_score, pair[1].raw_score
        );
        assert!(
            pair[0].raw_score <= pair[1].raw_score,
            "Results not sorted by distance: {} > {}",
            pair[0].raw_score,
            pair[1].raw_score
        );
    }
    println!("Completed test_large_insertion_and_ranking_accuracy");
}

#[test]
fn test_dot_product_prefers_larger_magnitudes() {
    println!("Starting test_dot_product_prefers_larger_magnitudes");
    let mut hnsw = HNSWIndex::new(DistanceMetric::Dot, 16, 50, 16, 2);
    println!("Created HNSW index with Dot product metric and dimension 2");

    println!("Inserting vector with ID 1: [1.0, 1.0]");
    hnsw.insert(1, vecf(&[1.0, 1.0])).unwrap();

    println!("Inserting vector with ID 2: [10.0, 10.0]");
    hnsw.insert(2, vecf(&[10.0, 10.0])).unwrap();

    println!("Inserting vector with ID 3: [-1.0, -1.0]");
    hnsw.insert(3, vecf(&[-1.0, -1.0])).unwrap();

    let query = vecf(&[1.0, 1.0]);
    println!("Searching for query vector: {:?}", query);
    let results = hnsw.search(&query, 3).unwrap();
    println!("Search returned {} results", results.len());

    for (i, result) in results.iter().enumerate() {
        println!("Result {}: ID={}, score={}", i, result.id, result.raw_score);
    }

    assert_eq!(
        results[0].id, 2,
        "First result should be ID 2 (with larger magnitude)"
    );
    println!("Completed test_dot_product_prefers_larger_magnitudes");
}

#[test]
fn test_cosine_distance_with_opposite_vectors() {
    println!("Starting test_cosine_distance_with_opposite_vectors");
    let mut hnsw = HNSWIndex::new(DistanceMetric::Cosine, 16, 50, 16, 3);
    println!("Created HNSW index with Cosine metric and dimension 3");

    println!("Inserting vector with ID 1: [1.0, 0.0, 0.0]");
    hnsw.insert(1, vecf(&[1.0, 0.0, 0.0])).unwrap();

    println!("Inserting vector with ID 2: [-1.0, 0.0, 0.0]");
    hnsw.insert(2, vecf(&[-1.0, 0.0, 0.0])).unwrap();

    let query = vecf(&[1.0, 0.0, 0.0]);
    println!("Searching for query vector: {:?}", query);
    let results = hnsw.search(&query, 2).unwrap();
    println!("Search returned {} results", results.len());

    for (i, result) in results.iter().enumerate() {
        println!("Result {}: ID={}, score={}", i, result.id, result.raw_score);
    }

    assert_eq!(
        results[0].id, 1,
        "First result should be ID 1 (same direction)"
    );
    assert_eq!(
        results[1].id, 2,
        "Second result should be ID 2 (opposite direction)"
    );
    println!("Completed test_cosine_distance_with_opposite_vectors");
}

#[test]
fn test_idempotent_insert_and_query() {
    println!("Starting test_idempotent_insert_and_query");
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    println!("Created HNSW index with Euclidean metric and dimension 2");

    println!("Inserting vector with ID 1: [3.0, 4.0]");
    hnsw.insert(1, vecf(&[3.0, 4.0])).unwrap();

    println!("Inserting same vector with same ID again (should be idempotent)");
    hnsw.insert(1, vecf(&[3.0, 4.0])).unwrap(); // Should be ignored

    let query = vecf(&[3.0, 4.0]);
    println!("Searching for query vector: {:?}", query);
    let results = hnsw.search(&query, 1).unwrap();
    println!("Search returned {} results", results.len());

    for (i, result) in results.iter().enumerate() {
        println!("Result {}: ID={}, score={}", i, result.id, result.raw_score);
    }

    assert_eq!(results.len(), 1, "Should return exactly 1 result");
    assert_eq!(results[0].id, 1, "Result should have ID 1");
    println!("Completed test_idempotent_insert_and_query");
}

#[test]
fn test_empty_index_search_returns_empty() {
    println!("Starting test_empty_index_search_returns_empty");
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    println!("Created empty HNSW index with dimension 2");

    let query = vecf(&[0.0, 0.0]);
    println!("Searching in empty index for query vector: {:?}", query);
    let results = hnsw.search(&query, 10).unwrap();
    println!("Search returned {} results", results.len());

    assert!(
        results.is_empty(),
        "Results should be empty for empty index"
    );
    println!("Completed test_empty_index_search_returns_empty");
}

#[test]
fn test_dimension_mismatch_is_handled() {
    println!("Starting test_dimension_mismatch_is_handled");
    let mut hnsw = HNSWIndex::new(DistanceMetric::Cosine, 16, 50, 16, 5);
    println!("Created HNSW index with dimension 5");

    println!("Inserting valid vector with ID 1: [1.0, 1.0, 1.0, 1.0, 1.0]");
    hnsw.insert(1, vecf(&[1.0, 1.0, 1.0, 1.0, 1.0])).unwrap();

    let bad_vec = vecf(&[1.0, 2.0]);
    println!(
        "Created mismatched vector with dimension 2 (index expects 5): {:?}",
        bad_vec
    );

    println!("Testing insertion with mismatched dimension vector");
    match hnsw.insert(2, bad_vec.clone()) {
        Err(DBError::VectorLengthMismatch { expected, actual }) => {
            println!(
                "Correctly got VectorLengthMismatch error: expected={}, actual={}",
                expected, actual
            );
            assert_eq!(expected, 5, "Expected dimension should be 5");
            assert_eq!(actual, 2, "Actual dimension should be 2");
        }
        other => {
            println!("Unexpected result: {:?}", other);
            panic!("Expected VectorLengthMismatch, got: {:?}", other);
        }
    }

    println!("Testing search with mismatched dimension vector");
    match hnsw.search(&bad_vec, 1) {
        Err(DBError::VectorLengthMismatch { expected, actual }) => {
            println!(
                "Correctly got VectorLengthMismatch error: expected={}, actual={}",
                expected, actual
            );
            assert_eq!(expected, 5, "Expected dimension should be 5");
            assert_eq!(actual, 2, "Actual dimension should be 2");
        }
        other => {
            println!("Unexpected result: {:?}", other);
            panic!("Expected VectorLengthMismatch, got: {:?}", other);
        }
    }
    println!("Completed test_dimension_mismatch_is_handled");
}

#[test]
fn test_search_k_greater_than_total_points() {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    hnsw.insert(1, vecf(&[1.0, 2.0])).unwrap();
    hnsw.insert(2, vecf(&[2.0, 3.0])).unwrap();

    let results = hnsw.search(&vecf(&[1.5, 2.5]), 10).unwrap();
    assert_eq!(results.len(), 2); // Only 2 points in index
}

#[test]
fn test_single_insertion_exact_retrieval() {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    let vec = vecf(&[3.15, 2.71]);
    hnsw.insert(42, vec.clone()).unwrap();

    let results = hnsw.search(&vec, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 42);
    assert!(results[0].raw_score <= 1e-6, "Distance should be ~0.0");
}

#[test]
fn test_insertion_order_independence() {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);

    // Insert in non-sorted order
    hnsw.insert(100, vecf(&[5.0, 5.0])).unwrap();
    hnsw.insert(200, vecf(&[1.0, 1.0])).unwrap();
    hnsw.insert(300, vecf(&[3.0, 3.0])).unwrap();

    // Closest to [2.9, 2.9] should be ID 300
    let results = hnsw.search(&vecf(&[2.9, 2.9]), 1).unwrap();
    assert_eq!(results[0].id, 300);
}

#[test]
fn test_dense_cloud_retrieval_accuracy() {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 64, 16, 3);
    let center = vecf(&[0.0, 0.0, 0.0]);

    // Insert many points around the origin
    for i in 0..500 {
        let offset = i as f32 * 0.01;
        hnsw.insert(i, vecf(&[offset, offset, offset])).unwrap();
    }

    // Closest to center should be ID 0
    let results = hnsw.search(&center, 5).unwrap();
    assert_eq!(results[0].id, 0);
}

#[test]
fn test_high_dimensional_accuracy() {
    // Use 64 as the dimensionality
    let dim = 64;
    let m = 16;
    let ef = 64;
    let max_level_cap = 16;

    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, m, ef, max_level_cap, dim);

    // Insert two far-apart high-dimensional vectors
    hnsw.insert(1, vecf(&vec![1.0; dim])).unwrap();
    hnsw.insert(2, vecf(&vec![100.0; dim])).unwrap();

    let query = vecf(&vec![1.0; dim]);
    let results = hnsw.search(&query, 1).unwrap();

    assert_eq!(results[0].id, 1, "Expected ID 1 to be closest to query");
}

#[test]
fn reorder_rcm_preserves_search_results() {
    use annex::utils::types::DistanceMetric;
    use annex::vector::hnsw::HNSWIndex;
    let mut index = HNSWIndex::new(DistanceMetric::Cosine, 8, 50, 4, 16);
    let mut rng_state = 12345u64;
    let mut lcg = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f32 / u32::MAX as f32
    };
    for i in 0..100u64 {
        let v: Vec<f32> = (0..16).map(|_| lcg()).collect();
        index.insert(i, v).unwrap();
    }
    let query: Vec<f32> = (0..16).map(|j| if j == 0 { 1.0 } else { 0.0 }).collect();
    let before = index.search(&query, 10).unwrap();
    index.reorder_rcm();
    let after = index.search(&query, 10).unwrap();
    let before_ids: Vec<u64> = before.iter().map(|r| r.id).collect();
    let after_ids: Vec<u64> = after.iter().map(|r| r.id).collect();
    assert_eq!(
        before_ids, after_ids,
        "reorder must not change search results"
    );
}

#[test]
fn ti_skip_matches_baseline_recall() {
    use annex::vector::hnsw::SearchRuntimeOptions;
    let mut index = HNSWIndex::new(DistanceMetric::Cosine, 16, 200, 4, 8);
    let vecs: Vec<Vec<f32>> = (0..200u64)
        .map(|i| {
            let mut v = vec![0.0f32; 8];
            v[i as usize % 8] = 1.0;
            v[(i as usize + 1) % 8] = 0.5;
            v
        })
        .collect();
    for (i, v) in vecs.iter().enumerate() {
        index.insert(i as u64, v.clone()).unwrap();
    }
    let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let baseline_opts = SearchRuntimeOptions {
        ef_search: Some(50),
        use_ti_skip: Some(false),
        ..Default::default()
    };
    let ti_opts = SearchRuntimeOptions {
        ef_search: Some(50),
        use_ti_skip: Some(true),
        ..Default::default()
    };
    let baseline = index
        .search_with_options(&query, 10, &baseline_opts)
        .unwrap();
    let ti_result = index.search_with_options(&query, 10, &ti_opts).unwrap();
    // TI skip is an approximation; top-1 must match exactly.
    assert_eq!(baseline[0].id, ti_result[0].id, "top-1 must match");
    // At least 80% recall for the top-10 set.
    let baseline_ids: std::collections::HashSet<_> = baseline.iter().map(|r| r.id).collect();
    let overlap = ti_result
        .iter()
        .filter(|r| baseline_ids.contains(&r.id))
        .count();
    assert!(overlap >= 8, "TI skip recall vs baseline: {}/10", overlap);
}

#[test]
fn sq8_rerank_top1_matches_f32() {
    use annex::utils::types::DistanceMetric;
    use annex::vector::hnsw::{HNSWIndex, SearchRuntimeOptions};
    let mut index = HNSWIndex::new(DistanceMetric::Cosine, 8, 50, 4, 16);
    let mut rng = 99u64;
    let lcg = |r: &mut u64| -> f32 {
        *r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*r >> 33) as f32 / u32::MAX as f32
    };
    for i in 0..150u64 {
        let v: Vec<f32> = (0..16).map(|_| lcg(&mut rng)).collect();
        index.insert(i, v).unwrap();
    }
    index.quantize_all();
    let query: Vec<f32> = (0..16).map(|j| if j < 4 { 0.7 } else { 0.0 }).collect();
    let f32_opts = SearchRuntimeOptions {
        ef_search: Some(50),
        sq8_rerank_factor: Some(0),
        ..Default::default()
    };
    let sq8_opts = SearchRuntimeOptions {
        ef_search: Some(50),
        sq8_rerank_factor: Some(4),
        ..Default::default()
    };
    let f32_res = index.search_with_options(&query, 5, &f32_opts).unwrap();
    let sq8_res = index.search_with_options(&query, 5, &sq8_opts).unwrap();
    assert_eq!(f32_res[0].id, sq8_res[0].id, "top-1 must match");
    let f32_ids: std::collections::HashSet<_> = f32_res.iter().map(|r| r.id).collect();
    let overlap = sq8_res.iter().filter(|r| f32_ids.contains(&r.id)).count();
    assert!(overlap >= 3, "SQ8 recall vs f32: {}/5", overlap);
}

#[test]
fn lid_sort_does_not_regress_recall() {
    use annex::utils::types::DistanceMetric;
    use annex::vector::hnsw::{HNSWIndex, SearchRuntimeOptions};

    fn build_index(apply_lid: bool) -> HNSWIndex {
        let mut index = HNSWIndex::new(DistanceMetric::Cosine, 8, 100, 4, 32);
        let mut rng = 42u64;
        let mut lcg = || -> f32 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as f32 / u32::MAX as f32
        };
        let mut entries: Vec<(u64, Vec<f32>)> = (0..500u64)
            .map(|i| (i, (0..32).map(|_| lcg()).collect()))
            .collect();
        if apply_lid {
            HNSWIndex::sort_by_lid(&mut entries);
        }
        for (id, v) in entries {
            index.insert(id, v).unwrap();
        }
        index
    }

    let mut rng = 999u64;
    let mut lcg = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as f32 / u32::MAX as f32
    };
    let queries: Vec<Vec<f32>> = (0..50).map(|_| (0..32).map(|_| lcg()).collect()).collect();

    let baseline = build_index(false);
    let lid = build_index(true);

    let eval = |index: &HNSWIndex| -> f64 {
        let truth_opts = SearchRuntimeOptions {
            ef_search: Some(450),
            ..Default::default()
        };
        let eval_opts = SearchRuntimeOptions {
            ef_search: Some(20),
            ..Default::default()
        };
        let mut hits = 0usize;
        for q in &queries {
            let truth = index.search_with_options(q, 10, &truth_opts).unwrap();
            let res = index.search_with_options(q, 10, &eval_opts).unwrap();
            let truth_ids: std::collections::HashSet<_> = truth.iter().map(|r| r.id).collect();
            hits += res.iter().filter(|r| truth_ids.contains(&r.id)).count();
        }
        hits as f64 / (queries.len() * 10) as f64
    };

    let r_base = eval(&baseline);
    let r_lid = eval(&lid);
    assert!(
        r_lid >= r_base - 0.02,
        "LID recall {:.3} < baseline {:.3} - 0.02",
        r_lid,
        r_base
    );
}

// ── VectorArena tests ────────────────────────────────────────────────────────

#[test]
fn vector_arena_push_and_get() {
    use annex::vector::hnsw::arena::{VectorArena, VectorArenaView};
    let arena = VectorArena::new(4, 8);
    let v0 = vec![1.0f32, 2.0, 3.0, 4.0];
    let v1 = vec![5.0f32, 6.0, 7.0, 8.0];
    let idx0 = arena.push(&v0);
    let idx1 = arena.push(&v1);
    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    let view = arena.view();
    assert_eq!(view.get(0), v0.as_slice());
    assert_eq!(view.get(1), v1.as_slice());
    assert_eq!(arena.len(), 2);
    // Suppress unused-import warning
    let _: VectorArenaView = view;
}

#[test]
fn vector_arena_crosses_chunk_boundary() {
    use annex::vector::hnsw::arena::VectorArena;
    let arena = VectorArena::new(2, 4); // 4 vectors per chunk, dim=2
    for i in 0..10u64 {
        let v = vec![i as f32, i as f32 * 2.0];
        arena.push(&v);
    }
    assert_eq!(arena.len(), 10);
    let view = arena.view();
    for i in 0..10u64 {
        let got = view.get(i as usize);
        assert_eq!(got[0], i as f32);
        assert_eq!(got[1], i as f32 * 2.0);
    }
}

#[test]
fn vector_arena_view_survives_concurrent_push() {
    use annex::vector::hnsw::arena::VectorArena;
    use std::sync::Arc;
    let arena = Arc::new(VectorArena::new(2, 4));
    arena.push(&[1.0f32, 2.0]);
    arena.push(&[3.0f32, 4.0]);
    let view = arena.view();
    let arena2 = arena.clone();
    let handle = std::thread::spawn(move || {
        for i in 0..20i32 {
            arena2.push(&[i as f32, i as f32]);
        }
    });
    // Original view still valid and correct while concurrent pushes happen
    assert_eq!(view.get(0), &[1.0f32, 2.0]);
    assert_eq!(view.get(1), &[3.0f32, 4.0]);
    handle.join().unwrap();
    assert_eq!(arena.len(), 22);
}

#[test]
fn vector_arena_addresses_stable_across_chunk_growth() {
    // Capture a raw pointer into the arena before growth, verify it's unchanged after.
    use annex::vector::hnsw::arena::VectorArena;
    let arena = VectorArena::new(4, 4); // 4 per chunk
    arena.push(&[1.0f32, 2.0, 3.0, 4.0]);
    let view_before = arena.view();
    let ptr_before: *const f32 = view_before.get(0).as_ptr();
    // Push enough to force multiple new chunks
    for i in 0..20i32 {
        arena.push(&[i as f32, 0.0, 0.0, 0.0]);
    }
    let view_after = arena.view();
    let ptr_after: *const f32 = view_after.get(0).as_ptr();
    // Address must be identical — chunk data never moves
    assert_eq!(
        ptr_before, ptr_after,
        "vector address changed after growth — not stable!"
    );
    assert_eq!(view_after.get(0), &[1.0f32, 2.0, 3.0, 4.0]);
}
