use annex::utils::types::DistanceMetric;
use annex::vector::hnsw::{HNSWIndex, SearchRuntimeOptions};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn dot_search_respects_ef_and_matches_full_budget_oracle() {
    const N: usize = 2048;
    const DIM: usize = 16;
    const K: usize = 10;
    let mut rng = StdRng::seed_from_u64(914);
    let mut index = HNSWIndex::new(DistanceMetric::Dot, 16, 128, 16, DIM);
    index.set_exact_fallback_enabled(false);
    // Unequal norms and mixed signs matter: normalization would change ranks.
    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let scale = 0.2 + (i % 13) as f32 * 0.1;
            (0..DIM)
                .map(|_| scale * rng.random_range(-1.0..1.0))
                .collect()
        })
        .collect();
    for (id, vector) in vectors.iter().enumerate() {
        index.insert(id as u64, vector.clone()).unwrap();
    }
    let mut narrow_work = 0;
    let mut full_work = 0;
    let mut recalled = 0;
    for _ in 0..20 {
        let query: Vec<f32> = (0..DIM).map(|_| rng.random_range(-2.0..2.0)).collect();
        let mut exact: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(id, vector)| {
                let score: f32 = query.iter().zip(vector).map(|(a, b)| a * b).sum();
                (id as u64, score)
            })
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let options = SearchRuntimeOptions {
            ef_search: Some(64),
            early_exit_patience: Some(0),
            neighbor_scan_cap_level0: Some(0),
            neighbor_scan_patience: Some(0),
            num_entry_seeds: Some(1),
            ..Default::default()
        };
        let (narrow, stats) = index
            .search_with_stats_with_options(&query, K, &options)
            .unwrap();
        let fast = index.search_with_options(&query, K, &options).unwrap();
        assert!(!stats.exact);
        assert_eq!(
            narrow.iter().map(|r| r.id).collect::<Vec<_>>(),
            fast.iter().map(|r| r.id).collect::<Vec<_>>()
        );
        for hit in &narrow {
            let expected = exact.iter().find(|(id, _)| *id == hit.id).unwrap().1;
            assert!((hit.raw_score - expected).abs() < 1e-4);
            assert_eq!(hit.sort_key, -hit.raw_score);
            recalled += usize::from(exact[..K].iter().any(|(id, _)| *id == hit.id));
        }
        let all_options = SearchRuntimeOptions {
            ef_search: Some(N),
            ..options
        };
        let (full, full_stats) = index
            .search_with_stats_with_options(&query, K, &all_options)
            .unwrap();
        assert_eq!(
            full.iter().map(|r| r.id).collect::<Vec<_>>(),
            exact[..K].iter().map(|r| r.0).collect::<Vec<_>>()
        );
        narrow_work += stats.distance_computations;
        full_work += full_stats.distance_computations;
    }
    eprintln!(
        "Dot synthetic: recall@10={:.3}, L0 distances ef64={} vs ef2048={} (20 queries)",
        recalled as f32 / 200.,
        narrow_work,
        full_work
    );
    assert!(
        recalled >= 160,
        "approximate recall unexpectedly low: {recalled}/200"
    );
    assert!(
        narrow_work * 4 < full_work * 3,
        "ef64 still traverses most of the graph: {narrow_work}/{full_work}"
    );
}

#[test]
fn dot_search_orders_negative_scores_without_normalizing_vectors() {
    let mut index = HNSWIndex::new(DistanceMetric::Dot, 8, 32, 16, 2);
    index.set_exact_fallback_enabled(false);
    for id in 1..=64 {
        index.insert(id, vec![id as f32, 1.]).unwrap();
    }
    let options = SearchRuntimeOptions {
        ef_search: Some(64),
        ..Default::default()
    };
    for (query, expected) in [
        (vec![-1., 0.], vec![1, 2, 3]),
        (vec![1., 0.], vec![64, 63, 62]),
    ] {
        let result = index.search_with_options(&query, 3, &options).unwrap();
        assert_eq!(result.iter().map(|r| r.id).collect::<Vec<_>>(), expected);
        for hit in result {
            assert_eq!(hit.raw_score, query[0] * hit.id as f32);
        }
    }
}
