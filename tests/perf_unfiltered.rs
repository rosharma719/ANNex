use std::env;
use std::time::Instant;

mod common;
use annex::segment::segment::Segment;
use annex::utils::types::DistanceMetric;
use annex::vector::hnsw::HNSWIndex;
use common::{generate_vector_dim, log_peak_rss};

/// Heavier unfiltered search latency check to approximate production scale.
#[test]
#[ignore]
fn bench_unfiltered_large_scale() {
    let size = 20_000;
    let metric = DistanceMetric::Euclidean;
    let dim = 1536;
    let hnsw = HNSWIndex::new(metric, 16, 64, 16, dim);
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(100);

    println!(
        "\n🚀 Inserting {} vectors for large-scale unfiltered benchmark",
        size
    );
    let start_insert = Instant::now();
    let mut last_log = start_insert;
    for i in 0..size {
        segment.insert(generate_vector_dim(i, dim), None).unwrap();
        if i != 0 && i % 1000 == 0 {
            let now = Instant::now();
            println!(
                "Inserted {} vectors... (+{:?}, chunk={:?})",
                i,
                now - start_insert,
                now - last_log
            );
            last_log = now;
        }
    }
    let insert_dur = start_insert.elapsed();
    let vps = size as f64 / insert_dur.as_secs_f64();
    println!(
        "✅ Inserted {} vectors in {:?} ({:.2} vec/s)",
        size, insert_dur, vps
    );
    log_peak_rss("unfiltered_large_inserted");

    let queries: Vec<_> = (0..500)
        .map(|i| generate_vector_dim(i + size, dim))
        .collect();
    let start_search = Instant::now();
    for q in &queries {
        let res = segment.search(q, 20).unwrap();
        assert!(!res.is_empty(), "Search returned empty result");
    }
    let search_dur = start_search.elapsed();
    let avg_ms = search_dur.as_secs_f64() * 1000.0 / queries.len() as f64;
    println!(
        "🔍 Ran {} unfiltered searches in {:?} (avg {:.3} ms/search)",
        queries.len(),
        search_dur,
        avg_ms
    );
    log_peak_rss("unfiltered_large_complete");
    segment.hnsw().flush_unfiltered_search_stats();
}

/// Parametrizable unfiltered benchmark; size/dim can be set via env vars.
#[test]
#[ignore]
fn bench_unfiltered_param_scale() {
    let size = env::var("VECTORDB_BENCH_SIZE")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(100_000);
    let dim = env::var("VECTORDB_BENCH_DIM")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(1536);
    let metric = DistanceMetric::Euclidean;
    let hnsw = HNSWIndex::new(metric, 16, 64, 16, dim);
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(100);

    println!(
        "\n🚀 Inserting {} vectors (dim = {}) for parametrized unfiltered benchmark",
        size, dim
    );
    let start_insert = Instant::now();
    let mut last_log = start_insert;
    for i in 0..size {
        segment.insert(generate_vector_dim(i, dim), None).unwrap();
        if i != 0 && i % 5000 == 0 {
            let now = Instant::now();
            println!(
                "Inserted {} vectors... (+{:?}, chunk={:?})",
                i,
                now - start_insert,
                now - last_log
            );
            last_log = now;
        }
    }
    let insert_dur = start_insert.elapsed();
    let vps = size as f64 / insert_dur.as_secs_f64();
    println!(
        "✅ Inserted {} vectors in {:?} ({:.2} vec/s)",
        size, insert_dur, vps
    );
    log_peak_rss("unfiltered_param_inserted");

    let queries: Vec<_> = (0..500)
        .map(|i| generate_vector_dim(i + size, dim))
        .collect();
    let start_search = Instant::now();
    for q in &queries {
        let res = segment.search(q, 20).unwrap();
        assert!(!res.is_empty(), "Search returned empty result");
    }
    let search_dur = start_search.elapsed();
    let avg_ms = search_dur.as_secs_f64() * 1000.0 / queries.len() as f64;
    println!(
        "🔍 Ran {} unfiltered searches in {:?} (avg {:.3} ms/search)",
        queries.len(),
        search_dur,
        avg_ms
    );
    log_peak_rss("unfiltered_param_complete");
    segment.hnsw().flush_unfiltered_search_stats();
}

/// Very large-scale unfiltered benchmark (opt-in; runs longer).
#[test]
#[ignore]
fn bench_unfiltered_million_scale() {
    let size = 1_000_000;
    let metric = DistanceMetric::Euclidean;
    let dim = 1536;
    let hnsw = HNSWIndex::new(metric, 16, 64, 16, dim);
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(100);

    println!(
        "\n🚀 Inserting {} vectors for million-scale unfiltered benchmark",
        size
    );
    let start_insert = Instant::now();
    let mut last_log = start_insert;
    for i in 0..size {
        segment.insert(generate_vector_dim(i, dim), None).unwrap();
        if i != 0 && i % 20000 == 0 {
            let now = Instant::now();
            println!(
                "Inserted {} vectors... (+{:?}, chunk={:?})",
                i,
                now - start_insert,
                now - last_log
            );
            last_log = now;
        }
    }
    let insert_dur = start_insert.elapsed();
    let vps = size as f64 / insert_dur.as_secs_f64();
    println!(
        "✅ Inserted {} vectors in {:?} ({:.2} vec/s)",
        size, insert_dur, vps
    );
    log_peak_rss("unfiltered_million_inserted");

    let queries: Vec<_> = (0..500)
        .map(|i| generate_vector_dim(i + size, dim))
        .collect();
    let start_search = Instant::now();
    for q in &queries {
        let res = segment.search(q, 20).unwrap();
        assert!(!res.is_empty(), "Search returned empty result");
    }
    let search_dur = start_search.elapsed();
    let avg_ms = search_dur.as_secs_f64() * 1000.0 / queries.len() as f64;
    println!(
        "🔍 Ran {} unfiltered searches in {:?} (avg {:.3} ms/search)",
        queries.len(),
        search_dur,
        avg_ms
    );
    log_peak_rss("unfiltered_million_complete");
    segment.hnsw().flush_unfiltered_search_stats();
}
