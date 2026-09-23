use std::collections::HashSet;
use std::env;
use std::time::Instant;

use rand::distr::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use annex::payload_storage::filters::Filter;
use annex::segment::segment::Segment;
use annex::utils::payload::PayloadValue;
use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;
use annex::vector::metric::score;

mod common;
use common::{env_usize_first, env_usize_list_first, generate_vector_dim};

fn parse_metric_env(env_key: &str, default: DistanceMetric) -> DistanceMetric {
    match env::var(env_key).ok().as_deref() {
        Some(s) if s.eq_ignore_ascii_case("cosine") => DistanceMetric::Cosine,
        Some(s) if s.eq_ignore_ascii_case("dot") => DistanceMetric::Dot,
        Some(s) if s.eq_ignore_ascii_case("euclidean") => DistanceMetric::Euclidean,
        _ => default,
    }
}

fn random_vector(rng: &mut SmallRng, dim: usize, noise: f32) -> Vector {
    // Fast uniform filler avoids trig and keeps RNG overhead low.
    let uniform = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let mut base: Vec<f32> = uniform.sample_iter(&mut *rng).take(dim).collect();
    if noise == 0.0 {
        return base;
    }
    let noise_dist = Uniform::new(-noise, noise).unwrap();
    for v in base.iter_mut() {
        *v += rng.sample(noise_dist);
    }
    base
}

#[test]
#[ignore]
fn recall_unfiltered_euclidean() {
    let t0 = Instant::now();
    let size = env_usize_first(&["VECTORDB_SIZE", "VECTORDB_RECALL_SIZE"]).unwrap_or(20_000);
    let dim = env_usize_first(&["VECTORDB_DIM", "VECTORDB_RECALL_DIM"]).unwrap_or(1536);
    let ef_values = env_usize_list_first(&["VECTORDB_EF_SEARCH", "VECTORDB_RECALL_EF_SEARCH"])
        .unwrap_or_else(|| vec![32, 64, 128]);
    let top_k = env_usize_first(&["VECTORDB_TOPK", "VECTORDB_RECALL_TOPK"]).unwrap_or(20);
    let num_queries =
        env_usize_first(&["VECTORDB_QUERIES", "VECTORDB_RECALL_QUERIES"]).unwrap_or(20);
    let seed = env::var("VECTORDB_RECALL_SEED")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(42);
    let noise = env::var("VECTORDB_RECALL_NOISE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.002);
    let use_random = env::var("VECTORDB_RECALL_RANDOM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let metric = DistanceMetric::Euclidean;
    let m = 16;
    let ef_construct = 200;

    println!(
        "\n🧪 Building recall harness: size={}, dim={}, top_k={}, queries={}, ef_search={:?}, ef_construct={}",
        size, dim, top_k, num_queries, ef_values, ef_construct
    );

    // Build dataset and index once; keep stored vectors for anchored queries.
    let mut data_rng = SmallRng::seed_from_u64(seed);
    let mut dataset: Vec<(u64, Vector)> = Vec::with_capacity(size);
    let hnsw = HNSWIndex::new(
        metric,
        m,
        ef_values.iter().copied().max().unwrap_or(32).max(top_k),
        16,
        dim,
    );
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(ef_construct);
    println!("⏱️  Setup in {:?} (initialization)", t0.elapsed());
    let insert_start = Instant::now();
    for i in 0..size {
        let v = if use_random {
            random_vector(&mut data_rng, dim, 0.0)
        } else {
            generate_vector_dim(i, dim)
        };
        let id = segment.insert(v.clone(), None).unwrap();
        dataset.push((id, v));
        if i != 0 && i % 1000 == 0 {
            println!("Inserted {} vectors... (+{:?})", i, insert_start.elapsed());
        }
    }

    // Anchored queries: sample stored vectors and add optional noise.
    let mut query_rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B97F4A7C15);
    let mut queries: Vec<Vector> = Vec::with_capacity(num_queries);
    while queries.len() < num_queries {
        let idx = query_rng.random_range(0..size);
        let mut q = dataset[idx].1.clone();
        if noise > 0.0 {
            for v in q.iter_mut() {
                *v += query_rng.random_range(-noise..noise);
            }
        }
        queries.push(q);
    }

    let mut summary: Vec<(usize, f64)> = Vec::new();

    for &ef_requested in &ef_values {
        let ef_search = ef_requested.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        println!(
            "ℹ️  Running sweep entry with ef_search={} (ef_construct={})",
            ef_search, ef_construct
        );

        let mut avg_recall = 0.0;
        let mut total_search = 0.0f64;
        for (qi, q) in queries.iter().enumerate() {
            // ground truth via brute force using the exact metric implementation
            let mut brute: Vec<(u64, f32)> = dataset
                .iter()
                .map(|(id, v)| (*id, score(q, v, metric)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            brute.truncate(top_k);
            let truth: HashSet<u64> = brute.iter().map(|(id, _)| *id).collect();

            let start = Instant::now();
            let approx = segment.search(q, top_k).unwrap();
            total_search += start.elapsed().as_secs_f64();
            let hits = approx.iter().filter(|r| truth.contains(&r.id)).count() as f64;
            let recall = hits / top_k as f64;
            avg_recall += recall;
            println!(
                "[ef_search={}] Query {} recall: {:.3} (hits {}/{})",
                ef_search, qi, recall, hits as usize, top_k
            );
        }
        avg_recall /= num_queries as f64;
        let avg_search_ms = (total_search / num_queries as f64) * 1000.0;
        println!(
            "✅ [ef_search={}] Average recall over {} queries: {:.3} | avg search {:.3} ms/query",
            ef_search, num_queries, avg_recall, avg_search_ms
        );
        segment.hnsw().flush_unfiltered_search_stats();
        summary.push((ef_search, avg_recall));
    }

    println!("\nSummary (ef_search -> avg recall):");
    for (ef, r) in summary {
        println!("  {} -> {:.3}", ef, r);
    }
}

#[test]
#[ignore]
fn recall_in_place_filtered() {
    let t0 = Instant::now();
    let size = env_usize_first(&["VECTORDB_SIZE", "VECTORDB_RECALL_SIZE"]).unwrap_or(20_000);
    let dim = env_usize_first(&["VECTORDB_DIM", "VECTORDB_RECALL_DIM"]).unwrap_or(1536);
    let ef_values = env_usize_list_first(&["VECTORDB_EF_SEARCH", "VECTORDB_RECALL_EF_SEARCH"])
        .unwrap_or_else(|| vec![32, 64, 128]);
    let top_k = env_usize_first(&["VECTORDB_TOPK", "VECTORDB_RECALL_TOPK"]).unwrap_or(20);
    let num_queries =
        env_usize_first(&["VECTORDB_QUERIES", "VECTORDB_RECALL_QUERIES"]).unwrap_or(20);
    let seed = env::var("VECTORDB_RECALL_SEED")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(42);
    let noise = env::var("VECTORDB_RECALL_NOISE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.002);
    let use_random = env::var("VECTORDB_RECALL_RANDOM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let metric = parse_metric_env("VECTORDB_RECALL_METRIC", DistanceMetric::Euclidean);
    let m = 16;
    let ef_construct = env::var("VECTORDB_RECALL_EF_CONSTRUCT")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(100);

    println!(
        "\n🧪 In-place filtered recall: size={}, dim={}, top_k={}, queries={}, ef_search={:?}, ef_construct={}, metric={:?}",
        size, dim, top_k, num_queries, ef_values, ef_construct, metric
    );

    // Build dataset and index once; keep stored vectors and payloads.
    let mut data_rng = SmallRng::seed_from_u64(seed);
    let mut dataset: Vec<(u64, Vector, PayloadValue, i64)> = Vec::with_capacity(size);
    let hnsw = HNSWIndex::new(
        metric,
        m,
        ef_values.iter().copied().max().unwrap_or(32).max(top_k),
        16,
        dim,
    );
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(ef_construct);
    println!("⏱️  Setup in {:?} (initialization)", t0.elapsed());

    let insert_start = Instant::now();
    for i in 0..size {
        let v = if use_random {
            random_vector(&mut data_rng, dim, 0.0)
        } else {
            generate_vector_dim(i, dim)
        };
        let group = if i % 2 == 0 { "even" } else { "odd" };
        let score_val = i as i64;
        let mut payload = annex::utils::payload::Payload::default();
        payload.set("group", PayloadValue::Str(group.to_string()));
        payload.set("score", PayloadValue::Int(score_val));
        let id = segment.insert(v.clone(), Some(payload)).unwrap();
        dataset.push((id, v, PayloadValue::Str(group.to_string()), score_val));
        if i != 0 && i % 1000 == 0 {
            println!("Inserted {} vectors... (+{:?})", i, insert_start.elapsed());
        }
    }

    // Anchored queries: sample stored vectors and add optional noise.
    let mut query_rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B97F4A7C15);
    let mut queries: Vec<Vector> = Vec::with_capacity(num_queries);
    while queries.len() < num_queries {
        let idx = query_rng.random_range(0..size);
        let mut q = dataset[idx].1.clone();
        if noise > 0.0 {
            for v in q.iter_mut() {
                *v += query_rng.random_range(-noise..noise);
            }
        }
        queries.push(q);
    }

    // Filter: group == even AND score >= threshold
    let score_threshold = (size as i64 * 2 / 3) as i64;
    let filter = Filter::And(vec![
        Filter::Match {
            key: "group".into(),
            value: PayloadValue::Str("even".into()),
        },
        Filter::Compare {
            key: "score".into(),
            op: annex::utils::payload::ScalarComparisonOp::Gte,
            value: PayloadValue::Int(score_threshold),
        },
    ]);

    let mut summary: Vec<(usize, f64, f64)> = Vec::new();

    for &ef_requested in &ef_values {
        let ef_search = ef_requested.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        println!(
            "ℹ️  Running filtered sweep entry with ef_search={} (ef_construct={})",
            ef_search, ef_construct
        );

        let mut avg_recall = 0.0;
        let mut total_search = 0.0f64;
        for (qi, q) in queries.iter().enumerate() {
            // ground truth via brute force with filter
            let mut brute: Vec<(u64, f32)> = dataset
                .iter()
                .filter(|(_, _, g, s)| {
                    g == &PayloadValue::Str("even".into()) && *s >= score_threshold
                })
                .map(|(id, v, _, _)| (*id, score(q, v, metric)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            brute.truncate(top_k);
            let truth: HashSet<u64> = brute.iter().map(|(id, _)| *id).collect();

            let start = Instant::now();
            let approx = segment.search_with_filter(q, top_k, Some(&filter)).unwrap();
            total_search += start.elapsed().as_secs_f64();
            let hits = approx.iter().filter(|r| truth.contains(&r.id)).count() as f64;
            let recall = hits / top_k as f64;
            avg_recall += recall;
            println!(
                "[ef_search={}] Query {} recall: {:.3} (hits {}/{})",
                ef_search, qi, recall, hits as usize, top_k
            );
        }
        avg_recall /= num_queries as f64;
        let avg_search_ms = (total_search / num_queries as f64) * 1000.0;
        println!(
            "✅ [ef_search={}] Average filtered recall over {} queries: {:.3} | avg search {:.3} ms/query",
            ef_search, num_queries, avg_recall, avg_search_ms
        );
        segment.hnsw().flush_unfiltered_search_stats();
        summary.push((ef_search, avg_recall, avg_search_ms));
    }

    println!("\nSummary (ef_search -> avg recall, ms/query):");
    for (ef, r, ms) in summary {
        println!("  {} -> {:.3}, {:.3} ms/query", ef, r, ms);
    }
}
