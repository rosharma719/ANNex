//! Query-planning overhead diagnostics; reporting rules: docs/benchmarks.md.

use std::collections::BTreeMap;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use multivector::{
    Durability, IndexConfig, MultiVectorIndex, Representation, RetrievalDocument, RetrieveRequest,
};
use serde_json::{Value, json};

const DIMENSION: usize = 128;

struct Fixture {
    _directory: tempfile::TempDir,
    index: MultiVectorIndex,
}

fn fixture(documents: usize) -> Fixture {
    let directory = tempfile::tempdir().expect("create planner benchmark directory");
    let index = MultiVectorIndex::open_with_durability(
        directory.path(),
        IndexConfig::new(DIMENSION),
        Durability::Buffered,
    )
    .expect("open planner benchmark index");

    let batch = (0..documents)
        .map(|position| {
            let mut vector = vec![0.0; DIMENSION];
            vector[position % DIMENSION] = 1.0;
            RetrievalDocument {
                id: format!("doc-{position}"),
                metadata: json!({
                    "tenant": if position % 10 == 0 { "selected" } else { "other" },
                    "bucket": position % 100,
                    "position": position,
                }),
                text: Some(format!(
                    "vector database query planning document {position}"
                )),
                representations: BTreeMap::from([(
                    "semantic".into(),
                    Representation::Dense { vector },
                )]),
                ..RetrievalDocument::default()
            }
        })
        .collect();
    index
        .upsert_records(batch)
        .expect("populate planner benchmark index");

    Fixture {
        _directory: directory,
        index,
    }
}

fn request(value: Value) -> RetrieveRequest {
    serde_json::from_value(value).expect("valid planner benchmark request")
}

fn requests() -> Vec<(&'static str, RetrieveRequest)> {
    let mut vector = vec![0.0; DIMENSION];
    vector[0] = 1.0;
    vec![
        (
            "manual_dense",
            request(json!({
                "prefetch": [{
                    "kind": "dense",
                    "field": "semantic",
                    "vector": vector,
                    "limit": 100,
                    "backend": "auto",
                    "ef_search": 256
                }],
                "limit": 10
            })),
        ),
        (
            "manual_hybrid",
            request(json!({
                "prefetch": [
                    {
                        "kind": "dense",
                        "field": "semantic",
                        "vector": vector,
                        "limit": 100,
                        "backend": "auto",
                        "ef_search": 256
                    },
                    {
                        "kind": "bm25",
                        "text": "vector database query planning",
                        "limit": 100
                    }
                ],
                "limit": 10
            })),
        ),
        (
            "manual_hybrid_filtered_10pct",
            request(json!({
                "prefetch": [
                    {
                        "kind": "dense",
                        "field": "semantic",
                        "vector": vector,
                        "limit": 100,
                        "backend": "auto",
                        "ef_search": 256
                    },
                    {
                        "kind": "bm25",
                        "text": "vector database query planning",
                        "limit": 100
                    }
                ],
                "filter": {
                    "op": "eq",
                    "field": "tenant",
                    "value": "selected"
                },
                "limit": 10
            })),
        ),
        (
            "auto_text_dense",
            request(json!({
                "planning_mode": "auto",
                "query": {
                    "text": "vector database query planning",
                    "dense": {"semantic": vector}
                },
                "objective": {"quality": "balanced"},
                "limit": 10
            })),
        ),
    ]
}

fn bench_planner(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_planner");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(5));

    for documents in [100usize, 10_000] {
        let fixture = fixture(documents);
        let requests = requests();
        group.throughput(Throughput::Elements(documents as u64));

        for (name, request) in &requests {
            group.bench_with_input(
                BenchmarkId::new(*name, format!("documents={documents}")),
                request,
                |b, request| {
                    b.iter(|| {
                        black_box(
                            fixture
                                .index
                                .plan(black_box(request))
                                .expect("plan benchmark request"),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_filtered_execution(c: &mut Criterion) {
    let fixture = fixture(10_000);
    fixture
        .index
        .build_dense_ann("semantic", 16, 100)
        .expect("build dense ANN for execution benchmark");
    let mut vector = vec![0.0; DIMENSION];
    vector[0] = 1.0;
    let cases = [
        ("exact/unfiltered", "exact", None),
        ("hnsw/unfiltered", "hnsw", None),
        (
            "exact/filtered_10pct",
            "exact",
            Some(json!({"op": "eq", "field": "tenant", "value": "selected"})),
        ),
        (
            "hnsw/filtered_10pct",
            "hnsw",
            Some(json!({"op": "eq", "field": "tenant", "value": "selected"})),
        ),
        (
            "exact/filtered_1pct",
            "exact",
            Some(json!({"op": "eq", "field": "bucket", "value": 0})),
        ),
        (
            "hnsw/filtered_1pct",
            "hnsw",
            Some(json!({"op": "eq", "field": "bucket", "value": 0})),
        ),
    ];
    let mut group = c.benchmark_group("query_execution");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    for (name, backend, filter) in cases {
        let request = request(json!({
            "prefetch": [{
                "kind": "dense",
                "field": "semantic",
                "vector": vector,
                "limit": 100,
                "backend": backend,
                "ef_search": 64
            }],
            "filter": filter,
            "limit": 10
        }));
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(
                    fixture
                        .index
                        .retrieve(black_box(&request))
                        .expect("execute benchmark request"),
                )
            })
        });
    }
    group.finish();
}

/// Planner regret: for each selectivity band, compare the auto-chosen plan
/// against forced-exact. Reports recall@k overlap and latency ratio so future
/// cost-model changes can be calibrated against real performance.
fn bench_planner_regret(c: &mut Criterion) {
    let fixture = fixture(10_000);
    fixture
        .index
        .build_dense_ann("semantic", 16, 100)
        .expect("build dense ANN for regret benchmark");
    let mut vector = vec![0.0; DIMENSION];
    vector[0] = 1.0;

    let cases = [
        ("unfiltered", None),
        (
            "filtered_10pct",
            Some(serde_json::json!({"op": "eq", "field": "tenant", "value": "selected"})),
        ),
        (
            "filtered_1pct",
            Some(serde_json::json!({"op": "eq", "field": "bucket", "value": 0})),
        ),
    ];

    let mut group = c.benchmark_group("planner_regret");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));

    for (name, filter) in cases {
        let auto_req = request(serde_json::json!({
            "prefetch": [{"kind": "dense", "field": "semantic",
                          "vector": vector, "limit": 10, "backend": "auto", "ef_search": 64}],
            "filter": filter,
            "limit": 10
        }));
        let exact_req = request(serde_json::json!({
            "prefetch": [{"kind": "dense", "field": "semantic",
                          "vector": vector, "limit": 10, "backend": "exact"}],
            "filter": filter,
            "limit": 10
        }));

        // Measure recall@10: |auto ∩ exact| / |exact|.
        // Printed once before the timed loop so it appears in bench output.
        let exact_ids: std::collections::HashSet<String> = fixture
            .index
            .retrieve(&exact_req)
            .expect("exact baseline")
            .matches
            .into_iter()
            .map(|r| r.id)
            .collect();
        let auto_ids: std::collections::HashSet<String> = fixture
            .index
            .retrieve(&auto_req)
            .expect("auto plan")
            .matches
            .into_iter()
            .map(|r| r.id)
            .collect();
        let recall = if exact_ids.is_empty() {
            1.0
        } else {
            auto_ids.intersection(&exact_ids).count() as f64 / exact_ids.len() as f64
        };
        eprintln!("regret/{name}: recall@10={recall:.3}");

        group.throughput(criterion::Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("auto", name), &auto_req, |b, req| {
            b.iter(|| {
                black_box(
                    fixture
                        .index
                        .retrieve(black_box(req))
                        .expect("auto retrieve"),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("exact", name), &exact_req, |b, req| {
            b.iter(|| {
                black_box(
                    fixture
                        .index
                        .retrieve(black_box(req))
                        .expect("exact retrieve"),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(
    planner,
    bench_planner,
    bench_filtered_execution,
    bench_planner_regret
);
criterion_main!(planner);
