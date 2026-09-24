mod common;
use annex::utils::types::DistanceMetric;
use common::*;

#[test]
#[ignore]
fn bench_all_euclidean_100() {
    let size = 100;
    let metric = DistanceMetric::Euclidean;
    let mut segment = bench_insertion(metric, size);
    let query = vecf(&[1.0, 0.0, 0.0]);
    bench_search(&segment, &query);
    bench_deletion(&mut segment, size);
}

#[test]
#[ignore]
fn bench_all_cosine_1000() {
    let size = 1000;
    let metric = DistanceMetric::Cosine;
    let mut segment = bench_insertion(metric, size);
    let query = vecf(&[1.0, 0.0, 0.0]);
    bench_search(&segment, &query);
    bench_deletion(&mut segment, size);
}

#[test]
#[ignore]
fn bench_all_dot_500() {
    let size = 500;
    let metric = DistanceMetric::Dot;
    let mut segment = bench_insertion(metric, size);
    let query = vecf(&[1.0, 0.0, 0.0]);
    bench_search(&segment, &query);
    bench_deletion(&mut segment, size);
}
