use std::path::PathBuf;

use annex::segment::Segment;
use annex::utils::types::DistanceMetric;
use annex::vector::hnsw::HNSWIndex;

const POINTS: &[(u64, [f32; 2])] = &[
    (101, [0.0, 0.0]),
    (103, [1.0, 0.0]),
    (107, [0.0, 2.0]),
    (109, [3.0, 1.0]),
    (113, [-2.0, -1.0]),
    (127, [1.5, 2.5]),
    (131, [4.0, -2.0]),
    (137, [-3.0, 3.0]),
];

fn main() {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: make_test_fixture SNAPSHOT_PATH");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixture directory");
    }

    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 8, 64, 8, 2);
    let mut segment = Segment::new(hnsw);
    for &(id, point) in POINTS {
        segment
            .insert_with_id(id, point.to_vec(), None)
            .expect("insert fixture point");
    }
    segment.save_to_path(&path).expect("save fixture snapshot");
    println!("{}", path.display());
}
