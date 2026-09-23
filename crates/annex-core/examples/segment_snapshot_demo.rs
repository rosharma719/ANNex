use std::path::PathBuf;

use annex::segment::Segment;
use annex::utils::types::{DistanceMetric, Vector};
use annex::vector::hnsw::HNSWIndex;

fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}

fn main() {
    let path = PathBuf::from("data/demo_segment_snapshot.bin");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create data directory");
    }
    let mut segment = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2));

    for i in 0..25u64 {
        let v = vecf(&[i as f32, (i % 5) as f32]);
        segment.insert_with_id(i, v, None).expect("insert failed");
    }

    segment
        .save_to_path(&path)
        .expect("failed to save snapshot");
    println!("saved snapshot to {}", path.display());

    let restored = Segment::load_from_path(&path).expect("failed to load snapshot");
    println!(
        "restored points={} payloads={}",
        restored.hnsw().len(),
        restored.payloads().len()
    );
}
