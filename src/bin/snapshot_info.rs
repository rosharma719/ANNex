use std::env;
use std::fs;
use std::path::Path;

use annex::segment::Segment;

fn main() {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: snapshot_info <path>");
            std::process::exit(2);
        }
    };
    let path_ref = Path::new(&path);
    let meta = fs::metadata(path_ref).ok();
    let size = meta.map(|m| m.len()).unwrap_or(0);

    let (segment, metadata) = match Segment::load_from_path_with_metadata(path_ref) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("failed to load snapshot: {err}");
            std::process::exit(1);
        }
    };

    println!("snapshot_path={}", path_ref.display());
    println!("file_size_bytes={}", size);
    println!("points={}", segment.hnsw().len());
    println!("payloads={}", segment.payloads().len());
    if let Some(meta) = metadata {
        println!("metadata_created_at_ms={}", meta.created_at_ms);
        println!(
            "metadata_hnsw=metric={:?} dim={} m={} m0={} ef={} ef_construct={} level_cap={} level_scale={:.3} max_level={} exact_fallback={} threshold={}",
            meta.hnsw.metric,
            meta.hnsw.dim,
            meta.hnsw.m,
            meta.hnsw.m0,
            meta.hnsw.ef,
            meta.hnsw.ef_construct,
            meta.hnsw.max_level_cap,
            meta.hnsw.level_scale,
            meta.hnsw.current_max_level,
            meta.hnsw.exact_fallback_enabled,
            meta.hnsw.exact_fallback_threshold
        );
        println!(
            "metadata_counts=points={} payloads={}",
            meta.points, meta.payloads
        );
    }
}
