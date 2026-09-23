use crate::utils::types::DistanceMetric;

/// Main distance dispatcher
pub fn score(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Dot => dot_product_similarity(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
    }
}

/// Cosine distance: 1 - cosine similarity
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>();
    let norm_a = a
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    let norm_b = b
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();

    if norm_a == 0.0 && norm_b == 0.0 {
        return 0.0;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    let mut sim = dot / (norm_a * norm_b);
    if sim > 1.0 {
        sim = 1.0;
    } else if sim < -1.0 {
        sim = -1.0;
    }

    (1.0 - sim) as f32
}

/// Dot product similarity (higher is closer)
fn dot_product_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>() as f32
}

/// Euclidean squared distance, matching the raw_score returned by HNSW search.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = (*x as f64) - (*y as f64);
            diff * diff
        })
        .sum::<f64>() as f32
}
