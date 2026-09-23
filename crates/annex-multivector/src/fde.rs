pub type Vector = Vec<f32>;

pub fn normalize(vector: &[f32]) -> Vector {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        vec![0.0; vector.len()]
    } else {
        vector.iter().map(|x| x / norm).collect()
    }
}

pub fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// ColBERT's late-interaction score: sum of per-query-token maxima.
pub fn maxsim(query: &[Vector], document: &[Vector]) -> f32 {
    if query.is_empty() || document.is_empty() {
        return 0.0;
    }
    let document: Vec<_> = document.iter().map(|v| normalize(v)).collect();
    query
        .iter()
        .map(|query| {
            let query = normalize(query);
            document
                .iter()
                .map(|doc| dot(&query, doc))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// MaxSim over already-normalized query vectors and a flat document matrix.
pub fn maxsim_flat(query: &[Vector], document: &[f32], dimension: usize) -> f32 {
    query
        .iter()
        .map(|q| {
            document
                .chunks_exact(dimension)
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}
