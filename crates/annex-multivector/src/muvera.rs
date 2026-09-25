use crate::fde::{Vector, dot};

/// MUVERA's asymmetric, data-oblivious fixed-dimensional encoding.
pub struct FdeEncoder {
    dimension: usize,
    ksim: usize,
    projected: usize,
    repetitions: Vec<Repetition>,
    encoding_version: u32,
}
struct Repetition {
    hash_planes: Vec<Vector>,
    projection: Vec<Vector>,
}
impl FdeEncoder {
    pub fn new(
        dimension: usize,
        ksim: usize,
        projected: usize,
        repetitions: usize,
        seed: u64,
    ) -> Self {
        Self::with_version(dimension, ksim, projected, repetitions, seed, 2)
    }
    pub fn with_version(
        dimension: usize,
        ksim: usize,
        projected: usize,
        repetitions: usize,
        seed: u64,
        version: u32,
    ) -> Self {
        assert!(matches!(version, 1 | 2), "unsupported FDE encoding version");
        let mut rng = SplitMix64(seed);
        let repetitions = (0..repetitions)
            .map(|_| {
                let hash_planes = (0..ksim)
                    .map(|_| (0..dimension).map(|_| rng.sign()).collect())
                    .collect();
                let scale = 1.0 / (projected as f32).sqrt();
                let projection = (0..projected)
                    .map(|_| (0..dimension).map(|_| rng.sign() * scale).collect())
                    .collect();
                Repetition {
                    hash_planes,
                    projection,
                }
            })
            .collect();
        Self {
            dimension,
            ksim,
            projected,
            repetitions,
            encoding_version: version,
        }
    }
    pub fn encoding_version(&self) -> u32 {
        self.encoding_version
    }
    pub fn output_dimension(&self) -> usize {
        (1usize << self.ksim) * self.projected * self.repetitions.len()
    }
    pub fn encode_query(&self, v: &[Vector]) -> Vector {
        self.encode(v, false)
    }
    pub fn encode_document(&self, v: &[Vector]) -> Vector {
        self.encode(v, true)
    }
    fn encode(&self, vectors: &[Vector], document: bool) -> Vector {
        let buckets = 1usize << self.ksim;
        let mut output = Vec::with_capacity(self.output_dimension());
        for repetition in &self.repetitions {
            let mut sums = vec![vec![0.; self.dimension]; buckets];
            let mut counts = vec![0usize; buckets];
            for vector in vectors {
                let bucket = hash(vector, &repetition.hash_planes);
                counts[bucket] += 1;
                for (target, value) in sums[bucket].iter_mut().zip(vector) {
                    *target += value;
                }
            }
            if document && self.encoding_version == 1 {
                // Preserve the original one-pass encoding for persisted v1
                // indexes, including its second division when the source
                // bucket has already been averaged.
                for bucket in 0..buckets {
                    if counts[bucket] > 0 {
                        for x in &mut sums[bucket] {
                            *x /= counts[bucket] as f32;
                        }
                    } else if let Some(nearest) = nearest_occupied(bucket, &counts) {
                        sums[bucket] = sums[nearest]
                            .iter()
                            .map(|x| x / counts[nearest] as f32)
                            .collect();
                    }
                }
            } else if document {
                // Fill every empty bucket with the average of its
                // Hamming-nearest occupied bucket. This selects a bucket
                // average, not an individual token. Copy from raw sums, THEN
                // normalize the originally-occupied buckets. Doing it in a
                // single pass silently double-divides whenever the nearest
                // occupied bucket comes earlier in iteration order — its
                // sums are already an average by then, so the fill divides
                // by count a second time.
                let occupied: Vec<bool> = counts.iter().map(|&n| n > 0).collect();
                for bucket in 0..buckets {
                    if !occupied[bucket] {
                        if let Some(nearest) = nearest_occupied(bucket, &counts) {
                            let scale = 1.0 / counts[nearest] as f32;
                            sums[bucket] = sums[nearest].iter().map(|x| x * scale).collect();
                        }
                    }
                }
                for bucket in 0..buckets {
                    if occupied[bucket] {
                        let scale = 1.0 / counts[bucket] as f32;
                        for x in &mut sums[bucket] {
                            *x *= scale;
                        }
                    }
                }
            }
            for bucket in sums {
                for plane in &repetition.projection {
                    output.push(dot(&bucket, plane));
                }
            }
        }
        output
    }
}
fn hash(v: &[f32], planes: &[Vector]) -> usize {
    planes
        .iter()
        .enumerate()
        .fold(0, |bits, (i, p)| bits | (((dot(v, p) > 0.) as usize) << i))
}
fn nearest_occupied(target: usize, counts: &[usize]) -> Option<usize> {
    counts
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0)
        .min_by_key(|(b, _)| (target ^ b).count_ones())
        .map(|(b, _)| b)
}
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn sign(&mut self) -> f32 {
        if self.next() & 1 == 0 { -1. } else { 1. }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matching_set_scores_higher() {
        let e = FdeEncoder::new(2, 2, 2, 8, 13);
        let q = vec![vec![1., 0.], vec![0., 1.]];
        assert!(
            dot(&e.encode_query(&q), &e.encode_document(&q))
                > dot(
                    &e.encode_query(&q),
                    &e.encode_document(&[vec![-1., 0.], vec![0., -1.]])
                )
        );
    }

    // Two tokens in the same bucket expose the legacy second division.
    #[test]
    fn corrected_empty_bucket_fill_preserves_average_and_versions_legacy_behavior() {
        let e = FdeEncoder::new(4, 3, 2, 1, 7);
        let legacy = FdeEncoder::with_version(4, 3, 2, 1, 7, 1);
        let token = vec![1., 0.5, -0.5, 1.];
        let occupied = hash(&token, &e.repetitions[0].hash_planes);
        assert_eq!(occupied, 1); // Empty buckets occur both before and after it.
        let doc = vec![token.clone(), token.clone()];
        let encoded = e.encode_document(&doc);
        let legacy_encoded = legacy.encode_document(&doc);
        let expected: Vec<_> = e.repetitions[0]
            .projection
            .iter()
            .map(|plane| dot(&token, plane))
            .collect();
        let stride = 2;
        let buckets = 1 << 3;
        assert_eq!(e.encoding_version(), 2);
        assert_eq!(legacy.encoding_version(), 1);
        assert_eq!(encoded.len(), buckets * stride);
        for bucket in 0..buckets {
            for (slot, expected) in expected.iter().enumerate() {
                let actual = encoded[bucket * stride + slot];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "bucket {bucket} differs from the occupied average"
                );
                let legacy_expected = if bucket > occupied {
                    expected / 2.0
                } else {
                    *expected
                };
                assert!((legacy_encoded[bucket * stride + slot] - legacy_expected).abs() < 1e-5);
            }
        }
        assert_ne!(encoded, legacy_encoded);
        assert_eq!(e.encode_query(&doc), legacy.encode_query(&doc));
    }
}
