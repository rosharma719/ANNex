//! Immutable, neighbor-local storage for compressed L0 graph expansion.

use crate::utils::types::DistanceMetric;

pub(crate) const NEIGHBOR_BLOCK_WIDTH: usize = 32;

/// One graph-expansion unit. Codes are stored neighbor-major so expanding a
/// node reads its neighbor IDs and SQ8 vectors sequentially.
#[derive(Debug, Clone)]
pub(crate) struct NeighborBlock {
    /// One allocation: fixed-width little-endian IDs, followed by valid SQ8 codes.
    data: Box<[u8]>,
    valid: u8,
    dim: usize,
}

impl NeighborBlock {
    fn new(neighbors: &[usize], quantized: &[u8], dim: usize) -> Self {
        debug_assert!(neighbors.len() <= NEIGHBOR_BLOCK_WIDTH);
        const ID_BYTES: usize = NEIGHBOR_BLOCK_WIDTH * size_of::<u32>();
        let mut data = vec![0; ID_BYTES + neighbors.len() * dim];
        for (slot, &neighbor) in neighbors.iter().enumerate() {
            let id = u32::try_from(neighbor)
                .expect("neighbor index exceeds u32")
                .to_le_bytes();
            data[slot * 4..slot * 4 + 4].copy_from_slice(&id);
            let source = &quantized[neighbor * dim..(neighbor + 1) * dim];
            let code_start = ID_BYTES + slot * dim;
            data[code_start..code_start + dim].copy_from_slice(source);
        }
        Self {
            data: data.into_boxed_slice(),
            valid: neighbors.len() as u8,
            dim,
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.valid as usize
    }

    #[inline]
    pub(crate) fn id(&self, slot: usize) -> usize {
        debug_assert!(slot < self.len());
        let start = slot * size_of::<u32>();
        u32::from_le_bytes(self.data[start..start + 4].try_into().unwrap()) as usize
    }

    /// Score every valid lane from the block's contiguous SQ8 code payload.
    #[inline]
    pub(crate) fn score(&self, query: &[i16], metric: DistanceMetric) -> [i32; 32] {
        debug_assert_eq!(query.len(), self.dim);
        let mut scores = [0; NEIGHBOR_BLOCK_WIDTH];
        let centered = metric == DistanceMetric::Cosine;
        let codes_start = NEIGHBOR_BLOCK_WIDTH * size_of::<u32>();
        for (slot, score) in scores.iter_mut().enumerate().take(self.len()) {
            let start = codes_start + slot * self.dim;
            let code = &self.data[start..start + self.dim];
            *score = code
                .iter()
                .zip(query)
                .map(|(&stored, &q)| {
                    let stored = if centered {
                        stored as i32 - 128
                    } else {
                        stored as i32
                    };
                    stored * q as i32
                })
                .sum();
        }
        scores
    }
}

/// Immutable L0 blocks indexed by source-node index.
#[derive(Debug, Clone, Default)]
pub(crate) struct NeighborBlockStore {
    nodes: Vec<Vec<NeighborBlock>>,
}

impl NeighborBlockStore {
    pub(crate) fn build(
        adjacency: impl IntoIterator<Item = Vec<usize>>,
        quantized: &[u8],
        dim: usize,
    ) -> Self {
        let nodes = adjacency
            .into_iter()
            .map(|neighbors| {
                neighbors
                    .chunks(NEIGHBOR_BLOCK_WIDTH)
                    .map(|chunk| NeighborBlock::new(chunk, quantized, dim))
                    .collect()
            })
            .collect();
        Self { nodes }
    }

    #[inline]
    pub(crate) fn get(&self, node: usize) -> Option<&[NeighborBlock]> {
        self.nodes.get(node).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_neighbors_and_copies_codes_in_order() {
        let dim = 3;
        let quantized: Vec<u8> = (0..40 * dim).map(|v| v as u8).collect();
        let store = NeighborBlockStore::build(vec![(0..35).collect()], &quantized, dim);
        let blocks = store.get(0).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].len(), 32);
        assert_eq!(blocks[1].len(), 3);
        assert_eq!(blocks[1].id(0), 32);
    }

    #[test]
    fn block_scores_match_global_sq8_scoring() {
        let dim = 4;
        let quantized = vec![128, 129, 127, 130, 130, 120, 128, 129];
        let query = [2, -3, 4, 5];
        let store = NeighborBlockStore::build(vec![vec![1, 0]], &quantized, dim);
        let block = &store.get(0).unwrap()[0];
        let scores = block.score(&query, DistanceMetric::Cosine);
        for slot in 0..block.len() {
            let id = block.id(slot);
            let expected: i32 = quantized[id * dim..(id + 1) * dim]
                .iter()
                .zip(query)
                .map(|(&stored, q)| (stored as i32 - 128) * q as i32)
                .sum();
            assert_eq!(scores[slot], expected);
        }
    }

    #[test]
    fn uncentered_block_scores_match_global_sq8_scoring() {
        let dim = 3;
        let quantized = vec![1, 2, 3, 7, 8, 9];
        let query = [4, -2, 5];
        let store = NeighborBlockStore::build(vec![vec![0, 1]], &quantized, dim);
        let block = &store.get(0).unwrap()[0];
        let scores = block.score(&query, DistanceMetric::Euclidean);
        assert_eq!(scores[0], 15);
        assert_eq!(scores[1], 57);
    }
}
