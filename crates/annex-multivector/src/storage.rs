use crate::fde::Vector;
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ObjectLocation {
    pub offset: u64,
    pub length: u64,
}
pub struct FlatVectors {
    pub values: Vec<f32>,
    pub dimension: usize,
}
pub struct FixedVectorStore {
    path: PathBuf,
    writer: Mutex<File>,
}
impl FixedVectorStore {
    pub fn new(root: impl AsRef<Path>) -> io::Result<Self> {
        fs::create_dir_all(root.as_ref())?;
        let path = root.as_ref().join("fde.bin");
        let writer = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: Mutex::new(writer),
        })
    }
    pub fn put(&self, vector: &[f32]) -> io::Result<ObjectLocation> {
        let bytes = bytemuck::cast_slice(vector);
        let mut file = self.writer.lock().unwrap();
        let offset = file.metadata()?.len();
        file.write_all(bytes)?;
        Ok(ObjectLocation {
            offset,
            length: bytes.len() as u64,
        })
    }
    pub fn map(&self) -> io::Result<Mmap> {
        let file = File::open(&self.path)?;
        if file.metadata()?.len() == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "empty FDE segment",
            ));
        }
        unsafe { MmapOptions::new().map(&file) }
    }
    pub fn get(mapped: &[u8], location: ObjectLocation, dimension: usize) -> io::Result<&[f32]> {
        let start = usize::try_from(location.offset).map_err(|_| io::ErrorKind::InvalidData)?;
        let length = usize::try_from(location.length).map_err(|_| io::ErrorKind::InvalidData)?;
        if length != dimension * size_of::<f32>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid FDE length",
            ));
        }
        let bytes = mapped
            .get(start..start + length)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid FDE location"))?;
        bytemuck::try_cast_slice(bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "unaligned FDE"))
    }
}

/// Append-only, contiguous PLAID residual segment. Superseded records are
/// reclaimed by a future compaction rather than creating per-document files.
pub struct CompressedVectorStore {
    path: PathBuf,
    writer: Mutex<File>,
}
impl CompressedVectorStore {
    pub fn new(root: impl AsRef<Path>) -> io::Result<Self> {
        fs::create_dir_all(root.as_ref())?;
        let path = root.as_ref().join("vectors.plaid");
        let writer = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: Mutex::new(writer),
        })
    }
    pub fn put(
        &self,
        vectors: &[Vector],
        centroid_ids: &[u32],
        centroids: &[Vector],
        residual_codebook: &[f32],
        bits: u8,
    ) -> io::Result<(ObjectLocation, u64)> {
        let dimension = vectors[0].len();
        if residual_codebook.len() != 1usize << bits {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "residual codebook size does not match bits",
            ));
        }
        let mut codes = Vec::with_capacity(vectors.len() * dimension);
        for (vector, &centroid) in vectors.iter().zip(centroid_ids) {
            for (value, center) in vector.iter().zip(&centroids[centroid as usize]) {
                let residual = value - center;
                let code = residual_codebook
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (residual - **a).abs().total_cmp(&(residual - **b).abs())
                    })
                    .unwrap()
                    .0;
                codes.push(code as u8);
            }
        }
        let packed = pack(&codes, bits);
        let mut bytes = Vec::with_capacity(16 + centroid_ids.len() * 4 + packed.len());
        bytes.extend_from_slice(&(vectors.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(dimension as u32).to_le_bytes());
        bytes.push(bits);
        bytes.extend_from_slice(&[0; 3]);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        for id in centroid_ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        bytes.extend_from_slice(&packed);
        let mut file = self.writer.lock().unwrap();
        let offset = file.metadata()?.len();
        file.write_all(&bytes)?;
        Ok((
            ObjectLocation {
                offset,
                length: bytes.len() as u64,
            },
            bytes.len() as u64,
        ))
    }
    pub fn map(&self) -> io::Result<Mmap> {
        let file = File::open(&self.path)?;
        if file.metadata()?.len() == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "empty vector segment",
            ));
        }
        unsafe { MmapOptions::new().map(&file) }
    }
    pub fn decode(
        mapped: &[u8],
        location: ObjectLocation,
        centroids: &[Vector],
        residual_codebook: &[f32],
    ) -> io::Result<FlatVectors> {
        let mut values = Vec::new();
        let dimension =
            Self::decode_into(mapped, location, centroids, residual_codebook, &mut values)?;
        Ok(FlatVectors { values, dimension })
    }

    /// Decode into a caller-supplied scratch buffer. Reuses the buffer's
    /// existing allocation and only grows it if the doc is larger than any
    /// previously decoded doc on this thread. Returns the doc's dimension.
    ///
    /// Callers (rescoring) hold one scratch buffer per rayon worker thread
    /// via `thread_local!` — over a 250-candidate query on FiQA that saves
    /// ~25MB of Vec::with_capacity allocations without changing any of the
    /// per-candidate math.
    pub fn decode_into(
        mapped: &[u8],
        location: ObjectLocation,
        centroids: &[Vector],
        residual_codebook: &[f32],
        scratch: &mut Vec<f32>,
    ) -> io::Result<usize> {
        let start = usize::try_from(location.offset).map_err(|_| io::ErrorKind::InvalidData)?;
        let length = usize::try_from(location.length).map_err(|_| io::ErrorKind::InvalidData)?;
        let bytes = mapped
            .get(start..start + length)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid object location"))?;
        if bytes.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated PLAID object",
            ));
        }
        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let dimension = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let bits = bytes[8];
        let ids_end = 16 + count * 4;
        if bits == 0 || bits > 8 || bytes.len() < ids_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid PLAID header",
            ));
        }
        let ids: Vec<_> = bytes[16..ids_end]
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()) as usize)
            .collect();
        let codes = unpack(&bytes[ids_end..], bits, count * dimension)?;
        if residual_codebook.len() != 1usize << bits {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "residual codebook size mismatch",
            ));
        }
        let total = count * dimension;
        scratch.clear();
        scratch.reserve(total.saturating_sub(scratch.capacity()));
        for (row, id) in ids.into_iter().enumerate() {
            if id >= centroids.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown centroid",
                ));
            }
            for col in 0..dimension {
                let residual = residual_codebook[codes[row * dimension + col] as usize];
                scratch.push(centroids[id][col] + residual);
            }
        }
        Ok(dimension)
    }
}
fn pack(values: &[u8], bits: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity((values.len() * bits as usize).div_ceil(8));
    let (mut acc, mut used) = (0_u64, 0_u8);
    for &v in values {
        acc |= (v as u64) << used;
        used += bits;
        while used >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            used -= 8;
        }
    }
    if used > 0 {
        out.push(acc as u8);
    }
    out
}
fn unpack(bytes: &[u8], bits: u8, count: usize) -> io::Result<Vec<u8>> {
    if bytes.len() * 8 < count * bits as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated residuals",
        ));
    }
    let mut out = Vec::with_capacity(count);
    // Fast path for the default 2-bit residual encoding: 4 codes per byte,
    // no inner while loop, no acc/used bookkeeping. About 4-6x faster than
    // the general path in decode's rescoring hot loop and eliminates the
    // per-code branch.
    if bits == 2 {
        let full_bytes = count / 4;
        for &byte in &bytes[..full_bytes] {
            out.push(byte & 0b11);
            out.push((byte >> 2) & 0b11);
            out.push((byte >> 4) & 0b11);
            out.push((byte >> 6) & 0b11);
        }
        let tail = count % 4;
        if tail > 0 {
            let byte = bytes[full_bytes];
            for i in 0..tail {
                out.push((byte >> (i * 2)) & 0b11);
            }
        }
        return Ok(out);
    }
    let mask = (1_u64 << bits) - 1;
    let (mut acc, mut used, mut input) = (0_u64, 0_u8, bytes.iter());
    while out.len() < count {
        while used < bits {
            acc |= (*input.next().unwrap() as u64) << used;
            used += 8;
        }
        out.push((acc & mask) as u8);
        acc >>= bits;
        used -= bits;
    }
    Ok(out)
}
pub fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slow bit-by-bit unpack that mirrors the original general-path logic.
    /// Kept in the test module as a reference for cross-checking the fast
    /// path in every bits/count combination we care about.
    fn unpack_reference(bytes: &[u8], bits: u8, count: usize) -> Vec<u8> {
        let mask = (1_u64 << bits) - 1;
        let mut out = Vec::with_capacity(count);
        let (mut acc, mut used, mut input) = (0_u64, 0_u8, bytes.iter());
        while out.len() < count {
            while used < bits {
                acc |= (*input.next().unwrap() as u64) << used;
                used += 8;
            }
            out.push((acc & mask) as u8);
            acc >>= bits;
            used -= bits;
        }
        out
    }

    fn deterministic_bytes(n: usize) -> Vec<u8> {
        // Simple LCG so tests are reproducible without depending on rand.
        let mut s: u32 = 0xC0DEBEEF;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 16) as u8
            })
            .collect()
    }

    #[test]
    fn unpack_bits2_fast_path_matches_reference_for_aligned_counts() {
        // For bits=2, count codes need (count * 2 + 7) / 8 bytes.
        let raw = deterministic_bytes(8000);
        for &count in &[4usize, 16, 128, 512, 25600] {
            let expected = unpack_reference(&raw, 2, count);
            let actual = unpack(&raw, 2, count).unwrap();
            assert_eq!(actual, expected, "count={count}");
        }
    }

    #[test]
    fn unpack_bits2_fast_path_matches_reference_for_ragged_counts() {
        let raw = deterministic_bytes(8000);
        // Every non-multiple-of-4 tail length.
        for &count in &[1usize, 2, 3, 5, 6, 7, 9, 17, 25599] {
            let expected = unpack_reference(&raw, 2, count);
            let actual = unpack(&raw, 2, count).unwrap();
            assert_eq!(actual, expected, "count={count}");
        }
    }

    #[test]
    fn unpack_other_bits_still_uses_general_path() {
        let raw = deterministic_bytes(2000);
        for bits in [1u8, 3, 4, 5, 8] {
            let count = 32;
            let expected = unpack_reference(&raw, bits, count);
            let actual = unpack(&raw, bits, count).unwrap();
            assert_eq!(actual, expected, "bits={bits}");
        }
    }
}
