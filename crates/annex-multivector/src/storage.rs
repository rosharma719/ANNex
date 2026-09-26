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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<[u8; 32]>,
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
        let length = file.metadata()?.len();
        // A prior failed write can leave a partial f32 in the uncommitted tail.
        // Keep subsequent referenced records aligned without touching live data.
        let padding = (4 - length % 4) % 4;
        file.write_all(&[0; 3][..padding as usize])?;
        let offset = length
            .checked_add(padding)
            .ok_or_else(|| invalid("FDE offset overflow"))?;
        append_record(&mut file, bytes, "fde_partial_write")?;
        Ok(ObjectLocation {
            offset,
            length: bytes.len() as u64,
            checksum: Some(*blake3::hash(bytes).as_bytes()),
        })
    }
    pub fn len(&self) -> io::Result<u64> {
        Ok(self.writer.lock().unwrap().metadata()?.len())
    }
    pub fn sync(&self) -> io::Result<()> {
        self.writer.lock().unwrap().sync_all()
    }
    pub fn recover(&self, committed: u64) -> io::Result<()> {
        let file = self.writer.lock().unwrap();
        if file.metadata()?.len() < committed {
            return Err(invalid("segment shorter than committed boundary"));
        }
        file.set_len(committed)
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
        let bytes = record_bytes(mapped, location)?;
        if Some(bytes.len()) != dimension.checked_mul(size_of::<f32>()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid FDE length",
            ));
        }
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
        let dimension = vectors.first().map_or(0, Vec::len);
        if dimension == 0
            || vectors.len() > u32::MAX as usize
            || dimension > u32::MAX as usize
            || centroid_ids.len() != vectors.len()
            || !(1..=8).contains(&bits)
            || vectors.iter().any(|v| v.len() != dimension)
            || centroid_ids.iter().any(|&id| {
                centroids
                    .get(id as usize)
                    .is_none_or(|v| v.len() != dimension)
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid PLAID input shape",
            ));
        }
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
        append_record(&mut file, &bytes, "object_partial_write")?;
        Ok((
            ObjectLocation {
                offset,
                length: bytes.len() as u64,
                checksum: Some(*blake3::hash(&bytes).as_bytes()),
            },
            bytes.len() as u64,
        ))
    }
    pub fn len(&self) -> io::Result<u64> {
        Ok(self.writer.lock().unwrap().metadata()?.len())
    }
    pub fn sync(&self) -> io::Result<()> {
        self.writer.lock().unwrap().sync_all()
    }
    pub fn recover(&self, committed: u64) -> io::Result<()> {
        let file = self.writer.lock().unwrap();
        if file.metadata()?.len() < committed {
            return Err(invalid("segment shorter than committed boundary"));
        }
        file.set_len(committed)
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
        let bytes = record_bytes(mapped, location)?;
        if bytes.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated PLAID object",
            ));
        }
        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let dimension = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let bits = bytes[8];
        let ids_end = count
            .checked_mul(4)
            .and_then(|n| n.checked_add(16))
            .ok_or_else(|| invalid("PLAID count overflow"))?;
        let total = count
            .checked_mul(dimension)
            .ok_or_else(|| invalid("PLAID dimensions overflow"))?;
        if count == 0
            || dimension == 0
            || bits == 0
            || bits > 8
            || bytes.len() < ids_end
            || bytes[9..12] != [0; 3]
            || bytes[12..16] != 1.0f32.to_le_bytes()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid PLAID header",
            ));
        }
        let ids: Vec<_> = bytes[16..ids_end]
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()) as usize)
            .collect();
        if ids
            .iter()
            .any(|&id| centroids.get(id).is_none_or(|v| v.len() != dimension))
        {
            return Err(invalid("unknown centroid or mismatched dimension"));
        }
        let codes = unpack(&bytes[ids_end..], bits, total)?;
        if residual_codebook.len() != 1usize << bits {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "residual codebook size mismatch",
            ));
        }
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
    let expected = count
        .checked_mul(bits as usize)
        .and_then(|n| n.checked_add(7))
        .map(|n| n / 8)
        .ok_or_else(|| invalid("residual count overflow"))?;
    if bytes.len() != expected {
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
fn append_record(file: &mut File, bytes: &[u8], _stage: &'static str) -> io::Result<()> {
    #[cfg(test)]
    {
        let split = (bytes.len() / 2 + 1).min(bytes.len());
        file.write_all(&bytes[..split])?;
        commit_boundary(_stage)?;
        file.write_all(&bytes[split..])
    }
    #[cfg(not(test))]
    file.write_all(bytes)
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub fn record_bytes(mapped: &[u8], location: ObjectLocation) -> io::Result<&[u8]> {
    let start = usize::try_from(location.offset).map_err(|_| invalid("offset overflow"))?;
    let length = usize::try_from(location.length).map_err(|_| invalid("length overflow"))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| invalid("record range overflow"))?;
    mapped
        .get(start..end)
        .ok_or_else(|| invalid("record outside segment"))
}

/// Verify once on open, outside the scoring hot path. Legacy records without a
/// digest still receive structural validation from their decoder.
pub fn verify_record(mapped: &[u8], location: ObjectLocation, required: bool) -> io::Result<()> {
    let bytes = record_bytes(mapped, location)?;
    match location.checksum {
        Some(expected) if blake3::hash(bytes).as_bytes() != &expected => {
            Err(invalid("record checksum mismatch"))
        }
        None if required => Err(invalid("missing record checksum")),
        _ => Ok(()),
    }
}

/// The caller must publish its staged in-memory state even when a failure
/// after rename makes durability uncertain. Rolling it back would diverge
/// from the manifest currently visible to a subsequent opener.
pub struct CommitError {
    pub source: io::Error,
    pub published: bool,
}

pub fn atomic_write(path: &Path, bytes: &[u8], sync: bool) -> Result<(), CommitError> {
    let temporary = path.with_extension("pending");
    let before_publish = || -> io::Result<()> {
        let mut file = File::create(&temporary)?;
        append_record(&mut file, bytes, "manifest_partial_write")?;
        commit_boundary("manifest_written")?;
        if sync {
            file.sync_all()?;
        }
        commit_boundary("manifest_synced")?;
        fs::rename(&temporary, path)?;
        Ok(())
    };
    before_publish().map_err(|source| CommitError {
        source,
        published: false,
    })?;
    let after_publish = || -> io::Result<()> {
        commit_boundary("manifest_renamed")?;
        if sync {
            File::open(path.parent().unwrap())?.sync_all()?;
        }
        commit_boundary("directory_synced")?;
        Ok(())
    };
    after_publish().map_err(|source| CommitError {
        source,
        published: true,
    })
}

// Fault injection exists only in test binaries, never in a production build.
#[cfg(test)]
thread_local! {
    pub(crate) static FAIL_COMMIT: std::cell::Cell<Option<(&'static str, usize)>> = const { std::cell::Cell::new(None) };
}
pub(crate) fn commit_boundary(_stage: &'static str) -> io::Result<()> {
    #[cfg(test)]
    {
        if std::env::var("ANNEX_TEST_CRASH_STAGE").as_deref() == Ok(_stage) {
            // No destructors, rollback, or implicit close/flush.
            std::process::exit(86);
        }
        let fail = FAIL_COMMIT.with(|point| match point.get() {
            Some((name, remaining)) if name == _stage => {
                point.set(if remaining > 1 {
                    Some((name, remaining - 1))
                } else {
                    None
                });
                remaining <= 1
            }
            _ => false,
        });
        if fail {
            return Err(io::Error::other(format!("injected failure: {_stage}")));
        }
    }
    Ok(())
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
            let actual = unpack(&raw[..(count * 2).div_ceil(8)], 2, count).unwrap();
            assert_eq!(actual, expected, "count={count}");
        }
    }

    #[test]
    fn unpack_bits2_fast_path_matches_reference_for_ragged_counts() {
        let raw = deterministic_bytes(8000);
        // Every non-multiple-of-4 tail length.
        for &count in &[1usize, 2, 3, 5, 6, 7, 9, 17, 25599] {
            let expected = unpack_reference(&raw, 2, count);
            let actual = unpack(&raw[..(count * 2).div_ceil(8)], 2, count).unwrap();
            assert_eq!(actual, expected, "count={count}");
        }
    }

    #[test]
    fn unpack_other_bits_still_uses_general_path() {
        let raw = deterministic_bytes(2000);
        for bits in [1u8, 3, 4, 5, 8] {
            let count = 32;
            let expected = unpack_reference(&raw, bits, count);
            let actual = unpack(&raw[..(count * bits as usize).div_ceil(8)], bits, count).unwrap();
            assert_eq!(actual, expected, "bits={bits}");
        }
    }
    #[test]
    fn corrupt_headers_and_ranges_return_errors_without_panicking() {
        let centers = vec![vec![0.; 3]];
        let residuals = vec![0.; 4];
        for (count, dimension) in [(u32::MAX, 3), (1, u32::MAX), (0, 3), (1, 0), (1, 4)] {
            let mut bytes = vec![0u8; 21];
            bytes[..4].copy_from_slice(&count.to_le_bytes());
            bytes[4..8].copy_from_slice(&dimension.to_le_bytes());
            bytes[8] = 2;
            bytes[12..16].copy_from_slice(&1f32.to_le_bytes());
            let location = ObjectLocation {
                offset: 0,
                length: bytes.len() as u64,
                checksum: None,
            };
            assert!(CompressedVectorStore::decode(&bytes, location, &centers, &residuals).is_err());
        }
        let location = ObjectLocation {
            offset: u64::MAX,
            length: 2,
            checksum: None,
        };
        assert!(FixedVectorStore::get(&[0; 16], location, 1).is_err());
        assert!(CompressedVectorStore::decode(&[0; 16], location, &centers, &residuals).is_err());
        let location = ObjectLocation {
            offset: 0,
            length: 4,
            checksum: None,
        };
        assert!(FixedVectorStore::get(&[0; 16], location, usize::MAX).is_err());
    }
}
