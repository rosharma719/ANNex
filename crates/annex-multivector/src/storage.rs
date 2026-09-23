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
        let mut values = Vec::with_capacity(count * dimension);
        for (row, id) in ids.into_iter().enumerate() {
            if id >= centroids.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown centroid",
                ));
            }
            for col in 0..dimension {
                let residual = residual_codebook[codes[row * dimension + col] as usize];
                values.push(centroids[id][col] + residual);
            }
        }
        Ok(FlatVectors { values, dimension })
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
    let mask = (1_u64 << bits) - 1;
    let (mut out, mut acc, mut used, mut input) =
        (Vec::with_capacity(count), 0_u64, 0_u8, bytes.iter());
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
