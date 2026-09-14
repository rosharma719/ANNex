use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::utils::errors::DBError;

const ADLER32_MOD: u32 = 65521;

pub fn write_atomic<P, F>(path: P, write_fn: F) -> Result<(), DBError>
where
    P: AsRef<Path>,
    F: FnOnce(&mut BufWriter<File>) -> Result<(), DBError>,
{
    let path = path.as_ref();
    let tmp_path = path.with_extension("tmp");

    let file = File::create(&tmp_path)?;
    let mut writer = BufWriter::new(file);
    write_fn(&mut writer)?;
    writer.flush()?;

    let file = writer
        .into_inner()
        .map_err(|e| DBError::IOError(e.into_error()))?;
    file.sync_all()?;

    if cfg!(windows) {
        let _ = std::fs::remove_file(path);
    }
    std::fs::rename(&tmp_path, path)?;

    if let Some(parent) = path.parent()
        && let Ok(dir) = File::open(parent)
    {
        let _ = dir.sync_all();
    }

    Ok(())
}

pub fn write_atomic_with_checksum<P, F>(
    path: P,
    footer_magic: [u8; 4],
    write_fn: F,
) -> Result<(), DBError>
where
    P: AsRef<Path>,
    F: FnOnce(&mut ChecksumWriter<&mut BufWriter<File>>) -> Result<(), DBError>,
{
    write_atomic(path, |writer| {
        let mut checksum_writer = ChecksumWriter::new(&mut *writer);
        write_fn(&mut checksum_writer)?;
        let checksum = checksum_writer.finish();
        writer.write_all(&footer_magic)?;
        writer.write_all(&checksum.to_le_bytes())?;
        Ok(())
    })
}

pub fn adler32(bytes: &[u8]) -> u32 {
    const NMAX: usize = 5552;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for chunk in bytes.chunks(NMAX) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= ADLER32_MOD;
        b %= ADLER32_MOD;
    }
    (b << 16) | a
}

pub struct ChecksumWriter<W: Write> {
    inner: W,
    a: u32,
    b: u32,
}

impl<W: Write> ChecksumWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner, a: 1, b: 0 }
    }

    pub fn finish(self) -> u32 {
        (self.b << 16) | self.a
    }
}

impl<W: Write> Write for ChecksumWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        const NMAX: usize = 5552;
        for chunk in buf[..written].chunks(NMAX) {
            for &byte in chunk {
                self.a += byte as u32;
                self.b += self.a;
            }
            self.a %= ADLER32_MOD;
            self.b %= ADLER32_MOD;
        }
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
