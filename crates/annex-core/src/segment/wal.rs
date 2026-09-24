use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::anyhow;
use serde::{Deserialize, Serialize};

use crate::utils::env::{env_bool, env_u64, env_usize};
use crate::utils::errors::DBError;
use crate::utils::io::adler32;
use crate::utils::payload::Payload;
use crate::utils::types::{PointId, Vector};

const WAL_MAGIC: [u8; 4] = *b"VDBW";
const WAL_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct WalConfig {
    pub path: PathBuf,
    pub fsync: bool,
    pub fsync_every: usize,
    pub fsync_interval: Duration,
}

impl WalConfig {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            fsync: true,
            fsync_every: 100,
            fsync_interval: Duration::from_millis(100),
        }
    }

    pub fn from_env<P: Into<PathBuf>>(path: P) -> Self {
        let mut cfg = Self::new(path);
        if let Some(value) = env_bool("VECTORDB_WAL_FSYNC") {
            cfg.fsync = value;
        }
        if let Some(value) = env_usize("VECTORDB_WAL_FSYNC_EVERY") {
            cfg.fsync_every = value.max(1);
        }
        if let Some(value) = env_u64("VECTORDB_WAL_FSYNC_MS") {
            cfg.fsync_interval = Duration::from_millis(value);
        }
        cfg
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum WalRecord {
    Insert {
        point_id: PointId,
        vector: Vector,
        payload: Option<Payload>,
    },
    Delete {
        point_id: PointId,
    },
    UpdatePayload {
        point_id: PointId,
        payload: Payload,
    },
}

pub struct WalWriter {
    path: PathBuf,
    file: File,
    fsync: bool,
    fsync_every: usize,
    fsync_interval: Duration,
    ops_since_sync: usize,
    last_sync: Instant,
}

impl WalWriter {
    pub fn open(config: WalConfig) -> Result<Self, DBError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&config.path)?;
        let mut writer = Self {
            path: config.path,
            file,
            fsync: config.fsync,
            fsync_every: config.fsync_every,
            fsync_interval: config.fsync_interval,
            ops_since_sync: 0,
            last_sync: Instant::now(),
        };
        writer.ensure_header()?;
        Ok(writer)
    }

    pub fn append(&mut self, record: &WalRecord) -> Result<(), DBError> {
        let payload =
            bincode::serialize(record).map_err(|e| DBError::SerializationError(anyhow!(e)))?;
        let len = payload.len() as u32;
        let checksum = adler32(&payload);
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.write_all(&checksum.to_le_bytes())?;
        self.file.flush()?;
        self.ops_since_sync = self.ops_since_sync.saturating_add(1);
        if self.should_fsync() {
            self.file.sync_data()?;
            self.ops_since_sync = 0;
            self.last_sync = Instant::now();
        }
        Ok(())
    }

    pub fn truncate(&mut self) -> Result<(), DBError> {
        self.file.set_len(0)?;
        if self.fsync {
            self.file.sync_data()?;
            self.last_sync = Instant::now();
            self.ops_since_sync = 0;
        }
        self.ensure_header()?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn ensure_header(&mut self) -> Result<(), DBError> {
        let len = self.file.metadata()?.len();
        if len == 0 {
            self.file.write_all(&WAL_MAGIC)?;
            self.file.write_all(&WAL_VERSION.to_le_bytes())?;
            self.file.flush()?;
            if self.fsync {
                self.file.sync_data()?;
                self.last_sync = Instant::now();
                self.ops_since_sync = 0;
            }
        }
        Ok(())
    }

    fn should_fsync(&self) -> bool {
        if !self.fsync {
            return false;
        }
        if self.fsync_every <= 1 && self.fsync_interval.is_zero() {
            return true;
        }
        let op_ready = self.fsync_every > 0 && self.ops_since_sync >= self.fsync_every;
        let time_ready =
            !self.fsync_interval.is_zero() && self.last_sync.elapsed() >= self.fsync_interval;
        op_ready || time_ready
    }
}

pub struct WalReader {
    path: PathBuf,
}

impl WalReader {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn replay<F>(&self, mut apply: F) -> Result<(), DBError>
    where
        F: FnMut(WalRecord) -> Result<(), DBError>,
    {
        if !self.path.exists() {
            return Ok(());
        }
        let mut file = BufReader::new(File::open(&self.path)?);
        let mut header = [0u8; 8];
        if let Err(err) = file.read_exact(&mut header) {
            if err.kind() == std::io::ErrorKind::UnexpectedEof {
                return Ok(());
            }
            return Err(DBError::IOError(err));
        }
        if header[..4] != WAL_MAGIC {
            return Err(DBError::WALCorrupt("missing WAL header".into()));
        }
        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if version != WAL_VERSION {
            return Err(DBError::WALCorrupt(format!(
                "unsupported WAL version {}",
                version
            )));
        }

        loop {
            let mut len_buf = [0u8; 4];
            match file.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(DBError::IOError(err)),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            if let Err(err) = file.read_exact(&mut payload) {
                if err.kind() == std::io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(DBError::IOError(err));
            }
            let mut checksum_buf = [0u8; 4];
            if let Err(err) = file.read_exact(&mut checksum_buf) {
                if err.kind() == std::io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(DBError::IOError(err));
            }
            let expected = u32::from_le_bytes(checksum_buf);
            let actual = adler32(&payload);
            if actual != expected {
                return Err(DBError::WALCorrupt("checksum mismatch".into()));
            }
            let record: WalRecord = bincode::deserialize(&payload)
                .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
            apply(record)?;
        }

        Ok(())
    }
}
