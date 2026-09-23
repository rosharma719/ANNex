use std::path::{Path, PathBuf};
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::utils::errors::DBError;

use super::Segment;

pub type SharedSegment = Arc<RwLock<Segment>>;

#[derive(Clone, Debug)]
pub struct SnapshotConfig {
    pub path: PathBuf,
    pub interval: Duration,
    pub max_ops: u64,
    pub check_every: Duration,
    pub retain_last: usize,
}

impl SnapshotConfig {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            interval: Duration::from_secs(60),
            max_ops: 10_000,
            check_every: Duration::from_secs(1),
            retain_last: 2,
        }
    }
}

pub struct SnapshotterHandle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl SnapshotterHandle {
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for SnapshotterHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub fn start_background_snapshots(
    segment: SharedSegment,
    config: SnapshotConfig,
) -> SnapshotterHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let join = thread::spawn(move || {
        let mut last_snapshot = Instant::now();
        let mut last_ops = 0u64;
        let check_every = if config.check_every.is_zero() {
            Duration::from_secs(1)
        } else {
            config.check_every
        };

        loop {
            if stop_thread.load(Ordering::Relaxed) {
                break;
            }

            let mut snapshot = None;
            let ops_now;
            {
                let guard = match segment.read() {
                    Ok(guard) => guard,
                    Err(e) => {
                        log::warn!(target: "segment::snapshot", "snapshotter lock poisoned: {}", e);
                        break;
                    }
                };
                ops_now = guard.op_count();
                let time_due =
                    !config.interval.is_zero() && last_snapshot.elapsed() >= config.interval;
                let ops_due =
                    config.max_ops > 0 && ops_now.saturating_sub(last_ops) >= config.max_ops;
                if time_due || ops_due {
                    snapshot = Some((guard.build_snapshot(), guard.snapshot_metadata()));
                }
            }

            if let Some((snapshot, metadata)) = snapshot {
                let rotated = rotate_snapshot(&config.path, config.retain_last);
                if let Err(err) = &rotated {
                    log::warn!(
                        target: "segment::snapshot",
                        "snapshot retention rotate failed: {}",
                        err
                    );
                }
                let res = Segment::persist_snapshot(&snapshot, Some(&metadata), &config.path);
                match &res {
                    Ok(()) => {
                        last_snapshot = Instant::now();
                        last_ops = ops_now;
                        if let Err(err) = prune_snapshots(&config.path, config.retain_last) {
                            log::warn!(
                                target: "segment::snapshot",
                                "snapshot retention cleanup failed: {}",
                                err
                            );
                        }
                    }
                    Err(DBError::SerializationError(err)) => {
                        log::warn!(
                            target: "segment::snapshot",
                            "snapshot serialization failed: {}",
                            err
                        );
                    }
                    Err(err) => {
                        log::warn!(target: "segment::snapshot", "snapshot write failed: {}", err);
                    }
                }
                if res.is_err()
                    && let Ok(Some(rotated_path)) = rotated
                    && !config.path.exists()
                {
                    let _ = std::fs::rename(rotated_path, &config.path);
                }
            }

            thread::sleep(check_every);
        }
    });

    SnapshotterHandle {
        stop,
        join: Some(join),
    }
}

fn rotate_snapshot(path: &PathBuf, retain_last: usize) -> Result<Option<PathBuf>, DBError> {
    if retain_last == 0 || !path.exists() {
        return Ok(None);
    }
    let file_name = match path.file_name().and_then(|f| f.to_str()) {
        Some(name) => name,
        None => return Ok(None),
    };
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let rotated_name = format!("{}.{}", file_name, stamp);
    let rotated_path = path.with_file_name(rotated_name);
    std::fs::rename(path, &rotated_path)?;
    Ok(Some(rotated_path))
}

fn prune_snapshots(path: &Path, retain_last: usize) -> Result<(), DBError> {
    if retain_last == 0 {
        return Ok(());
    }
    let Some(dir) = path.parent() else {
        return Ok(());
    };
    let base = match path.file_name().and_then(|f| f.to_str()) {
        Some(name) => name.to_string(),
        None => return Ok(()),
    };
    let prefix = format!("{}.", base);
    let mut entries: Vec<(u128, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        if !name.starts_with(&prefix) {
            continue;
        }
        let suffix = &name[prefix.len()..];
        if let Ok(ts) = suffix.parse::<u128>() {
            entries.push((ts, path));
        }
    }
    entries.sort_by(|a, b| b.0.cmp(&a.0));
    for (_ts, path) in entries.into_iter().skip(retain_last) {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}
