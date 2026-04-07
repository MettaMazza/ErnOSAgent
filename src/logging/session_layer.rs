//! Custom `tracing`-compatible writer that routes log output to a per-session file.
//!
//! The writer can be atomically swapped to a new file when the user switches sessions,
//! ensuring no log entries are lost during the transition.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// A thread-safe log file writer that can be rotated to a new session directory.
///
/// Implements `std::io::Write` so it can be used as a `tracing_appender` writer target.
pub struct SessionLogWriter {
    inner: Mutex<SessionLogInner>,
}

struct SessionLogInner {
    file: File,
    path: PathBuf,
}

impl SessionLogWriter {
    /// Create a new session log writer targeting a file in `session_dir`.
    ///
    /// The log file is named `{timestamp}.jsonl` based on the current time.
    pub fn new(session_dir: &Path) -> Result<Self> {
        let (file, path) = Self::open_log_file(session_dir)?;
        Ok(Self {
            inner: Mutex::new(SessionLogInner { file, path }),
        })
    }

    /// Rotate to a new session directory. Flushes the current file and opens a new one.
    pub fn rotate(&self, new_session_dir: &Path) -> Result<()> {
        let (new_file, new_path) = Self::open_log_file(new_session_dir)?;

        let mut inner = self.inner.lock().map_err(|e| {
            anyhow::anyhow!("Log writer lock poisoned during rotation: {}", e)
        })?;

        // Flush current file before switching
        inner
            .file
            .flush()
            .context("Failed to flush log file during rotation")?;

        inner.file = new_file;
        inner.path = new_path;

        Ok(())
    }

    /// Get the path of the current log file.
    pub fn current_path(&self) -> PathBuf {
        self.inner
            .lock()
            .map(|inner| inner.path.clone())
            .unwrap_or_default()
    }

    fn open_log_file(session_dir: &Path) -> Result<(File, PathBuf)> {
        let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
        let filename = format!("{}.jsonl", timestamp);
        let path = session_dir.join(&filename);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("Failed to open log file: {}", path.display()))?;

        Ok((file, path))
    }
}

impl Write for SessionLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        inner.file.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        inner.file.flush()
    }
}

/// `tracing_appender::non_blocking` requires `Write` on `&Self` (shared reference).
impl Write for &SessionLogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        inner.file.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        inner.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_new_creates_file() {
        let tmp = TempDir::new().unwrap();
        let writer = SessionLogWriter::new(tmp.path()).unwrap();
        let path = writer.current_path();
        assert!(path.exists());
        assert!(path.extension().map_or(false, |e| e == "jsonl"));
    }

    #[test]
    fn test_write_appends_to_file() {
        let tmp = TempDir::new().unwrap();
        let mut writer = SessionLogWriter::new(tmp.path()).unwrap();
        let path = writer.current_path();

        writer.write_all(b"test log line 1\n").unwrap();
        writer.write_all(b"test log line 2\n").unwrap();
        writer.flush().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("test log line 1"));
        assert!(content.contains("test log line 2"));
    }

    #[test]
    fn test_rotate_switches_file() {
        let tmp = TempDir::new().unwrap();
        let dir1 = tmp.path().join("sess-1");
        let dir2 = tmp.path().join("sess-2");
        std::fs::create_dir_all(&dir1).unwrap();
        std::fs::create_dir_all(&dir2).unwrap();

        let mut writer = SessionLogWriter::new(&dir1).unwrap();
        writer.write_all(b"before rotation\n").unwrap();
        writer.flush().unwrap();

        let path1 = writer.current_path();
        writer.rotate(&dir2).unwrap();

        writer.write_all(b"after rotation\n").unwrap();
        writer.flush().unwrap();

        let path2 = writer.current_path();

        // First file has pre-rotation content
        let content1 = std::fs::read_to_string(&path1).unwrap();
        assert!(content1.contains("before rotation"));

        // Second file has post-rotation content
        let content2 = std::fs::read_to_string(&path2).unwrap();
        assert!(content2.contains("after rotation"));
    }

    #[test]
    fn test_shared_ref_write() {
        let tmp = TempDir::new().unwrap();
        let writer = SessionLogWriter::new(tmp.path()).unwrap();
        let path = writer.current_path();

        // Write via shared reference (as tracing_appender does)
        let writer_ref: &SessionLogWriter = &writer;
        let mut w = writer_ref;
        w.write_all(b"shared ref write\n").unwrap();
        w.flush().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("shared ref write"));
    }
}
