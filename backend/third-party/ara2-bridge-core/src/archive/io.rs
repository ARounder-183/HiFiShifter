//! Position-based archive I/O.

use crate::{AraError, ArchiveError};
use std::ops::Range;

/// Random-access exact archive reads.
pub trait ReadAt {
    /// Returns the archive length.
    fn len(&self) -> Result<u64, AraError>;

    /// Returns whether the archive is empty.
    fn is_empty(&self) -> Result<bool, AraError> {
        self.len().map(|length| length == 0)
    }

    /// Returns an optional nonempty ASCII archive identifier.
    fn archive_id(&self) -> Option<&str> {
        None
    }

    /// Reads exactly `out.len()` bytes starting at `position`.
    fn read_at(&self, position: u64, out: &mut [u8]) -> Result<(), AraError>;
}

/// Random-access exact archive writes.
pub trait WriteAt {
    /// Writes all bytes starting at `position`, growing sparse storage as needed.
    fn write_at(&mut self, position: u64, data: &[u8]) -> Result<(), AraError>;

    /// Completes pending transport writes.
    fn flush(&mut self) -> Result<(), AraError> {
        Ok(())
    }
}

/// In-memory random-access archive used by applications and tests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryArchive {
    bytes: Vec<u8>,
    archive_id: Option<String>,
}

impl From<Vec<u8>> for MemoryArchive {
    fn from(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            archive_id: None,
        }
    }
}

impl MemoryArchive {
    /// Creates memory storage carrying a validated archive identifier.
    pub fn with_id(bytes: Vec<u8>, archive_id: impl Into<String>) -> Result<Self, AraError> {
        let archive_id = archive_id.into();
        if archive_id.is_empty() || !archive_id.is_ascii() || archive_id.contains('\0') {
            return Err(AraError::Archive(ArchiveError::InvalidFilter(
                "archive ID must be nonempty ASCII",
            )));
        }
        Ok(Self {
            bytes,
            archive_id: Some(archive_id),
        })
    }

    /// Returns the archive bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes the archive and returns its bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl ReadAt for MemoryArchive {
    fn len(&self) -> Result<u64, AraError> {
        u64::try_from(self.bytes.len()).map_err(|_| AraError::ArchiveTooLargeForTarget)
    }

    fn archive_id(&self) -> Option<&str> {
        self.archive_id.as_deref()
    }

    fn read_at(&self, position: u64, out: &mut [u8]) -> Result<(), AraError> {
        let range = checked_range(position, out.len())?;
        let source = self
            .bytes
            .get(range)
            .ok_or(AraError::Archive(ArchiveError::OutOfBounds))?;
        out.copy_from_slice(source);
        Ok(())
    }
}

impl WriteAt for MemoryArchive {
    fn write_at(&mut self, position: u64, data: &[u8]) -> Result<(), AraError> {
        let range = checked_range(position, data.len())?;
        if range.end > self.bytes.len() {
            self.bytes.resize(range.end, 0);
        }
        self.bytes[range].copy_from_slice(data);
        Ok(())
    }
}

fn checked_range(position: u64, length: usize) -> Result<Range<usize>, AraError> {
    let length = u64::try_from(length).map_err(|_| AraError::ArchiveTooLargeForTarget)?;
    let end = position
        .checked_add(length)
        .ok_or(AraError::Archive(ArchiveError::RangeOverflow))?;
    if position > usize::MAX as u64 || end > usize::MAX as u64 {
        return Err(AraError::ArchiveTooLargeForTarget);
    }
    Ok(position as usize..end as usize)
}

/// Validates finite monotonic archive-operation progress.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArchiveProgress {
    last: Option<f32>,
}

impl ArchiveProgress {
    /// Records the next progress value.
    pub fn update(&mut self, progress: f32) -> Result<(), AraError> {
        if !progress.is_finite()
            || !(0.0..=1.0).contains(&progress)
            || self.last.is_some_and(|last| progress < last)
        {
            return Err(AraError::Archive(ArchiveError::InvalidProgress));
        }
        self.last = Some(progress);
        Ok(())
    }

    /// Returns the most recently accepted progress value.
    pub fn current(&self) -> Option<f32> {
        self.last
    }

    /// Starts a new independent operation.
    pub fn reset(&mut self) {
        self.last = None;
    }
}
