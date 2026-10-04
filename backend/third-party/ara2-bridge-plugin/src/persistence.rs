//! Complete and filtered persistence adapters over call-scoped host archives.

use crate::{HostArchiveReader, HostArchiveWriter, PartialPersistence, Persistence};
use ara2_bridge_core::{AraError, FilterSelection, RestoreFilter, StoreFilter};

/// Enforces dedicated restore/store call scopes around an application persistence delegate.
pub struct PersistenceAdapter<P> {
    delegate: P,
    restoring: bool,
}

impl<P> PersistenceAdapter<P> {
    /// Wraps a persistence delegate in an idle lifecycle.
    pub const fn new(delegate: P) -> Self {
        Self {
            delegate,
            restoring: false,
        }
    }

    /// Borrows the underlying delegate.
    pub const fn delegate(&self) -> &P {
        &self.delegate
    }

    /// Mutably borrows the underlying delegate.
    pub fn delegate_mut(&mut self) -> &mut P {
        &mut self.delegate
    }
}

impl<P: Persistence> PersistenceAdapter<P> {
    /// Begins a generation-1 complete-document restore and consumes the current archive bytes.
    pub fn begin_restore(&mut self, reader: HostArchiveReader<'_>) -> Result<(), AraError> {
        if self.restoring {
            return Err(AraError::InvalidState("document restore is already active"));
        }
        let mut bytes = vec![0_u8; reader.len()];
        reader.read_at(0, &mut bytes)?;
        self.delegate.restore_document(&bytes)?;
        self.restoring = true;
        Ok(())
    }

    /// Ends a balanced generation-1 complete-document restore.
    pub fn end_restore(&mut self) -> Result<(), AraError> {
        if !self.restoring {
            return Err(AraError::InvalidState("document restore is not active"));
        }
        self.restoring = false;
        Ok(())
    }

    /// Stores the complete document into the call-scoped host writer.
    pub fn store_document(&mut self, writer: HostArchiveWriter<'_>) -> Result<(), AraError> {
        let bytes = self.delegate.store_document()?;
        writer.write_at(0, &bytes)
    }
}

impl<P: PartialPersistence> PersistenceAdapter<P> {
    /// Restores an ARA 2 validated object subset from the current archive.
    pub fn restore_objects(
        &mut self,
        filter: &FilterSelection<RestoreFilter>,
        reader: HostArchiveReader<'_>,
    ) -> Result<(), AraError> {
        let mut bytes = vec![0_u8; reader.len()];
        reader.read_at(0, &mut bytes)?;
        self.delegate.restore_objects(filter, &bytes)
    }

    /// Stores an ARA 2 validated object subset into the current archive.
    pub fn store_objects(
        &mut self,
        filter: &FilterSelection<StoreFilter>,
        writer: HostArchiveWriter<'_>,
    ) -> Result<(), AraError> {
        let bytes = self.delegate.store_objects(filter)?;
        writer.write_at(0, &bytes)
    }
}
