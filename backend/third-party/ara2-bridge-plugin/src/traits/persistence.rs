//! Required and optional persistence capability traits.

use ara2_bridge_core::{AraError, FilterSelection, RestoreFilter, StoreFilter};

/// Stores and restores the complete document state.
pub trait Persistence {
    /// Restores a complete document archive.
    fn restore_document(&mut self, _bytes: &[u8]) -> Result<(), AraError>;

    /// Produces a complete document archive.
    fn store_document(&mut self) -> Result<Vec<u8>, AraError>;
}

/// Stores and restores ARA 2 filtered object subsets.
pub trait PartialPersistence: Persistence {
    /// Restores only the objects selected by the validated filter.
    fn restore_objects(
        &mut self,
        _filter: &FilterSelection<RestoreFilter>,
        _bytes: &[u8],
    ) -> Result<(), AraError>;

    /// Stores only the objects selected by the validated filter.
    fn store_objects(
        &mut self,
        _filter: &FilterSelection<StoreFilter>,
    ) -> Result<Vec<u8>, AraError>;
}
