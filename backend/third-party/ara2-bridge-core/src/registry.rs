//! Bounded append-only storage for runtime-owned ARA reference cells.

use crate::{AraError, Handle, ModelRef};
use std::marker::PhantomData;
use std::num::{NonZeroU32, NonZeroU64};
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

struct RegistryCell<T> {
    value: Option<T>,
}

fn next_session() -> NonZeroU64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    loop {
        let candidate = NEXT.fetch_add(1, Ordering::Relaxed);
        if let Some(session) = NonZeroU64::new(candidate) {
            return session;
        }
    }
}

/// Shared identity for every typed registry owned by one document session.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RegistrySession(NonZeroU64);

impl RegistrySession {
    /// Allocates a process-unique document-session identity.
    pub fn new() -> Self {
        Self(next_session())
    }

    /// Returns the nonzero numeric identity.
    pub const fn get(self) -> NonZeroU64 {
        self.0
    }
}

impl Default for RegistrySession {
    fn default() -> Self {
        Self::new()
    }
}

/// A bounded registry of stable, never-reused handle cells.
///
/// Each insertion appends a separately allocated cell whose address remains stable as the registry
/// grows. Removal tombstones that cell before returning the value. Tombstones remain until the
/// registry is dropped, so stale handles can never name a later object in the same session.
pub struct Registry<K, T> {
    session: NonZeroU64,
    capacity: usize,
    cells: Vec<Box<RegistryCell<T>>>,
    _kind: PhantomData<fn(K) -> K>,
}

impl<K: 'static, T> Registry<K, T> {
    /// Default maximum number of cells retained by one document registry.
    pub const DEFAULT_CAPACITY: usize = 1_048_576;

    /// Creates an empty registry with a hard append-only cell cap.
    pub fn new(capacity: usize) -> Self {
        Self::in_session(RegistrySession::new(), capacity)
    }

    /// Creates an empty typed registry sharing a document identity with sibling registries.
    pub fn in_session(session: RegistrySession, capacity: usize) -> Self {
        Self {
            session: session.get(),
            capacity: capacity.min(u32::MAX as usize),
            cells: Vec::new(),
            _kind: PhantomData,
        }
    }

    /// Creates an empty registry with [`Self::DEFAULT_CAPACITY`].
    pub fn with_default_capacity() -> Self {
        Self::new(Self::DEFAULT_CAPACITY)
    }

    /// Returns this registry's session identity.
    pub const fn session(&self) -> NonZeroU64 {
        self.session
    }

    /// Returns the shared typed document-session identity.
    pub const fn session_id(&self) -> RegistrySession {
        RegistrySession(self.session)
    }

    /// Returns the configured maximum number of cells.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of cells, including tombstones.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Iterates over live values in stable insertion order, skipping tombstones.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.cells.iter().filter_map(|cell| cell.value.as_ref())
    }

    /// Iterates over live typed handles in stable insertion order, skipping tombstones.
    pub fn handles(&self) -> impl Iterator<Item = Handle<K>> + '_ {
        self.cells.iter().enumerate().filter_map(|(offset, cell)| {
            cell.value.as_ref()?;
            let index = NonZeroU32::new(u32::try_from(offset + 1).ok()?)?;
            Some(Handle::new(index, self.session))
        })
    }

    /// Appends a stable live cell and returns its typed identity.
    pub fn insert(&mut self, value: T) -> Result<Handle<K>, AraError> {
        if self.cells.len() >= self.capacity {
            return Err(AraError::InvalidState("registry capacity exhausted"));
        }
        let index = NonZeroU32::new(
            u32::try_from(self.cells.len() + 1)
                .map_err(|_| AraError::InvalidState("registry capacity exhausted"))?,
        )
        .ok_or(AraError::InvalidState("registry capacity exhausted"))?;
        self.cells
            .push(Box::new(RegistryCell { value: Some(value) }));
        Ok(Handle::new(index, self.session))
    }

    /// Borrows a live value after checking its session and tombstone state.
    pub fn get(&self, handle: Handle<K>) -> Result<&T, AraError> {
        let cell = self.checked_cell(handle)?;
        cell.value
            .as_ref()
            .ok_or(AraError::InvalidArgument("stale handle"))
    }

    /// Mutably borrows a live value after checking its session and tombstone state.
    pub fn get_mut(&mut self, handle: Handle<K>) -> Result<&mut T, AraError> {
        let cell = self.checked_cell_mut(handle)?;
        cell.value
            .as_mut()
            .ok_or(AraError::InvalidArgument("stale handle"))
    }

    /// Tombstones a live cell before returning its value to the caller.
    pub fn remove(&mut self, handle: Handle<K>) -> Result<T, AraError> {
        let cell = self.checked_cell_mut(handle)?;
        cell.value
            .take()
            .ok_or(AraError::InvalidArgument("stale handle"))
    }

    /// Returns the stable opaque address for a live cell.
    pub fn opaque_pointer(&self, handle: Handle<K>) -> Result<NonNull<c_void>, AraError> {
        let cell = self.checked_cell(handle)?;
        if cell.value.is_none() {
            return Err(AraError::InvalidArgument("stale handle"));
        }
        Ok(NonNull::from(cell).cast())
    }

    /// Creates a typed model reference backed by a live stable cell.
    pub fn model_ref(&self, handle: Handle<K>) -> Result<ModelRef<K>, AraError> {
        self.opaque_pointer(handle).map(ModelRef::new)
    }

    /// Recovers a live handle by address without dereferencing the foreign pointer.
    ///
    /// Unknown and null addresses are rejected by identity comparison against owned cells.
    pub fn handle_from_opaque(&self, pointer: *mut c_void) -> Result<Handle<K>, AraError> {
        let pointer =
            NonNull::new(pointer).ok_or(AraError::InvalidArgument("foreign handle pointer"))?;
        for (offset, cell) in self.cells.iter().enumerate() {
            if NonNull::from(cell.as_ref()).cast::<c_void>() == pointer {
                if cell.value.is_none() {
                    return Err(AraError::InvalidArgument("stale handle"));
                }
                let index = NonZeroU32::new(
                    u32::try_from(offset + 1)
                        .map_err(|_| AraError::InvalidArgument("foreign handle pointer"))?,
                )
                .ok_or(AraError::InvalidArgument("foreign handle pointer"))?;
                return Ok(Handle::new(index, self.session));
            }
        }
        Err(AraError::InvalidArgument("foreign handle pointer"))
    }

    fn checked_cell(&self, handle: Handle<K>) -> Result<&RegistryCell<T>, AraError> {
        if handle.session != self.session {
            return Err(AraError::InvalidArgument("foreign handle"));
        }
        self.cells
            .get(handle.index.get() as usize - 1)
            .map(Box::as_ref)
            .ok_or(AraError::InvalidArgument("foreign handle"))
    }

    fn checked_cell_mut(&mut self, handle: Handle<K>) -> Result<&mut RegistryCell<T>, AraError> {
        if handle.session != self.session {
            return Err(AraError::InvalidArgument("foreign handle"));
        }
        self.cells
            .get_mut(handle.index.get() as usize - 1)
            .map(Box::as_mut)
            .ok_or(AraError::InvalidArgument("foreign handle"))
    }
}

impl<K: 'static, T> Default for Registry<K, T> {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}
