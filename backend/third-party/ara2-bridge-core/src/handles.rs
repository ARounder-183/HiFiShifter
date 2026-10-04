//! Typed, session-bound identities for ARA model objects.

use crate::AraError;
use std::any::TypeId;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::{NonZeroU32, NonZeroU64};
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::rc::Rc;

/// An erased handle representation that retains its originating Rust kind.
///
/// Raw handles can be stored by generic infrastructure and converted back only to their original
/// kind with [`Handle::try_from_raw`]. Their fields are intentionally private.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RawHandle {
    index: NonZeroU32,
    session: NonZeroU64,
    kind: TypeId,
}

/// A typed identity owned by one document session.
///
/// Handles are copyable identifiers but deliberately neither `Send` nor `Sync`. The owning runtime
/// validates liveness and session membership on every registry access.
pub struct Handle<K> {
    pub(crate) index: NonZeroU32,
    pub(crate) session: NonZeroU64,
    pub(crate) _kind: PhantomData<fn(K) -> K>,
    pub(crate) _not_send_sync: PhantomData<Rc<()>>,
}

impl<K> Handle<K> {
    /// Returns the one-based append-only cell index.
    pub const fn index(self) -> NonZeroU32 {
        self.index
    }

    /// Returns the owning document-session identity.
    pub const fn session(self) -> NonZeroU64 {
        self.session
    }
}

impl<K: 'static> Handle<K> {
    /// Erases the compile-time kind while retaining a checked runtime kind tag.
    pub fn into_raw(self) -> RawHandle {
        RawHandle {
            index: self.index,
            session: self.session,
            kind: TypeId::of::<K>(),
        }
    }

    /// Restores a typed handle after validating its runtime kind tag.
    pub fn try_from_raw(raw: RawHandle) -> Result<Self, AraError> {
        if raw.kind != TypeId::of::<K>() {
            return Err(AraError::InvalidArgument("wrong handle kind"));
        }
        Ok(Self::new(raw.index, raw.session))
    }

    pub(crate) const fn new(index: NonZeroU32, session: NonZeroU64) -> Self {
        Self {
            index,
            session,
            _kind: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

impl<K> Clone for Handle<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for Handle<K> {}

impl<K> fmt::Debug for Handle<K> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Handle")
            .field("index", &self.index)
            .field("session", &self.session)
            .finish_non_exhaustive()
    }
}

impl<K> PartialEq for Handle<K> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.session == other.session
    }
}

impl<K> Eq for Handle<K> {}

impl<K> Hash for Handle<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.session.hash(state);
    }
}

/// A typed stable opaque pointer backed by a live registry cell.
///
/// Like [`Handle`], model references are neither `Send` nor `Sync`. Local references are created
/// by [`crate::Registry::model_ref`]. Foreign references received across the ARA ABI can be
/// admitted with [`ModelRef::from_raw`] when the caller can uphold the same stability contract.
pub struct ModelRef<K> {
    pointer: NonNull<c_void>,
    _kind: PhantomData<fn(K) -> K>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<K> ModelRef<K> {
    pub(crate) const fn new(pointer: NonNull<c_void>) -> Self {
        Self {
            pointer,
            _kind: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Admits a non-null foreign model reference into the typed graph API.
    ///
    /// # Safety
    ///
    /// `pointer` must identify a live object of kind `K`, remain stable for every use of the
    /// returned reference, and only be used on the thread permitted by its ARA owner.
    pub unsafe fn from_raw(pointer: *mut c_void) -> Result<Self, AraError> {
        NonNull::new(pointer)
            .map(Self::new)
            .ok_or(AraError::InvalidArgument("null model reference"))
    }

    /// Returns the non-null stable opaque address for ABI construction.
    pub const fn as_raw(self) -> *mut c_void {
        self.pointer.as_ptr()
    }
}

impl<K> Clone for ModelRef<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for ModelRef<K> {}

impl<K> fmt::Debug for ModelRef<K> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ModelRef")
            .field(&self.pointer)
            .finish()
    }
}

impl<K> PartialEq for ModelRef<K> {
    fn eq(&self, other: &Self) -> bool {
        self.pointer == other.pointer
    }
}

impl<K> Eq for ModelRef<K> {}

impl<K> Hash for ModelRef<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pointer.hash(state);
    }
}
