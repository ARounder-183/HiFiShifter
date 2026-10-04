//! Owned aligned mirrors and pinned outbound guards for ARA property records.

mod document;
mod model;
mod selection;

use crate::{AraError, ForeignStr};
use ara2_bridge_sys::access;
use std::ffi::CString;
use std::marker::{PhantomData, PhantomPinned};
use std::os::raw::c_char;
use std::pin::Pin;

pub use document::DocumentProperties;
pub use model::{
    AudioModificationKind, AudioModificationProperties, AudioSourceKind, AudioSourceProperties,
    Color, MusicalContextKind, MusicalContextProperties, PlaybackRegionKind,
    PlaybackRegionProperties, RawChannelArrangement, RegionSequenceKind, RegionSequenceProperties,
};
pub use selection::{ContentTimeRange, ViewSelection};

pub(crate) const MAX_PROPERTY_STRING_BYTES: usize = 1_048_576;

/// A pinned call-scoped raw property record borrowing its owner's backing allocations.
///
/// The guard is allocated and pinned by property builders. Its raw record and all internal
/// pointers remain stable until the guard is dropped.
pub struct FfiProperties<'a, T> {
    raw: T,
    _backing: PhantomData<&'a ()>,
    _pinned: PhantomPinned,
}

impl<'a, T> FfiProperties<'a, T> {
    pub(crate) fn pin(raw: T) -> Pin<Box<Self>> {
        Box::pin(Self {
            raw,
            _backing: PhantomData,
            _pinned: PhantomPinned,
        })
    }

    /// Returns a stable pointer to the raw call-scoped record.
    pub fn as_ptr(self: Pin<&Self>) -> *const T {
        std::ptr::addr_of!(self.get_ref().raw)
    }

    /// Returns every initialized byte of the raw record, including zeroed padding.
    pub fn raw_bytes(self: Pin<&Self>) -> &[u8] {
        let pointer = self.as_ptr().cast::<u8>();
        // SAFETY: all `FfiProperties` constructors start from zeroed valid raw storage and write
        // fields by value, so padding is initialized. Pinning keeps the record live and stable.
        unsafe { std::slice::from_raw_parts(pointer, std::mem::size_of::<T>()) }
    }
}

pub(crate) fn display_string(value: Option<&str>) -> Result<Option<CString>, AraError> {
    value
        .map(|value| {
            CString::new(value)
                .map_err(|_| AraError::InvalidArgument("display string contains NUL"))
        })
        .transpose()
}

pub(crate) fn persistent_id(value: &str) -> Result<CString, AraError> {
    if value.is_empty() || !value.is_ascii() {
        return Err(AraError::InvalidArgument(
            "persistent ID must be nonempty ASCII",
        ));
    }
    CString::new(value).map_err(|_| AraError::InvalidArgument("persistent ID contains NUL"))
}

pub(crate) unsafe fn copy_optional_display(
    pointer: *const c_char,
) -> Result<Option<CString>, AraError> {
    if pointer.is_null() {
        return Ok(None);
    }
    // SAFETY: the enclosing property-copy precondition includes readable nested strings.
    let value = unsafe { ForeignStr::copy_display(pointer, MAX_PROPERTY_STRING_BYTES)? };
    display_string(Some(value.as_str()))
}

pub(crate) unsafe fn copy_required_id(pointer: *const c_char) -> Result<CString, AraError> {
    // SAFETY: the enclosing property-copy precondition includes readable nested strings; the
    // foreign-string validator rejects null and unterminated pointers.
    let value = unsafe { ForeignStr::copy_persistent_id(pointer, MAX_PROPERTY_STRING_BYTES)? };
    persistent_id(value.as_str())
}

pub(crate) unsafe fn zeroed_raw<T>() -> T {
    // SAFETY: callers use only generated raw ARA structs whose fields all accept the zero bit
    // pattern. Starting zeroed also initializes every padding byte before field writes.
    unsafe { std::mem::MaybeUninit::<T>::zeroed().assume_init() }
}

pub(crate) unsafe fn write_raw<T, F>(raw: &mut T, offset: usize, value: F) {
    // SAFETY: callers provide a generated offset and the matching raw field type. The record is
    // exclusively borrowed and fully initialized for overwriting.
    unsafe { access::write_field(std::ptr::from_mut(raw).cast::<u8>(), offset, value) }
}
