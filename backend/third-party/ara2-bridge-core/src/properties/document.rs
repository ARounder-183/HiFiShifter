//! Owned document properties.

use super::{copy_optional_display, display_string, write_raw, zeroed_raw, FfiProperties};
use crate::{AraError, SizedInput};
use ara2_bridge_sys::{layout, ARADocumentProperties, ARASize, ARAUtf8String};
use std::ffi::CString;
use std::mem::offset_of;
use std::pin::Pin;

/// Owned document display properties.
#[derive(Clone, Debug)]
pub struct DocumentProperties {
    name: Option<CString>,
}

impl DocumentProperties {
    /// Creates document properties, copying the optional display name.
    pub fn new(name: Option<&str>) -> Result<Self, AraError> {
        Ok(Self {
            name: display_string(name)?,
        })
    }

    /// Copies an ephemeral packed ARA document property record.
    ///
    /// # Safety
    ///
    /// `pointer` and any represented non-null string must satisfy [`SizedInput::from_ptr`]'s
    /// caller-valid storage precondition and remain readable for this call.
    pub unsafe fn copy_from_ffi(pointer: *const ARADocumentProperties) -> Result<Self, AraError> {
        // SAFETY: forwarded from this function's caller contract.
        let input = unsafe { SizedInput::from_ptr(pointer)? };
        // SAFETY: generated offset/type/extent describe `name`; the sized input is validated.
        let name = unsafe {
            input.copy_field::<ARAUtf8String>(
                offset_of!(ARADocumentProperties, name),
                layout::ARADOCUMENT_PROPERTIES_NAME,
            )?
        };
        // SAFETY: the outer contract covers represented nested string storage.
        let name = unsafe { copy_optional_display(name)? };
        Ok(Self { name })
    }

    /// Returns the optional display name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().map(|name| {
            name.to_str()
                .expect("document display names originate from UTF-8")
        })
    }

    /// Builds a pinned raw property record borrowing this owned backing.
    pub fn as_ffi(&self) -> Pin<Box<FfiProperties<'_, ARADocumentProperties>>> {
        // SAFETY: every field of this raw property type accepts the zero bit pattern.
        let mut raw = unsafe { zeroed_raw::<ARADocumentProperties>() };
        // SAFETY: offsets and types are generated from `ARADocumentProperties`.
        unsafe {
            write_raw(
                &mut raw,
                offset_of!(ARADocumentProperties, structSize),
                layout::ARADOCUMENT_PROPERTIES_NAME as ARASize,
            );
            write_raw(
                &mut raw,
                offset_of!(ARADocumentProperties, name),
                self.name
                    .as_ref()
                    .map_or(std::ptr::null(), |name| name.as_ptr()),
            );
        }
        FfiProperties::pin(raw)
    }
}
