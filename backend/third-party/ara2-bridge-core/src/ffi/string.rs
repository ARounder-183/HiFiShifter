//! Bounded copying and validation of ARA C strings.

use crate::AraError;
use std::os::raw::c_char;

/// An owned string copied from bounded caller-owned C storage.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ForeignStr(String);

impl ForeignStr {
    /// Copies a bounded NUL-terminated UTF-8 display string.
    ///
    /// # Safety
    ///
    /// `pointer` must be non-null and readable byte-by-byte through a NUL terminator within
    /// `maximum_bytes`. The storage must remain live for the duration of this call.
    pub unsafe fn copy_display(
        pointer: *const c_char,
        maximum_bytes: usize,
    ) -> Result<Self, AraError> {
        // SAFETY: forwarded from this function's bounded caller-valid string contract.
        let bytes = unsafe { copy_c_bytes(pointer, maximum_bytes)? };
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| AraError::InvalidArgument("display string is not UTF-8"))?;
        Ok(Self(text.to_owned()))
    }

    /// Copies a bounded NUL-terminated nonempty seven-bit ARA persistent ID.
    ///
    /// # Safety
    ///
    /// `pointer` must be non-null and readable byte-by-byte through a NUL terminator within
    /// `maximum_bytes`. The storage must remain live for the duration of this call.
    pub unsafe fn copy_persistent_id(
        pointer: *const c_char,
        maximum_bytes: usize,
    ) -> Result<Self, AraError> {
        // SAFETY: forwarded from this function's bounded caller-valid string contract.
        let bytes = unsafe { copy_c_bytes(pointer, maximum_bytes)? };
        if bytes.is_empty() || !bytes.is_ascii() {
            return Err(AraError::InvalidArgument(
                "persistent ID must be nonempty ASCII",
            ));
        }
        // ASCII is valid UTF-8.
        Ok(Self(String::from_utf8(bytes).expect("ASCII is UTF-8")))
    }

    /// Returns the copied Rust string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the wrapper and returns the copied Rust string.
    pub fn into_string(self) -> String {
        self.0
    }
}

unsafe fn copy_c_bytes(pointer: *const c_char, maximum_bytes: usize) -> Result<Vec<u8>, AraError> {
    if pointer.is_null() {
        return Err(AraError::InvalidArgument("null string pointer"));
    }
    if maximum_bytes == 0 {
        return Err(AraError::InvalidArgument("string bound must be nonzero"));
    }
    let pointer = pointer.cast::<u8>();
    let mut bytes = Vec::new();
    for offset in 0..maximum_bytes {
        // SAFETY: the caller guarantees readable storage through a terminator within the bound.
        let byte = unsafe { pointer.add(offset).read() };
        if byte == 0 {
            return Ok(bytes);
        }
        bytes.push(byte);
    }
    Err(AraError::InvalidArgument(
        "string is not terminated within bound",
    ))
}
