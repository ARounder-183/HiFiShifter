//! Checked copying of caller-owned foreign arrays.

use crate::AraError;
use std::mem::{align_of, size_of};

/// An aligned owned copy of a caller-owned foreign array.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForeignSlice<T> {
    values: Vec<T>,
}

impl<T: Copy> ForeignSlice<T> {
    /// Validates and copies `count` elements from foreign storage.
    ///
    /// # Safety
    ///
    /// When `count` is nonzero, `pointer` must be aligned and readable for `count` initialized
    /// values of `T`. The allocation must remain live for the duration of this call. A null pointer
    /// is permitted only when `count` is zero.
    pub unsafe fn copy_from_raw(pointer: *const T, count: usize) -> Result<Self, AraError> {
        if count == 0 {
            return Ok(Self { values: Vec::new() });
        }
        if pointer.is_null() {
            return Err(AraError::InvalidArgument("null array with nonzero count"));
        }
        let element_size = size_of::<T>();
        if element_size == 0 {
            return Err(AraError::InvalidArgument("zero-sized FFI array element"));
        }
        let extent = count
            .checked_mul(element_size)
            .filter(|extent| *extent <= isize::MAX as usize)
            .ok_or(AraError::InvalidArgument("array extent overflow"))?;
        let _ = extent;
        if pointer as usize % align_of::<T>() != 0 {
            return Err(AraError::InvalidArgument("misaligned array pointer"));
        }
        // SAFETY: the caller guarantees an aligned initialized extent; arithmetic and the Rust
        // slice maximum have been checked above. The data is copied before returning.
        let values = unsafe { std::slice::from_raw_parts(pointer, count) }.to_vec();
        Ok(Self { values })
    }

    /// Returns the aligned owned values.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    /// Consumes the wrapper and returns the aligned owned values.
    pub fn into_vec(self) -> Vec<T> {
        self.values
    }
}
