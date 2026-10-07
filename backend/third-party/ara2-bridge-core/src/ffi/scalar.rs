//! Canonical scalar conversions for the ARA ABI.

use ara2_bridge_sys::{kARAFalse, kARATrue, ARABool};

/// A safe Rust boolean with canonical ARA wire conversion.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AraBool(bool);

impl AraBool {
    /// Creates a safe ARA boolean from a Rust value.
    pub const fn new(value: bool) -> Self {
        Self(value)
    }

    /// Applies ARA's rule that every nonzero inbound integer means true.
    pub const fn from_raw(raw: ARABool) -> Self {
        Self(raw != 0)
    }

    /// Returns the Rust value.
    pub const fn get(self) -> bool {
        self.0
    }

    /// Returns exactly [`ara2_bridge_sys::kARAFalse`] or [`ara2_bridge_sys::kARATrue`].
    pub const fn into_raw(self) -> ARABool {
        if self.0 {
            kARATrue
        } else {
            kARAFalse
        }
    }
}

impl From<bool> for AraBool {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

impl From<AraBool> for bool {
    fn from(value: AraBool) -> Self {
        value.get()
    }
}
