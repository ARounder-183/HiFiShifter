//! Target-aware representation of released ARA API generations.

use crate::AraError;
use ara2_bridge_sys::ARAAPIGeneration;

/// A released ARA API generation in wire-order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(i32)]
pub enum ApiGeneration {
    /// Private ARA 1.0 draft compatibility generation.
    V1Draft = 1,
    /// Final ARA 1.0 generation.
    V1Final = 2,
    /// Transitional ARA 2.0 draft generation.
    V2Draft = 3,
    /// Final ARA 2.0 generation.
    V2Final = 4,
    /// Released ARA 2.x development generation.
    V2xDraft = 5,
    /// Final ARA 2.3 generation.
    V23Final = 6,
}

impl ApiGeneration {
    /// Every released generation, in ascending wire order.
    pub const ALL: [Self; 6] = [
        Self::V1Draft,
        Self::V1Final,
        Self::V2Draft,
        Self::V2Final,
        Self::V2xDraft,
        Self::V23Final,
    ];

    /// Returns whether the released headers expose this generation on the current target family.
    pub const fn supported_on_target(self) -> bool {
        cfg!(not(target_arch = "aarch64")) || (self as i32) >= Self::V2Final as i32
    }

    /// Returns the raw ARA generation value.
    pub const fn as_raw(self) -> ARAAPIGeneration {
        self as ARAAPIGeneration
    }

    /// Validates and converts a raw ARA generation value.
    pub fn try_from_raw(raw: ARAAPIGeneration) -> Result<Self, AraError> {
        match raw {
            1 => Ok(Self::V1Draft),
            2 => Ok(Self::V1Final),
            3 => Ok(Self::V2Draft),
            4 => Ok(Self::V2Final),
            5 => Ok(Self::V2xDraft),
            6 => Ok(Self::V23Final),
            _ => Err(AraError::InvalidArgument("unknown API generation")),
        }
    }
}
