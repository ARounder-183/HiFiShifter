//! Error categories shared by host and plug-in runtimes.

/// Failures encountered while transporting or interpreting archive data.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ArchiveError {
    /// Archive position-plus-length arithmetic overflowed.
    #[error("archive range overflow")]
    RangeOverflow,
    /// An archive operation exceeded the available bytes.
    #[error("archive range is out of bounds")]
    OutOfBounds,
    /// Progress was non-finite, outside `0..=1`, or moved backwards.
    #[error("invalid archive progress")]
    InvalidProgress,
    /// The peer archive transport failed.
    #[error("archive transport failed: {0}")]
    Transport(&'static str),
    /// Archive bytes could not be decoded or validated.
    #[error("invalid archive data: {0}")]
    Decode(&'static str),
    /// A partial-persistence filter is internally inconsistent.
    #[error("invalid archive filter: {0}")]
    InvalidFilter(&'static str),
}

/// Failures encountered while discovering, binding, or tearing down companion APIs.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CompanionError {
    /// An ARA endpoint could not be discovered through the companion API.
    #[error("companion discovery failed: {0}")]
    Discovery(&'static str),
    /// A discovered endpoint could not be bound to the requested ARA role.
    #[error("companion binding failed: {0}")]
    Binding(&'static str),
    /// Companion and controller lifecycles were used in an invalid order.
    #[error("invalid companion lifecycle: {0}")]
    Lifecycle(&'static str),
    /// The companion mechanism is unavailable on the current platform.
    #[error("unsupported companion platform: {0}")]
    UnsupportedPlatform(&'static str),
}

/// Errors reported by safe ARA operations.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AraError {
    /// The peer supplied an invalid ABI representation.
    #[error("invalid ABI: {0}")]
    Abi(&'static str),
    /// An argument is invalid independently of runtime state.
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
    /// The operation is not legal in the current lifecycle state.
    #[error("invalid state: {0}")]
    InvalidState(&'static str),
    /// The operation was invoked from a thread forbidden by ARA.
    #[error("invalid thread: {0}")]
    InvalidThread(&'static str),
    /// The requested optional capability is unavailable.
    #[error("unsupported capability: {0}")]
    Unsupported(&'static str),
    /// Archive transport, filtering, or decoding failed.
    #[error(transparent)]
    Archive(#[from] ArchiveError),
    /// Companion API discovery, binding, or lifecycle management failed.
    #[error(transparent)]
    Companion(#[from] CompanionError),
    /// A foreign ARA peer reported failure.
    #[error("peer failure: {0}")]
    Peer(&'static str),
    /// A previous panic or unrecoverable invariant failure poisoned the instance.
    #[error("instance poisoned")]
    Poisoned,
    /// An archive position or extent cannot be represented on this target.
    #[error("archive too large for target")]
    ArchiveTooLargeForTarget,
}
