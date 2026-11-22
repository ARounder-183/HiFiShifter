//! Aggregating facade for the ARA 2.3 Rust workspace.
//!
//! # Role and boundaries
//!
//! The focused crates remain independently usable. This crate exposes them under stable module
//! names so applications can depend on one package without collapsing their safety boundaries.
//! The facade has **No direct C counterpart**; it selects the Rust authoring surfaces layered over
//! the upstream [ARA API](https://github.com/Celemony/ARA_API).
//!
//! # Lifecycle and threading
//!
//! The facade adds no runtime state. Ownership, model-thread admission, realtime rules, fallible
//! validation, and teardown remain documented by the selected focused crate.
//!
//! # Features and platforms
//!
//! `plugin` is the default. `host`, `testkit`, `clap`, `vst3`, and `audio-unit-v2` are additive;
//! `full-portable` combines plug-in, host, CLAP, and VST3, while `full-apple` adds Audio Unit v2.
//! VST3 requires `ARA_VST3_SDK_DIR`; Audio Unit v2 is Apple-only and requires
//! `ARA_AUDIO_UNIT_SDK_DIR`. Builds never download SDKs.
//!
//! # Compatibility and licensing
//!
//! Version 0.2 targets Rust 1.82 and ARA through 2.3 Final. See the migration guide for the removed
//! 0.1 raw-pointer API. Project crates are MIT OR Apache-2.0; companion SDK licenses are separate.
//!
//! # Example
//!
//! ```
//! use ara2_bridge::plugin::FactoryBuilder;
//!
//! let factory = FactoryBuilder::new("org.example.ara", "org.example.archive")
//!     .display("Example", "Example Audio", "https://example.invalid", "2.0")
//!     .build()?;
//! assert_eq!(factory.id(), "org.example.ara");
//! # Ok::<(), ara2_bridge::core::AraError>(())
//! ```
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::missing_safety_doc)]
#![deny(clippy::undocumented_unsafe_blocks)]

/// Raw, pregenerated ARA 2.3 C ABI bindings.
pub use ara2_bridge_sys as sys;

/// Shared safe types, validation, and dispatch infrastructure.
pub use ara2_bridge_core as core;

/// Safe ARA plug-in authoring runtime.
#[cfg(feature = "plugin")]
pub use ara2_bridge_plugin as plugin;

/// Safe ARA host authoring runtime.
#[cfg(feature = "host")]
pub use ara2_bridge_host as host;

/// CLAP, VST3, and Audio Unit companion adapters.
///
/// The bare `companion` feature exposes only the format-neutral binding layer
/// (`CompanionFactory`, `CompanionProcessorBinding`, `CompanionControllerBinding`,
/// `CompanionRoles`). It enables no format submodule, so it triggers no SDK checkout
/// and no C++ compilation; format crates add `vst3` / `clap` / `audio-unit-v2` on top
/// of it. (HiFiShifter local patch — see `PATCHED.md`.)
#[cfg(any(
    feature = "companion",
    feature = "clap",
    feature = "vst3",
    feature = "audio-unit-v2"
))]
pub use ara2_bridge_companion as companion;

/// Mock peers and conformance utilities.
#[cfg(feature = "testkit")]
pub use ara2_bridge_testkit as testkit;
