//! Mock ARA peers, deterministic fixtures, and shared conformance scenarios.
//!
//! # Role and boundaries
//!
//! This crate supplies a capability-rich Rust TestHost/TestPlugIn pair, named upstream-equivalent
//! scenarios, native interoperability harnesses, and coverage joins. Test helpers have **No direct C
//! counterpart**; they exercise the released ARA host, plug-in, and companion contracts. Production
//! crates never depend on this crate.
//!
//! # Lifecycle and threading
//!
//! Each [`TestHost`] owns its services and sequence-numbered trace. Scenarios construct, mutate,
//! validate, and tear down one document deterministically. Native assertions and C++ exceptions are
//! bounded diagnostics. Realtime and concurrency suites have dedicated runners and timeouts.
//!
//! # Features and platforms
//!
//! `cpp-interop` uses the project-local Celemony SDK installed by `scripts/install-ara-sdk.sh`.
//! `clap`, `vst3`, and `audio-unit-v2` forward to their companion adapters and use the same Cargo
//! SDK configuration. Default Rust scenarios need no SDK checkout.
//!
//! # Compatibility and licensing
//!
//! The crate targets Rust 1.82 and ARA through 2.3 Final. It is MIT OR Apache-2.0; fixtures and
//! upstream-derived scenarios retain source hashes and Apache-2.0 provenance.
//!
//! # Example
//!
//! ```
//! use ara2_bridge_core::ApiGeneration;
//! use ara2_bridge_testkit::TestHost;
//!
//! let host = TestHost::new(ApiGeneration::V23Final)?;
//! assert_eq!(host.generation(), ApiGeneration::V23Final);
//! # Ok::<(), ara2_bridge_core::AraError>(())
//! ```
//!
//! See `docs/conformance/` and the upstream
//! [ARA SDK examples](https://github.com/Celemony/ARA_SDK).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::missing_safety_doc)]
#![deny(clippy::undocumented_unsafe_blocks)]

pub mod coverage;
mod host;
#[cfg(feature = "cpp-interop")]
pub mod native;
mod plugin;
pub mod scenarios;

pub use host::{TestHost, TestHostEvent, TestHostTrace};

pub use plugin::{
    all_content_types, all_extension_roles, all_transformations, build_minimal_test_factory,
    build_minimal_test_plugin, build_test_extension, build_test_factory, build_test_plugin,
    test_audio_source_properties, test_update_scopes, TestPluginModel, TestPluginTrace,
};
