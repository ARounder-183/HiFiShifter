//! Safe authoring runtime for ARA plug-ins.
//!
//! # Role and boundaries
//!
//! This crate owns ARA factories, document-controller dispatch, plug-in model state, content,
//! persistence, analysis, processing metadata, and extension roles. It depends on core validation
//! and the raw ABI but never on the host runtime, companion formats, or testkit. Builders and guards
//! have **No direct C counterpart**; they support the `ARAFactory`,
//! `ARADocumentControllerInterface`, and `ARAPlugInExtensionInstance` contracts.
//!
//! # Lifecycle and threading
//!
//! Retain [`PluginRegistry`] for dynamic-library lifetime. A factory creates one [`PluginRuntime`]
//! per document. Model mutation occurs inside [`EditSession`] on the admitted model thread; reader,
//! analysis, render, and extension lifetimes are explicit. Callback panics are contained and poison
//! only the affected controller. Realtime helpers do not make arbitrary user callbacks realtime-safe.
//!
//! # Features and platforms
//!
//! The runtime has no format feature and needs no SDK checkout. CLAP, VST3, and Audio Unit adapters
//! live in `ara2-bridge-companion`; generation-1 availability follows the target ABI.
//!
//! # Compatibility and licensing
//!
//! The crate targets Rust 1.82 and ARA through 2.3 Final. It is MIT OR Apache-2.0 and preserves
//! Apache-2.0 provenance for behavior derived from Celemony's SDK examples.
//!
//! # Example
//!
//! ```
//! use ara2_bridge_plugin::FactoryBuilder;
//!
//! let factory = FactoryBuilder::new("org.example.ara", "org.example.archive")
//!     .display("Example", "Example Audio", "https://example.invalid", "2.0")
//!     .build()?;
//! assert_eq!(factory.id(), "org.example.ara");
//! # Ok::<(), ara2_bridge_core::AraError>(())
//! ```
//!
//! See the workspace plug-in specification and the upstream
//! [ARA API](https://github.com/Celemony/ARA_API).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::missing_safety_doc)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod analysis;
mod builder;
mod content;
mod controller;
mod entry;
mod extension;
mod factory;
pub mod ffi;
mod host;
mod model;
mod persistence;
mod processing;
mod realtime;
mod runtime;
mod traits;
mod updates;

pub use analysis::{AnalysisCoordinator, AnalysisEmitter, AnalysisProgress};
pub use builder::{FactoryBuilder, FactoryCapabilities, PluginRegistry, PluginRegistryBuilder};
pub use content::{ContentObject, ContentReaderSnapshot, ContentSnapshot};
pub use entry::PluginEntry;
pub use extension::{
    ExtensionBinding, ExtensionControllerLease, ExtensionRoles, ExtensionViewSelection,
};
pub use factory::Factory;
pub use ffi::{
    document_controller_interface, CallbackContract, ControllerCapabilities, ControllerInterface,
    Delegate, PLUGIN_CONTRACT_TESTS, PLUGIN_DELEGATES,
};
pub use host::{
    ArchiveAccess, AudioAccess, HostArchiveReader, HostArchiveWriter, HostAudioReader,
    HostAudioSourceRef, HostClients, HostContentReader, HostContentScope, HostMusicalContextRef,
    ModelUpdateAccess, PlaybackAccess, SampleFormat,
};
pub use persistence::PersistenceAdapter;
pub use processing::{AudioFileChunk, Plugin, PluginBuilder, SemanticCapabilities};
pub use realtime::RealtimeHeadTailAdapter;
pub use runtime::{CreateContext, EditSession, PluginModel, PluginRuntime};
pub use traits::{
    AnalysisProvider, AudioModifications, AudioSources, ContentProvider, DocumentLifecycle,
    MusicalContexts, PartialPersistence, Persistence, PlaybackRegions, RegionSequences,
};
pub use updates::{UpdateEmitter, UpdateNotification, UpdateOrigin, UpdateTracker};
