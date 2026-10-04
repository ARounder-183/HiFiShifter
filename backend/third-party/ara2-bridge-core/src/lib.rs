//! Shared validation, dispatch, and safety primitives for ARA host and plug-in runtimes.
//!
//! # Role and boundaries
//!
//! This crate owns the narrow boundary between caller-valid foreign storage and aligned Rust data.
//! Normal application code should use `ara2-bridge-plugin` or `ara2-bridge-host`; raw callback
//! authors must uphold every pointer extent documented on [`SizedInput::from_ptr`]. Opaque model
//! identities are runtime-owned [`Handle`] values backed by stable bounded [`Registry`] cells.
//! Factory generations remain local while [`AssertCoordinator`] shares the assertion-cell address
//! required by ARA. Callback implementations should use the panic-contained dispatch helpers.
//!
//! Bridge-native safe types in this crate have **No direct C counterpart**; their rustdoc names the
//! ARA record, callback, or lifecycle behavior they validate.
//!
//! # Lifecycle and threading
//!
//! Model handles stay inside their registry session. Edit, restore, render, sample-access, and
//! teardown guards enforce ordering. [`ModelThread`] admits model mutation, while realtime helpers
//! use bounded nonblocking storage and never make allocation or blocking guarantees for arbitrary
//! caller code. Fallible admission reports [`AraError`] before invalid data crosses FFI.
//!
//! # Features and platforms
//!
//! The crate has no features or SDK dependency and is portable wherever `ara2-bridge-sys` has a
//! proven layout. Generation-1 availability remains target-dependent.
//!
//! # Compatibility and licensing
//!
//! The API targets Rust 1.82 and ARA generations through 2.3 Final. The crate is MIT OR Apache-2.0;
//! ported ARA utilities and vectors retain their recorded Apache-2.0 provenance.
//!
//! # Example
//!
//! ```
//! use ara2_bridge_core::{AraError, Registry};
//!
//! enum AudioSource {}
//! let mut sources = Registry::<AudioSource, String>::new(8);
//! let source = sources.insert(String::from("take.wav"))?;
//! assert_eq!(sources.get(source)?, "take.wav");
//! sources.remove(source)?;
//! assert!(matches!(sources.get(source), Err(AraError::InvalidArgument(_))));
//! # Ok::<(), AraError>(())
//! ```
//!
//! See the workspace specifications and the upstream
//! [ARA API](https://github.com/Celemony/ARA_API) for normative ABI behavior.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::missing_safety_doc)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod archive;
mod assertions;
mod audio_file;
mod channel;
mod content;
mod diagnostics;
mod dispatch;
mod error;
mod ffi;
mod generation;
mod handles;
mod lifecycle;
mod poison;
mod processing;
mod properties;
mod realtime;
mod registry;
mod threading;
mod util;

pub use archive::{
    ArchiveProgress, FfiRestoreFilter, FfiStoreFilter, FilterSelection, MemoryArchive, ReadAt,
    RestoreFilter, RestoreFilterBuilder, RestoreMapping, RestorePhase, StoreFilter,
    StoreFilterBuilder, WriteAt,
};
pub use assertions::{AssertCoordinator, FactoryInitialization};
pub use audio_file::{
    read_ixml, read_ixml_with_limit, replace_ara_in_path, rewrite_ixml, AraChunkSet,
    AudioFileError, AudioFileKind, AudioSourceArchive, ChunkError, ChunkLimits, PathRewriteError,
    SuggestedPlugIn,
};
pub use channel::{
    ChannelArrangement, ChannelFormat, CoreAudioChannelDescription, CoreAudioChannelLayout,
    OpaqueChannelArrangement,
};
pub use content::{
    copy_event_from_ffi, validate_event_sequence, BarSignatureEvent, BarSignatures, ChordEvent,
    ChordIntervalUsage, ContentGrade, ContentKind, ContentReader, ContentReaderBackend,
    ContentReaderGate, ContentReaderLease, ContentUpdateScopes, DynamicContentReader, EventRef,
    KeySignatureEvent, KeySignatureIntervalUsage, KeySignatures, NoContentReaderBackend, NoteEvent,
    Notes, SheetChords, StaticTuning, Tempo, TempoEvent, TuningEvent,
};
pub use diagnostics::{BoundedDiagnosticSink, Diagnostic, DiagnosticSink, DocumentId, InstanceId};
pub use dispatch::{
    dispatch_bool, dispatch_i32, dispatch_ref, dispatch_time_pair, dispatch_void, DispatchRuntime,
};
pub use error::{AraError, ArchiveError, CompanionError};
pub use ffi::{AraBool, ForeignSlice, ForeignStr, SizedInput, SizedRecord};
pub use generation::ApiGeneration;
pub use handles::{Handle, ModelRef, RawHandle};
pub use lifecycle::{
    ContentCallGuard, EditGuard, Lifecycle, RenderActivationGuard, RestoreGuard, SampleAccessGuard,
    TeardownGuard,
};
pub use poison::PoisonState;
pub use processing::{
    LicenseCapabilities, LicenseRequest, PlaybackTransformationFlags, ProcessingAlgorithmCatalog,
    ProcessingAlgorithmFfi, ProcessingAlgorithmProperties,
};
pub use properties::{
    AudioModificationKind, AudioModificationProperties, AudioSourceKind, AudioSourceProperties,
    Color, ContentTimeRange, DocumentProperties, FfiProperties, MusicalContextKind,
    MusicalContextProperties, PlaybackRegionKind, PlaybackRegionProperties, RawChannelArrangement,
    RegionSequenceKind, RegionSequenceProperties, ViewSelection,
};
pub use realtime::{
    HeadTailEntry, RealtimeFailureCode, RealtimeFailureQueue, RealtimeHeadTailView,
};
pub use registry::{Registry, RegistrySession};
pub use threading::ModelThread;
pub use util::{
    intersect_content_ranges, sample_to_time, time_to_sample, BarMap, PitchInterpreter, ScaleMode,
    TempoMap,
};
