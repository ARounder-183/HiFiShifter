//! Feature-gated interoperability with the pinned Celemony C++ examples.

use crate::{build_test_factory, test_audio_source_properties, TestHost, TestPluginTrace};
use ara2_bridge_core::{
    ApiGeneration, AraError, AudioModificationProperties, ContentGrade, ContentUpdateScopes,
    DocumentProperties, MusicalContextProperties, Notes, RegionSequenceProperties, RestoreFilter,
};
use ara2_bridge_host::{DocumentSession, LoadedFactory};
use ara2_bridge_sys::{ARAAssertCategory, ARAFactory};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::AtomicU8;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

const NATIVE_OK: i32 = 0;
const BASIC_DOCUMENT_ID: i32 = 1;

static ARCHIVE_READER: AtomicU8 = AtomicU8::new(0);
static ARCHIVE_WRITER: AtomicU8 = AtomicU8::new(0);
static CHUNK_WRITER: AtomicU8 = AtomicU8::new(0);
static NATIVE_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static ASSERTIONS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Stable native scenario names supported by the interoperability harness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeScenario {
    /// Update document and model-object properties.
    PropertyUpdates,
    /// Notify the plug-in about changed host content.
    ContentUpdates,
    /// Request analysis and read typed plug-in content.
    ContentReading,
    /// Clone an audio modification.
    ModificationCloning,
    /// Store and restore one complete document archive.
    FullArchive,
    /// Store and restore filtered archives.
    SplitPartialArchives,
    /// Import filtered objects into another document.
    DragDropImport,
    /// Enumerate and select processing algorithms.
    ProcessingAlgorithms,
    /// Ask the plug-in to author an audio-file chunk archive.
    AudioFileChunkSave,
    /// Create, update, and destroy a basic ARA document graph.
    BasicDocument,
}

impl NativeScenario {
    /// Returns the stable manifest name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::PropertyUpdates => "property-updates",
            Self::ContentUpdates => "content-updates",
            Self::ContentReading => "content-reading",
            Self::ModificationCloning => "audio-modification-cloning",
            Self::FullArchive => "full-archive",
            Self::SplitPartialArchives => "split-partial-archives",
            Self::DragDropImport => "drag-drop-import",
            Self::ProcessingAlgorithms => "processing-algorithms",
            Self::AudioFileChunkSave => "audio-file-chunk-save",
            Self::BasicDocument => "basic-document",
        }
    }

    /// Returns every upstream scenario buildable through a direct ARA factory pairing.
    pub const fn buildable() -> &'static [Self] {
        &[
            Self::PropertyUpdates,
            Self::ContentUpdates,
            Self::ContentReading,
            Self::ModificationCloning,
            Self::FullArchive,
            Self::SplitPartialArchives,
            Self::DragDropImport,
            Self::ProcessingAlgorithms,
            Self::AudioFileChunkSave,
            Self::BasicDocument,
        ]
    }

    const fn as_raw(self) -> i32 {
        match self {
            Self::PropertyUpdates => 2,
            Self::ContentUpdates => 3,
            Self::ContentReading => 4,
            Self::ModificationCloning => 5,
            Self::FullArchive => 6,
            Self::SplitPartialArchives => 7,
            Self::DragDropImport => 8,
            Self::ProcessingAlgorithms => 9,
            Self::AudioFileChunkSave => 10,
            Self::BasicDocument => BASIC_DOCUMENT_ID,
        }
    }
}

/// Configuration shared by both native interoperability directions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeScenarioConfig {
    /// ARA API generation selected for the factory initialization.
    pub generation: ApiGeneration,
    /// Named behavior exercised across the boundary.
    pub scenario: NativeScenario,
}

/// Observable evidence returned by one native interoperability run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeScenarioResult {
    /// Selected API generation.
    pub generation: ApiGeneration,
    /// Stable scenario name.
    pub scenario: &'static str,
    /// Successful ABI callbacks observed during the run.
    pub callback_count: usize,
    /// Structured native diagnostics emitted by the shim.
    pub diagnostics: Vec<String>,
    /// Native objects still live after teardown.
    pub live_objects: usize,
}

/// Errors reported by the optional C++ interoperability harness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeInteropError {
    /// The feature boundary exists but the native bridge has not been installed.
    NativeBridgeUnavailable,
    /// A Rust-side ARA operation failed.
    Ara(AraError),
    /// The native shim rejected the request or caught a C++ exception.
    NativeFailure {
        /// Stable native status code.
        status: i32,
        /// Bounded diagnostic copied from the shim.
        diagnostic: String,
    },
    /// The two sides observed different numbers of semantic callbacks.
    CallbackCountMismatch {
        /// Count reported by Celemony's TestHost shim.
        native: usize,
        /// Count recorded by the Rust TestPlugIn fixture.
        rust: usize,
    },
}

impl From<AraError> for NativeInteropError {
    fn from(error: AraError) -> Self {
        Self::Ara(error)
    }
}

#[repr(C)]
struct RawNativeResult {
    status: i32,
    generation: i32,
    callback_count: u64,
    live_objects: u64,
    diagnostic: [c_char; 512],
}

impl Default for RawNativeResult {
    fn default() -> Self {
        Self {
            status: 0,
            generation: 0,
            callback_count: 0,
            live_objects: 0,
            diagnostic: [0; 512],
        }
    }
}

unsafe extern "C" {
    fn ara2_cpp_test_plugin_factory(result: *mut RawNativeResult) -> *const ARAFactory;
    fn ara2_cpp_assert_scope_begin(result: *mut RawNativeResult);
    fn ara2_cpp_assert_scope_end();
    fn ara2_cpp_test_host_run(
        factory: *const ARAFactory,
        generation: i32,
        scenario: i32,
        result: *mut RawNativeResult,
    ) -> i32;
}

struct NativeAssertScope;

impl NativeAssertScope {
    fn begin(result: &mut RawNativeResult) -> Self {
        // SAFETY: the native scope is serialized and `result` remains live until this guard drops.
        unsafe { ara2_cpp_assert_scope_begin(result) };
        Self
    }
}

impl Drop for NativeAssertScope {
    fn drop(&mut self) {
        // SAFETY: this guard owns the one serialized native assertion scope.
        unsafe { ara2_cpp_assert_scope_end() };
    }
}

fn lock_native() -> MutexGuard<'static, ()> {
    NATIVE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

unsafe extern "C" fn assertion(
    category: ARAAssertCategory,
    _problem: *const c_void,
    file: *const c_char,
) {
    let file = if file.is_null() {
        "unknown file".to_owned()
    } else {
        // SAFETY: ARA assertion callbacks provide a call-scoped NUL-terminated file name.
        unsafe { CStr::from_ptr(file) }
            .to_string_lossy()
            .into_owned()
    };
    ASSERTIONS.with(|assertions| {
        assertions
            .borrow_mut()
            .push(format!("ARA assertion category {category}: {file}"));
    });
}

fn take_assertions() -> Vec<String> {
    ASSERTIONS.with(|assertions| std::mem::take(&mut *assertions.borrow_mut()))
}

fn native_diagnostic(raw: &RawNativeResult) -> String {
    // SAFETY: the C++ shim always NUL-terminates this fixed-size array.
    unsafe { CStr::from_ptr(raw.diagnostic.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn require_native_success(raw: &RawNativeResult) -> Result<(), NativeInteropError> {
    if raw.status == NATIVE_OK {
        Ok(())
    } else {
        Err(NativeInteropError::NativeFailure {
            status: raw.status,
            diagnostic: native_diagnostic(raw),
        })
    }
}

/// Runs Rust TestHost against Celemony's C++ TestPlugIn.
pub fn run_rust_host_cpp_plugin(
    config: NativeScenarioConfig,
) -> Result<NativeScenarioResult, NativeInteropError> {
    with_cpp_document(config, |host, session| match config.scenario {
        NativeScenario::PropertyUpdates => cpp_property_updates(session),
        NativeScenario::ContentUpdates => cpp_content_updates(session),
        NativeScenario::ContentReading => cpp_content_reading(session),
        NativeScenario::ModificationCloning => cpp_modification_cloning(session),
        NativeScenario::FullArchive => cpp_full_archive(host, session),
        NativeScenario::SplitPartialArchives => cpp_split_archive(host, session),
        NativeScenario::DragDropImport => cpp_drag_drop(host, session),
        NativeScenario::ProcessingAlgorithms => cpp_processing_algorithms(session),
        NativeScenario::AudioFileChunkSave => cpp_audio_file_chunk_save(session),
        NativeScenario::BasicDocument => cpp_basic_document(session),
    })
}

fn with_cpp_document(
    config: NativeScenarioConfig,
    operation: impl FnOnce(&TestHost, &mut DocumentSession<'_, '_>) -> Result<usize, AraError>,
) -> Result<NativeScenarioResult, NativeInteropError> {
    let _native_lock = lock_native();
    let _ = take_assertions();
    let mut raw = RawNativeResult::default();
    // SAFETY: the result POD is writable for the call and the returned factory has static backing.
    let factory = unsafe { ara2_cpp_test_plugin_factory(&mut raw) };
    require_native_success(&raw)?;
    if factory.is_null() {
        return Err(NativeInteropError::NativeBridgeUnavailable);
    }
    let assert_scope = NativeAssertScope::begin(&mut raw);

    let host = TestHost::new(config.generation)?;
    // SAFETY: the pinned C++ TestPlugIn factory has process-static backing and obeys ARA's
    // initialize/create/uninitialize ordering for the lifetime of `loaded`.
    let loaded = unsafe { LoadedFactory::load(factory, config.generation, Some(assertion))? };
    host.set_archive_id(loaded.metadata().document_archive_id());
    let mut session = DocumentSession::new(
        &loaded,
        host.services(),
        DocumentProperties::new(Some("C++ TestPlugIn / Rust TestHost"))?,
    )?;
    let operation_result = operation(&host, &mut session);
    let close_result = session
        .close()
        .map_err(|_| AraError::Peer("native Rust-host document close failed"));
    drop(loaded);
    drop(assert_scope);
    require_native_success(&raw)?;
    let callback_count = operation_result?;
    close_result?;

    Ok(NativeScenarioResult {
        generation: config.generation,
        scenario: config.scenario.name(),
        callback_count,
        diagnostics: take_assertions(),
        live_objects: 0,
    })
}

fn cpp_basic_document(session: &mut DocumentSession<'_, '_>) -> Result<usize, AraError> {
    let mut edit = session.edit()?;
    let context = edit.create_musical_context(MusicalContextProperties::new(
        Some("Interop context"),
        0,
        None,
    )?)?;
    let sequence = edit.create_region_sequence(RegionSequenceProperties::new(
        Some("Interop sequence"),
        0,
        edit.musical_context_ref(context)?,
        None,
    )?)?;
    edit.finish()?;

    let mut edit = session.edit()?;
    edit.update_document_properties(DocumentProperties::new(Some(
        "C++ TestPlugIn / Rust TestHost updated",
    ))?)?;
    edit.update_musical_context(
        context,
        MusicalContextProperties::new(Some("Interop context updated"), 1, None)?,
    )?;
    edit.update_region_sequence(
        sequence,
        RegionSequenceProperties::new(
            Some("Interop sequence updated"),
            1,
            edit.musical_context_ref(context)?,
            None,
        )?,
    )?;
    edit.finish()?;
    Ok(15)
}

fn cpp_property_updates(session: &mut DocumentSession<'_, '_>) -> Result<usize, AraError> {
    let mut edit = session.edit()?;
    edit.update_document_properties(DocumentProperties::new(Some("Updated document"))?)?;
    let source = edit.create_audio_source(test_audio_source_properties()?)?;
    edit.update_audio_source(source, test_audio_source_properties()?)?;
    edit.finish()?;
    Ok(6)
}

fn cpp_content_updates(session: &mut DocumentSession<'_, '_>) -> Result<usize, AraError> {
    let mut edit = session.edit()?;
    let source = edit.create_audio_source(test_audio_source_properties()?)?;
    edit.update_audio_source_content(source, None, ContentUpdateScopes::empty())?;
    edit.finish()?;
    Ok(5)
}

fn cpp_content_reading(session: &mut DocumentSession<'_, '_>) -> Result<usize, AraError> {
    let source = {
        let mut edit = session.edit()?;
        let source = edit.create_audio_source(test_audio_source_properties()?)?;
        edit.finish()?;
        source
    };
    session.set_audio_source_samples_access(source, true)?;
    session.request_audio_source_content_analysis::<Notes>(source)?;
    let mut completed = false;
    for _ in 0..2_000 {
        session.notify_model_updates()?;
        if !session.audio_source_content_analysis_incomplete::<Notes>(source)? {
            completed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    if !completed || !session.audio_source_content_available::<Notes>(source)? {
        return Err(AraError::Peer("C++ TestPlugIn analysis did not complete"));
    }
    if session.audio_source_content_grade::<Notes>(source)? == ContentGrade::INITIAL {
        return Err(AraError::Peer(
            "C++ TestPlugIn returned an initial content grade",
        ));
    }
    let mut reader = session
        .audio_source_content_reader::<Notes>(source, None)?
        .ok_or(AraError::Peer("C++ TestPlugIn returned no note reader"))?;
    if reader.is_empty() {
        return Err(AraError::Peer("C++ TestPlugIn note reader was empty"));
    }
    let _ = reader.event(0)?;
    drop(reader);
    session.set_audio_source_samples_access(source, false)?;
    Ok(10)
}

fn cpp_modification_cloning(session: &mut DocumentSession<'_, '_>) -> Result<usize, AraError> {
    let mut edit = session.edit()?;
    let source = edit.create_audio_source(test_audio_source_properties()?)?;
    let original = edit.create_audio_modification(
        source,
        AudioModificationProperties::new(Some("Original"), "cpp-original")?,
    )?;
    let clone = edit.clone_audio_modification(
        original,
        AudioModificationProperties::new(Some("Clone"), "cpp-clone")?,
    )?;
    edit.finish()?;
    if session.audio_modification_ref(original)? == session.audio_modification_ref(clone)? {
        return Err(AraError::Peer(
            "C++ TestPlugIn returned the original clone reference",
        ));
    }
    Ok(7)
}

fn cpp_full_archive(
    host: &TestHost,
    session: &mut DocumentSession<'_, '_>,
) -> Result<usize, AraError> {
    session.store_document_to_archive(&ARCHIVE_WRITER)?;
    let bytes = host
        .written_archive()
        .ok_or(AraError::Peer("C++ TestPlugIn produced no full archive"))?;
    if bytes.is_empty() {
        return Err(AraError::Peer(
            "C++ TestPlugIn produced an empty full archive",
        ));
    }
    Ok(3)
}

fn cpp_split_archive(
    host: &TestHost,
    session: &mut DocumentSession<'_, '_>,
) -> Result<usize, AraError> {
    let (source, modification) = {
        let mut edit = session.edit()?;
        let source = edit.create_audio_source(test_audio_source_properties()?)?;
        let modification = edit.create_audio_modification(
            source,
            AudioModificationProperties::new(None, "cpp-split-modification")?,
        )?;
        edit.finish()?;
        (source, modification)
    };
    let filter = session
        .store_filter_builder()
        .audio_source(source)
        .audio_modification(modification)
        .document_data(true)
        .build()?;
    session.store_objects_to_archive(&ARCHIVE_WRITER, Some(&filter))?;
    let bytes = host
        .written_archive()
        .ok_or(AraError::Peer("C++ TestPlugIn produced no partial archive"))?;
    host.seed_archive(bytes);
    let restore = RestoreFilter::builder()
        .audio_source("test-source", "test-source")
        .audio_modification("cpp-split-modification", "cpp-split-modification")
        .document_data(true)
        .build()?;
    let mut edit = session.edit()?;
    edit.restore_objects_from_archive(&ARCHIVE_READER, Some(&restore))?;
    edit.finish()?;
    Ok(9)
}

fn cpp_drag_drop(
    host: &TestHost,
    session: &mut DocumentSession<'_, '_>,
) -> Result<usize, AraError> {
    let source = {
        let mut edit = session.edit()?;
        let source = edit.create_audio_source(test_audio_source_properties()?)?;
        edit.finish()?;
        source
    };
    let store = session
        .store_filter_builder()
        .audio_source(source)
        .build()?;
    session.store_objects_to_archive(&ARCHIVE_WRITER, Some(&store))?;
    let bytes = host
        .written_archive()
        .ok_or(AraError::Peer("C++ TestPlugIn produced no drag archive"))?;
    host.seed_archive(bytes);
    let restore = RestoreFilter::builder()
        .audio_source("test-source", "test-source")
        .build()?;
    let mut edit = session.edit()?;
    edit.restore_objects_from_archive(&ARCHIVE_READER, Some(&restore))?;
    edit.finish()?;
    Ok(7)
}

fn cpp_processing_algorithms(session: &mut DocumentSession<'_, '_>) -> Result<usize, AraError> {
    let source = {
        let mut edit = session.edit()?;
        let source = edit.create_audio_source(test_audio_source_properties()?)?;
        edit.finish()?;
        source
    };
    let algorithms = session.processing_algorithms()?;
    if algorithms.is_empty() {
        return Err(AraError::Peer(
            "C++ TestPlugIn published no processing algorithms",
        ));
    }
    let selected = algorithms.len() - 1;
    let mut edit = session.edit()?;
    edit.request_processing_algorithm(source, selected)?;
    edit.finish()?;
    if session.processing_algorithm_for_audio_source(source)? != selected {
        return Err(AraError::Peer(
            "C++ TestPlugIn did not select the algorithm",
        ));
    }
    Ok(7)
}

fn cpp_audio_file_chunk_save(session: &mut DocumentSession<'_, '_>) -> Result<usize, AraError> {
    let source = {
        let mut edit = session.edit()?;
        let source = edit.create_audio_source(test_audio_source_properties()?)?;
        edit.finish()?;
        source
    };
    let stored = session.store_audio_source_to_audio_file_chunk(&CHUNK_WRITER, source)?;
    if stored.document_archive_id().is_empty() {
        return Err(AraError::Peer(
            "C++ TestPlugIn returned an empty chunk archive ID",
        ));
    }
    Ok(5)
}

/// Runs Celemony's C++ TestHost against the Rust TestPlugIn.
pub fn run_cpp_host_rust_plugin(
    config: NativeScenarioConfig,
) -> Result<NativeScenarioResult, NativeInteropError> {
    let _native_lock = lock_native();
    let trace = TestPluginTrace::new();
    let factory = build_test_factory(trace.clone())?;
    let mut raw = RawNativeResult::default();
    // SAFETY: `factory` retains stable ARA factory backing for the synchronous native run, and
    // the writable POD contains no pointers retained by the C++ shim.
    let status = unsafe {
        ara2_cpp_test_host_run(
            factory.as_raw(),
            config.generation.as_raw(),
            config.scenario.as_raw(),
            &mut raw,
        )
    };
    if status != raw.status {
        return Err(NativeInteropError::NativeFailure {
            status,
            diagnostic: "native return status disagreed with result POD".to_owned(),
        });
    }
    require_native_success(&raw)?;
    let rust_callbacks = trace.records().len();
    let native_callbacks =
        usize::try_from(raw.callback_count).map_err(|_| NativeInteropError::NativeFailure {
            status: raw.status,
            diagnostic: "native callback count exceeds usize".to_owned(),
        })?;
    if native_callbacks != 0 && native_callbacks != rust_callbacks {
        return Err(NativeInteropError::CallbackCountMismatch {
            native: native_callbacks,
            rust: rust_callbacks,
        });
    }

    Ok(NativeScenarioResult {
        generation: ApiGeneration::try_from_raw(raw.generation)?,
        scenario: config.scenario.name(),
        callback_count: rust_callbacks,
        diagnostics: Vec::new(),
        live_objects: usize::try_from(raw.live_objects).map_err(|_| {
            NativeInteropError::NativeFailure {
                status: raw.status,
                diagnostic: "native live-object count exceeds usize".to_owned(),
            }
        })?,
    })
}
