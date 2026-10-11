use ara2_bridge_core::{
    ApiGeneration, AudioModificationProperties, DocumentProperties, MusicalContextProperties,
    RawHandle, Registry,
};
use ara2_bridge_plugin::PLUGIN_CONTRACT_TESTS;
use ara2_bridge_sys::*;
use ara2_bridge_testkit::{
    all_content_types, all_extension_roles, all_transformations, build_test_factory,
    build_test_plugin, test_audio_source_properties, TestPluginTrace,
};
use std::collections::BTreeSet;
use std::ffi::{c_void, CStr};
use std::mem::offset_of;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn capability_rich_fixture_represents_every_controller_callback_without_skips() {
    let trace = TestPluginTrace::new();
    let mut plugin = build_test_plugin(trace.clone()).unwrap();
    let interface = plugin
        .document_controller_interface(ApiGeneration::V23Final)
        .unwrap();
    assert_eq!(PLUGIN_CONTRACT_TESTS.len(), 54);
    assert_eq!(interface.represented_callback_count(), 54);
    assert!(interface.represented_callbacks_are_non_null());

    let algorithms = plugin.capabilities().algorithms().unwrap();
    assert_eq!(algorithms.len_i32().unwrap(), 2);
    let handle = fixture_handle();
    let chunk = plugin
        .capabilities_mut()
        .store_audio_file_chunk(handle)
        .unwrap();
    assert_eq!(chunk.document_archive_id, "org.ara2-bridge.test.archive");
    assert!(plugin.capabilities().preserves_signal(handle));
    assert_eq!(trace.count("store_audio_file_chunk"), 1);
    assert_eq!(trace.count("query_signal_preservation"), 1);
}

#[test]
fn fixture_factory_and_extension_surface_advertise_every_supported_capability() {
    let factory = build_test_factory(TestPluginTrace::new()).unwrap();
    let raw = factory.raw_copy();
    // SAFETY: `raw` is a complete local copy of the immutable packed factory record.
    let analyzable_count =
        unsafe { std::ptr::addr_of!(raw.analyzeableContentTypesCount).read_unaligned() };
    // SAFETY: same complete local packed record.
    let transformations =
        unsafe { std::ptr::addr_of!(raw.supportedPlaybackTransformationFlags).read_unaligned() };
    // SAFETY: same complete local packed record.
    let stores_chunks =
        unsafe { std::ptr::addr_of!(raw.supportsStoringAudioFileChunks).read_unaligned() };
    assert_eq!(analyzable_count, all_content_types().len());
    assert_eq!(transformations as u32, all_transformations().bits());
    assert_ne!(stores_chunks, ara2_bridge_sys::kARAFalse);
    assert_eq!(all_extension_roles().bits().count_ones(), 3);
}

#[test]
fn raw_fixture_drives_all_54_document_controller_callbacks() {
    let trace = TestPluginTrace::new();
    let factory = build_test_factory(trace).unwrap();
    let mut assertion: ARAAssertFunction = None;
    factory
        .entry()
        .initialize(ApiGeneration::V23Final, &raw mut assertion)
        .unwrap();
    let host = RawHost::new();
    let host_instance = host.instance();
    let document = DocumentProperties::new(Some("Raw Contract")).unwrap();
    let raw_document = document.as_ffi();
    let raw_factory = factory.raw_copy();
    // SAFETY: copied from the complete live packed factory record.
    let factory_archive_id =
        unsafe { std::ptr::addr_of!(raw_factory.documentArchiveID).read_unaligned() };
    // SAFETY: copied from the complete live factory record.
    let create_controller = unsafe {
        std::ptr::addr_of!(raw_factory.createDocumentControllerWithDocument).read_unaligned()
    }
    .unwrap();
    // SAFETY: host and document backing outlive the created controller.
    let instance =
        unsafe { create_controller(&raw const host_instance, raw_document.as_ref().as_ptr()) };
    assert!(!instance.is_null());
    // SAFETY: the factory returned a complete live instance.
    let controller = unsafe {
        ara2_bridge_sys::access::read_field::<ARADocumentControllerRef>(
            instance.cast(),
            offset_of!(ARADocumentControllerInstance, documentControllerRef),
        )
    };
    // SAFETY: same complete live instance contract.
    let interface = unsafe {
        ara2_bridge_sys::access::read_field::<*const ARADocumentControllerInterface>(
            instance.cast(),
            offset_of!(ARADocumentControllerInstance, documentControllerInterface),
        )
    };
    let mut driven = BTreeSet::new();
    macro_rules! drive {
        ($name:literal, $call:expr) => {{
            assert!(driven.insert($name));
            // SAFETY: each call site below maintains the callback's live identities and call scope.
            unsafe { $call }
        }};
    }
    macro_rules! cb {
        ($field:ident, $type:ty) => {
            callback::<$type>(
                interface,
                offset_of!(ARADocumentControllerInterface, $field),
            )
        };
    }

    let get_factory = cb!(
        getFactory,
        unsafe extern "C" fn(ARADocumentControllerRef) -> *const ARAFactory
    );
    assert_eq!(
        drive!("getFactory", get_factory(controller)),
        factory.as_raw()
    );
    let legacy_begin = cb!(
        beginRestoringDocumentFromArchive,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAArchiveReaderHostRef) -> ARABool
    );
    let legacy_end = cb!(
        endRestoringDocumentFromArchive,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAArchiveReaderHostRef) -> ARABool
    );
    assert_ne!(
        drive!(
            "beginRestoringDocumentFromArchive",
            legacy_begin(controller, raw_ref())
        ),
        kARAFalse
    );
    assert_ne!(
        drive!(
            "endRestoringDocumentFromArchive",
            legacy_end(controller, raw_ref())
        ),
        kARAFalse
    );
    let legacy_store = cb!(
        storeDocumentToArchive,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAArchiveWriterHostRef) -> ARABool
    );
    assert_ne!(
        drive!(
            "storeDocumentToArchive",
            legacy_store(controller, raw_ref())
        ),
        kARAFalse
    );

    let begin = cb!(beginEditing, unsafe extern "C" fn(ARADocumentControllerRef));
    drive!("beginEditing", begin(controller));
    let update_document = cb!(
        updateDocumentProperties,
        unsafe extern "C" fn(ARADocumentControllerRef, *const ARADocumentProperties)
    );
    drive!(
        "updateDocumentProperties",
        update_document(controller, raw_document.as_ref().as_ptr())
    );

    let context_properties = MusicalContextProperties::new(Some("Context"), 0, None).unwrap();
    let raw_context = context_properties.as_ffi(ApiGeneration::V23Final).unwrap();
    let create_context = cb!(
        createMusicalContext,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAMusicalContextHostRef,
            *const ARAMusicalContextProperties,
        ) -> ARAMusicalContextRef
    );
    let context = drive!(
        "createMusicalContext",
        create_context(controller, raw_ref(), raw_context.as_ref().as_ptr())
    );
    assert!(!context.is_null());
    let update_context = cb!(
        updateMusicalContextProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAMusicalContextRef,
            *const ARAMusicalContextProperties,
        )
    );
    drive!(
        "updateMusicalContextProperties",
        update_context(controller, context, raw_context.as_ref().as_ptr())
    );
    let range = ARAContentTimeRange {
        start: 0.0,
        duration: 1.0,
    };
    let update_context_content = cb!(
        updateMusicalContextContent,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAMusicalContextRef,
            *const ARAContentTimeRange,
            ARAContentUpdateFlags,
        )
    );
    drive!(
        "updateMusicalContextContent",
        update_context_content(controller, context, &raw const range, 0)
    );

    let source_properties = test_audio_source_properties().unwrap();
    let raw_source = source_properties.as_ffi(ApiGeneration::V23Final).unwrap();
    let create_source = cb!(
        createAudioSource,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceHostRef,
            *const ARAAudioSourceProperties,
        ) -> ARAAudioSourceRef
    );
    let source = drive!(
        "createAudioSource",
        create_source(controller, raw_ref(), raw_source.as_ref().as_ptr())
    );
    assert!(!source.is_null());
    let update_source = cb!(
        updateAudioSourceProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            *const ARAAudioSourceProperties,
        )
    );
    drive!(
        "updateAudioSourceProperties",
        update_source(controller, source, raw_source.as_ref().as_ptr())
    );
    let update_source_content = cb!(
        updateAudioSourceContent,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            *const ARAContentTimeRange,
            ARAContentUpdateFlags,
        )
    );
    drive!(
        "updateAudioSourceContent",
        update_source_content(controller, source, &raw const range, 0)
    );
    let enable_source = cb!(
        enableAudioSourceSamplesAccess,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef, ARABool)
    );
    drive!(
        "enableAudioSourceSamplesAccess",
        enable_source(controller, source, kARATrue)
    );

    let modification_properties =
        AudioModificationProperties::new(Some("Modification"), "modification-1").unwrap();
    let raw_modification = modification_properties.as_ffi();
    let create_modification = cb!(
        createAudioModification,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAAudioModificationHostRef,
            *const ARAAudioModificationProperties,
        ) -> ARAAudioModificationRef
    );
    let modification = drive!(
        "createAudioModification",
        create_modification(
            controller,
            source,
            raw_ref(),
            raw_modification.as_ref().as_ptr(),
        )
    );
    assert!(!modification.is_null());
    let clone_modification = cb!(
        cloneAudioModification,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAAudioModificationHostRef,
            *const ARAAudioModificationProperties,
        ) -> ARAAudioModificationRef
    );
    let clone = drive!(
        "cloneAudioModification",
        clone_modification(
            controller,
            modification,
            raw_ref(),
            raw_modification.as_ref().as_ptr(),
        )
    );
    assert!(!clone.is_null());
    let update_modification = cb!(
        updateAudioModificationProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            *const ARAAudioModificationProperties,
        )
    );
    drive!(
        "updateAudioModificationProperties",
        update_modification(controller, modification, raw_modification.as_ref().as_ptr())
    );

    let sequence_properties = ARARegionSequenceProperties {
        structSize: std::mem::size_of::<ARARegionSequenceProperties>(),
        name: c"Sequence".as_ptr(),
        orderIndex: 0,
        musicalContextRef: context,
        color: null(),
    };
    let create_sequence = cb!(
        createRegionSequence,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARARegionSequenceHostRef,
            *const ARARegionSequenceProperties,
        ) -> ARARegionSequenceRef
    );
    let sequence = drive!(
        "createRegionSequence",
        create_sequence(controller, raw_ref(), &raw const sequence_properties)
    );
    assert!(!sequence.is_null());
    let update_sequence = cb!(
        updateRegionSequenceProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARARegionSequenceRef,
            *const ARARegionSequenceProperties,
        )
    );
    drive!(
        "updateRegionSequenceProperties",
        update_sequence(controller, sequence, &raw const sequence_properties)
    );

    let playback_properties = ARAPlaybackRegionProperties {
        structSize: std::mem::size_of::<ARAPlaybackRegionProperties>(),
        transformationFlags: 0,
        startInModificationTime: 0.0,
        durationInModificationTime: 1.0,
        startInPlaybackTime: 0.0,
        durationInPlaybackTime: 1.0,
        musicalContextRef: null_mut(),
        regionSequenceRef: sequence,
        name: c"Region".as_ptr(),
        color: null(),
    };
    let create_region = cb!(
        createPlaybackRegion,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAPlaybackRegionHostRef,
            *const ARAPlaybackRegionProperties,
        ) -> ARAPlaybackRegionRef
    );
    let region = drive!(
        "createPlaybackRegion",
        create_region(
            controller,
            modification,
            raw_ref(),
            &raw const playback_properties,
        )
    );
    // SAFETY: the clone, sequence, host identity, and properties are live in this edit session.
    let clone_region =
        unsafe { create_region(controller, clone, raw_ref(), &raw const playback_properties) };
    assert!(!region.is_null() && !clone_region.is_null());
    let update_region = cb!(
        updatePlaybackRegionProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            *const ARAPlaybackRegionProperties,
        )
    );
    drive!(
        "updatePlaybackRegionProperties",
        update_region(controller, region, &raw const playback_properties)
    );

    drive_content_callbacks(
        &mut driven,
        controller,
        interface,
        source,
        modification,
        region,
    );

    let request_algorithm = cb!(
        requestProcessingAlgorithmForAudioSource,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef, ARAInt32)
    );
    drive!(
        "requestProcessingAlgorithmForAudioSource",
        request_algorithm(controller, source, 1)
    );
    let request_analysis = cb!(
        requestAudioSourceContentAnalysis,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARASize,
            *const ARAContentType,
        )
    );
    let content_types = all_content_types();
    RAW_ANALYSIS_PROGRESS_COUNT.store(0, Ordering::SeqCst);
    drive!(
        "requestAudioSourceContentAnalysis",
        request_analysis(
            controller,
            source,
            content_types.len(),
            content_types.as_ptr(),
        )
    );
    assert_eq!(RAW_ANALYSIS_PROGRESS_COUNT.load(Ordering::SeqCst), 0);
    let analysis_incomplete = cb!(
        isAudioSourceContentAnalysisIncomplete,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAContentType,
        ) -> ARABool
    );
    drive!(
        "isAudioSourceContentAnalysisIncomplete",
        analysis_incomplete(controller, source, kARAContentTypeNotes as i32)
    );

    let head_tail = cb!(
        getPlaybackRegionHeadAndTailTime,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            *mut ARATimeDuration,
            *mut ARATimeDuration,
        )
    );
    let mut head = 0.0;
    let mut tail = 0.0;
    drive!(
        "getPlaybackRegionHeadAndTailTime",
        head_tail(controller, region, &raw mut head, &raw mut tail)
    );
    assert_eq!((head, tail), (0.125, 0.25));

    let restore_objects = cb!(
        restoreObjectsFromArchive,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAArchiveReaderHostRef,
            *const ARARestoreObjectsFilter,
        ) -> ARABool
    );
    assert_ne!(
        drive!(
            "restoreObjectsFromArchive",
            restore_objects(controller, raw_ref(), null())
        ),
        kARAFalse
    );
    let license = cb!(
        isLicensedForCapabilities,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARABool,
            ARASize,
            *const ARAContentType,
            ARAPlaybackTransformationFlags,
        ) -> ARABool
    );
    assert_ne!(
        drive!(
            "isLicensedForCapabilities",
            license(
                controller,
                kARAFalse,
                content_types.len(),
                content_types.as_ptr(),
                all_transformations().bits() as i32,
            )
        ),
        kARAFalse
    );
    let signal = cb!(
        isAudioModificationPreservingAudioSourceSignal,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioModificationRef) -> ARABool
    );
    assert_ne!(
        drive!(
            "isAudioModificationPreservingAudioSourceSignal",
            signal(controller, modification)
        ),
        kARAFalse
    );

    let end = cb!(endEditing, unsafe extern "C" fn(ARADocumentControllerRef));
    drive!("endEditing", end(controller));
    let notify = cb!(
        notifyModelUpdates,
        unsafe extern "C" fn(ARADocumentControllerRef)
    );
    drive!("notifyModelUpdates", notify(controller));
    assert_eq!(RAW_ANALYSIS_PROGRESS_COUNT.load(Ordering::SeqCst), 3);
    drive_processing_and_storage_callbacks(
        &mut driven,
        controller,
        interface,
        source,
        factory_archive_id,
    );

    // SAFETY: starts a fresh edit after the prior balanced session.
    unsafe { begin(controller) };
    let destroy_region = cb!(
        destroyPlaybackRegion,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAPlaybackRegionRef)
    );
    drive!("destroyPlaybackRegion", destroy_region(controller, region));
    // SAFETY: the second region is a live graph leaf.
    unsafe { destroy_region(controller, clone_region) };
    let deactivate_modification = cb!(
        deactivateAudioModificationForUndoHistory,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioModificationRef, ARABool)
    );
    drive!(
        "deactivateAudioModificationForUndoHistory",
        deactivate_modification(controller, modification, kARATrue)
    );
    // SAFETY: the clone is childless after its playback region was destroyed.
    unsafe { deactivate_modification(controller, clone, kARATrue) };
    let deactivate_source = cb!(
        deactivateAudioSourceForUndoHistory,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef, ARABool)
    );
    drive!(
        "deactivateAudioSourceForUndoHistory",
        deactivate_source(controller, source, kARATrue)
    );
    // SAFETY: reactivation follows the required parent-before-child order.
    unsafe {
        deactivate_source(controller, source, kARAFalse);
        deactivate_modification(controller, modification, kARAFalse);
        deactivate_modification(controller, clone, kARAFalse);
    }
    let destroy_modification = cb!(
        destroyAudioModification,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioModificationRef)
    );
    drive!(
        "destroyAudioModification",
        destroy_modification(controller, modification)
    );
    // SAFETY: the clone remains a live childless modification.
    unsafe { destroy_modification(controller, clone) };
    let destroy_source = cb!(
        destroyAudioSource,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef)
    );
    drive!("destroyAudioSource", destroy_source(controller, source));
    let destroy_sequence = cb!(
        destroyRegionSequence,
        unsafe extern "C" fn(ARADocumentControllerRef, ARARegionSequenceRef)
    );
    drive!(
        "destroyRegionSequence",
        destroy_sequence(controller, sequence)
    );
    let destroy_context = cb!(
        destroyMusicalContext,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAMusicalContextRef)
    );
    drive!(
        "destroyMusicalContext",
        destroy_context(controller, context)
    );
    // SAFETY: balances the destruction edit session.
    unsafe { end(controller) };

    let destroy_controller = cb!(
        destroyDocumentController,
        unsafe extern "C" fn(ARADocumentControllerRef)
    );
    drive!("destroyDocumentController", destroy_controller(controller));
    factory.entry().uninitialize().unwrap();

    let expected = PLUGIN_CONTRACT_TESTS
        .iter()
        .map(|contract| contract.c_name)
        .collect::<BTreeSet<_>>();
    assert_eq!(driven, expected);
}

#[test]
fn every_callback_rejects_a_null_controller_with_its_abi_fallback() {
    let plugin = build_test_plugin(TestPluginTrace::new()).unwrap();
    let interface = plugin
        .document_controller_interface(ApiGeneration::V23Final)
        .unwrap();
    let driven = drive_null_controller_callbacks(interface.as_raw());
    let expected = PLUGIN_CONTRACT_TESTS
        .iter()
        .map(|contract| contract.c_name)
        .collect::<BTreeSet<_>>();
    assert_eq!(driven, expected);
}

fn drive_null_controller_callbacks(
    interface: *const ARADocumentControllerInterface,
) -> BTreeSet<&'static str> {
    let mut driven = BTreeSet::new();
    macro_rules! reject {
        ($name:literal, $field:ident, $type:ty $(, $argument:expr)*) => {{
            let callback = callback::<$type>(
                interface,
                offset_of!(ARADocumentControllerInterface, $field),
            );
            assert!(driven.insert($name));
            // SAFETY: a null controller selects the generated fallback before arguments are read.
            unsafe { callback(null_mut() $(, $argument)*) }
        }};
    }
    reject!(
        "destroyDocumentController",
        destroyDocumentController,
        unsafe extern "C" fn(ARADocumentControllerRef)
    );
    reject!(
        "getFactory",
        getFactory,
        unsafe extern "C" fn(ARADocumentControllerRef) -> *const ARAFactory
    );
    reject!(
        "beginEditing",
        beginEditing,
        unsafe extern "C" fn(ARADocumentControllerRef)
    );
    reject!(
        "endEditing",
        endEditing,
        unsafe extern "C" fn(ARADocumentControllerRef)
    );
    reject!(
        "notifyModelUpdates",
        notifyModelUpdates,
        unsafe extern "C" fn(ARADocumentControllerRef)
    );
    reject!(
        "beginRestoringDocumentFromArchive",
        beginRestoringDocumentFromArchive,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAArchiveReaderHostRef) -> ARABool,
        null_mut()
    );
    reject!(
        "endRestoringDocumentFromArchive",
        endRestoringDocumentFromArchive,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAArchiveReaderHostRef) -> ARABool,
        null_mut()
    );
    reject!(
        "storeDocumentToArchive",
        storeDocumentToArchive,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAArchiveWriterHostRef) -> ARABool,
        null_mut()
    );
    reject!(
        "updateDocumentProperties",
        updateDocumentProperties,
        unsafe extern "C" fn(ARADocumentControllerRef, *const ARADocumentProperties),
        null()
    );
    reject!(
        "createMusicalContext",
        createMusicalContext,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAMusicalContextHostRef,
            *const ARAMusicalContextProperties,
        ) -> ARAMusicalContextRef,
        null_mut(),
        null()
    );
    reject!(
        "updateMusicalContextProperties",
        updateMusicalContextProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAMusicalContextRef,
            *const ARAMusicalContextProperties,
        ),
        null_mut(),
        null()
    );
    reject!(
        "updateMusicalContextContent",
        updateMusicalContextContent,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAMusicalContextRef,
            *const ARAContentTimeRange,
            ARAContentUpdateFlags,
        ),
        null_mut(),
        null(),
        0
    );
    reject!(
        "destroyMusicalContext",
        destroyMusicalContext,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAMusicalContextRef),
        null_mut()
    );
    reject!(
        "createAudioSource",
        createAudioSource,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceHostRef,
            *const ARAAudioSourceProperties,
        ) -> ARAAudioSourceRef,
        null_mut(),
        null()
    );
    reject!(
        "updateAudioSourceProperties",
        updateAudioSourceProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            *const ARAAudioSourceProperties,
        ),
        null_mut(),
        null()
    );
    reject!(
        "updateAudioSourceContent",
        updateAudioSourceContent,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            *const ARAContentTimeRange,
            ARAContentUpdateFlags,
        ),
        null_mut(),
        null(),
        0
    );
    reject!(
        "enableAudioSourceSamplesAccess",
        enableAudioSourceSamplesAccess,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef, ARABool),
        null_mut(),
        kARAFalse
    );
    reject!(
        "deactivateAudioSourceForUndoHistory",
        deactivateAudioSourceForUndoHistory,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef, ARABool),
        null_mut(),
        kARAFalse
    );
    reject!(
        "destroyAudioSource",
        destroyAudioSource,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef),
        null_mut()
    );
    reject!(
        "createAudioModification",
        createAudioModification,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAAudioModificationHostRef,
            *const ARAAudioModificationProperties,
        ) -> ARAAudioModificationRef,
        null_mut(),
        null_mut(),
        null()
    );
    reject!(
        "cloneAudioModification",
        cloneAudioModification,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAAudioModificationHostRef,
            *const ARAAudioModificationProperties,
        ) -> ARAAudioModificationRef,
        null_mut(),
        null_mut(),
        null()
    );
    reject!(
        "updateAudioModificationProperties",
        updateAudioModificationProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            *const ARAAudioModificationProperties,
        ),
        null_mut(),
        null()
    );
    reject!(
        "deactivateAudioModificationForUndoHistory",
        deactivateAudioModificationForUndoHistory,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioModificationRef, ARABool),
        null_mut(),
        kARAFalse
    );
    reject!(
        "destroyAudioModification",
        destroyAudioModification,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioModificationRef),
        null_mut()
    );
    reject!(
        "createPlaybackRegion",
        createPlaybackRegion,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAPlaybackRegionHostRef,
            *const ARAPlaybackRegionProperties,
        ) -> ARAPlaybackRegionRef,
        null_mut(),
        null_mut(),
        null()
    );
    reject!(
        "updatePlaybackRegionProperties",
        updatePlaybackRegionProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            *const ARAPlaybackRegionProperties,
        ),
        null_mut(),
        null()
    );
    reject!(
        "destroyPlaybackRegion",
        destroyPlaybackRegion,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAPlaybackRegionRef),
        null_mut()
    );
    reject!(
        "isAudioSourceContentAvailable",
        isAudioSourceContentAvailable,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAContentType,
        ) -> ARABool,
        null_mut(),
        0
    );
    reject!(
        "isAudioSourceContentAnalysisIncomplete",
        isAudioSourceContentAnalysisIncomplete,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAContentType,
        ) -> ARABool,
        null_mut(),
        0
    );
    reject!(
        "requestAudioSourceContentAnalysis",
        requestAudioSourceContentAnalysis,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARASize,
            *const ARAContentType,
        ),
        null_mut(),
        0,
        null()
    );
    reject!(
        "getAudioSourceContentGrade",
        getAudioSourceContentGrade,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAContentType,
        ) -> ARAContentGrade,
        null_mut(),
        0
    );
    reject!(
        "createAudioSourceContentReader",
        createAudioSourceContentReader,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAContentType,
            *const ARAContentTimeRange,
        ) -> ARAContentReaderRef,
        null_mut(),
        0,
        null()
    );
    reject!(
        "isAudioModificationContentAvailable",
        isAudioModificationContentAvailable,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAContentType,
        ) -> ARABool,
        null_mut(),
        0
    );
    reject!(
        "getAudioModificationContentGrade",
        getAudioModificationContentGrade,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAContentType,
        ) -> ARAContentGrade,
        null_mut(),
        0
    );
    reject!(
        "createAudioModificationContentReader",
        createAudioModificationContentReader,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAContentType,
            *const ARAContentTimeRange,
        ) -> ARAContentReaderRef,
        null_mut(),
        0,
        null()
    );
    reject!(
        "isPlaybackRegionContentAvailable",
        isPlaybackRegionContentAvailable,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            ARAContentType,
        ) -> ARABool,
        null_mut(),
        0
    );
    reject!(
        "getPlaybackRegionContentGrade",
        getPlaybackRegionContentGrade,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            ARAContentType,
        ) -> ARAContentGrade,
        null_mut(),
        0
    );
    reject!(
        "createPlaybackRegionContentReader",
        createPlaybackRegionContentReader,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            ARAContentType,
            *const ARAContentTimeRange,
        ) -> ARAContentReaderRef,
        null_mut(),
        0,
        null()
    );
    reject!(
        "getContentReaderEventCount",
        getContentReaderEventCount,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAContentReaderRef) -> ARAInt32,
        null_mut()
    );
    reject!(
        "getContentReaderDataForEvent",
        getContentReaderDataForEvent,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAContentReaderRef,
            ARAInt32,
        ) -> *const c_void,
        null_mut(),
        0
    );
    reject!(
        "destroyContentReader",
        destroyContentReader,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAContentReaderRef),
        null_mut()
    );
    reject!(
        "createRegionSequence",
        createRegionSequence,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARARegionSequenceHostRef,
            *const ARARegionSequenceProperties,
        ) -> ARARegionSequenceRef,
        null_mut(),
        null()
    );
    reject!(
        "updateRegionSequenceProperties",
        updateRegionSequenceProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARARegionSequenceRef,
            *const ARARegionSequenceProperties,
        ),
        null_mut(),
        null()
    );
    reject!(
        "destroyRegionSequence",
        destroyRegionSequence,
        unsafe extern "C" fn(ARADocumentControllerRef, ARARegionSequenceRef),
        null_mut()
    );
    reject!(
        "getPlaybackRegionHeadAndTailTime",
        getPlaybackRegionHeadAndTailTime,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            *mut ARATimeDuration,
            *mut ARATimeDuration,
        ),
        null_mut(),
        null_mut(),
        null_mut()
    );
    reject!(
        "restoreObjectsFromArchive",
        restoreObjectsFromArchive,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAArchiveReaderHostRef,
            *const ARARestoreObjectsFilter,
        ) -> ARABool,
        null_mut(),
        null()
    );
    reject!(
        "storeObjectsToArchive",
        storeObjectsToArchive,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAArchiveWriterHostRef,
            *const ARAStoreObjectsFilter,
        ) -> ARABool,
        null_mut(),
        null()
    );
    reject!(
        "getProcessingAlgorithmsCount",
        getProcessingAlgorithmsCount,
        unsafe extern "C" fn(ARADocumentControllerRef) -> ARAInt32
    );
    reject!(
        "getProcessingAlgorithmProperties",
        getProcessingAlgorithmProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAInt32,
        ) -> *const ARAProcessingAlgorithmProperties,
        0
    );
    reject!(
        "getProcessingAlgorithmForAudioSource",
        getProcessingAlgorithmForAudioSource,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef) -> ARAInt32,
        null_mut()
    );
    reject!(
        "requestProcessingAlgorithmForAudioSource",
        requestProcessingAlgorithmForAudioSource,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef, ARAInt32),
        null_mut(),
        0
    );
    reject!(
        "isLicensedForCapabilities",
        isLicensedForCapabilities,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARABool,
            ARASize,
            *const ARAContentType,
            ARAPlaybackTransformationFlags,
        ) -> ARABool,
        kARAFalse,
        0,
        null(),
        0
    );
    reject!(
        "storeAudioSourceToAudioFileChunk",
        storeAudioSourceToAudioFileChunk,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAArchiveWriterHostRef,
            ARAAudioSourceRef,
            *mut ARAPersistentID,
            *mut ARABool,
        ) -> ARABool,
        null_mut(),
        null_mut(),
        null_mut(),
        null_mut()
    );
    reject!(
        "isAudioModificationPreservingAudioSourceSignal",
        isAudioModificationPreservingAudioSourceSignal,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioModificationRef) -> ARABool,
        null_mut()
    );
    driven
}

fn drive_content_callbacks(
    driven: &mut BTreeSet<&'static str>,
    controller: ARADocumentControllerRef,
    interface: *const ARADocumentControllerInterface,
    source: ARAAudioSourceRef,
    modification: ARAAudioModificationRef,
    region: ARAPlaybackRegionRef,
) {
    macro_rules! cb {
        ($field:ident, $type:ty) => {
            callback::<$type>(
                interface,
                offset_of!(ARADocumentControllerInterface, $field),
            )
        };
    }
    let source_available = cb!(
        isAudioSourceContentAvailable,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAContentType,
        ) -> ARABool
    );
    let source_grade = cb!(
        getAudioSourceContentGrade,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAContentType,
        ) -> ARAContentGrade
    );
    let source_reader = cb!(
        createAudioSourceContentReader,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioSourceRef,
            ARAContentType,
            *const ARAContentTimeRange,
        ) -> ARAContentReaderRef
    );
    let modification_available = cb!(
        isAudioModificationContentAvailable,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAContentType,
        ) -> ARABool
    );
    let modification_grade = cb!(
        getAudioModificationContentGrade,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAContentType,
        ) -> ARAContentGrade
    );
    let modification_reader = cb!(
        createAudioModificationContentReader,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAAudioModificationRef,
            ARAContentType,
            *const ARAContentTimeRange,
        ) -> ARAContentReaderRef
    );
    let region_available = cb!(
        isPlaybackRegionContentAvailable,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            ARAContentType,
        ) -> ARABool
    );
    let region_grade = cb!(
        getPlaybackRegionContentGrade,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            ARAContentType,
        ) -> ARAContentGrade
    );
    let region_reader = cb!(
        createPlaybackRegionContentReader,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAPlaybackRegionRef,
            ARAContentType,
            *const ARAContentTimeRange,
        ) -> ARAContentReaderRef
    );
    let count = cb!(
        getContentReaderEventCount,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAContentReaderRef) -> ARAInt32
    );
    let data = cb!(
        getContentReaderDataForEvent,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAContentReaderRef,
            ARAInt32,
        ) -> *const c_void
    );
    let destroy = cb!(
        destroyContentReader,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAContentReaderRef)
    );
    driven.insert("isAudioSourceContentAvailable");
    driven.insert("getAudioSourceContentGrade");
    driven.insert("createAudioSourceContentReader");
    driven.insert("isAudioModificationContentAvailable");
    driven.insert("getAudioModificationContentGrade");
    driven.insert("createAudioModificationContentReader");
    driven.insert("isPlaybackRegionContentAvailable");
    driven.insert("getPlaybackRegionContentGrade");
    driven.insert("createPlaybackRegionContentReader");
    driven.insert("getContentReaderEventCount");
    driven.insert("getContentReaderDataForEvent");
    driven.insert("destroyContentReader");
    // SAFETY: all object identities are live and each reader is consumed before another is made.
    unsafe {
        assert_ne!(
            source_available(controller, source, kARAContentTypeNotes as i32),
            kARAFalse
        );
        assert_eq!(
            source_grade(controller, source, kARAContentTypeNotes as i32),
            kARAContentGradeApproved as i32
        );
        let reader = source_reader(controller, source, kARAContentTypeNotes as i32, null());
        assert!(!reader.is_null());
        assert_eq!(count(controller, reader), 1);
        assert!(!data(controller, reader, 0).is_null());
        destroy(controller, reader);

        assert_ne!(
            modification_available(controller, modification, kARAContentTypeNotes as i32),
            kARAFalse
        );
        assert_eq!(
            modification_grade(controller, modification, kARAContentTypeNotes as i32),
            kARAContentGradeApproved as i32
        );
        let reader = modification_reader(
            controller,
            modification,
            kARAContentTypeNotes as i32,
            null(),
        );
        assert!(!reader.is_null());
        destroy(controller, reader);

        assert_ne!(
            region_available(controller, region, kARAContentTypeNotes as i32),
            kARAFalse
        );
        assert_eq!(
            region_grade(controller, region, kARAContentTypeNotes as i32),
            kARAContentGradeApproved as i32
        );
        let reader = region_reader(controller, region, kARAContentTypeNotes as i32, null());
        assert!(!reader.is_null());
        destroy(controller, reader);
    }
}

fn drive_processing_and_storage_callbacks(
    driven: &mut BTreeSet<&'static str>,
    controller: ARADocumentControllerRef,
    interface: *const ARADocumentControllerInterface,
    source: ARAAudioSourceRef,
    factory_archive_id: ARAPersistentID,
) {
    macro_rules! cb {
        ($field:ident, $type:ty) => {
            callback::<$type>(
                interface,
                offset_of!(ARADocumentControllerInterface, $field),
            )
        };
    }
    let algorithm_count = cb!(
        getProcessingAlgorithmsCount,
        unsafe extern "C" fn(ARADocumentControllerRef) -> ARAInt32
    );
    let algorithm_properties = cb!(
        getProcessingAlgorithmProperties,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAInt32,
        ) -> *const ARAProcessingAlgorithmProperties
    );
    let current_algorithm = cb!(
        getProcessingAlgorithmForAudioSource,
        unsafe extern "C" fn(ARADocumentControllerRef, ARAAudioSourceRef) -> ARAInt32
    );
    let store_objects = cb!(
        storeObjectsToArchive,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAArchiveWriterHostRef,
            *const ARAStoreObjectsFilter,
        ) -> ARABool
    );
    let store_chunk = cb!(
        storeAudioSourceToAudioFileChunk,
        unsafe extern "C" fn(
            ARADocumentControllerRef,
            ARAArchiveWriterHostRef,
            ARAAudioSourceRef,
            *mut ARAPersistentID,
            *mut ARABool,
        ) -> ARABool
    );
    driven.insert("getProcessingAlgorithmsCount");
    driven.insert("getProcessingAlgorithmProperties");
    driven.insert("getProcessingAlgorithmForAudioSource");
    driven.insert("storeObjectsToArchive");
    driven.insert("storeAudioSourceToAudioFileChunk");
    let mut archive_id = null();
    let mut open_automatically = kARAFalse;
    // SAFETY: all callbacks receive live controller/source identities and valid scalar outputs.
    unsafe {
        assert_eq!(algorithm_count(controller), 2);
        assert!(!algorithm_properties(controller, 1).is_null());
        assert_eq!(current_algorithm(controller, source), 1);
        assert_ne!(store_objects(controller, raw_ref(), null()), kARAFalse);
        assert_ne!(
            store_chunk(
                controller,
                raw_ref(),
                source,
                &raw mut archive_id,
                &raw mut open_automatically,
            ),
            kARAFalse
        );
    }
    assert_eq!(archive_id, factory_archive_id);
    assert_ne!(open_automatically, kARAFalse);
}

fn fixture_handle() -> RawHandle {
    enum Kind {}
    let mut registry = Registry::<Kind, ()>::new(1);
    registry.insert(()).unwrap().into_raw()
}

struct RawHost {
    audio: Box<ARAAudioAccessControllerInterface>,
    archive: Box<ARAArchivingControllerInterface>,
    content: Box<ARAContentAccessControllerInterface>,
    updates: Box<ARAModelUpdateControllerInterface>,
    playback: Box<ARAPlaybackControllerInterface>,
}

impl RawHost {
    fn new() -> Self {
        Self {
            audio: Box::new(ARAAudioAccessControllerInterface {
                structSize: std::mem::size_of::<ARAAudioAccessControllerInterface>(),
                createAudioReaderForSource: Some(raw_create_audio_reader),
                readAudioSamples: Some(raw_read_audio),
                destroyAudioReader: Some(raw_destroy_audio_reader),
            }),
            archive: Box::new(ARAArchivingControllerInterface {
                structSize: std::mem::size_of::<ARAArchivingControllerInterface>(),
                getArchiveSize: Some(raw_archive_size),
                readBytesFromArchive: Some(raw_archive_read),
                writeBytesToArchive: Some(raw_archive_write),
                notifyDocumentArchivingProgress: Some(raw_archive_progress),
                notifyDocumentUnarchivingProgress: Some(raw_archive_progress),
                getDocumentArchiveID: Some(raw_archive_id),
            }),
            content: Box::new(ARAContentAccessControllerInterface {
                structSize: std::mem::size_of::<ARAContentAccessControllerInterface>(),
                isMusicalContextContentAvailable: Some(raw_musical_available),
                getMusicalContextContentGrade: Some(raw_musical_grade),
                createMusicalContextContentReader: Some(raw_create_musical_reader),
                isAudioSourceContentAvailable: Some(raw_source_available),
                getAudioSourceContentGrade: Some(raw_source_grade),
                createAudioSourceContentReader: Some(raw_create_source_reader),
                getContentReaderEventCount: Some(raw_host_reader_count),
                getContentReaderDataForEvent: Some(raw_host_reader_data),
                destroyContentReader: Some(raw_destroy_host_reader),
            }),
            updates: Box::new(ARAModelUpdateControllerInterface {
                structSize: std::mem::size_of::<ARAModelUpdateControllerInterface>(),
                notifyAudioSourceAnalysisProgress: Some(raw_analysis_progress),
                notifyAudioSourceContentChanged: Some(raw_source_changed),
                notifyAudioModificationContentChanged: Some(raw_modification_changed),
                notifyPlaybackRegionContentChanged: Some(raw_region_changed),
                notifyDocumentDataChanged: Some(raw_document_changed),
            }),
            playback: Box::new(ARAPlaybackControllerInterface {
                structSize: std::mem::size_of::<ARAPlaybackControllerInterface>(),
                requestStartPlayback: Some(raw_start_playback),
                requestStopPlayback: Some(raw_stop_playback),
                requestSetPlaybackPosition: Some(raw_set_position),
                requestSetCycleRange: Some(raw_set_cycle),
                requestEnableCycle: Some(raw_enable_cycle),
            }),
        }
    }

    fn instance(&self) -> ARADocumentControllerHostInstance {
        ARADocumentControllerHostInstance {
            structSize: std::mem::size_of::<ARADocumentControllerHostInstance>(),
            audioAccessControllerHostRef: raw_ref(),
            audioAccessControllerInterface: self.audio.as_ref(),
            archivingControllerHostRef: raw_ref(),
            archivingControllerInterface: self.archive.as_ref(),
            contentAccessControllerHostRef: raw_ref(),
            contentAccessControllerInterface: self.content.as_ref(),
            modelUpdateControllerHostRef: raw_ref(),
            modelUpdateControllerInterface: self.updates.as_ref(),
            playbackControllerHostRef: raw_ref(),
            playbackControllerInterface: self.playback.as_ref(),
        }
    }
}

fn callback<T: Copy>(interface: *const ARADocumentControllerInterface, offset: usize) -> T {
    // SAFETY: contract tests request represented fields from the complete live 2.3 interface.
    unsafe {
        ara2_bridge_sys::access::read_field::<Option<T>>(interface.cast(), offset)
            .expect("represented callback")
    }
}

fn raw_ref<T>() -> *mut T {
    std::ptr::from_ref(&RAW_SENTINEL).cast_mut().cast()
}

static RAW_SENTINEL: AtomicUsize = AtomicUsize::new(0);
static RAW_ANALYSIS_PROGRESS_COUNT: AtomicUsize = AtomicUsize::new(0);
static RAW_ARCHIVE_ID: &CStr = c"org.ara2-bridge.test.archive";
static RAW_ARCHIVE: [u8; 3] = *b"ARA";

unsafe extern "C" fn raw_create_audio_reader(
    _: ARAAudioAccessControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: ARABool,
) -> ARAAudioReaderHostRef {
    raw_ref()
}

unsafe extern "C" fn raw_read_audio(
    _: ARAAudioAccessControllerHostRef,
    _: ARAAudioReaderHostRef,
    _: ARASamplePosition,
    _: ARASampleCount,
    _: *const *mut c_void,
) -> ARABool {
    kARATrue
}

unsafe extern "C" fn raw_destroy_audio_reader(
    _: ARAAudioAccessControllerHostRef,
    _: ARAAudioReaderHostRef,
) {
}

unsafe extern "C" fn raw_archive_size(
    _: ARAArchivingControllerHostRef,
    _: ARAArchiveReaderHostRef,
) -> ARASize {
    RAW_ARCHIVE.len()
}

unsafe extern "C" fn raw_archive_read(
    _: ARAArchivingControllerHostRef,
    _: ARAArchiveReaderHostRef,
    position: ARASize,
    length: ARASize,
    bytes: *mut ARAByte,
) -> ARABool {
    let Some(source) = RAW_ARCHIVE.get(position..position.saturating_add(length)) else {
        return kARAFalse;
    };
    if bytes.is_null() {
        return kARAFalse;
    }
    // SAFETY: the host callback contract supplies `length` writable output bytes.
    unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), bytes, length) };
    kARATrue
}

unsafe extern "C" fn raw_archive_write(
    _: ARAArchivingControllerHostRef,
    _: ARAArchiveWriterHostRef,
    _: ARASize,
    _: ARASize,
    _: *const ARAByte,
) -> ARABool {
    kARATrue
}

unsafe extern "C" fn raw_archive_progress(_: ARAArchivingControllerHostRef, _: f32) {}

unsafe extern "C" fn raw_archive_id(
    _: ARAArchivingControllerHostRef,
    _: ARAArchiveReaderHostRef,
) -> ARAPersistentID {
    RAW_ARCHIVE_ID.as_ptr()
}

unsafe extern "C" fn raw_musical_available(
    _: ARAContentAccessControllerHostRef,
    _: ARAMusicalContextHostRef,
    _: ARAContentType,
) -> ARABool {
    kARAFalse
}

unsafe extern "C" fn raw_musical_grade(
    _: ARAContentAccessControllerHostRef,
    _: ARAMusicalContextHostRef,
    _: ARAContentType,
) -> ARAContentGrade {
    kARAContentGradeInitial as i32
}

unsafe extern "C" fn raw_create_musical_reader(
    _: ARAContentAccessControllerHostRef,
    _: ARAMusicalContextHostRef,
    _: ARAContentType,
    _: *const ARAContentTimeRange,
) -> ARAContentReaderHostRef {
    null_mut()
}

unsafe extern "C" fn raw_source_available(
    _: ARAContentAccessControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: ARAContentType,
) -> ARABool {
    kARAFalse
}

unsafe extern "C" fn raw_source_grade(
    _: ARAContentAccessControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: ARAContentType,
) -> ARAContentGrade {
    kARAContentGradeInitial as i32
}

unsafe extern "C" fn raw_create_source_reader(
    _: ARAContentAccessControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: ARAContentType,
    _: *const ARAContentTimeRange,
) -> ARAContentReaderHostRef {
    null_mut()
}

unsafe extern "C" fn raw_host_reader_count(
    _: ARAContentAccessControllerHostRef,
    _: ARAContentReaderHostRef,
) -> ARAInt32 {
    0
}

unsafe extern "C" fn raw_host_reader_data(
    _: ARAContentAccessControllerHostRef,
    _: ARAContentReaderHostRef,
    _: ARAInt32,
) -> *const c_void {
    null()
}

unsafe extern "C" fn raw_destroy_host_reader(
    _: ARAContentAccessControllerHostRef,
    _: ARAContentReaderHostRef,
) {
}

unsafe extern "C" fn raw_analysis_progress(
    _: ARAModelUpdateControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: ARAAnalysisProgressState,
    _: f32,
) {
    RAW_ANALYSIS_PROGRESS_COUNT.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn raw_source_changed(
    _: ARAModelUpdateControllerHostRef,
    _: ARAAudioSourceHostRef,
    _: *const ARAContentTimeRange,
    _: ARAContentUpdateFlags,
) {
}

unsafe extern "C" fn raw_modification_changed(
    _: ARAModelUpdateControllerHostRef,
    _: ARAAudioModificationHostRef,
    _: *const ARAContentTimeRange,
    _: ARAContentUpdateFlags,
) {
}

unsafe extern "C" fn raw_region_changed(
    _: ARAModelUpdateControllerHostRef,
    _: ARAPlaybackRegionHostRef,
    _: *const ARAContentTimeRange,
    _: ARAContentUpdateFlags,
) {
}

unsafe extern "C" fn raw_document_changed(_: ARAModelUpdateControllerHostRef) {}
unsafe extern "C" fn raw_start_playback(_: ARAPlaybackControllerHostRef) {}
unsafe extern "C" fn raw_stop_playback(_: ARAPlaybackControllerHostRef) {}
unsafe extern "C" fn raw_set_position(_: ARAPlaybackControllerHostRef, _: ARATimePosition) {}
unsafe extern "C" fn raw_set_cycle(
    _: ARAPlaybackControllerHostRef,
    _: ARATimePosition,
    _: ARATimeDuration,
) {
}
unsafe extern "C" fn raw_enable_cycle(_: ARAPlaybackControllerHostRef, _: ARABool) {}
