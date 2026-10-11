use ara2_bridge_core::{
    ApiGeneration, AudioModificationProperties, DocumentProperties, MusicalContextProperties,
    PlaybackRegionProperties, RegionSequenceProperties,
};
use ara2_bridge_host::DocumentSession;
use ara2_bridge_testkit::{
    build_test_factory, test_audio_source_properties, TestHost, TestPluginTrace,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct TrackingAllocator;

// SAFETY: allocation and deallocation are forwarded unchanged to the system allocator; the
// thread-local counter is observational and does not alter allocation semantics.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TRACKING.with(|tracking| {
            if tracking.get() {
                ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
            }
        });
        // SAFETY: this forwards the allocator contract and exact layout to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: this forwards the pointer and layout received from the matching allocation.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn designated_head_tail_callback_is_bounded_and_allocation_free() {
    let generation = ApiGeneration::V23Final;
    let factory = build_test_factory(TestPluginTrace::new()).unwrap();
    let host = TestHost::new(generation).unwrap();
    let loaded = host.load_factory(&factory).unwrap();
    let mut session = DocumentSession::new(
        &loaded,
        host.services(),
        DocumentProperties::new(Some("Realtime audit")).unwrap(),
    )
    .unwrap();
    let region = {
        let mut edit = session.edit().unwrap();
        let context = edit
            .create_musical_context(MusicalContextProperties::new(Some("Music"), 0, None).unwrap())
            .unwrap();
        let sequence = edit
            .create_region_sequence(
                RegionSequenceProperties::new(
                    Some("Sequence"),
                    0,
                    edit.musical_context_ref(context).unwrap(),
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let source = edit
            .create_audio_source(test_audio_source_properties().unwrap())
            .unwrap();
        let modification = edit
            .create_audio_modification(
                source,
                AudioModificationProperties::new(Some("Modification"), "realtime-modification")
                    .unwrap(),
            )
            .unwrap();
        let region = edit
            .create_playback_region(
                modification,
                PlaybackRegionProperties::for_ara2(
                    0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    edit.region_sequence_ref(sequence).unwrap(),
                    Some("Region"),
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        edit.finish().unwrap();
        region
    };

    ALLOCATIONS.with(|allocations| allocations.set(0));
    TRACKING.with(|tracking| tracking.set(true));
    let result = session.playback_region_head_and_tail_time(region);
    TRACKING.with(|tracking| tracking.set(false));

    assert_eq!(result.unwrap(), (0.125, 0.25));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    session.close().unwrap();
}

#[test]
fn designated_callback_source_has_no_blocking_io_or_logging_operations() {
    let source = include_str!("../../ara2-bridge-plugin/src/realtime.rs");
    let callback = source
        .split("impl crate::ffi::generated_callbacks::ControllerDelegate for HeadTailDispatch")
        .nth(1)
        .unwrap()
        .split("impl RealtimeHeadTailAdapter")
        .next()
        .unwrap();

    for forbidden in [
        ".lock(",
        ".read()",
        ".write()",
        "std::fs",
        "File::",
        "OpenOptions",
        "println!",
        "eprintln!",
        "dbg!",
        "log::",
        "tracing::",
    ] {
        assert!(
            !callback.contains(forbidden),
            "realtime callback contains forbidden operation {forbidden}"
        );
    }
}
