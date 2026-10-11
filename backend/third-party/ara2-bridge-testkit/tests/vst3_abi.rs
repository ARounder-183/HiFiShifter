#![cfg(feature = "vst3")]

use ara2_bridge_companion::vst3::ffi::*;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

#[repr(C)]
struct CallbackState {
    binds: AtomicU32,
    known: AtomicI32,
    assigned: AtomicI32,
    factory: *const c_void,
    extension: *const c_void,
}

unsafe extern "C" fn get_factory(context: *mut c_void) -> *const c_void {
    // SAFETY: the adapter retains the boxed fixture context until its final release.
    unsafe { (*context.cast::<CallbackState>()).factory }
}

unsafe extern "C" fn bind(
    context: *mut c_void,
    _: *mut c_void,
    known_roles: i32,
    assigned_roles: i32,
) -> *const c_void {
    // SAFETY: the adapter retains the boxed fixture context until its final release.
    let state = unsafe { &*context.cast::<CallbackState>() };
    state.binds.fetch_add(1, Ordering::SeqCst);
    state.known.store(known_roles, Ordering::SeqCst);
    state.assigned.store(assigned_roles, Ordering::SeqCst);
    state.extension
}

unsafe extern "C" fn drop_context(context: *mut c_void) {
    // SAFETY: this is the single final callback for the Box transferred to the adapter.
    drop(unsafe { Box::from_raw(context.cast::<CallbackState>()) });
}

fn fixture_state() -> Box<CallbackState> {
    Box::new(CallbackState {
        binds: AtomicU32::new(0),
        known: AtomicI32::new(-1),
        assigned: AtomicI32::new(-1),
        factory: 0x101usize as *const c_void,
        extension: 0x202usize as *const c_void,
    })
}

#[test]
fn shim_matches_iids_category_and_exception_boundary() {
    let cases = [
        (Ara2Vst3InterfaceKind::Unknown, [0, 0, 0xC0000000, 0x46]),
        (
            Ara2Vst3InterfaceKind::MainFactory,
            [0xDB2A1669, 0xFAFD42A5, 0xA82F864F, 0x7B6872EA],
        ),
        (
            Ara2Vst3InterfaceKind::PluginEntry,
            [0x12814E54, 0xA1CE4076, 0x82B96813, 0x16950BD6],
        ),
        (
            Ara2Vst3InterfaceKind::PluginEntry2,
            [0xCD9A5913, 0xC9EB46D7, 0x96CA53AD, 0xD1DB89F5],
        ),
    ];
    for (kind, expected) in cases {
        let mut actual = Ara2Vst3InterfaceId { words: [0; 4] };
        // SAFETY: output points to writable storage for the complete IID record.
        assert_eq!(
            unsafe { ara2_vst3_interface_id(kind, &mut actual) },
            ARA2_VST3_OK
        );
        assert_eq!(actual.words, expected);
    }
    // SAFETY: the shim returns a process-lifetime static NUL-terminated category string.
    let category = unsafe { CStr::from_ptr(ara2_vst3_main_factory_category()) };
    assert_eq!(category.to_bytes(), b"ARA Main Factory Class");
    // SAFETY: the closed probe has no pointer preconditions.
    assert_eq!(
        unsafe { ara2_vst3_probe_exception_boundary(0) },
        ARA2_VST3_OK
    );
    // SAFETY: the closed probe catches its deliberately thrown C++ exception.
    assert_eq!(
        unsafe { ara2_vst3_probe_exception_boundary(1) },
        ARA2_VST3_EXCEPTION
    );
}

#[test]
fn main_factory_query_and_reference_ownership_are_balanced() {
    let state = fixture_state();
    let factory = state.factory;
    let callbacks = Ara2Vst3MainFactoryCallbacks {
        context: Box::into_raw(state).cast(),
        get_factory: Some(get_factory),
        drop: Some(drop_context),
    };
    let mut object = std::ptr::null_mut();
    // SAFETY: callbacks and context remain valid until the final native release.
    assert_eq!(
        unsafe { ara2_vst3_main_factory_create(&callbacks, &mut object) },
        ARA2_VST3_OK
    );
    let mut queried = std::ptr::null_mut();
    // SAFETY: object owns a live COM reference and output is writable.
    assert_eq!(
        unsafe {
            ara2_vst3_query_interface(object, Ara2Vst3InterfaceKind::MainFactory, &mut queried)
        },
        ARA2_VST3_OK
    );
    let mut actual_factory = std::ptr::null();
    // SAFETY: queried supports IMainFactory and output is writable.
    assert_eq!(
        unsafe { ara2_vst3_main_factory_get_factory(queried, &mut actual_factory) },
        ARA2_VST3_OK
    );
    assert_eq!(actual_factory, factory);
    let mut references = 0;
    // SAFETY: queried owns the reference returned by queryInterface.
    assert_eq!(
        unsafe { ara2_vst3_release(queried, &mut references) },
        ARA2_VST3_OK
    );
    assert_eq!(references, 1);
    // SAFETY: object retains the original owning reference.
    assert_eq!(
        unsafe { ara2_vst3_release(object, &mut references) },
        ARA2_VST3_OK
    );
    assert_eq!(references, 0);
}

#[test]
fn entry_supports_legacy_and_role_aware_binding() {
    let state = fixture_state();
    let factory = state.factory;
    let extension = state.extension;
    let callbacks = Ara2Vst3PluginEntryCallbacks {
        context: Box::into_raw(state).cast(),
        get_factory: Some(get_factory),
        bind: Some(bind),
        drop: Some(drop_context),
    };
    let mut object = std::ptr::null_mut();
    // SAFETY: callbacks and context remain valid until the final native release.
    assert_eq!(
        unsafe { ara2_vst3_plugin_entry_create(&callbacks, &mut object) },
        ARA2_VST3_OK
    );
    let mut actual_factory = std::ptr::null();
    // SAFETY: object supports IPlugInEntryPoint and output is writable.
    assert_eq!(
        unsafe { ara2_vst3_plugin_entry_get_factory(object, &mut actual_factory) },
        ARA2_VST3_OK
    );
    assert_eq!(actual_factory, factory);
    let mut entry2 = std::ptr::null_mut();
    // SAFETY: object owns a live COM reference and output is writable.
    assert_eq!(
        unsafe {
            ara2_vst3_query_interface(object, Ara2Vst3InterfaceKind::PluginEntry2, &mut entry2)
        },
        ARA2_VST3_OK
    );
    let mut canonical_unknown = std::ptr::null_mut();
    // SAFETY: entry2 owns a live secondary-interface reference and output is writable.
    assert_eq!(
        unsafe {
            ara2_vst3_query_interface(
                entry2,
                Ara2Vst3InterfaceKind::Unknown,
                &mut canonical_unknown,
            )
        },
        ARA2_VST3_OK
    );
    assert_eq!(canonical_unknown, object);
    let controller = 0x303usize as *mut c_void;
    let mut actual_extension = std::ptr::null();
    // SAFETY: callbacks treat controller as opaque and return stable sentinel backing.
    assert_eq!(
        unsafe { ara2_vst3_plugin_entry_bind(object, controller, 7, 5, 1, &mut actual_extension) },
        ARA2_VST3_OK
    );
    assert_eq!(actual_extension, extension);
    actual_extension = std::ptr::null();
    // SAFETY: legacy entry uses the same opaque fixture contract.
    assert_eq!(
        unsafe { ara2_vst3_plugin_entry_bind(object, controller, 7, 5, 0, &mut actual_extension) },
        ARA2_VST3_OK
    );
    assert_eq!(actual_extension, extension);
    let mut references = 0;
    // SAFETY: canonical_unknown owns the reference returned by queryInterface.
    assert_eq!(
        unsafe { ara2_vst3_release(canonical_unknown, &mut references) },
        ARA2_VST3_OK
    );
    assert_eq!(references, 2);
    // SAFETY: entry2 owns the reference returned by queryInterface.
    assert_eq!(
        unsafe { ara2_vst3_release(entry2, &mut references) },
        ARA2_VST3_OK
    );
    assert_eq!(references, 1);
    // SAFETY: object retains its original owning reference.
    assert_eq!(
        unsafe { ara2_vst3_release(object, &mut references) },
        ARA2_VST3_OK
    );
    assert_eq!(references, 0);
}
