#![cfg(all(feature = "audio-unit-v2", target_vendor = "apple"))]

use ara2_bridge_companion::audio_unit::ffi::*;
use ara2_bridge_companion::audio_unit::AudioUnitPluginAdapter;
use ara2_bridge_companion::{
    CompanionFactory, CompanionProcessorBinding, CompanionRoles, LifecycleEvent,
};
use ara2_bridge_core::AraError;
use ara2_bridge_sys::{ARAFactory, ARAPlugInExtensionInstance};
use ara2_bridge_testkit::{build_test_factory, TestPluginTrace};
use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, Ordering};

struct CallbackState {
    known: AtomicI32,
    assigned: AtomicI32,
    factory: *const ARAFactory,
    extension: *const ARAPlugInExtensionInstance,
}

unsafe extern "C" fn get_factory(context: *mut c_void) -> *const ARAFactory {
    // SAFETY: the native handler owns this boxed state until destruction.
    unsafe { (*context.cast::<CallbackState>()).factory }
}

unsafe extern "C" fn bind(
    context: *mut c_void,
    _: ara2_bridge_sys::ARADocumentControllerRef,
    known_roles: i32,
    assigned_roles: i32,
) -> *const ARAPlugInExtensionInstance {
    // SAFETY: the native handler owns this boxed state until destruction.
    let state = unsafe { &*context.cast::<CallbackState>() };
    state.known.store(known_roles, Ordering::SeqCst);
    state.assigned.store(assigned_roles, Ordering::SeqCst);
    state.extension
}

unsafe extern "C" fn drop_context(context: *mut c_void) {
    // SAFETY: the native handler invokes this exactly once at final destruction.
    drop(unsafe { Box::from_raw(context.cast::<CallbackState>()) });
}

#[test]
fn audio_unit_constants_match_ara_2_3() {
    assert_eq!(ARA_AUDIO_COMPONENT_TAG, "ARA");
    assert_eq!(ARA_AUDIO_UNIT_MAGIC, 0x4172_6121);
    assert_eq!(AUDIO_UNIT_PROPERTY_ARA_FACTORY, 0x4172_6146);
    assert_eq!(AUDIO_UNIT_PROPERTY_ARA_BINDING, 0x4172_6142);
    assert_eq!(AUDIO_UNIT_PROPERTY_ARA_BINDING_WITH_ROLES, 0x4172_6145);
    assert_eq!(AUDIO_UNIT_SCOPE_GLOBAL, 0);
}

#[test]
fn plugin_properties_validate_scope_size_magic_and_preserve_failures() {
    let factory = 0x101usize as *const ARAFactory;
    let extension = 0x202usize as *const ARAPlugInExtensionInstance;
    let state = Box::new(CallbackState {
        known: AtomicI32::new(-1),
        assigned: AtomicI32::new(-1),
        factory,
        extension,
    });
    let callbacks = Ara2AudioUnitPluginCallbacks {
        context: Box::into_raw(state).cast(),
        get_factory: Some(get_factory),
        bind: Some(bind),
        drop: Some(drop_context),
    };
    let mut handler = std::ptr::null_mut();
    // SAFETY: callbacks remain valid until the native handler is destroyed below.
    assert_eq!(
        unsafe { ara2_audio_unit_plugin_create(&callbacks, &mut handler) },
        0
    );

    let mut size = 0;
    let mut writable = 1;
    // SAFETY: handler and output pointers are live.
    assert_eq!(
        unsafe {
            ara2_audio_unit_plugin_get_property_info(
                handler,
                AUDIO_UNIT_PROPERTY_ARA_FACTORY,
                AUDIO_UNIT_SCOPE_GLOBAL,
                0,
                &mut size,
                &mut writable,
            )
        },
        0
    );
    assert_eq!(size as usize, std::mem::size_of::<AraAudioUnitFactory>());
    assert_eq!(writable, 0);

    let sentinel = 0x303usize as *const ARAFactory;
    let mut factory_record = AraAudioUnitFactory {
        in_out_magic_number: 0,
        out_factory: sentinel,
    };
    // SAFETY: record is writable and its exact size is supplied.
    assert_ne!(
        unsafe {
            ara2_audio_unit_plugin_get_property(
                handler,
                AUDIO_UNIT_PROPERTY_ARA_FACTORY,
                AUDIO_UNIT_SCOPE_GLOBAL,
                0,
                std::ptr::from_mut(&mut factory_record).cast(),
                std::mem::size_of_val(&factory_record) as u32,
            )
        },
        0
    );
    assert_eq!(factory_record.out_factory, sentinel);
    factory_record.in_out_magic_number = ARA_AUDIO_UNIT_MAGIC;
    // SAFETY: same exact live record with the required magic.
    assert_eq!(
        unsafe {
            ara2_audio_unit_plugin_get_property(
                handler,
                AUDIO_UNIT_PROPERTY_ARA_FACTORY,
                AUDIO_UNIT_SCOPE_GLOBAL,
                0,
                std::ptr::from_mut(&mut factory_record).cast(),
                std::mem::size_of_val(&factory_record) as u32,
            )
        },
        0
    );
    assert_eq!(factory_record.out_factory, factory);

    let mut binding = AraAudioUnitPluginExtensionBinding {
        in_out_magic_number: ARA_AUDIO_UNIT_MAGIC,
        in_document_controller_ref: 0x404usize as _,
        out_plugin_extension: std::ptr::null(),
        known_roles: 7,
        assigned_roles: 5,
    };
    // SAFETY: the role-aware record is complete and all pointers are opaque fixture sentinels.
    assert_eq!(
        unsafe {
            ara2_audio_unit_plugin_get_property(
                handler,
                AUDIO_UNIT_PROPERTY_ARA_BINDING_WITH_ROLES,
                AUDIO_UNIT_SCOPE_GLOBAL,
                0,
                std::ptr::from_mut(&mut binding).cast(),
                std::mem::size_of_val(&binding) as u32,
            )
        },
        0
    );
    assert_eq!(binding.out_plugin_extension, extension);

    // SAFETY: this is the unique destruction of the live handler.
    unsafe { ara2_audio_unit_plugin_destroy(handler) };
}

fn extension_instance(
    _: CompanionRoles,
    _: CompanionRoles,
) -> Result<*const ARAPlugInExtensionInstance, AraError> {
    Ok(0x505usize as *const ARAPlugInExtensionInstance)
}

#[test]
fn safe_plugin_adapter_binds_once_before_audio_unit_boundaries() {
    let factory = build_test_factory(TestPluginTrace::new()).unwrap();
    // SAFETY: fixture factory backing remains live until after adapter destruction.
    let factory_ref = unsafe { &*factory.as_raw() };
    // SAFETY: explicit test drop order gives this callback registration static backing.
    let factory_ref =
        unsafe { std::mem::transmute::<&ARAFactory, &'static ARAFactory>(factory_ref) };
    // SAFETY: manually extended backing remains live throughout this test.
    let association =
        unsafe { CompanionFactory::from_raw("test.audio-unit", factory_ref) }.unwrap();
    let processor = CompanionProcessorBinding::new([association], CompanionRoles::all()).unwrap();
    let adapter =
        AudioUnitPluginAdapter::new(processor, "test.audio-unit", extension_instance).unwrap();

    let mut factory_record = AraAudioUnitFactory {
        in_out_magic_number: ARA_AUDIO_UNIT_MAGIC,
        out_factory: std::ptr::null(),
    };
    // SAFETY: complete live factory record carries the required input magic.
    assert_eq!(
        unsafe {
            adapter.get_property(
                AUDIO_UNIT_PROPERTY_ARA_FACTORY,
                AUDIO_UNIT_SCOPE_GLOBAL,
                0,
                std::ptr::from_mut(&mut factory_record).cast(),
                std::mem::size_of_val(&factory_record) as u32,
            )
        },
        0
    );
    assert_eq!(factory_record.out_factory, factory.as_raw());

    let mut binding = AraAudioUnitPluginExtensionBinding {
        in_out_magic_number: ARA_AUDIO_UNIT_MAGIC,
        in_document_controller_ref: 0x606usize as _,
        out_plugin_extension: std::ptr::null(),
        known_roles: CompanionRoles::all().bits(),
        assigned_roles: CompanionRoles::all().bits(),
    };
    // SAFETY: complete live binding record uses an opaque controller kept valid through teardown.
    assert_eq!(
        unsafe {
            adapter.get_property(
                AUDIO_UNIT_PROPERTY_ARA_BINDING_WITH_ROLES,
                AUDIO_UNIT_SCOPE_GLOBAL,
                0,
                std::ptr::from_mut(&mut binding).cast(),
                std::mem::size_of_val(&binding) as u32,
            )
        },
        0
    );
    assert_eq!(binding.out_plugin_extension, 0x505usize as _);
    // SAFETY: repeated property call is well-formed but must be rejected by one-shot state.
    assert_ne!(
        unsafe {
            adapter.get_property(
                AUDIO_UNIT_PROPERTY_ARA_BINDING_WITH_ROLES,
                AUDIO_UNIT_SCOPE_GLOBAL,
                0,
                std::ptr::from_mut(&mut binding).cast(),
                std::mem::size_of_val(&binding) as u32,
            )
        },
        0
    );
    adapter.observe(LifecycleEvent::Activate).unwrap();
    adapter.observe(LifecycleEvent::Deactivate).unwrap();
    adapter.observe_controller_destruction().unwrap();
}
