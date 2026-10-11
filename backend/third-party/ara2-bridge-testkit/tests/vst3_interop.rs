#![cfg(feature = "vst3")]

use ara2_bridge_companion::vst3::{
    match_vst3_classes, Vst3AraMainClass, Vst3HostMainFactory, Vst3HostPlugin,
    Vst3MainFactoryAdapter, Vst3PluginEntryAdapter, Vst3ProcessorClass,
};
use ara2_bridge_companion::{
    CompanionFactory, CompanionProcessorBinding, CompanionRoles, LifecycleEvent,
};
use ara2_bridge_core::AraError;
use ara2_bridge_sys::ARAPlugInExtensionInstance;
use ara2_bridge_testkit::{build_test_factory, TestPluginTrace};
use std::sync::atomic::AtomicU8;

static EXTENSION_INSTANCE: AtomicU8 = AtomicU8::new(0);

fn extension_instance(
    _: CompanionRoles,
    _: CompanionRoles,
) -> Result<*const ARAPlugInExtensionInstance, AraError> {
    Ok(std::ptr::from_ref(&EXTENSION_INSTANCE).cast())
}

#[test]
fn main_factory_and_processor_entry_publish_identical_factory_pointers() {
    let factory = build_test_factory(TestPluginTrace::new()).unwrap();
    // SAFETY: fixture backing remains stable and adapters are dropped before the factory.
    let factory_ref = unsafe { &*factory.as_raw() };
    // SAFETY: declared drop order gives this test the process-lifetime callback contract.
    let factory_ref = unsafe {
        std::mem::transmute::<&ara2_bridge_sys::ARAFactory, &'static ara2_bridge_sys::ARAFactory>(
            factory_ref,
        )
    };
    // SAFETY: manually extended fixture backing remains valid through all adapters.
    let association =
        unsafe { CompanionFactory::from_raw("ARA2 Bridge Test Plug-In", factory_ref) }.unwrap();
    let main =
        Vst3MainFactoryAdapter::new("ARA2 Bridge Test Plug-In", association.clone()).unwrap();
    let processor = CompanionProcessorBinding::new([association], CompanionRoles::all()).unwrap();
    let plugin =
        Vst3PluginEntryAdapter::new(processor, "ARA2 Bridge Test Plug-In", extension_instance)
            .unwrap();

    // SAFETY: both native adapters retain live COM objects through host discovery.
    let host_main = unsafe { Vst3HostMainFactory::discover(main.as_raw()) }.unwrap();
    // SAFETY: same live COM identity contract for the processor entry.
    let host_plugin = unsafe { Vst3HostPlugin::discover(plugin.as_raw()) }.unwrap();
    assert!(host_plugin.supports_role_aware_binding());
    assert_eq!(host_main.factory().unwrap(), factory.as_raw());
    assert_eq!(host_plugin.factory().unwrap(), factory.as_raw());

    let mut controller_storage = Box::new(0_u8);
    let controller = (&mut *controller_storage as *mut u8).cast();
    let roles = CompanionRoles::all();
    // SAFETY: controller storage remains live until the controller-first notification below.
    let extension = unsafe { host_plugin.bind(controller, roles, roles, true) }.unwrap();
    assert_eq!(extension, std::ptr::from_ref(&EXTENSION_INSTANCE).cast());
    assert!(unsafe { host_plugin.bind(controller, roles, roles, true) }.is_err());
    plugin.observe(LifecycleEvent::Activate).unwrap();
    plugin.observe(LifecycleEvent::Process).unwrap();
    plugin.observe(LifecycleEvent::Deactivate).unwrap();
    plugin.observe_controller_destruction().unwrap();
    assert!(plugin.observe(LifecycleEvent::ModelMutation).is_err());
}

#[test]
fn queried_com_reference_keeps_callback_state_alive_after_adapter_drop() {
    let factory = build_test_factory(TestPluginTrace::new()).unwrap();
    // SAFETY: factory outlives all native references in this test.
    let factory_ref = unsafe { &*factory.as_raw() };
    // SAFETY: test drop order provides the required stable callback lifetime.
    let factory_ref = unsafe {
        std::mem::transmute::<&ara2_bridge_sys::ARAFactory, &'static ara2_bridge_sys::ARAFactory>(
            factory_ref,
        )
    };
    // SAFETY: stable fixture backing remains live until test exit.
    let association =
        unsafe { CompanionFactory::from_raw("ARA2 Bridge Test Plug-In", factory_ref) }.unwrap();
    let processor = CompanionProcessorBinding::new([association], CompanionRoles::all()).unwrap();
    let plugin =
        Vst3PluginEntryAdapter::new(processor, "ARA2 Bridge Test Plug-In", extension_instance)
            .unwrap();
    // SAFETY: discovery acquires its own owning COM reference.
    let host = unsafe { Vst3HostPlugin::discover(plugin.as_raw()) }.unwrap();
    drop(plugin);
    assert_eq!(host.factory().unwrap(), factory.as_raw());
}

#[test]
fn class_matching_rejects_name_id_and_factory_ambiguity() {
    let factory = build_test_factory(TestPluginTrace::new()).unwrap();
    // SAFETY: fixture factory and display-string backing remain live through all copies.
    let main = unsafe {
        Vst3AraMainClass::from_raw([1; 16], "ARA2 Bridge Test Plug-In", factory.as_raw())
    }
    .unwrap();
    let processor = Vst3ProcessorClass::new([2; 16], "ARA2 Bridge Test Plug-In").unwrap();
    let matches = match_vst3_classes([main.clone()], [processor.clone()]).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].main_factory.factory(), factory.as_raw());
    assert_eq!(matches[0].processor, processor);

    assert!(match_vst3_classes([main.clone(), main.clone()], [processor.clone()]).is_err());
    let wrong_processor = Vst3ProcessorClass::new([3; 16], "Different Processor").unwrap();
    assert!(match_vst3_classes([main], [wrong_processor]).is_err());
    // SAFETY: same live fixture backing; only the intentionally wrong class name changes.
    let inconsistent =
        unsafe { Vst3AraMainClass::from_raw([4; 16], "Different Processor", factory.as_raw()) }
            .unwrap();
    assert!(match_vst3_classes([inconsistent], [processor]).is_err());
}
