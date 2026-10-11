#![cfg(feature = "clap")]

use ara2_bridge_companion::clap::sys::{ClapPlugin, CLAP_EXT_ARA_FACTORY_COMPAT};
use ara2_bridge_companion::clap::{
    clap_ara_get_extension, ClapAraEntry, ClapAraHostFactory, ClapAraHostPlugin,
    ClapAraPluginAdapter,
};
use ara2_bridge_companion::{CompanionFactory, CompanionProcessorBinding, CompanionRoles};
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
fn clap_factory_discovers_only_associated_ara_plugins_with_stable_identity() {
    let first = build_test_factory(TestPluginTrace::new()).unwrap();
    let second = build_test_factory(TestPluginTrace::new()).unwrap();
    // SAFETY: fixture factories retain their stable raw records through entry teardown.
    let first_association =
        unsafe { CompanionFactory::from_raw("test.ara.first", &*first.as_raw()).unwrap() };
    // SAFETY: same entry-lifetime fixture contract.
    let second_association =
        unsafe { CompanionFactory::from_raw("test.ara.second", &*second.as_raw()).unwrap() };
    let entry = ClapAraEntry::new([first_association, second_association]).unwrap();
    assert_eq!(entry.factory_count(), 2);
    assert!(!entry.supports_factory_id("unknown.factory"));
    assert!(entry.supports_factory_id(CLAP_EXT_ARA_FACTORY_COMPAT));

    // SAFETY: entry retains its complete factory interface and association backing.
    let discovered = unsafe { ClapAraHostFactory::discover(entry.as_raw()) }.unwrap();
    assert_eq!(discovered.factory_count(), 2);
    let first_discovered = discovered.factory(0).unwrap();
    assert_eq!(first_discovered.plugin_id(), "test.ara.first");
    assert_eq!(first_discovered.ara_factory(), first.as_raw());
    let second_discovered = discovered.factory(1).unwrap();
    assert_eq!(second_discovered.plugin_id(), "test.ara.second");
    assert_eq!(second_discovered.ara_factory(), second.as_raw());
}

#[test]
fn clap_plugin_extension_binds_once_before_lifecycle_boundaries() {
    let factory = build_test_factory(TestPluginTrace::new()).unwrap();
    // SAFETY: factory raw backing is stable. The adapter is declared later and therefore drops
    // before `factory`; this test manually supplies the process-lifetime registry contract without
    // leaking the fixture.
    let factory_ref = unsafe { &*factory.as_raw() };
    // SAFETY: drop order described immediately above keeps the reference live for all uses.
    let factory_ref = unsafe {
        std::mem::transmute::<&ara2_bridge_sys::ARAFactory, &'static ara2_bridge_sys::ARAFactory>(
            factory_ref,
        )
    };
    // SAFETY: manually extended fixture backing remains valid through adapter teardown.
    let association =
        unsafe { CompanionFactory::from_raw("test.ara.plugin", factory_ref).unwrap() };
    let processor = CompanionProcessorBinding::new([association], CompanionRoles::all()).unwrap();
    let mut plugin = Box::new(ClapPlugin {
        desc: std::ptr::null(),
        plugin_data: std::ptr::null_mut(),
        init: None,
        destroy: None,
        activate: None,
        deactivate: None,
        start_processing: None,
        stop_processing: None,
        reset: None,
        process: None,
        get_extension: Some(clap_ara_get_extension),
        on_main_thread: None,
    });
    let plugin_pointer = std::ptr::from_mut(&mut *plugin).cast_const();
    // SAFETY: the boxed CLAP identity remains stable until the adapter is dropped.
    let adapter = unsafe {
        ClapAraPluginAdapter::attach(
            plugin_pointer,
            processor,
            "test.ara.plugin",
            extension_instance,
        )
    }
    .unwrap();
    assert_eq!(adapter.factory(), factory.as_raw());
    {
        // SAFETY: plug-in and registered extension adapter remain live.
        let host = unsafe { ClapAraHostPlugin::discover(plugin_pointer) }.unwrap();
        assert_eq!(host.factory().unwrap(), factory.as_raw());

        let mut controller_storage = Box::new(0_u8);
        let controller = (&mut *controller_storage as *mut u8).cast();
        let roles = CompanionRoles::all();
        // SAFETY: controller storage remains live through adapter teardown.
        let extension = unsafe { host.bind(controller, roles, roles) }.unwrap();
        assert_eq!(extension, std::ptr::from_ref(&EXTENSION_INSTANCE).cast());
        assert!(unsafe { host.bind(controller, roles, roles) }.is_err());
        adapter.observe_activation().unwrap();
        adapter.observe_processing().unwrap();
        assert!(adapter.observe_state_load().is_err());
        adapter.observe_deactivation().unwrap();
        adapter.observe_controller_destruction().unwrap();
        assert!(adapter.observe_state_load().is_err());
    }
    drop(adapter);
}
