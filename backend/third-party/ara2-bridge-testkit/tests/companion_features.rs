use ara2_bridge_companion::{CompanionRoles, LifecycleEvent};

#[test]
fn neutral_companion_surface_is_available_without_native_features() {
    assert!(CompanionRoles::all().contains(CompanionRoles::PLAYBACK_RENDERER));
    assert_eq!(LifecycleEvent::Activate, LifecycleEvent::Activate);
}

#[cfg(feature = "clap")]
#[test]
fn clap_feature_is_additive() {
    assert_eq!(
        ara2_bridge_companion::clap::sys::CLAP_EXT_ARA_FACTORY,
        "org.ara-audio.ara.factory/2"
    );
}

#[cfg(feature = "vst3")]
#[test]
fn vst3_feature_is_additive() {
    assert_eq!(
        ara2_bridge_companion::vst3::ffi::Ara2Vst3InterfaceKind::MainFactory as i32,
        1
    );
}

#[cfg(feature = "audio-unit-v2")]
#[test]
fn audio_unit_feature_is_additive() {
    assert_eq!(
        ara2_bridge_companion::audio_unit::ffi::ARA_AUDIO_UNIT_MAGIC,
        0x4172_6121
    );
}
