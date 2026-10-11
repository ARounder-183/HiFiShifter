#![cfg(feature = "clap")]

use ara2_bridge_companion::clap::sys::*;
use std::ffi::{c_char, CStr};
use std::mem::{align_of, offset_of, size_of};

#[link(name = "ara2_clap_probe", kind = "static")]
extern "C" {
    fn ara2_clap_version_major() -> u32;
    fn ara2_clap_version_minor() -> u32;
    fn ara2_clap_version_revision() -> u32;
    fn ara2_clap_ara_factory_id() -> *const c_char;
    fn ara2_clap_ara_factory_compat_id() -> *const c_char;
    fn ara2_clap_ara_plugin_extension_id() -> *const c_char;
    fn ara2_clap_ara_plugin_extension_compat_id() -> *const c_char;
    fn ara2_clap_ara_supported_feature() -> *const c_char;
    fn ara2_clap_ara_required_feature() -> *const c_char;
    fn ara2_sizeof_clap_ara_factory() -> usize;
    fn ara2_alignof_clap_ara_factory() -> usize;
    fn ara2_offset_clap_ara_factory_count() -> usize;
    fn ara2_offset_clap_ara_factory_factory() -> usize;
    fn ara2_offset_clap_ara_factory_plugin_id() -> usize;
    fn ara2_sizeof_clap_ara_plugin_extension() -> usize;
    fn ara2_alignof_clap_ara_plugin_extension() -> usize;
    fn ara2_offset_clap_ara_plugin_extension_factory() -> usize;
    fn ara2_offset_clap_ara_plugin_extension_bind() -> usize;
    fn ara2_sizeof_clap_plugin_entry() -> usize;
    fn ara2_sizeof_clap_plugin() -> usize;
    fn ara2_sizeof_clap_plugin_factory() -> usize;
}

unsafe fn c_string(function: unsafe extern "C" fn() -> *const c_char) -> String {
    // SAFETY: each native probe returns one process-lifetime static header string.
    unsafe { CStr::from_ptr(function()) }
        .to_str()
        .unwrap()
        .to_owned()
}

#[test]
fn direct_clap_ara_declarations_match_the_pinned_headers() {
    // SAFETY: all probes are pure constant/layout queries compiled from pinned headers.
    unsafe {
        assert_eq!(ara2_clap_version_major(), CLAP_VERSION.major);
        assert_eq!(ara2_clap_version_minor(), CLAP_VERSION.minor);
        assert_eq!(ara2_clap_version_revision(), CLAP_VERSION.revision);
        assert_eq!(c_string(ara2_clap_ara_factory_id), CLAP_EXT_ARA_FACTORY);
        assert_eq!(
            c_string(ara2_clap_ara_factory_compat_id),
            CLAP_EXT_ARA_FACTORY_COMPAT
        );
        assert_eq!(
            c_string(ara2_clap_ara_plugin_extension_id),
            CLAP_EXT_ARA_PLUGIN_EXTENSION
        );
        assert_eq!(
            c_string(ara2_clap_ara_plugin_extension_compat_id),
            CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT
        );
        assert_eq!(
            c_string(ara2_clap_ara_supported_feature),
            CLAP_PLUGIN_FEATURE_ARA_SUPPORTED
        );
        assert_eq!(
            c_string(ara2_clap_ara_required_feature),
            CLAP_PLUGIN_FEATURE_ARA_REQUIRED
        );

        assert_eq!(ara2_sizeof_clap_ara_factory(), size_of::<ClapAraFactory>());
        assert_eq!(
            ara2_alignof_clap_ara_factory(),
            align_of::<ClapAraFactory>()
        );
        assert_eq!(
            ara2_offset_clap_ara_factory_count(),
            offset_of!(ClapAraFactory, get_factory_count)
        );
        assert_eq!(
            ara2_offset_clap_ara_factory_factory(),
            offset_of!(ClapAraFactory, get_ara_factory)
        );
        assert_eq!(
            ara2_offset_clap_ara_factory_plugin_id(),
            offset_of!(ClapAraFactory, get_plugin_id)
        );
        assert_eq!(
            ara2_sizeof_clap_ara_plugin_extension(),
            size_of::<ClapAraPluginExtension>()
        );
        assert_eq!(
            ara2_alignof_clap_ara_plugin_extension(),
            align_of::<ClapAraPluginExtension>()
        );
        assert_eq!(
            ara2_offset_clap_ara_plugin_extension_factory(),
            offset_of!(ClapAraPluginExtension, get_factory)
        );
        assert_eq!(
            ara2_offset_clap_ara_plugin_extension_bind(),
            offset_of!(ClapAraPluginExtension, bind_to_document_controller)
        );
        assert_eq!(
            ara2_sizeof_clap_plugin_entry(),
            size_of::<ClapPluginEntry>()
        );
        assert_eq!(ara2_sizeof_clap_plugin(), size_of::<ClapPlugin>());
        assert_eq!(
            ara2_sizeof_clap_plugin_factory(),
            size_of::<ClapPluginFactory>()
        );
    }
}
