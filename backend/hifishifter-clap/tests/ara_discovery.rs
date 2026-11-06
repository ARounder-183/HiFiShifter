//! 阶段 2 前置判据：ARA-over-CLAP 的 ID 与发现表 ABI 必须与上游一致。
//!
//! 【为什么用 dev-dependency】`ara2-bridge-companion` 的 `clap` feature 是纯
//! checked-in 声明，build.rs 对它不要求任何 SDK；只挂在 dev-dependencies 上，
//! 既不进产品依赖图，也不会把 VST3 SDK / `audio-unit-v2` 拖进六平台构建。
//! 这条测试回答的是"等共享 ARA runtime 就绪时，我们能不能直接接上"，
//! 而不是"现在就暴露 ARA 能力"——后者仍按阶段 1 契约返回 null。

use ara2_bridge_companion::clap::sys::{
    CLAP_EXT_ARA_FACTORY, CLAP_EXT_ARA_FACTORY_COMPAT, CLAP_EXT_ARA_PLUGIN_EXTENSION,
    CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT, CLAP_PLUGIN_FEATURE_ARA_SUPPORTED,
};
use ara2_bridge_companion::clap::{ClapAraEntry, ClapAraHostFactory};
use ara2_bridge_companion::{CompanionFactory, CompanionRoles};
use ara2_bridge_sys::ARAFactory;
use hifishifter_clap::ara as ours;
use std::mem::size_of;

/// 与上游 `tests/binding.rs` 同形的最小工厂：本测试只查 ID 与表的布局，
/// 不消费任何 ARA 回调，因此把回调留空是安全的。
fn factory_fixture() -> Box<ARAFactory> {
    Box::new(ARAFactory {
        structSize: size_of::<ARAFactory>(),
        ..unsafe { std::mem::zeroed() }
    })
}

#[test]
fn the_four_ara_extension_ids_match_the_pinned_companion() {
    assert_eq!(ours::CLAP_EXT_ARA_FACTORY, CLAP_EXT_ARA_FACTORY);
    assert_eq!(
        ours::CLAP_EXT_ARA_FACTORY_COMPAT,
        CLAP_EXT_ARA_FACTORY_COMPAT
    );
    assert_eq!(
        ours::CLAP_EXT_ARA_PLUGIN_EXTENSION,
        CLAP_EXT_ARA_PLUGIN_EXTENSION
    );
    assert_eq!(
        ours::CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT,
        CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT
    );
    assert_eq!(CLAP_PLUGIN_FEATURE_ARA_SUPPORTED, "ara:supported");
}

#[test]
fn a_real_clap_ara_entry_exposes_one_stable_factory_association() {
    let factory = factory_fixture();
    // SAFETY: the boxed factory stays at a stable address for the whole test.
    let companion =
        unsafe { CompanionFactory::from_raw(hifishifter_clap::abi::PLUGIN_ID_STR, &factory) }
            .expect("the CLAP descriptor ID must be a valid companion association ID");
    let entry = ClapAraEntry::new([companion]).expect("one association is valid");
    assert_eq!(entry.factory_count(), 1);
    assert!(entry.supports_factory_id(CLAP_EXT_ARA_FACTORY));
    assert!(entry.supports_factory_id(CLAP_EXT_ARA_FACTORY_COMPAT));
    assert!(!entry.supports_factory_id("clap.state"));
    assert!(!entry.extension(CLAP_EXT_ARA_FACTORY).is_null());
    assert!(entry.extension("clap.state").is_null());

    // 宿主侧视图：从真实表读回工厂数量与 descriptor ID，验证 get_plugin_id
    // 与 descriptor 逐字相同 —— 这正是 ARA-over-CLAP 的配对键。
    let raw = entry.as_raw();
    let host = unsafe { ClapAraHostFactory::discover(raw) }.expect("discover");
    assert_eq!(host.factory_count(), 1);
    let association = host.factory(0).expect("factory 0");
    assert_eq!(
        association.plugin_id(),
        hifishifter_clap::abi::PLUGIN_ID_STR
    );
    assert_eq!(association.ara_factory(), &*factory as *const ARAFactory);
}

#[test]
fn the_unwired_stage_one_still_returns_null_for_every_ara_query() {
    for id in [
        CLAP_EXT_ARA_FACTORY,
        CLAP_EXT_ARA_FACTORY_COMPAT,
        CLAP_EXT_ARA_PLUGIN_EXTENSION,
        CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT,
    ] {
        assert!(
            ours::get_extension(std::ptr::null(), id).is_null(),
            "stage 1 must not expose {id}"
        );
    }
    // 角色集合只是确认 companion 的 CLAP 通道可用，不作为能力声明。
    assert!(CompanionRoles::all().contains(CompanionRoles::PLAYBACK_RENDERER));
}
