//! ARA-over-CLAP 的发现表判据：ID、布局、配对键与真实入口行为。
//!
//! 【为什么用真实 companion 类型】这些 ID 与结构体是 ARA 官方头文件的逐字镜像；
//! 用自己的常量去断言自己的常量是空转。这里从 `ara2-bridge-companion` 读回同一组
//! 值，再与 `hifishifter_clap::ara` 暴露的那一份逐字比对。
//!
//! 【现在断言什么】entry 级工厂表已接上共享 ARA runtime（与 VST3 同一个
//! `ARAFactory`，只是关联名换成 CLAP 的 descriptor ID），所以"工厂 ID 必须给出表、
//! 别的 ID 必须给 null"是产品行为；实例级扩展在没有绑定器的实例上仍必须返回 null。
//!
//! 【REAPER 的私有草案 ID 是**反向**判据】它不是 ARACLAP.h 的任何常量，其宿主实现
//! 在需要真正渲染时会崩在自身代码里（见 `hifishifter_clap::ara` 模块文档）。因此
//! 它在这里的角色是"必须被拒绝的 ID"，而不是"另一个可接受的别名" —— 把它从应答
//! 分支删掉这件事，必须有可执行的用例守着。

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

/// REAPER 的旧草案 ID 不是 ARACLAP.h 的任何一个常量，且**不得**被应答。
#[test]
fn the_reaper_legacy_id_is_a_distinct_and_rejected_constant() {
    assert_eq!(
        ours::CLAP_EXT_ARA_PLUGIN_EXTENSION_REAPER_LEGACY,
        "com.celemony.ara.plugin_extension.draft/1"
    );
    assert_ne!(
        ours::CLAP_EXT_ARA_PLUGIN_EXTENSION_REAPER_LEGACY,
        CLAP_EXT_ARA_PLUGIN_EXTENSION
    );
    assert_ne!(
        ours::CLAP_EXT_ARA_PLUGIN_EXTENSION_REAPER_LEGACY,
        CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT
    );
    // 应答它 = 把 REAPER 的空指针缺陷引到用户面前；这条断言就是那道闸门。
    assert!(
        ours::get_extension(std::ptr::null(), ours::CLAP_EXT_ARA_PLUGIN_EXTENSION_REAPER_LEGACY)
            .is_null(),
        "the REAPER private draft ID must never be answered"
    );
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
fn an_instance_without_a_binder_returns_null_for_every_ara_query() {
    for id in [
        CLAP_EXT_ARA_FACTORY,
        CLAP_EXT_ARA_FACTORY_COMPAT,
        CLAP_EXT_ARA_PLUGIN_EXTENSION,
        CLAP_EXT_ARA_PLUGIN_EXTENSION_COMPAT,
    ] {
        assert!(
            ours::get_extension(std::ptr::null(), id).is_null(),
            "an instance without an ARA binder must not expose {id}"
        );
    }
    // 角色集合只是确认 companion 的 CLAP 通道可用，不作为能力声明。
    assert!(CompanionRoles::all().contains(CompanionRoles::PLAYBACK_RENDERER));
}

/// **entry 级配对键**：工厂扩展 ID 必须给出表，其余 ID 必须给 null。
///
/// 【为什么这条是产品行为而不是前置判据】entry 现在接的是**共享** ARA runtime 的
/// 同一个 `ARAFactory`；宿主拿到的表里，`get_plugin_id` 必须与 descriptor 逐字相同，
/// 否则宿主无法把工厂与实例配成一对，表现为"插件能加载但从不进入 ARA"。
#[test]
fn the_entry_publishes_the_ara_factory_under_the_clap_descriptor_id() {
    let table = ours::entry_extension(CLAP_EXT_ARA_FACTORY);
    assert!(!table.is_null(), "the ARA factory table must be published");
    let host = unsafe { ClapAraHostFactory::discover(table.cast()) }.expect("discover");
    assert_eq!(host.factory_count(), 1);
    let association = host.factory(0).expect("factory 0");
    assert_eq!(
        association.plugin_id(),
        hifishifter_clap::abi::PLUGIN_ID_STR
    );
    assert!(!association.ara_factory().is_null());
    assert!(ours::entry_extension("clap.plugin-factory").is_null());
    assert!(ours::entry_extension(CLAP_EXT_ARA_PLUGIN_EXTENSION).is_null());
}
