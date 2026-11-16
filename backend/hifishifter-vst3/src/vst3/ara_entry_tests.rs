//! `HostEntry`（ARA↔VST3 COM 入口）的两条端到端用例，随壳从共享层搬来。
//!
//! 【为什么在壳里】它们驱动的是 `ara_entry.rs` 的 COM 面（`ara2_vst3_plugin_entry_bind`
//! 与真 C++ companion shim），而那面只存在于本 crate。绑定决策本身（角色校验、幂等
//! 复位）仍由共享层 `ara/binder.rs` 的用例覆盖。

use super::*;
use crate::ara_entry::HostEntry;
use ara2_bridge::companion::vst3::ffi::{ara2_vst3_plugin_entry_bind, ARA2_VST3_OK};
use ara2_bridge::companion::{CompanionFactory, CompanionRoles};
use ara2_bridge::core::ApiGeneration;
use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
use ara2_bridge::sys::*;

/// 真实空文档绑定也必须收到销毁；不依赖 first region assignment 猜 document。
#[test]
fn bound_native_entry_is_tombstoned_by_actual_document_teardown() {
    let factory = Box::leak(Box::new(
        FactoryBuilder::new("org.hfs.bound", "org.hfs.bound.archive")
            .display(
                "bound",
                "HiFiShifter",
                "https://github.com/ARounder-183/HiFiShifter",
                "1",
            )
            .document_controller(|| {
                let model = hifishifter_plugin::ara::model::ModelHandle::new();
                let session = model.session();
                PluginBuilder::new(model)
                    .controller_identity(move |key| session.register(key))
                    .build()
            })
            .build()
            .unwrap(),
    ));
    factory
        .entry()
        .initialize(
            ApiGeneration::V2Final,
            hifishifter_plugin::test_support::assert_address(),
        )
        .unwrap();
    let mut fixture = hifishifter_plugin::test_support::HostFixture::new(vec![]);
    let host = fixture.instance();
    let properties = ARADocumentProperties {
        structSize: std::mem::size_of::<ARADocumentProperties>(),
        name: c"bound empty".as_ptr(),
    };
    // SAFETY: test host、factory 和 properties 保持到实际 controller 终止。
    let raw = unsafe {
        (factory
            .raw_copy()
            .createDocumentControllerWithDocument
            .unwrap())(&host, &properties)
    };
    assert!(!raw.is_null());
    // SAFETY: 原生工厂返回完整 packed 实例。
    let controller = unsafe { raw.read_unaligned() };
    let document = hifishifter_plugin::render::document::DocumentSession::lookup(
        controller.documentControllerRef as usize,
    )
    .unwrap();
    // SAFETY: test factory backing 已保留到进程结束。
    let association = unsafe { CompanionFactory::from_raw("bound", &*factory.as_raw()) }.unwrap();
    let owner = Arc::new(ExtensionOwner::default());
    // `HostEntry` 现在自己建 companion（重开工程时要能重建），所以传原始 factory。
    let entry = HostEntry::new(association, CompanionRoles::all(), "bound", owner.clone()).unwrap();
    // 探针必须指向**真正被绑定**的那一份 (HostEntry 内部的绑定)。
    let probe = entry
        .lifetime_probe()
        .expect("HostEntry 必须持有初始绑定探针");
    let mut extension_raw = std::ptr::null();
    // SAFETY: real C++ shim 调真实产品 bind，controller 存活且已登记。
    assert_eq!(
        unsafe {
            ara2_vst3_plugin_entry_bind(
                entry.as_raw(),
                controller.documentControllerRef.cast(),
                7,
                1,
                1,
                &raw mut extension_raw,
            )
        },
        ARA2_VST3_OK
    );
    assert!(!extension_raw.is_null());
    assert!(probe.controller_alive());
    let identity = Box::new(0_u8);
    let key = (&*identity as *const u8) as u64;
    hifishifter_plugin::test_support::register_region_owner(key, document.id, 0).unwrap();
    // SAFETY: extension 与 controller 保留；region 仅作为不透明身份。
    unsafe {
        let extension = extension_raw
            .cast::<ARAPlugInExtensionInstance>()
            .read_unaligned();
        ((*extension.playbackRendererInterface)
            .addPlaybackRegion
            .unwrap())(extension.playbackRendererRef, key as *mut _);
    }
    assert_eq!(owner.assignments_snapshot(1), &[key]);
    // SAFETY: actual controller destructor invokes ModelHandle::destroy_document。
    unsafe {
        ((*controller.documentControllerInterface)
            .destroyDocumentController
            .unwrap())(controller.documentControllerRef)
    };
    assert!(owner.has_no_assignments());
    assert!(owner.assigned_regions().is_err());
    assert!(!probe.controller_alive());
    // SAFETY: native entry 仍保留 companion storage；销毁文档后回调必须被 tombstone 拒绝。
    unsafe {
        let extension = extension_raw
            .cast::<ARAPlugInExtensionInstance>()
            .read_unaligned();
        ((*extension.playbackRendererInterface)
            .addPlaybackRegion
            .unwrap())(extension.playbackRendererRef, key as *mut _);
    }
    assert!(owner.has_no_assignments());
    drop(entry);
    drop(owner);
    factory.entry().uninitialize().unwrap();
}

/// **重开工程：上一份文档关闭后，同一实例必须能重新绑定。**
///
/// 【为什么这是问题二的主回归】companion 的 `bound` 是一次性的：`Drop` 都不复位它。
/// 宿主重开工程复用同一个处理器实例，于是第二次 `bind` 必然被 "already bound" 拒绝。
/// 上一版把这条误当成"宿主重复绑定"，把 null 扩展实例交给宿主 —— 插件加载了却永远
/// 不绑，既有 ARA 数据与新导入音频都不加载，只有删掉插件再添加才好。
///
/// 本用例走真实 C++ shim：第一次绑定成功 → 关闭文档（`closed = true`，binding 保留）
/// → 再次绑定必须**复位 companion 后成功**，而不是返回 null。
#[test]
fn a_reopened_document_rebinds_through_a_reset_companion() {
    use ara2_bridge::companion::CompanionFactory;
    use ara2_bridge::plugin::PluginBuilder;

    let factory = Box::leak(Box::new(
        FactoryBuilder::new("org.hfs.rebind", "org.hfs.rebind.archive")
            .display(
                "rebind",
                "HiFiShifter",
                "https://github.com/ARounder-183/HiFiShifter",
                "1",
            )
            .document_controller(|| {
                let model = hifishifter_plugin::ara::model::ModelHandle::new();
                let session = model.session();
                PluginBuilder::new(model)
                    .controller_identity(move |key| session.register(key))
                    .build()
            })
            .build()
            .unwrap(),
    ));
    factory
        .entry()
        .initialize(
            ApiGeneration::V2Final,
            hifishifter_plugin::test_support::assert_address(),
        )
        .unwrap();
    let mut fixture = hifishifter_plugin::test_support::HostFixture::new(vec![]);
    let host = fixture.instance();
    let properties = ARADocumentProperties {
        structSize: std::mem::size_of::<ARADocumentProperties>(),
        name: c"rebind".as_ptr(),
    };
    // SAFETY: test host、factory 与 properties 保持到 controller 终止。
    let raw = unsafe {
        (factory
            .raw_copy()
            .createDocumentControllerWithDocument
            .unwrap())(&host, &properties)
    };
    assert!(!raw.is_null());
    // SAFETY: 原生工厂返回完整 packed 实例。
    let controller = unsafe { raw.read_unaligned() };
    // SAFETY: test factory backing 保留到进程结束。
    let association = unsafe { CompanionFactory::from_raw("rebind", &*factory.as_raw()) }.unwrap();
    let owner = Arc::new(ExtensionOwner::default());
    let entry =
        HostEntry::new(association, CompanionRoles::all(), "rebind", owner.clone()).unwrap();

    let mut first = std::ptr::null();
    // SAFETY: controller 存活且已登记。
    assert_eq!(
        unsafe {
            ara2_vst3_plugin_entry_bind(
                entry.as_raw(),
                controller.documentControllerRef.cast(),
                7,
                1,
                1,
                &raw mut first,
            )
        },
        ARA2_VST3_OK
    );
    assert!(!first.is_null(), "第一次绑定必须成功");
    assert_eq!(owner.companion_reset_count(), 0);

    // 工程关闭：文档撤销 lease，实例被关闭（binding 仍保留给宿主最后几次回调）。
    owner.document_closed();
    assert!(owner.was_bound_before(), "历史事实不回落");
    assert!(
        owner.existing_binding_raw().is_none(),
        "关闭后不再复用旧实例指针"
    );

    let mut second = std::ptr::null();
    // SAFETY: 同一 controller，宿主重开工程会这样调用。
    assert_eq!(
        unsafe {
            ara2_vst3_plugin_entry_bind(
                entry.as_raw(),
                controller.documentControllerRef.cast(),
                7,
                1,
                1,
                &raw mut second,
            )
        },
        ARA2_VST3_OK
    );
    assert!(
        !second.is_null(),
        "重开工程必须重新拿到非空扩展实例，而不是 null"
    );
    assert_eq!(owner.companion_reset_count(), 1, "必须复位一次 companion");
    assert_eq!(owner.rebind_generation_count(), 1, "必须走重绑路径");

    // SAFETY: 真实 controller 析构会撤销文档。
    unsafe {
        ((*controller.documentControllerInterface)
            .destroyDocumentController
            .unwrap())(controller.documentControllerRef)
    };
    drop(entry);
    drop(owner);
    factory.entry().uninitialize().unwrap();
}
