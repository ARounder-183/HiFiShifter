//! 每个 VST3 entry 的扩展所有权。强引用由 entry builder 保留，不在组件销毁时悬空。

use super::ownership::{region_owners, RegionKey};
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::{ExtensionBinding, ExtensionRoles};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

/// 仅模型线程访问；音频线程将在 Task 14 消费已发布快照，不能锁此 owner。
#[derive(Default)]
pub(crate) struct ExtensionOwner {
    binding: Mutex<Option<ExtensionBinding>>,
    document: Mutex<Option<std::sync::Weak<super::document::DocumentSession>>>,
    assignments: Mutex<HashMap<i32, Vec<RegionKey>>>,
    sequences: Mutex<HashMap<i32, Vec<u64>>>,
    role: AtomicI32,
    pub snapshots: [super::snapshot::SnapshotPublisher; 2],
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    use crate::ara_entry::HostEntry;
    use ara2_bridge::companion::vst3::ffi::{ara2_vst3_plugin_entry_bind, ARA2_VST3_OK};
    use ara2_bridge::companion::{CompanionFactory, CompanionProcessorBinding, CompanionRoles};
    use ara2_bridge::plugin::{FactoryBuilder, PluginBuilder};
    use ara2_bridge::sys::*;

    /// 真实空文档绑定也必须收到销毁；不依赖 first region assignment 猜 document。
    #[test]
    fn bound_native_entry_is_tombstoned_by_actual_document_teardown() {
        let factory = Box::leak(Box::new(
            FactoryBuilder::new("org.hfs.bound", "org.hfs.bound.archive")
                .display("bound", "HiFiShifter", "https://example.invalid", "1")
                .document_controller(|| {
                    let model = crate::ara::model::ModelHandle::new();
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
            .initialize(ApiGeneration::V2Final, crate::test_host::assert_address())
            .unwrap();
        let mut fixture = crate::test_host::HostFixture::new(vec![]);
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
        let document = crate::render::document::DocumentSession::lookup(
            controller.documentControllerRef as usize,
        )
        .unwrap();
        // SAFETY: test factory backing 已保留到进程结束。
        let association =
            unsafe { CompanionFactory::from_raw("bound", &*factory.as_raw()) }.unwrap();
        let processor =
            CompanionProcessorBinding::new([association], CompanionRoles::all()).unwrap();
        let probe = processor.lifetime_probe();
        let owner = Arc::new(ExtensionOwner::default());
        let entry = HostEntry::new(processor, "bound", owner.clone()).unwrap();
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
        region_owners()
            .lock()
            .unwrap()
            .register(key, document.id, 0)
            .unwrap();
        // SAFETY: extension 与 controller 保留；region 仅作为不透明身份。
        unsafe {
            let extension = extension_raw
                .cast::<ARAPlugInExtensionInstance>()
                .read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, key as *mut _);
        }
        assert_eq!(owner.assignments.lock().unwrap().get(&1).unwrap(), &[key]);
        // SAFETY: actual controller destructor invokes ModelHandle::destroy_document。
        unsafe {
            ((*controller.documentControllerInterface)
                .destroyDocumentController
                .unwrap())(controller.documentControllerRef)
        };
        assert!(owner.assignments.lock().unwrap().is_empty());
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
        assert!(owner.assignments.lock().unwrap().is_empty());
        drop(entry);
        drop(owner);
        factory.entry().uninitialize().unwrap();
    }

    /// sequence 预览只展开本 sequence；显式区域重复分配不能导致双倍混音。
    #[test]
    fn editor_sequence_selection_is_expanded_and_deduplicated() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document.clone(),
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::EDITOR_RENDERER,
                None,
            )
            .unwrap();
        let region_a = Box::new(0_u8);
        let region_b = Box::new(0_u8);
        let sequence = Box::new(0_u8);
        let a = (&*region_a as *const u8) as u64;
        let b = (&*region_b as *const u8) as u64;
        let seq = (&*sequence as *const u8) as u64;
        region_owners()
            .lock()
            .unwrap()
            .register(a, document.id, 0)
            .unwrap();
        region_owners()
            .lock()
            .unwrap()
            .register(b, document.id, 1)
            .unwrap();
        document
            .sequence_regions
            .lock()
            .unwrap()
            .insert(seq, [a, b].into_iter().collect());
        // SAFETY: binding、session 和不透明身份在调用期间均存活。
        unsafe {
            let extension = &*raw;
            let api = &*extension.editorRendererInterface;
            api.addPlaybackRegion.unwrap()(extension.editorRendererRef, a as *mut _);
            api.addRegionSequence.unwrap()(extension.editorRendererRef, seq as *mut _);
        }
        let mut expected = vec![a, b];
        expected.sort_unstable();
        assert_eq!(owner.assigned_regions().unwrap(), expected);
        document
            .sequence_regions
            .lock()
            .unwrap()
            .get_mut(&seq)
            .unwrap()
            .remove(&b);
        region_owners().lock().unwrap().remove(b);
        assert_eq!(owner.assigned_regions().unwrap(), [a]);
        // SAFETY: lease 保留接口，remove sequence 不会移除显式 region。
        unsafe {
            let extension = &*raw;
            ((*extension.editorRendererInterface)
                .removeRegionSequence
                .unwrap())(extension.editorRendererRef, seq as *mut _);
        }
        assert_eq!(owner.assigned_regions().unwrap(), [a]);
        drop(model);
        assert!(owner.assigned_regions().is_err());
    }

    /// companion 先销毁时 controller lease 必须独立保留真实 raw storage。
    #[test]
    fn actual_document_retains_bound_storage_after_owner_is_dropped() {
        let model = crate::ara::model::ModelHandle::new();
        let document = model.session();
        let owner = Arc::new(ExtensionOwner::default());
        let raw = owner
            .bind_to_document(
                document,
                ApiGeneration::V2Final,
                ExtensionRoles::all(),
                ExtensionRoles::PLAYBACK_RENDERER,
                None,
            )
            .unwrap();
        drop(owner);
        let mut identity = 0_u8;
        // SAFETY: actual model 的 session 独立持有控制器 lease，owner 已 drop 也不释放 storage。
        unsafe {
            let extension = raw.read_unaligned();
            ((*extension.playbackRendererInterface)
                .addPlaybackRegion
                .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
            ((*extension.playbackRendererInterface)
                .removePlaybackRegion
                .unwrap())(extension.playbackRendererRef, (&raw mut identity).cast());
        }
        drop(model);
        // 模型终止后所有 owner 都已释放，此后不得再访问 raw。
    }
}

impl ExtensionOwner {
    /// 把租约关联到真实文档；关闭文档时由其同步撤销。
    pub fn bind_to_document(
        self: &Arc<Self>,
        document: Arc<super::document::DocumentSession>,
        generation: ApiGeneration,
        known: ExtensionRoles,
        assigned: ExtensionRoles,
        companion: Option<ara2_bridge::companion::CompanionControllerBinding<'static>>,
    ) -> Result<*const ara2_bridge::sys::ARAPlugInExtensionInstance, AraError> {
        let mut current = self.binding.lock().unwrap_or_else(|p| p.into_inner());
        if current.is_some() {
            return Err(AraError::InvalidState("extension already bound"));
        }
        let owner = Arc::downgrade(self);
        let document_id = document.id;
        let observer = Arc::new(
            move |role: ExtensionRoles, keys: &[usize], sequences: &[usize]| {
                if let Some(owner) = owner.upgrade() {
                    let keys = keys.iter().map(|key| *key as RegionKey).collect::<Vec<_>>();
                    let valid = keys.is_empty()
                        || region_owners()
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .resolve(&keys)
                            .is_ok_and(|(owner, _)| owner == document_id);
                    let count = keys.len();
                    owner
                        .assignments
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(role.bits(), if valid { keys } else { Vec::new() });
                    owner.sequences.lock().unwrap().insert(
                        role.bits(),
                        sequences.iter().map(|key| *key as u64).collect(),
                    );
                    if valid {
                        log::info!(
                            "[ara] renderer assignment role={} regions={count}",
                            role.bits()
                        );
                    } else {
                        log::warn!("[ara] rejected unknown or cross-document renderer assignment");
                    }
                    owner.prepare();
                }
            },
        );
        let supported = ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER;
        let enabled = ExtensionRoles::resolve(known, assigned, supported)?;
        let owned = ExtensionBinding::new_with_renderer_observer(
            generation, known, assigned, supported, observer,
        )?;
        let raw = owned.0.as_raw();
        document.attach(self, owned.1, companion)?;
        *self.document.lock().unwrap() = Some(Arc::downgrade(&document));
        *current = Some(owned.0);
        self.role.store(
            if enabled.contains(ExtensionRoles::PLAYBACK_RENDERER) {
                1
            } else {
                2
            },
            Ordering::Release,
        );
        Ok(raw)
    }

    /// 文档侧已撤销 lease；保留 binding 以供宿主最后几次合法移除/释放回调访问。
    pub fn document_closed(&self) {
        self.assignments.lock().unwrap().clear();
        self.sequences.lock().unwrap().clear();
        self.document.lock().unwrap().take();
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
    }

    /// 模型线程展开当前 renderer 的显式区域与 editor sequence，并去重，拒绝跨文档身份。
    pub fn assigned_regions(&self) -> Result<Vec<u64>, AraError> {
        let document = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or(AraError::InvalidState("document no longer available"))?;
        let role = self.role.load(Ordering::Acquire);
        let mut keys = self
            .assignments
            .lock()
            .unwrap()
            .get(&role)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let sequences = self
            .sequences
            .lock()
            .unwrap()
            .get(&role)
            .cloned()
            .unwrap_or_default();
        let members = document.sequence_regions.lock().unwrap();
        for sequence in sequences {
            keys.extend(
                members
                    .get(&sequence)
                    .ok_or(AraError::InvalidState("unknown assigned sequence"))?
                    .iter()
                    .copied(),
            );
        }
        drop(members);
        let keys = keys.into_iter().collect::<Vec<_>>();
        if !keys.is_empty() {
            let (owner, _) = region_owners()
                .lock()
                .unwrap()
                .resolve(&keys)
                .map_err(|_| AraError::InvalidState("invalid assigned region"))?;
            if owner != document.id {
                return Err(AraError::InvalidState(
                    "assigned region belongs to another document",
                ));
            }
        }
        Ok(keys)
    }

    /// 非实时准备两个支持的输出采样率，callback 不重采样、不等待。
    pub fn prepare(&self) {
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
        let Some(document) = self
            .document
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
        else {
            return;
        };
        let Ok(keys) = self.assigned_regions() else {
            return;
        };
        if keys.is_empty() {
            return;
        }
        let geometry = document.regions.lock().unwrap();
        let Some(regions) = keys
            .iter()
            .map(|key| geometry.get(key).cloned())
            .collect::<Option<Vec<_>>>()
        else {
            return;
        };
        drop(geometry);
        let sources = document.sources.lock().unwrap().clone();
        for (publisher, rate) in self.snapshots.iter().zip([44100, 48000]) {
            match super::snapshot::mix_plain_regions(&regions, &sources, rate)
                .and_then(|snapshot| publisher.publish(snapshot))
            {
                Ok(()) => log::info!(
                    "[ara] snapshot ready role={} rate={rate} regions={}",
                    self.role.load(Ordering::Relaxed),
                    keys.len()
                ),
                Err(error) => log::warn!("[ara] snapshot unavailable: {error:?}"),
            }
        }
    }
}
