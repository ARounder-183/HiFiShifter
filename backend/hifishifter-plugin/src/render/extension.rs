//! 每个 VST3 entry 的扩展所有权。强引用由 entry builder 保留，不在组件销毁时悬空。

use super::ownership::{region_owners, RegionKey};
use ara2_bridge::core::{ApiGeneration, AraError};
use ara2_bridge::plugin::{ExtensionBinding, ExtensionControllerLease, ExtensionRoles};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 仅模型线程访问；音频线程将在 Task 14 消费已发布快照，不能锁此 owner。
#[derive(Default)]
pub(crate) struct ExtensionOwner {
    binding: Mutex<Option<(ExtensionBinding, ExtensionControllerLease)>>,
    assignments: Mutex<HashMap<i32, Vec<RegionKey>>>,
}

impl ExtensionOwner {
    /// 为 entry 创建并保留一个绑定，观察器用 Weak 避免 owner 与 binding 互相保活。
    pub fn bind(
        self: &Arc<Self>,
        generation: ApiGeneration,
        known: ExtensionRoles,
        assigned: ExtensionRoles,
    ) -> Result<*const ara2_bridge::sys::ARAPlugInExtensionInstance, AraError> {
        let mut current = self.binding.lock().unwrap_or_else(|p| p.into_inner());
        if current.is_some() {
            return Err(AraError::InvalidState("extension already bound"));
        }
        let owner = Arc::downgrade(self);
        let observer = Arc::new(move |role: ExtensionRoles, keys: &[usize]| {
            if let Some(owner) = owner.upgrade() {
                let keys = keys.iter().map(|key| *key as RegionKey).collect::<Vec<_>>();
                let valid = keys.is_empty()
                    || region_owners()
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .resolve(&keys)
                        .is_ok();
                let count = keys.len();
                owner
                    .assignments
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .insert(role.bits(), if valid { keys } else { Vec::new() });
                if valid {
                    log::info!(
                        "[ara] renderer assignment role={} regions={count}",
                        role.bits()
                    );
                } else {
                    log::warn!("[ara] rejected unknown or cross-document renderer assignment");
                }
            }
        });
        let owned = ExtensionBinding::new_with_assignment_observer(
            generation,
            known,
            assigned,
            ExtensionRoles::PLAYBACK_RENDERER | ExtensionRoles::EDITOR_RENDERER,
            observer,
        )?;
        let raw = owned.0.as_raw();
        *current = Some(owned);
        Ok(raw)
    }
}
