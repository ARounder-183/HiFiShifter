//! ARA 文档存档：把插件的编辑交给宿主的工程存档通道。
//!
//! 【为什么需要它】ARA 的 `storeDocumentToArchive` / `beginRestoringDocumentFromArchive`
//! 是宿主保存/重载工程时携带插件状态的通道。此前本插件**没有注册** persistence 能力
//! （`runtime.rs` 的 builder 链里没有 `.persistence(...)`），于是四个存档回调全部以
//! `Unsupported` 告终、返回 `kARAFalse`：宿主每次存盘都被告知"插件存档失败"，
//! 重载时状态全丢。
//!
//! 【为什么它不是唯一的持久化通道】插件的编辑也经 VST3 `IComponent::getState/setState`
//! 持久化（`render/extension.rs` 的 `encode_state` / `restore_state`）。两条通道携带的是
//! **同一份** `EditState`：VST3 组件状态按组件走，ARA 存档按文档走。这里复用同一份
//! 编码与合并逻辑，不另写一套。
//!
//! 【为什么不需要路由表】`PluginBuilder` 是**每个文档控制器一份**的（见 `runtime.rs` 的
//! `document_controller` 闭包），所以 provider 在构造时就能直接拿到所属文档的会话 ——
//! 不存在"该服务哪个文档"的歧义，也就不需要共享登记表。这是比"猜一个当前文档"更强的
//! 保证：猜错会把用户的编辑写到别的轨道上。

use ara2_bridge::core::AraError;
use ara2_bridge::plugin::Persistence;
use std::sync::{Arc, Weak};

use crate::render::document::DocumentSession;

/// ARA 文档存档 provider，绑定到一份具体的文档会话。
pub struct ArchivePersistence {
    /// 弱引用：文档由 `ModelHandle` 持有，provider 不应延长它的生命周期
    /// （与 `ExtensionOwner::document` 同一口径）。
    document: Weak<DocumentSession>,
}

impl ArchivePersistence {
    /// 绑定一份文档会话。
    pub(crate) fn new(document: &Arc<DocumentSession>) -> Self {
        Self {
            document: Arc::downgrade(document),
        }
    }
}

impl Persistence for ArchivePersistence {
    fn restore_document(&mut self, bytes: &[u8]) -> Result<(), AraError> {
        let Some(document) = self.document.upgrade() else {
            log::warn!("[ara] document archive restore dropped: the document is already gone");
            return Err(AraError::InvalidState(
                "the ARA document was destroyed before its archive could be restored",
            ));
        };
        document.restore_archive(bytes).map_err(|error| {
            // 细节进日志；返回给宿主的必须是 `&'static str`（`AraError` 的形状），
            // 所以用一条固定措辞，具体原因在上面那行日志里。
            log::error!("[ara] document archive restore failed: {error}");
            AraError::InvalidState("ARA document archive could not be decoded")
        })?;
        log::info!(
            "[ara] document archive restored: {} byte(s); edits are held until the model graph \
             is ready",
            bytes.len()
        );
        Ok(())
    }

    fn store_document(&mut self) -> Result<Vec<u8>, AraError> {
        let Some(document) = self.document.upgrade() else {
            // 文档已释放 ⇒ 没有编辑可存。返回空载荷而不是失败：宿主对失败的反应是提示
            // 用户"插件存档失败"，而这里并没有失败。
            log::info!("[ara] document archive requested after teardown; storing empty");
            return Ok(Vec::new());
        };
        let bytes = document.encode_archive().map_err(|error| {
            log::error!("[ara] document archive store failed: {error}");
            AraError::InvalidState("ARA document archive could not be encoded")
        })?;
        log::info!("[ara] document archive stored: {} byte(s)", bytes.len());
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::MutexExt;

    #[test]
    fn storing_after_the_document_is_gone_yields_an_empty_payload_instead_of_failing() {
        let mut provider = {
            let document = DocumentSession::new(1);
            ArchivePersistence::new(&document)
        };
        // 文档已释放：空载荷 + 成功（宿主据此不会提示"插件存档失败"）。
        assert!(provider.store_document().unwrap().is_empty());
    }

    #[test]
    fn restoring_after_the_document_is_gone_reports_the_real_reason() {
        let mut provider = {
            let document = DocumentSession::new(1);
            ArchivePersistence::new(&document)
        };
        let error = provider.restore_document(&[1, 2, 3]).unwrap_err();
        assert!(
            format!("{error:?}").contains("destroyed before its archive"),
            "恢复必须有可读的原因：{error:?}"
        );
    }

    #[test]
    fn an_empty_archive_is_a_no_op_even_with_a_live_document() {
        let document = DocumentSession::new(1);
        let mut provider = ArchivePersistence::new(&document);
        // 空载荷 = 没有存档内容。必须是 no-op 成功，不能当成"解码失败"。
        assert!(provider.restore_document(&[]).is_ok());
        assert!(document.pending_archive_restore.lock_or_recover().is_none());
    }

    /// **存档往返：编码出来的载荷必须能被恢复回来，并停在"等模型图就绪"的位置。**
    ///
    /// 【为什么这是回归测试】此前 persistence 能力从未注册，`storeDocumentToArchive`
    /// 一律返回 `kARAFalse` —— 宿主每次存盘都被告知"插件存档失败"，重载时编辑全丢。
    /// 这条测试钉住两端：编码产出非空载荷、恢复把它暂存到文档级等待收口。
    #[test]
    fn an_archive_round_trips_through_encode_and_restore() {
        use std::sync::atomic::Ordering;

        let document = DocumentSession::new(1);
        // `encode_archive` 只在宿主图就绪后允许存档（半张图存出去等于丢编辑）。
        document.ready.store(true, Ordering::Release);

        let mut provider = ArchivePersistence::new(&document);
        let bytes = provider.store_document().expect("archive encodes");
        assert!(!bytes.is_empty(), "存档必须产出实际载荷");

        // 恢复进一份**全新**文档：载荷应当被暂存，而不是立刻应用（模型图还没到）。
        let restored = DocumentSession::new(2);
        let mut consumer = ArchivePersistence::new(&restored);
        consumer.restore_document(&bytes).expect("archive restores");
        assert!(
            restored.pending_archive_restore.lock_or_recover().is_some(),
            "恢复必须暂存到文档级，等 `prepare_renderers` 收口时再合入"
        );
    }

    /// 未就绪的图**不能**存档：半张图存出去等于把编辑丢了。
    #[test]
    fn archiving_is_refused_while_the_host_graph_is_not_ready() {
        let document = DocumentSession::new(1);
        let mut provider = ArchivePersistence::new(&document);
        let error = provider.store_document().unwrap_err();
        assert!(
            format!("{error:?}").contains("could not be encoded"),
            "必须明确拒绝而不是存一份不完整的编辑：{error:?}"
        );
    }
}
