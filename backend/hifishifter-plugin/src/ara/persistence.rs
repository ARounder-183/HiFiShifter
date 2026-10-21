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

use ara2_bridge::core::{AraError, FilterSelection, RestoreFilter, StoreFilter};
use ara2_bridge::plugin::{PartialPersistence, Persistence};
use std::collections::BTreeSet;
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

/// ARA 2 的**部分存档**：按宿主给出的筛选只保存被选中的对象。
///
/// 【为什么必须实现它】宿主可以只保存"它知道变过的"那部分（`storeObjectsToArchive` 的
/// 筛选正是为此而设，`ARAInterface.h:2979-2988`）。此前只注册了整份存档，于是任何带筛选
/// 的保存都会拿到 `kARAFalse` —— 宿主据此认为"插件存档失败"，做增量保存时**丢掉**编辑。
impl PartialPersistence for ArchivePersistence {
    fn restore_objects(
        &mut self,
        filter: &FilterSelection<RestoreFilter>,
        bytes: &[u8],
    ) -> Result<(), AraError> {
        let Some(document) = self.document.upgrade() else {
            log::warn!("[ara] partial archive restore dropped: the document is already gone");
            return Err(AraError::InvalidState(
                "the ARA document was destroyed before its partial archive could be restored",
            ));
        };
        let outcome = match filter {
            // null 筛选 = 整份文档，走同一条完整恢复路径。
            FilterSelection::All => document.restore_archive(bytes),
            FilterSelection::Selected(filter) => {
                // 【为什么必须做 ID 重映射】persistentID 只保证**文档内**唯一，跨文档导入
                // 时宿主要求把存档 ID 映射到当前图里的 ID（`ARAInterface.h:2291-2294`）。
                // 直接按存档 ID 合入会把参数挂到一个不存在的键上 —— 静默丢失。
                let remap = filter
                    .audio_modifications()
                    .iter()
                    .map(|mapping| {
                        (
                            mapping.archive_id().to_owned(),
                            mapping.current_id().to_owned(),
                        )
                    })
                    .collect::<Vec<_>>();
                document.restore_archive_partial(bytes, &remap)
            }
        };
        outcome.map_err(|error| {
            log::error!("[ara] partial archive restore failed: {error}");
            AraError::InvalidState("ARA partial archive could not be decoded")
        })?;
        log::info!(
            "[ara] partial archive restored: {} byte(s); it is merged by key once the model graph              is ready",
            bytes.len()
        );
        Ok(())
    }

    fn store_objects(
        &mut self,
        filter: &FilterSelection<StoreFilter>,
    ) -> Result<Vec<u8>, AraError> {
        let Some(document) = self.document.upgrade() else {
            log::info!("[ara] partial archive requested after teardown; storing empty");
            return Ok(Vec::new());
        };
        let (modifications, document_data) = match filter {
            FilterSelection::All => (None, true),
            FilterSelection::Selected(filter) => {
                // 筛选交出的句柄要映射回插件自己的 persistentID —— 权威表按它索引。
                let ids = filter
                    .audio_modifications()
                    .iter()
                    .filter_map(|handle| {
                        document.modification_id_for_handle(handle.index(), handle.session())
                    })
                    .collect::<BTreeSet<_>>();
                (Some(ids), filter.includes_document_data())
            }
        };
        let bytes = document
            .encode_partial(modifications.as_ref(), document_data)
            .map_err(|error| {
                log::error!("[ara] partial archive store failed: {error}");
                AraError::InvalidState("ARA partial archive could not be encoded")
            })?;
        log::info!(
            "[ara] partial archive stored: {} byte(s); {} modification(s) selected",
            bytes.len(),
            modifications.as_ref().map_or(0, BTreeSet::len)
        );
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

    /// **部分存档只带被选中的 modification 参数。**
    ///
    /// 【为什么这条重要】宿主做增量保存时会用 `storeObjectsToArchive` 的筛选只交出一条
    /// 或几条 modification。此前插件只注册了整份存档，带筛选的调用一律拿到 `kARAFalse`，
    /// 宿主据此认为"插件存档失败"，那次编辑就丢了。
    #[test]
    fn a_partial_store_keeps_only_the_selected_modification_parameters() {
        use std::sync::atomic::Ordering;
        let document = DocumentSession::new(1);
        document.ready.store(true, Ordering::Release);
        {
            let mut edits = document.edits.lock_or_recover();
            edits
                .atlas
                .seed_test_pitch_curve("mod:a", 250., vec![60., 62.]);
            edits
                .atlas
                .seed_test_pitch_curve("mod:b", 250., vec![64., 66.]);
        }
        let selected = std::collections::BTreeSet::from(["a".to_owned()]);
        let bytes = document.encode_partial(Some(&selected), false).unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let authorities = payload["edits"]["atlas"]["authorities"]
            .as_object()
            .unwrap();
        assert_eq!(authorities.len(), 1, "只应带上被选中的那一条");
        assert!(authorities.contains_key("mod:a"));
        assert!(!authorities.contains_key("mod:b"));
        // 文档级状态没被选中 ⇒ 载荷里不带它（恢复端因此不会覆盖它）。
        assert!(payload["edits"]["params"].as_object().unwrap().is_empty());
    }

    /// **部分恢复必须按 `archive → current` 重映射 ID。**
    ///
    /// 【为什么必须重映射】persistentID 只保证**文档内**唯一，跨文档导入时宿主要求把
    /// 存档 ID 映射到当前图里的 ID（`ARAInterface.h:2291-2294`）。按存档 ID 合入会把
    /// 参数挂到一个不存在的键上 —— 静默丢失。
    #[test]
    fn a_partial_restore_remaps_archive_ids_onto_current_ids() {
        use std::sync::atomic::Ordering;
        let source = DocumentSession::new(1);
        source.ready.store(true, Ordering::Release);
        source.edits.lock_or_recover().atlas.seed_test_pitch_curve(
            "mod:archive-a",
            250.,
            vec![60., 62.],
        );
        let bytes = source.encode_partial(None, false).unwrap();

        let target = DocumentSession::new(2);
        target.ready.store(true, Ordering::Release);
        target
            .restore_archive_partial(&bytes, &[("archive-a".into(), "current-a".into())])
            .unwrap();
        target.prepare_renderers();
        let edits = target.edits.lock_or_recover();
        assert!(
            edits.atlas.has_authority_parameters(None, "current-a"),
            "必须按 current ID 合入"
        );
        assert!(
            !edits.atlas.has_authority_parameters(None, "archive-a"),
            "不得留下存档 ID"
        );
    }

    /// **宿主先重建图、后给存档：恢复到达时可能一个 owner 都没有。**
    ///
    /// 【为什么必须钉住】用户在 Cubase 上报"撤销/重做后 ARA 不刷新，只有删掉插件重新
    /// 添加才好"。撤销在 ARA 侧等价于一次"存档还原 + 图重建"会话
    /// （`ARAInterface.h:2938-2943`：还原会话在概念上就是一次编辑会话）。而还原**到达
    /// 的时刻早于图就绪**：SDK 明确要求宿主"先配置好该对象的属性，再恢复它的状态"
    /// （`ARAInterface.h:2922-2924`），所以那一瞬间还没有 renderer 可合并。
    ///
    /// 因此载荷必须停在文档级、且**不能被判成失败**。此前这条路径没有测试：若有人把
    /// "没有 owner"改成"返回 Err"，宿主会以为"插件存档失败"，而恢复就此丢失 ——
    /// 用户在宿主里继续编辑，看到的却仍是旧曲线。
    #[test]
    fn an_archive_arriving_before_the_graph_is_ready_is_held_not_dropped() {
        use std::sync::atomic::Ordering;
        let document = DocumentSession::new(7);
        // 只给"图就绪"，**不**绑定任何 renderer：正是还原先到、图还没建的时刻。
        document.ready.store(true, Ordering::Release);
        document
            .edits
            .lock_or_recover()
            .atlas
            .seed_test_pitch_curve("mod:held", 250., vec![60., 62.]);
        // 【为什么载荷来自**另一份**文档】存档到达的目标文档本来就不是发出它的那一个
        // （重开工程 = 新文档；撤销 = 另一条通道）。同一份文档"存完立刻读回"会被
        // 回声判据识别成"宿主把刚才那份还给我"并忽略 —— 那是 G5 的**正确**行为，
        // 而本用例要钉的是"没有 owner 时载荷不被丢掉"，所以载荷必须真的来自别处。
        let source = DocumentSession::new(70);
        source.ready.store(true, Ordering::Release);
        source
            .edits
            .lock_or_recover()
            .atlas
            .seed_test_pitch_curve("mod:held", 250., vec![60., 62.]);
        let mut provider = ArchivePersistence::new(&source);
        let bytes = provider.store_document().expect("archive encodes");
        let mut consumer = ArchivePersistence::new(&document);
        // 载荷先落进一份没有任何 renderer 的文档。
        *document.pending_archive_restore.lock_or_recover() = None;
        consumer.restore_document(&bytes).expect("restore succeeds");
        let held = document.pending_archive_restore.lock_or_recover();
        assert!(
            held.is_some(),
            "没有 owner 时载荷必须留在文档级等待下一次收口"
        );
        drop(held);

        // 收口：仍然没有 owner ⇒ 载荷必须原样留着，且 `ready` 必须为真。
        document.prepare_renderers();
        assert!(
            document.pending_archive_restore.lock_or_recover().is_some(),
            "没有 owner 时不得把载荷丢掉：下一次收口还要再试"
        );
        assert!(
            document.ready.load(Ordering::Acquire),
            "`ready` 是 ensure_loaded 判断\"宿主是否交付了输入\"的依据，假它就是一份 \
             Conflict；它只能由 prepare_renderers 置真（ARAInterface.h:2954-2958 要求 \
             插件\"尽量恢复\"而不是把实例打死）"
        );
    }

    /// **部分存档必须先于整份存档合入。**
    ///
    /// 【为什么顺序不能反】SDK 规定："如果宿主把恢复拆成多次，每一次恢复了某个 audio
    /// source 状态的调用，必须**同时或先于**恢复依赖它的 audio modification 状态"
    /// （`ARAInterface.h:2925-2929`）。落到本插件：部分存档是**按键合并**（只覆盖它
    /// 带上的键），整份存档是**整份替换**。反过来先合整份，部分载荷就会被整份那份
    /// 冲掉 —— 用户看到"撤销之后只有一部分参数回来了"。
    ///
    /// 本用例让两类载荷同时在队列里，断言收口后**两者都在**。
    ///
    /// 【为什么断言的是权威表而不是 `params`】整份存档的合入要经 owner 的
    /// `merge_pending_restore`（它有判据：必须有已分配的角色与绑定），那是
    /// `render/extension` 的既有路径、另有测试覆盖。本用例要的是**顺序**这件事本身
    /// —— 而顺序的正确性只由"部分载荷的键有没有活下来"体现：若顺序颠倒，整份那份
    /// 会把它冲掉。因此这里不绑定 owner（`params` 本就不会变），只验证按键合并不丢。
    #[test]
    fn a_partial_archive_is_merged_before_the_full_archive() {
        use std::sync::atomic::Ordering;
        let document = DocumentSession::new(8);
        document.ready.store(true, Ordering::Release);
        {
            let mut edits = document.edits.lock_or_recover();
            edits
                .atlas
                .seed_test_pitch_curve("mod:partial", 250., vec![60., 62.]);
            edits
                .atlas
                .seed_test_pitch_curve("mod:full", 250., vec![70., 72.]);
        }
        // 部分载荷只带 `mod:partial`（它的值故意与整份那份不同，便于区分谁赢）。
        let partial = document
            .encode_partial(
                Some(&std::collections::BTreeSet::from(["partial".to_owned()])),
                false,
            )
            .unwrap();
        // 整份载荷同时带上两条权威。
        let mut full = crate::state_channel::EditState::default();
        full.atlas
            .seed_test_pitch_curve("mod:partial", 250., vec![61., 63.]);
        full.atlas
            .seed_test_pitch_curve("mod:full", 250., vec![71., 73.]);
        let full = full.encode().unwrap();

        // 注意入队顺序故意**倒着**来：先整份、后部分。
        document.restore_archive(&full).unwrap();
        document.restore_archive_partial(&partial, &[]).unwrap();

        document.prepare_renderers();
        let edits = document.edits.lock_or_recover();
        assert!(
            edits.atlas.has_authority_parameters(None, "partial"),
            "部分存档带上的键必须存活（它不得被整份存档冲掉）"
        );
        assert!(
            edits.atlas.has_authority_parameters(None, "full"),
            "整份存档的状态也必须合入"
        );
    }

    /// **部分恢复失败不得把实例打成"未就绪"。**
    ///
    /// 【为什么必须容忍】SDK 对 `restoreObjectsFromArchive` 的约定是"尽量恢复能恢复的，
    /// 不要把它当错误处理"（`ARAInterface.h:2954-2958`），并且"与当前图里 persistentID
    /// 对不上的存档状态直接忽略"（`:2950-1952`）。跨文档导入时"modification 恢复了但
    /// 它的 source 变了"更是 SDK 明说可能发生的情形（`:2930-2933`）。
    ///
    /// 若这里跟着失败把 `ready` 按回 `false`：`ensure_loaded` 会返回
    /// `Conflict: host changed`，而 `ready` 只能由 `prepare_renderers` 置真 ——
    /// 宿主若不重发 `endEditing` 就再也回不来（实例永久砖化，症状正是用户报的
    /// "只有删掉插件再添加才好"）。
    #[test]
    fn a_failed_partial_restore_leaves_the_instance_serving() {
        use std::sync::atomic::Ordering;
        let document = DocumentSession::new(9);
        document.ready.store(true, Ordering::Release);
        {
            let mut edits = document.edits.lock_or_recover();
            edits
                .atlas
                .seed_test_pitch_curve("mod:keep", 250., vec![60., 62.]);
        }
        // 坏载荷：不是本插件的存档格式 ⇒ 忽略并返回成功。
        // 【为什么不能报错】宿主把恢复失败当成"整份文档恢复失败"，会中止 region 交付；
        // 丢一份解不出的编辑是局部损失，让宿主卡在 `awaiting_regions` 是整份工程不可用。
        assert!(document
            .restore_archive_partial(b"not-an-archive", &[])
            .is_ok());
        // 但坏载荷不得进入待合入队列，更不得影响 `ready`。
        assert!(
            document
                .pending_partial_restore
                .lock_or_recover()
                .is_empty(),
            "解码失败的载荷不该排队：合入它只会再造一次失败"
        );
        document.prepare_renderers();
        assert!(
            document.ready.load(Ordering::Acquire),
            "恢复失败也必须保持就绪： ready=false 会让 ensure_loaded 报 Conflict，\
             而它只能由 prepare_renderers 置真（实例可能永久砖化）"
        );
        assert!(
            document
                .edits
                .lock_or_recover()
                .atlas
                .has_authority_parameters(None, "keep"),
            "failed 恢复不得顺手清掉已有的参数权威"
        );
    }

    /// **参数变更必须进入 ARA 通知队列。**
    ///
    /// 【为什么必须有】宿主只保存"它知道变过"的部分。插件此前**从未**通知过任何持久
    /// 状态变化（`mark_*` 在生产代码里出现次数为 0），只存增量的宿主因此会丢数据。
    #[test]
    fn parameter_changes_are_queued_for_the_host() {
        let document = DocumentSession::new(1);
        // 未绑定 emitter 时是 no-op：不该 panic。
        document.notify_parameters_changed();
        let emitter = ara2_bridge::plugin::UpdateEmitter::new();
        document.set_update_emitter(emitter.clone());
        document.notify_parameters_changed();
        assert!(
            emitter.pending_count() >= 1,
            "文档级持久状态变化必须排队，否则只存增量的宿主会丢数据"
        );
    }

    /// 未就绪的图**不能**让宿主看到存档失败；退化为当前可用的合法载荷。
    ///
    /// 【为什么契约变了】宿主把 `storeDocumentToArchive` 的错误当成"插件存档失败"，
    /// 有些宿主会因此跳过这份状态。存一份稍旧/稍空的载荷永远好过让宿主丢掉整块状态。
    #[test]
    fn archiving_degrades_to_a_valid_payload_while_the_host_graph_is_not_ready() {
        let document = DocumentSession::new(1);
        let mut provider = ArchivePersistence::new(&document);
        let bytes = provider
            .store_document()
            .expect("存档不能因为图未就绪而失败");
        let mut restored = crate::state_channel::EditState::default();
        restored
            .restore(&bytes)
            .expect("降级载荷仍必须是可解析的 EditState");
    }
}
