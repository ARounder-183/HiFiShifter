//! 会话生命周期、编辑目标、持久化状态通道与恢复合并。
use super::*;
impl ExtensionOwner {
    /// 当前生效的编辑后端。
    ///
    /// 【为什么不能只返回字段里的那份】字段在 `initialize` 期间写入，而那时 ARA 绑定
    /// 窗口还开着（见 `HostBackend::from_native` 的说明），所以字段里那份在
    /// "宿主会绑定 ARA"的场合是 `AraOnly`、在"宿主不会绑定"的场合**也是** `AraOnly` ——
    /// 它区分不了这两种形态。形态由 [`Self::mode`] 在窗口关闭后给出，这里据此把
    /// "无原生扩展且无 ARA" 解析成 [`crate::host::adapter::HostBackend::Capture`]。
    ///
    /// 【为什么原生扩展优先】REAPER 既有原生扩展**又**会绑定 ARA；只要字段里有
    /// `Native`，形态问题就不存在（它一定走 ARA），直接返回它。
    pub(crate) fn host_backend(&self) -> crate::host::adapter::HostBackend {
        let stored = self.host.lock_or_recover().clone();
        if let Some(backend) = stored {
            if backend.native().is_some() {
                return backend;
            }
        }
        match self.mode() {
            InstanceMode::Capture => {
                crate::host::adapter::HostBackend::Capture(crate::host::capture::CaptureHost::new())
            }
            // 【为什么 `Undetermined` 也要看素材】形态要等 `setActive` 之后 5 秒才落定，
            // 而"宿主把本插件当普通插入"的会话里用户可能从不打开窗口 —— 那时形态永远
            // 停在 `Undetermined`，按 ARA-only 准备会让 Capture 的渲染准入失败。
            // 素材存在本身就是 Capture 的证明（见 `renders`）。
            InstanceMode::Undetermined if self.capture.has_material() => {
                crate::host::adapter::HostBackend::Capture(crate::host::capture::CaptureHost::new())
            }
            // 剩下的 `Undetermined` 是"还没确定"（此时按 ARA 准备是安全的，ARA 是更丰富的
            // 形态），`Ara` 本来就是这个后端。
            InstanceMode::Undetermined | InstanceMode::Ara => {
                crate::host::adapter::HostBackend::from_native(None)
            }
        }
    }
    /// 原生宿主适配器（未绑定时 `None`）。
    ///
    /// 【为什么单独一个访问器】它是通用层唯一允许取具体 `NativeHost` 的地方 —— 所有
    /// 需要宿主扩展的操作都从这里取、并显式处理 `None`，让"这一步依赖原生扩展"写在
    /// 调用点上，而不是藏在字段类型里。`NativeHost` 自身不带品牌：它只是一组宿主扩展
    /// API 的适配器，当前实现是 REAPER，但通用层从不命名具体宿主。
    pub(crate) fn native_host(&self) -> Option<Arc<crate::host::NativeHost>> {
        self.host
            .lock_or_recover()
            .clone()
            .and_then(|backend| backend.native().cloned())
    }
    /// **生效的**宿主能力位：唯一来源，前端只读这一份。
    ///
    /// 【为什么不能直接用 `host_backend().capabilities()`】后者按 `self.host`（本实例自己
    /// 的原生扩展）计算，而写能力实际由**本文档里任何**一个持有 project history 的实例
    /// 提供（`project_history_host` 的解析口径）。此前前端另有一组 `BootstrapFlags` 布尔，
    /// 用 `project_history_host().can_*()` 各算一遍 —— 两份判据必然漂移，而且"到底哪一份
    /// 算数"没人说得清。这里把同一口径收敛成一个值，布尔与能力位不再各说各话。
    ///
    /// 【为什么 `clip_write` 与 `undo_stack` 分开报】`has_project_history()` 等于
    /// "有撤销栈 **且** 可写片段"，前端 `canEditHostClips()` 用 `clipWrite && undoStack`
    /// 即可精确复现"没有历史 API 就不给拖拽"这条既有语义，而不是把它折进 `clip_write`。
    pub(crate) fn effective_host_capabilities(
        &self,
    ) -> hifishifter_kernel::editor::host::HostCapabilities {
        let mut caps = self.host_backend().capabilities();
        if let Some(host) = self.project_history_host() {
            caps.clip_write = host.can_edit_clips();
            caps.split = host.can_split_clips();
            caps.item_clipboard = host.can_clipboard_items();
            caps.media_import = host.can_import_audio();
            caps.track_create = host.can_create_audio_track();
            caps.undo_stack = host.has_project_history();
        }
        caps
    }
    /// 撤销历史的**权威**：宿主提供撤销栈时用宿主的，否则用插件自管的那份。
    ///
    /// 【为什么永远返回 `Some`】把"没有宿主历史"变成一个实现（`LocalHistory`）而不是
    /// 一条被跳过的分支，是 `undo_timeline` / `redo_timeline` / `set_history_position`
    /// 在非 REAPER 宿主上不再静默失效的关键。只有在文档/会话都还没建起来时才退到
    /// `NullHistory`（诚实的空栈），仍然不是 `None`。
    /// 撤销历史的承载者分类（`"host"` / `"local"`），**不创建会话**。
    ///
    /// 【为什么与 `history_authority` 分开】bootstrap/健康探活只需要一个字符串；
    /// `history_authority` 在没有宿主历史时会创建编辑器会话（见其实现），
    /// 而探活必须能在会话尚未创建时回答。
    pub(crate) fn history_backend(&self) -> &'static str {
        if self.project_history_host().is_some() {
            "host"
        } else {
            "local"
        }
    }

    pub(crate) fn history_authority(
        self: &Arc<Self>,
    ) -> Arc<dyn crate::host::history::HostHistory> {
        if let Some(host) = self.project_history_host() {
            return Arc::new(crate::host::history::ProjectHistory { host });
        }
        match self.editor_session() {
            Ok(editor) => Arc::new(crate::host::history::LocalHistory {
                editor: Arc::downgrade(&editor),
            }),
            Err(_) => Arc::new(crate::host::history::NullHistory),
        }
    }
    /// 明确指定轨道须已有真实assigned item；空轨未指定时仅允许本FX的直接parent轨道。
    pub(crate) fn audio_import_target(
        &self,
        track: Option<&str>,
        authorized: &impl Fn() -> bool,
    ) -> Result<crate::host::HostTrackTarget, String> {
        if let Some(id) = track {
            let document = self.editor_document()?;
            let target = {
                document
                    .ui_tracks
                    .lock_or_recover()
                    .values()
                    .find(|t| t.id == id)
                    .map(|t| t.target.clone())
            };
            if let Some(target) = target {
                return Ok(target);
            }
        }
        if let Some(track) = track {
            let document = self.editor_document()?;
            let clips = {
                let timeline = document.timeline.lock_or_recover();
                timeline
                    .as_ref()
                    .ok_or("host timeline unavailable")?
                    .clips
                    .iter()
                    .filter(|clip| clip.track_id == track)
                    .map(|clip| clip.id.clone())
                    .collect::<Vec<_>>()
            };
            for clip in clips {
                if let Ok(target) = self.host_edit_target(&clip) {
                    return target.import_target(authorized);
                }
            }
            return Err("import track has no directly bound host region".into());
        }
        let host = self.native_host().ok_or("primary host interface missing")?;
        host.direct_track_target(authorized)
    }
    /// 同文档实际renderer唯一region边解析写对象，不能按名字/位置/源路径查item。
    pub(crate) fn host_edit_target(
        &self,
        clip_id: &str,
    ) -> Result<crate::host::HostClipTarget, String> {
        self.refresh_ui_inventory();
        let document = self.editor_document()?;
        if let Some(item) = clip_id.strip_prefix("ara-item-") {
            let target = document
                .ui_tracks
                .lock_or_recover()
                .values()
                .flat_map(|track| &track.items)
                .find(|entry| entry.geometry.item_id == item)
                .map(|entry| entry.target.clone());
            if let Some(target) = target {
                return target.current(&|| {
                    !self.is_closed()
                        && document.is_alive()
                        && self
                            .editor_document()
                            .is_ok_and(|doc| Arc::ptr_eq(&doc, &document))
                });
            }
        }
        let key = {
            let ids = document.clip_ids.lock_or_recover();
            ids.iter()
                .find(|(_, id)| id.as_str() == clip_id)
                .map(|(key, _)| *key)
                .ok_or("unknown ARA clip identity")?
        };
        let mut target = None;
        for owner in document.renderer_owners() {
            if !owner.renders_playback()
                || !owner
                    .assigned_regions()
                    .is_ok_and(|keys| keys.as_slice() == [key])
            {
                continue;
            }
            let stamp = owner.host_query_stamp()?;
            let bound = owner
                .host_geometry()
                .map_err(GeometryFailure::into_message)?;
            if bound.region_key != key {
                return Err("host region binding changed".into());
            }
            let host = owner.native_host().ok_or("host interface missing")?;
            let next = host.clip_target(|| owner.host_query_authorized(&stamp))?;
            if target.is_some() {
                return Err("multiple host writers for one ARA clip".into());
            }
            target = Some(next);
        }
        target.ok_or_else(|| "clip has no unique directly bound host take".into())
    }
    /// 组件终止时停止实例 worker 并释放宿主适配器；不要在音频 process 调用。
    ///
    /// 【为什么与文档关闭分开】`stop_editor()` 同时服务两种语义：文档结束
    /// （组件还活着，下一份文档还会来）与组件结束（宿主适配器确实该释放）。
    /// 合并在一起会在一份文档关闭时把 REAPER 原生扩展适配器永久丢掉 ——
    /// 宿主复用处理器实例重开工程时，重绑后 `native_host()` 永远是 `None`，
    /// 于是不枚举 item、不填 `region_items`、界面停在"等待宿主音频"。
    /// 文档关闭必须走 [`Self::stop_editor_for_document`]。
    pub(crate) fn stop_editor(&self) {
        self.stop_editor_inner(false);
    }

    /// 文档结束：停止 worker、清快照与 per-document 缓存，**保留**宿主适配器。
    pub(crate) fn stop_editor_for_document(&self) {
        self.stop_editor_inner(true);
    }

    pub(super) fn stop_editor_inner(&self, keep_host: bool) {
        let document = {
            self.document
                .lock_or_recover()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
        };
        if let Some(document) = document {
            {
                let _transaction = document.transaction.lock_or_recover();
                if !self.closed.swap(true, Ordering::AcqRel) {
                    document.scope_revision.fetch_add(1, Ordering::AcqRel);
                }
                self.snapshots.iter().for_each(|snapshot| snapshot.clear());
            }
            // 不持文档事务取得views锁，避免enqueue/worker授权反向等待。
            document.revoke_editor_views();
        } else {
            self.closed.store(true, Ordering::Release);
            self.snapshots.iter().for_each(|snapshot| snapshot.clear());
        }
        if !keep_host {
            let host = self.host.lock_or_recover().take();
            drop(host);
        }
        self.host_geometry.lock_or_recover().take();
        *self.host_audio.lock_or_recover() = HostAudioStatus::default();
        self.binding_report.lock_or_recover().take();
        *self.last_enumeration.lock_or_recover() = None;
        self.cancel_preparation();
        // 【为什么 take 掉而不是留在槽位里】`close()` 是终结性的；留下一个
        // `closed == true` 的队列表只会让下一次重绑再次踩到同一个死队列。
        // 重绑时 `ensure_preparation()` 会建一条新的。
        if let Some(worker) = self.preparation.lock_or_recover().take() {
            worker.close();
        }
    }

    /// 把本实例的**每份文档**缓存清到"尚未绑定"状态（宿主切换文档时用）。
    ///
    /// 【为什么不能只清 `assignments / sequences / document`】这些字段的名字都带
    /// "当前文档"的语义，但它们的读者分布在音频线程、UI 定时器与诊断包里；
    /// 留下旧文档的恢复状态、几何缓存、宿主音频读数或快照，会让新文档的第一帧
    /// 输出/第一份 UI 载荷读到上一份工程的结论。宿主适配器（`host`）**不在**
    /// 这里清 —— 它属于组件，不属于文档。
    pub(super) fn prepare_rebind_from_previous_document(&self) {
        self.snapshots.iter().for_each(|snapshot| snapshot.clear());
        self.raw_tape.clear();
        *self.prepared.lock_or_recover() = None;
        self.reset_restore_report();
        self.binding_report.lock_or_recover().take();
        *self.host_audio.lock_or_recover() = HostAudioStatus::default();
        *self.last_enumeration.lock_or_recover() = None;
        *self.last_failed_revision.lock_or_recover() = None;
        self.enumeration_backoff.store(0, Ordering::Release);
        self.host_item_muted.store(false, Ordering::Release);
        self.playback_covered.store(false, Ordering::Release);
        self.rt_last_emitted_valid.store(false, Ordering::Release);
        self.rt_last_emitted_frames.store(0, Ordering::Release);
        self.trace_wanted.store(false, Ordering::Release);
        self.trace_logged.store(false, Ordering::Release);
        self.render_wanted.store(false, Ordering::Release);
        // 【为什么不在这里清 `pending_restore`】组件 `setState` 可能先于 ARA bind
        // 到达（宿主在 setActive 之前恢复组件状态）。那份载荷属于**即将到来的**
        // 新文档，清掉等于把用户参数丢掉。旧文档遗留的载荷由下一次成功合并覆盖。
        if self.pending_restore.lock_or_recover().is_some() {
            self.note_restore_waiting("waiting for the new document graph");
        }
    }
    /// 模型撤销时立即取消待准备作业；保持worker可供后续重新授权使用。
    pub(crate) fn cancel_preparation(&self) {
        if let Some(worker) = self.preparation() {
            worker.cancel();
        }
    }
    /// 原GUI合并后台宿主准备状态，不能把冷恢复尚未准备完写成已应用。
    pub(crate) fn preparation_state(&self) -> (bool, Option<String>) {
        if self.is_editor_only() {
            if let Some(document) = self
                .document
                .lock_or_recover()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
            {
                let states = document
                    .renderer_owners()
                    .into_iter()
                    .filter(|owner| owner.renders_playback())
                    .map(|owner| owner.local_preparation_state())
                    .collect::<Vec<_>>();
                return (
                    states.iter().any(|state| state.0),
                    states.into_iter().find_map(|state| state.1),
                );
            }
        }
        self.local_preparation_state()
    }
    pub(crate) fn local_preparation_state(&self) -> (bool, Option<String>) {
        match self.preparation() {
            Some(worker) => worker.state(),
            None => (false, None),
        }
    }
    /// 渲染器的准备状态快照（诊断包用）：模型/编辑代次、作用域与区间数。
    ///
    /// 【为什么单独抽出来】`Snapshot` 里那段诊断需要当前分配区间才能算逐输出读数；
    /// 诊断包在任意时刻被导出，不值得为它触发一次完整快照。这里只给"准备好了没有、
    /// 准备的是哪一版"，那正是排查"渲染结果陈旧"时要看的东西。
    pub(crate) fn prepared_snapshot(&self) -> Option<serde_json::Value> {
        self.prepared.lock_or_recover().as_ref().map(|version| {
            serde_json::json!({
                "model": version.model,
                "edit": version.edit,
                "epoch": version.epoch,
                "scope": version.scope,
                "regions": version.keys.len(),
            })
        })
    }
    /// 宿主给本 ARA 文档的可撤销播放租约（含它要求的线程）；不寻找全局 REAPER 窗口。
    ///
    /// 【为什么连线程一起返回】调用方必须把请求投递到**创建这条租约的那条线程**上
    /// （见 `render::document::PlaybackEndpoint`）。只交出句柄，调用方就只能去猜
    /// 线程，而猜错的代价是请求永远排在一个没人排空的队列里。
    pub(crate) fn host_playback(&self) -> Option<crate::render::document::PlaybackEndpoint> {
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .and_then(|document| document.playback.lock_or_recover().clone())
    }
    /// 组件释放/文档关闭前结束后台服务，避免DLL卸载后线程仍执行插件代码。
    /// 未绑定时暂存组件state；绑定后所有处理器读取同一文档的参数权威。
    // 组件state访问器保留，供未绑定路径读取。
    #[cfg(test)]
    pub fn edit_state(&self) -> Arc<Mutex<crate::state_channel::EditState>> {
        self.document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .map(|document| document.edits.clone())
            .unwrap_or_else(|| self.edits.clone())
    }

    /// 恢复/undo组件状态后刷新整张文档，各renderer保持原分配。
    ///
    /// 【为什么拆成两个】这个方法会在 `transaction` 之外被调用（`restore_state`），
    /// 也会在**已经持有** `transaction` 的地方被调用（`DocumentSession::prepare_renderers`
    /// 的存档合入路径）。`std::sync::Mutex` 不可重入，两条路径共用一个"自己取锁"的
    /// 实现就会自锁 —— 用户报的"重开 ARA 工程宿主未响应"正是那一次。
    ///
    /// 现在的契约：本方法是**无锁入口**，自己取 `transaction`；
    /// [`Self::refresh_document_locked`] 要求调用方已经持有它。任何未来调用点必须
    /// 显式二选一，不存在"隐含取锁"的版本。
    pub fn refresh_document(&self) {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        if let Some(document) = document {
            let _transaction = document.transaction.lock_or_recover();
            self.refresh_document_locked(&document);
        }
    }

    /// [`Self::refresh_document`] 的持锁版本：**调用方必须已持有** `document.transaction`。
    ///
    /// 【为什么参数是 `&DocumentSession` 而不是自己 upgrade】持锁调用方本来就有那个
    /// 引用（`prepare_renderers` 就是 `self`），传进来既省一次 Weak 升级，也让
    /// "哪份文档的锁被持有"在签名上是唯一的。
    pub(crate) fn refresh_document_locked(&self, document: &super::document::DocumentSession) {
        let owners = document.renderer_owners();
        for owner in &owners {
            if let Err(error) = owner.merge_pending_restore(document) {
                log::warn!("[ara] instance state unresolved: {error}");
            }
        }
        for owner in owners {
            owner.prepare();
        }
    }

    /// setState只暂存本组件恢复；宿主明确分配区域且模型ready后再合入共享权威。
    pub(crate) fn restore_state(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        // 容器解析对**历史格式**（裸 `EditState`）是透明的：没有魔数就整块按旧格式读。
        // 读不出来的后果是把用户已有的编辑弄丢，所以这条兼容不是可选项。
        let state = crate::component_state::ComponentState::decode(bytes)?;
        if let Some(manifest) = state.capture.as_deref() {
            match serde_json::from_slice::<serde_json::Value>(manifest) {
                // 【为什么清单失败只记日志】PCM 丢了不该让参数曲线也读不出来 ——
                // 那是两件独立的用户资产，而参数是更贵的那一份。
                Ok(value) => {
                    if let Err(error) = self.capture.restore_manifest(&value) {
                        log::warn!("[capture] manifest restore failed: {error}");
                    }
                }
                Err(error) => log::warn!("[capture] manifest is not valid JSON: {error}"),
            }
        }
        // 【Capture 的用户曲线走会话，不走 `pending_restore`】`pending_restore` 的读者
        // 是 `merge_pending_restore`，它只服务 ARA 文档；没有文档时那条路是 no-op
        // （见 `refresh_document`）。把曲线投给会话，它会在下一次时间线加载时叠上去。
        //
        // 会话可能还不存在（用户从没开过窗口）—— 那时先暂存在 owner 上，
        // `capture_session()` 创建会话时补投。
        if let Some(capture_edits) = state.capture_edits.as_deref() {
            match self.capture_session_if_initialized() {
                Some(session) => {
                    session.stash_serialized_capture_edits(capture_edits)?;
                    session.request_reload();
                }
                None => {
                    *self.pending_capture_edits.lock_or_recover() = Some(capture_edits.to_vec());
                }
            }
        }
        let mut restored = crate::state_channel::EditState::default();
        restored.restore(&state.edits)?;
        // 【G5】它到达时我们在不在一次编辑会话里（见 `EditState::arrived_in_edit_session`）。
        restored.arrived_in_edit_session = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .is_some_and(|document| document.in_edit_session());
        // 组件通道：这份载荷是**本实例**的存档，宿主用它做 undo / 复制轨 —— 恢复即权威。
        restored.from_archive = false;
        // 【T-D5】同一份载荷已经合入过就别再排队：宿主重复交回同一份存档时（重开编辑器、
        // 撤销后重做、文档重建），重放只会重复劳动，并在失败时重新点燃"没读回来"。
        if Self::restore_digest(&restored).is_some_and(|digest| {
            *self.restore_applied_digest.lock_or_recover() == Some(digest)
                && !self.has_pending_restore()
        }) {
            log::info!("[ara] persisted state unchanged since the last merge; not replaying it");
            self.refresh_document();
            return Ok(());
        }
        *self.pending_restore.lock_or_recover() = Some(restored);
        self.reset_restore_budget();
        self.note_restore_pending();
        self.refresh_document();
        Ok(())
    }

    /// 暂存 ARA 存档恢复出来的编辑，交给本 owner 的既有合并路径。
    ///
    /// 【为什么不直接写 `document.edits`】合并要做身份重绑、冲突判定、`needs_rebind`
    /// 处理 —— 这些都在 `merge_pending_restore` 里，而且已经过测试。存档通道只是另一个
    /// 来源，走同一条路才不会出现两份会漂移的实现。
    ///
    /// 【为什么覆盖已有的 `pending_restore`】VST3 组件状态与 ARA 存档携带的是同一份
    /// `EditState`。两者同时到达时后到的应当胜出（它是宿主更晚保存的那份）；而 ARA 存档
    /// 在宿主重载工程时是权威来源。
    ///
    /// 【为什么这里**不**刷新文档】唯一的调用点是 `DocumentSession::prepare_renderers`，
    /// 它已经持有 `transaction`，而刷新要取同一把不可重入的锁 —— 那就是用户报的
    /// "重开工程宿主未响应"。合入由 `prepare_renderers` 紧随其后的
    /// `merge_pending_restore` 循环完成，所以这里只负责**暂存**。
    pub(crate) fn stash_archive_restore(&self, restored: crate::state_channel::EditState) {
        // 【T-D5】与 `restore_state` 同一条纪律：已经合入过的同一份载荷不再重放。
        if Self::restore_digest(&restored).is_some_and(|digest| {
            *self.restore_applied_digest.lock_or_recover() == Some(digest)
                && !self.has_pending_restore()
        }) {
            log::info!("[ara] archive payload unchanged since the last merge; not replaying it");
            return;
        }
        *self.pending_restore.lock_or_recover() = Some(restored);
        self.reset_restore_budget();
        self.note_restore_pending();
    }
    /// 新载荷到达 ⇒ 等待账从头开始（否则上一份载荷用完的 10 s 会立刻压到这一份上）。
    pub(super) fn reset_restore_budget(&self) {
        self.restore_attempts.store(0, Ordering::Relaxed);
        *self.restore_progress.lock_or_recover() = RestoreProgress::default();
        *self.restore_next_attempt.lock_or_recover() = None;
    }
    /// 一份载荷的摘要（T-D5）：同一份存档重复交回时不再重放。
    ///
    /// 【为什么用编码后的字节】它能覆盖"用户真的改了参数"与"宿主只是把同一份又给了
    /// 一遍"这两种形状，而字段级比较会因为 schema 演进而漏判。
    pub(super) fn restore_digest(state: &crate::state_channel::EditState) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let bytes = state.encode().ok()?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        Some(hasher.finish())
    }
    /// 宿主图的形状摘要：**它是等待时间的唯一时钟**（T-D3）。
    ///
    /// 【为什么是摘要而不是计数】"宿主有没有动过"要一次比较得出结论。把
    /// `track_bindings` / `regions` / `region_items` 一起哈希，任何一次真实的交付
    /// 都会改变它，而我们的轮询不会 —— 于是"十秒"第一次真的是"十秒宿主没动"。
    pub(super) fn restore_signature(&self, document: &super::document::DocumentSession) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        {
            let bindings = document.track_bindings.lock_or_recover();
            for (id, members) in bindings.iter() {
                id.hash(&mut hasher);
                members.hash(&mut hasher);
            }
        }
        document.regions.lock_or_recover().len().hash(&mut hasher);
        document
            .region_items
            .lock_or_recover()
            .len()
            .hash(&mut hasher);
        hasher.finish()
    }
    /// 是否还有一份**没合入**的持久化载荷（组件状态或 ARA 存档）。
    ///
    /// 【为什么需要这个只读问题】存档路径必须能判断"内存里的编辑是不是用户的数据"。
    /// 载荷还在 ⇒ 不是。见 `DocumentSession::has_unmerged_restore`。
    pub(crate) fn has_pending_restore(&self) -> bool {
        self.pending_restore.lock_or_recover().is_some()
    }
    /// 未合入载荷的编码形式（存档路径的兜底：宁可交回它，也不交空块）。
    pub(crate) fn pending_restore_bytes(&self) -> Option<Vec<u8>> {
        self.pending_restore
            .lock_or_recover()
            .as_ref()
            .and_then(|state| state.encode().ok())
    }
    /// 调用方持文档transaction；恢复候选集由宿主区域分配收窄，不按名称/旧序号猜测。
    ///
    /// 【为什么外面再包一层】恢复失败此前只有调用点的 `log::warn!`，用户与诊断包都
    /// 看不到"参数到底读回来没有"。包装层把结果写进 `restore_status`，四个调用点
    /// 不必各自重复这件事。
    ///
    /// 【T-D3：等待只由宿主图的变化推进】旧实现用"我们 poll 了多少次"当预算，
    /// 于是界面刷新得越快、预算烧得越快，10 秒的窗口能打印出 38.2 秒。现在只有一个
    /// 起点：图签名最后一次变化的时刻。签名不变且超过 `RESTORE_WAIT_WINDOW` 才报一次
    /// "没等到"，之后**不再重报**，直到宿主图再次变化。
    pub(in crate::render) fn merge_pending_restore(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<(), String> {
        // 【为什么要先看有没有载荷】成功路径会 `take()` 掉 pending；只有"进来时有、
        // 出去时没有"才说明这一次真的把状态合进去了。没有这个前提，一个从未收到
        // 载荷的实例也会被记成"恢复成功"。
        if !self.pending_restore.lock_or_recover().is_some() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let signature = self.restore_signature(document);
        let (waited, reported, changed) = {
            let mut progress = self.restore_progress.lock_or_recover();
            let changed = progress.signature != signature;
            if progress.since.is_none() || changed {
                progress.signature = signature;
                progress.since = Some(now);
                progress.reported = false;
            }
            (
                now.duration_since(progress.since.unwrap_or(now)),
                progress.reported,
                changed,
            )
        };
        if changed {
            // 宿主动了 ⇒ 退避失效，立刻重试一次（否则"改完工程还得再等十秒"）。
            *self.restore_next_attempt.lock_or_recover() = None;
            *self.restore_pending_reason.lock_or_recover() = None;
        }
        {
            let mut next = self.restore_next_attempt.lock_or_recover();
            if let Some(at) = *next {
                if now < at {
                    return Ok(());
                }
                *next = None;
            }
        }
        self.restore_attempts.fetch_add(1, Ordering::Relaxed);
        // 摘要要在 inner 之前取：成功路径会把 pending `take()` 掉。
        let pending_digest = self
            .pending_restore
            .lock_or_recover()
            .as_ref()
            .and_then(Self::restore_digest);
        let result = self.merge_pending_restore_inner(document);
        match &result {
            Ok(()) => {
                if self.pending_restore.lock_or_recover().is_some() {
                    if waited >= RESTORE_WAIT_WINDOW && !reported {
                        let reason = self
                            .restore_pending_reason
                            .lock_or_recover()
                            .clone()
                            .unwrap_or_else(|| "waiting-for-host-model".to_owned());
                        self.note_restore_failed(&format!(
                            "the host graph has not changed for {:.1}s ({reason}); the saved payload is kept and retried every {}s",
                            waited.as_secs_f64(),
                            RESTORE_RETRY_INTERVAL.as_secs()
                        ));
                        self.restore_progress.lock_or_recover().reported = true;
                        *self.restore_next_attempt.lock_or_recover() =
                            Some(now + RESTORE_RETRY_INTERVAL);
                    } else if !reported {
                        // 仍有载荷挂着（文档未就绪 / 空轨 / 缺 fade 元数据）：不是失败，
                        // 只是还不能收口。
                        self.note_restore_pending();
                        if self.restore_pending_reason.lock_or_recover().is_none() {
                            *self.restore_pending_reason.lock_or_recover() =
                                Some("waiting-for-host-model".to_owned());
                        }
                    }
                } else {
                    self.note_restore_applied();
                    if let Some(digest) = pending_digest {
                        *self.restore_applied_digest.lock_or_recover() = Some(digest);
                    }
                    self.reset_restore_budget();
                }
            }
            Err(error) => {
                self.note_restore_failed(error);
                self.restore_progress.lock_or_recover().reported = true;
                *self.restore_next_attempt.lock_or_recover() = Some(now + RESTORE_RETRY_INTERVAL);
            }
        }
        result
    }

    /// 【T18】`reconcile` 的身份失败属于"等宿主图重建"还是"真失败"。
    ///
    /// 【为什么抽成纯函数】它是本轮的**唯一判据**，而它决定用户看到"还在等"还是
    /// "没读回来" —— 那两句话的可执行动作完全相反。纯函数让六个边界可单测，
    /// 不必构造一整份 ARA 文档。
    ///
    /// 【为什么按措辞判而不是错误类型】`reconcile` 返回 `Result<(), String>`
    /// （`state_channel` 不依赖本模块的错误类型），三条身份错误都带 `identity`：
    /// "ARA edit identity missing…" / "…missing or ambiguous in rebuilt host graph" /
    /// "ARA saved edit identities are ambiguous…"。其余错误（修订号耗尽）是真失败。
    pub(super) fn classify_restore_identity_error(error: &str) -> RestoreIdentityVerdict {
        // 【为什么是 `identit` 而不是 `identity`】`state_channel` 的三条措辞分别是
        // "identity missing"、"identity missing or ambiguous" 与
        // **"identities are ambiguous"** —— 第三条是复数，用全词会漏掉它，
        // 而那正是"两条存档记录指向同一条宿主轨"这个最需要继续等的形状。
        if error.contains("identit") {
            RestoreIdentityVerdict::AwaitingHostGraph
        } else {
            RestoreIdentityVerdict::Fatal
        }
    }
    pub(super) fn merge_pending_restore_inner(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<(), String> {
        let mut pending = self.pending_restore.lock_or_recover();
        let Some(saved) = pending.as_ref() else {
            return Ok(());
        };
        if !document.ready.load(Ordering::Acquire) {
            self.note_restore_waiting("document-not-ready");
            return Ok(());
        }
        // 【为什么先看 timeline】`assigned_timeline` 在时间线还没发布时会返回
        // "host timeline unavailable"；把它当失败会让正常重开过程显示
        // `RestoreUnresolved`。这里把它归为"还在等宿主收口"，并留下可读原因。
        if document.timeline.lock_or_recover().is_none() {
            self.note_restore_waiting("host-timeline-not-ready");
            return Ok(());
        }
        let host = self.assigned_timeline(document)?;
        if host.tracks.is_empty() {
            // 空轨组件仍能恢复私有参数分组；不借零音频scope恢复其它轨的源曲线。
            if saved.params.is_empty() && saved.tracks.is_empty() && saved.atlas.is_empty() {
                let mut edits = document.edits.lock_or_recover();
                edits.groups = saved.groups.clone();
                edits.revision = edits
                    .revision
                    .checked_add(1)
                    .ok_or("edit revision exhausted")?;
                *pending = None;
            } else {
                self.note_restore_waiting("host-timeline-empty");
            }
            return Ok(());
        }
        let allowed: BTreeSet<_> = host.tracks.iter().map(|t| t.id.clone()).collect();
        let bindings = document
            .track_bindings
            .lock_or_recover()
            .iter()
            .filter(|(id, _)| allowed.contains(*id))
            .map(|(id, identity)| (id.clone(), identity.clone()))
            .collect();
        let mut restored = saved.clone();
        // 【T18：身份对不上是"还在等"，不是"失败"】
        //
        // `reconcile` 的契约是严格的：任何一条旧轨在新图里找不到唯一身份，它就返回
        // `Err`（这条契约由 `restored_edits_refuse_ambiguous_empty_and_missing_host_identity`
        // 等用例钉住，本轮**不动**它 —— 猜测归属会把参数挂到别人的轨道上）。
        //
        // 但"对不上"有两种成因，处置相反：
        //   * 存档坏了 / 身份本身为空 —— 真失败，报 `failed`；
        //   * 宿主**这一次**的图还没长成（重开工程时 region 尚未交付、轨道还在重建）——
        //     该继续等，报 `pending` 并说明在等什么。
        // 此前两者都走 `?` ⇒ `note_restore_failed` ⇒ 界面显示"没能读回参数"，而实际
        // 上再等一轮（或用户在宿主里做一次编辑）就好了。
        if let Err(error) = restored.reconcile(&bindings) {
            match Self::classify_restore_identity_error(&error) {
                RestoreIdentityVerdict::AwaitingHostGraph => {
                    self.note_restore_waiting(&format!("host-graph-not-rebuilt: {error}"));
                    return Ok(());
                }
                RestoreIdentityVerdict::Fatal => return Err(error),
            }
        }
        restored.atlas.migrate_legacy_gaps(&restored.params)?;
        let mut changed_geometry = false;
        if !restored.atlas.is_empty() {
            let rebound = restored
                .atlas
                .rebind(&host, &document.parameter_identities_locked(&host)?)?;
            changed_geometry = !restored.atlas.same_layout(&rebound);
            restored.atlas = rebound;
        }
        let mut client = host.clone();
        restored.apply(&mut client);
        if changed_geometry {
            crate::editor::parameter_atlas::merge_root_projection(
                &mut client.params_by_root_track,
                restored
                    .atlas
                    .project_root_records(&host, &document.parameter_identities_locked(&host)?)?,
            );
        } else if restored.atlas.is_empty() && !restored.params.is_empty() {
            // 旧v2只有项目帧：在首次真实绑定的宿主布局建立source basis，随后移动/拉伸沿此迁移。
            // 不猜旧会话key/名字，也不能回推旧状态未保存的、首次加载前已改变的布局。
            restored.atlas = crate::editor::parameter_atlas::ParameterAtlas::default()
                .capture(&client, &document.parameter_identities_locked(&host)?)?;
        }
        let mut edits = document.edits.lock_or_recover();
        let mut merged = edits.merge(&host, &client, edits.revision)?;
        merged.groups = restored.groups.clone();
        let clip_ids = host
            .clips
            .iter()
            .map(|clip| clip.id.clone())
            .collect::<BTreeSet<_>>();
        // 【G5：权威侧合并分两种，不能一律覆盖】
        //
        // 旧规则是"恢复的那一份更权威"，理由写得很直白 —— 它只在**重开工程**时成立
        // （在线状态是空的）。而切屏重载 take 时交回来的载荷携带的是**更旧**的参数，
        // 那时"恢复的更权威"就等于"回退"。
        //
        // 三种情况分开处置：
        //   * 在线状态是**空的**（冷启动）⇒ 整份装上。没有别的数据源。
        //   * 载荷到达时我们正处在**一次编辑会话**里 ⇒ 覆盖。撤销/重做在 ARA 里就是
        //     一次还原会话（`ARAInterface.h:2938-2943`），undo 必须生效。
        //   * 其余（宿主交回一份旧载荷）⇒ **按键补缺**：只补在线状态里没有的键。
        let authoritative = !restored.from_archive
            || edits.is_pristine()
            || restored.arrived_in_edit_session
            || document.authoritative_restore();
        let mut kept_live = 0_usize;
        for (key, params) in restored.atlas.authorities.clone() {
            if authoritative || !merged.atlas.authorities.contains_key(&key) {
                merged.atlas.set_authority(&key, params);
            } else {
                kept_live += 1;
            }
        }
        if kept_live > 0 {
            log::info!(
                "[ara] restore kept {kept_live} live parameter set(s); the payload carries older \
                 values for them and the live state is newer"
            );
        }
        merged.atlas.regions.retain(|id, _| !clip_ids.contains(id));
        merged.atlas.regions.extend(restored.atlas.regions);
        merged
            .atlas
            .copy_seeds
            .retain(|_, seed| !allowed.contains(&seed.root));
        merged.atlas.copy_seeds.extend(restored.atlas.copy_seeds);
        merged.atlas.gaps.retain(|root, _| !allowed.contains(root));
        merged.atlas.gaps.extend(restored.atlas.gaps);
        let items = document.fade_items_locked(&clip_ids);
        if !merged.fades.is_empty() && items.is_empty() && !clip_ids.is_empty() {
            // 【为什么是 pending 而不是失败】fade 归属元数据要等宿主几何刷新后
            // 才能算出来；这是一次可重试的暂缓，不是身份冲突。保留载荷，下一次
            // `prepare_renderers` / 分配变化会再试。
            self.note_restore_waiting("fade-metadata-pending");
            return Ok(());
        }
        merged.fades.retain(|item, _| !items.contains(item));
        merged.fades.extend(restored.fades);
        merged.reconcile(&document.track_bindings.lock_or_recover())?;
        *edits = merged;
        *pending = None;
        Ok(())
    }

    /// 保存前重新核对完整宿主图；歧义或中间编辑态不能伪装成可恢复的state。
    ///
    /// 【契约】只有"调用方给的 state stream 非法"才可能失败；**任何**图未就绪、
    /// flush 超时、reconcile 失败、fade 元数据在途、payload 超限都降级为"返回当前
    /// 可用的最佳载荷"。宿主把 `getState` 的失败当成"插件存档失败"，而那比保存一份
    /// 稍旧的快照糟得多。
    pub(crate) fn encode_state(&self) -> Result<Vec<u8>, String> {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        let (edits, capture_edits) = if let Some(document) = document {
            // 【为什么 flush 失败不再传播】actor 可能正在跑一次长合成；等 10 秒再失败
            // 会把宿主保存线程冻住，而且仍然存不下。短预算 + 用当前权威继续编码。
            if let Err(error) =
                document.flush_editor_with_budget(std::time::Duration::from_millis(200))
            {
                log::warn!("[ara] flush before save failed: {error}; saving the committed state");
            }
            let _transaction = document.transaction.lock_or_recover();
            if !document.ready.load(Ordering::Acquire) {
                log::warn!(
                    "[ara] save requested before the host graph is ready; using the last good \
                     snapshot"
                );
                let edits = self
                    .pending_restore
                    .lock_or_recover()
                    .as_ref()
                    .and_then(|state| state.encode().ok())
                    .or_else(|| document.archive_cache.lock_or_recover().clone())
                    .or_else(|| document.edits.lock_or_recover().encode().ok())
                    .unwrap_or_else(default_edit_state_bytes);
                return self.wrap_component_state(edits, None);
            }
            if let Err(error) = self.merge_pending_restore(&document) {
                log::warn!("[ara] merge pending restore before save failed: {error}");
            }
            let mut edits = document.edits.lock_or_recover();
            if let Err(error) = edits.reconcile(&document.track_bindings.lock_or_recover()) {
                log::warn!(
                    "[ara] reconcile before save failed: {error}; saving the unpruned state"
                );
            }
            let mut local = edits.clone();
            match self.assigned_timeline(&document) {
                Ok(host) => {
                    let allowed: BTreeSet<_> = host.tracks.iter().map(|t| t.id.clone()).collect();
                    local.params.retain(|id, _| allowed.contains(id));
                    local.tracks.retain(|t| allowed.contains(&t.id));
                    let clips = host
                        .clips
                        .iter()
                        .map(|clip| clip.id.clone())
                        .collect::<BTreeSet<_>>();
                    local.atlas.regions.retain(|id, _| clips.contains(id));
                    // 权威表必须按**同一口径**裁剪：键从已裁剪的 regions 派生 —— 那正是本组件
                    // 真实持有的片段身份。不裁剪会让一个组件把另一个组件的参数写进自己的存档
                    // （两个组件共享同一 source/modification 身份时尤其明显）。
                    let allowed_authorities = local
                        .atlas
                        .regions
                        .values()
                        .filter_map(|record| {
                            crate::editor::parameter_atlas::authority_key(
                                record.identity.item.as_deref(),
                                &record.identity.modification,
                            )
                        })
                        .collect::<BTreeSet<_>>();
                    local
                        .atlas
                        .authorities
                        .retain(|key, _| allowed_authorities.contains(key));
                    local
                        .atlas
                        .copy_seeds
                        .retain(|_, seed| allowed.contains(&seed.root));
                    local.atlas.gaps.retain(|root, _| allowed.contains(root));
                    if !local.fades.is_empty() {
                        let items = document.fade_items_locked(&clips);
                        // 【为什么这里不再失败】fade 元数据在途时保留整份 fade 表；
                        // 恢复端会按 clip 身份逐条 reconcile，丢几条好过整份存不下。
                        if items.is_empty() && !clips.is_empty() {
                            log::warn!(
                                "[ara] fade ownership metadata pending; saving the unpruned fades"
                            );
                        } else {
                            local.fades.retain(|item, _| items.contains(item));
                        }
                    }
                    local.bindings.retain(|id, _| allowed.contains(id));
                }
                Err(error) => {
                    log::warn!(
                        "[ara] assigned timeline unavailable during save: {error}; saving the \
                         full state"
                    );
                }
            }
            match local.encode() {
                Ok(bytes) => {
                    *self.last_ara_state.lock_or_recover() = Some(bytes.clone());
                    (bytes, None)
                }
                Err(error) => {
                    log::error!("[ara] state encode failed: {error}; using the last good snapshot");
                    (
                        self.last_ara_state
                            .lock_or_recover()
                            .clone()
                            .unwrap_or_else(default_edit_state_bytes),
                        None,
                    )
                }
            }
        } else if self.bound_once.load(Ordering::Acquire) {
            // 【为什么不能落到 Capture 分支】这个实例曾经绑定过 ARA，只是文档已经
            // 关闭。Capture 的 `self.edits` 从未被 ARA 写过；序列化它等于用一份空
            // 状态覆盖用户全部编辑。保留最后一份成功载荷（或尚未合入的恢复载荷）。
            let edits = self
                .last_ara_state
                .lock_or_recover()
                .clone()
                .or_else(|| {
                    self.pending_restore
                        .lock_or_recover()
                        .as_ref()
                        .and_then(|state| state.encode().ok())
                })
                .unwrap_or_else(default_edit_state_bytes);
            (edits, None)
        } else {
            // 【Capture 形态：用户曲线在会话上，不在这里】`self.edits` 从未被任何
            // Capture 生产路径写过 —— 序列化它只会得到一个语法合法、语义为空的
            // `EditState`（那正是"重开工程参数全丢"的直接原因）。真正的权威是
            // `EditorSession::timeline`，见 `editor::capture_edits`。
            //
            // `edits` 仍然写一份默认 `EditState`：`restore_state` 要能把它读回来，
            // 而 ARA 字段在 Capture 下本来就该是空的。
            let edits = self
                .pending_restore
                .lock_or_recover()
                .as_ref()
                .and_then(|state| state.encode().ok())
                .or_else(|| self.edits.lock_or_recover().encode().ok())
                .unwrap_or_else(default_edit_state_bytes);
            let capture_edits = self.capture_edits_for_save();
            (edits, capture_edits)
        };
        self.wrap_component_state(edits, capture_edits)
    }

    /// 保存 Capture 用户编辑：优先 live 会话，其次恢复载荷，最后 owner 暂存。
    ///
    /// 【为什么顺序不能反】live 编辑是用户刚做的事，最权威；恢复载荷是"还没合入
    /// 时间线"的存档；owner 暂存是"连会话都还没建"。反过来会让一次保存把用户刚
    /// 画的曲线覆盖成旧存档。
    pub(super) fn capture_edits_for_save(&self) -> Option<Vec<u8>> {
        match self.capture_session_if_initialized() {
            Some(session) => {
                // 【为什么不是只看 `is_loaded`】恢复后、actor 第一次加载前，时间线可能
                // 已经被写入（恢复路径/测试）；那仍是 live 权威，不能回落到空载荷。
                let live = if session.is_loaded() || session.has_live_capture_edits() {
                    if let Err(error) =
                        session.flush_with_budget(std::time::Duration::from_millis(200))
                    {
                        log::warn!(
                            "[capture] flush before save exceeded the budget: {error}; using \
                             the last committed curves"
                        );
                    }
                    match session.serialize_capture_edits() {
                        Ok(bytes) => Some(bytes),
                        Err(error) => {
                            log::error!("[capture] serialize before save failed: {error}");
                            None
                        }
                    }
                } else {
                    None
                };
                live.or_else(|| session.serialize_pending_capture_edits())
            }
            None => self.pending_capture_edits.lock_or_recover().clone(),
        }
    }

    /// 把 `EditState` 字节与 Capture 清单包成组件 state。
    ///
    /// 【为什么清单只带路径】PCM 在 sidecar 文件里（见 `capture::sidecar`）：
    /// 组件 state 的预算是 `MAX_COMPONENT_STATE`（64 MiB），而 4 分钟立体声 96 kHz 已是 184 MB，何况组件 state
    /// 会进宿主的每一次保存与撤销快照。清单是几 KB。
    pub(super) fn wrap_component_state(
        &self,
        edits: Vec<u8>,
        capture_edits: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, String> {
        let capture = self.capture.manifest();
        let has_takes = capture["takes"]
            .as_array()
            .is_some_and(|takes| !takes.is_empty());
        // 【为什么没有 Capture 数据时返回**裸** `EditState` 字节】两重理由：
        //
        // 1. 那些实例没有任何额外信息要存，套一层容器纯是开销；
        // 2. 更重要的是**兼容性**：此前 `encode_state` 的输出就是 `EditState` 的
        //    编码（ARA 路径下是一段 JSON）。加容器会让每一个既有消费者与既有工程
        //    都读到看不懂的东西 —— 而它们本来完全正常。
        //
        // 于是格式变成"有 Capture 数据才是容器，否则是历史格式"，而 `decode`
        // 两种都认。这条不对称是有意的：新信息需要新格式，旧信息不需要。
        let capture_edits = capture_edits.filter(|bytes| !bytes.is_empty());
        if !has_takes && capture_edits.is_none() {
            return Ok(edits);
        }
        Ok(crate::component_state::ComponentState {
            edits,
            capture: has_takes.then(|| capture.to_string().into_bytes()),
            capture_edits,
        }
        .encode())
    }

    /// 文档级的宿主时间线（**不**按实例收窄）。
    ///
    /// 【为什么与 `assigned_timeline` 分开】后者把 clip 收窄到本实例被分配的 region，
    /// 因为渲染必须只碰自己的那一份。而导出与几何判据要的是**整个文档**：
    /// 用户的心智模型是「这个工程」，不是「这个插件窗口」。
    pub(super) fn document_timeline(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<hifishifter_kernel::state::TimelineState, String> {
        let mut timeline = document
            .timeline
            .lock_or_recover()
            .clone()
            .ok_or("host timeline unavailable")?;
        if let Some(tempo) = document.clock.tempo() {
            timeline.bpm = tempo;
        }
        document.project_plugin_musical_context_locked(&mut timeline);
        Ok(timeline)
    }
}
