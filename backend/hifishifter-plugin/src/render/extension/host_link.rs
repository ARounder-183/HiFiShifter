//! 工作区（host-link）导出与编辑回灌。
//!
//! 【文件名为什么不叫 `link.rs`】那样会在本模块里遮蔽 `render::link`，而这批方法
//! 正是靠裸名 `link::…` 工作的（拆分前它经祖先模块解析到 `render::link`）。
use super::*;
impl ExtensionOwner {
    /// 把**整个文档**导出成一个工作区。
    ///
    /// 【为什么是文档级而不是实例级】管道时代的服务端挂在实例上，于是导出的 clip
    /// 被收窄到「本实例拥有的 region」，10 条轨就有 10 个候选，而 `nMaxInstances = 1`
    /// 只能连一个。内核侧早就有这个唯一性（一个文档一个 `EditorSession`）。
    /// ARA 契约关心的是 per playback region 只有一个 editor renderer ——
    /// 导出不新增任何 renderer。
    ///
    /// 【为什么素材缺音频不再是失败】没有授权 PCM 的素材以
    /// [`hifishifter_host_link::AudioState::Partial`] 登记：几何保留、音频槽留空。
    /// 管道时代这会让**整份导入**失败，而用户看到的是「连接失败」。
    pub(crate) fn export_link(
        &self,
        root: &std::path::Path,
    ) -> Result<
        (
            hifishifter_host_link::LinkManifest,
            hifishifter_host_link::Workspace,
        ),
        String,
    > {
        use hifishifter_host_link as link;
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| crate::editor::errors::DOCUMENT_CLOSED.error())?;
        let _transaction = document.transaction.lock_or_recover();
        self.merge_pending_restore(&document)?;
        if !document.ready.load(Ordering::Acquire) {
            return Err(format!(
                "{}; edit something in the host and try again",
                crate::editor::errors::HOST_MODEL_NOT_READY
            ));
        }
        let model_revision = document.revision.load(Ordering::Acquire);
        let mut timeline = self.document_timeline(&document)?;
        let edits_revision = {
            let edits = document.edits.lock_or_recover();
            // 导出的是**用户现在看到的**那一份，所以要把已保存的曲线叠上去；
            // 否则每次重新导出都会让 App 丢掉它上次提交的编辑。
            edits.apply(&mut timeline);
            edits.revision
        };

        let host_label = self.host_backend().app_name().unwrap_or("Host").to_string();
        let doc_key = self.document_key(&document);
        let identity = link::link_id(link::SourceKind::Ara, &host_label, &doc_key);
        // 【为什么目录名就是身份】重新导出必须落在**同一个**工作区上，否则每次导出
        // 都在用户磁盘上留一个新目录。身份稳定，目录名就稳定。
        let workspace = link::Workspace::at(root.join(identity.replace(':', "-")));

        let mut sources = Vec::new();
        {
            let available = document.edit_sources.lock_or_recover();
            let keys: BTreeSet<String> = timeline
                .clips
                .iter()
                .filter_map(|clip| clip.source_path.clone())
                .collect();
            for key in keys {
                let Some(pcm) = available.get(&key) else {
                    sources.push(link::SourceEntry {
                        key,
                        sample_rate: 0,
                        channels: 0,
                        frames: 0,
                        state: link::AudioState::Partial,
                        file: None,
                        blake3: None,
                        bytes: None,
                        note: Some(crate::editor::errors::HOST_PCM_UNAVAILABLE.to_string()),
                    });
                    continue;
                };
                match workspace.write_audio(&key, pcm.sample_rate, &pcm.planes) {
                    Ok(audio) => sources.push(link::SourceEntry {
                        key,
                        sample_rate: pcm.sample_rate,
                        channels: pcm.planes.len() as u32,
                        frames: pcm.planes.first().map_or(0, Vec::len) as u64,
                        state: link::AudioState::Complete,
                        file: Some(audio.file),
                        blake3: Some(audio.blake3),
                        bytes: Some(audio.bytes),
                        note: None,
                    }),
                    Err(error) => {
                        log::warn!("[link] cannot write audio for {key}: {error}");
                        sources.push(link::SourceEntry {
                            key,
                            sample_rate: pcm.sample_rate,
                            channels: pcm.planes.len() as u32,
                            frames: pcm.planes.first().map_or(0, Vec::len) as u64,
                            state: link::AudioState::Partial,
                            file: None,
                            blake3: None,
                            bytes: None,
                            note: Some(error),
                        });
                    }
                }
            }
        }

        // 逐条降级报告：宿主里被跳过、进不来的东西。空表示**没有丢掉任何东西**。
        let losses: Vec<link::LossEntry> = document
            .skipped_regions
            .lock_or_recover()
            .iter()
            .map(|(index, reason)| link::LossEntry {
                id: format!("region #{index}"),
                reason: reason.clone(),
                detail: None,
            })
            .collect();

        let manifest = link::LinkManifest {
            format: link::FORMAT.into(),
            format_version: link::FORMAT_VERSION,
            link_id: identity.clone(),
            source_kind: link::SourceKind::Ara,
            host_label,
            created_unix_ms: link::now_ms(),
            producer: link::Producer {
                implementation: format!("hifishifter-ara {}", crate::VERSION),
                pid: Some(std::process::id()),
            },
            // 【为什么不写 `project_sec` 以外的推导值】时间轴就是宿主事实的投影；
            // 这里是唯一一处把 `TimelineState` 交给外部的出口。
            timeline: serde_json::to_value(&timeline).map_err(|error| error.to_string())?,
            // 【为什么在这里算，而不是让消费者算】见 `LinkManifest::host_geometry`：
            // 判据的两边必须出自同一次计算，否则任何浮点归一化差异都会变成假漂移。
            host_geometry: hifishifter_kernel::editor::host_timeline::host_geometry_projection(
                &timeline,
            ),
            sources,
            granted: vec![link::Capability::Commit, link::Capability::Audio],
            revisions: link::Revisions {
                edits: edits_revision,
                model: model_revision,
            },
            admission: link::Admission::Strict,
            losses,
            diagnostics: Some(self.link_diagnostics(&document, &timeline)),
        };
        workspace.write_manifest(&manifest)?;
        workspace.write_status(&link::LinkStatus::initial(&identity, manifest.revisions))?;
        Ok((manifest, workspace))
    }

    /// 文档身份：`TrackBindings` 的 `(modification, source)` 签名。
    ///
    /// 【为什么不用 region key】region key 只在会话内稳定。`TrackBindings` 是既有的、
    /// 被文档注释称为「跨会话身份」的那一份（见 `ara/model.rs` 的 `publish_track_bindings`），
    /// 它由宿主给的两个 persistentID 组成，重开宿主工程后仍然相同。
    pub(super) fn document_key(&self, document: &super::document::DocumentSession) -> String {
        let bindings = document.track_bindings.lock_or_recover();
        if bindings.is_empty() {
            return "unknown-document".to_string();
        }
        let mut pairs: Vec<String> = bindings
            .values()
            .flatten()
            .map(|(modification, source)| format!("{modification}|{source}"))
            .collect();
        pairs.sort();
        pairs.dedup();
        let joined = pairs.join(",");
        // 摘要而非原文：身份要短、要稳定、要能进文件名，而原文可以很长。
        blake3::hash(joined.as_bytes()).to_hex().to_string()[..16].to_string()
    }

    /// 只读现场诊断：渲染器就绪度，以及**已发布输出对片段范围的覆盖**。
    ///
    /// 【为什么这次真的会有人读】管道时代它被生产出来后在 App 侧**没有任何消费者** ——
    /// 一条只写不读的线。现在它随清单落盘，面板上直接显示
    /// 「宿主渲染器尚未就绪」「该片段在宿主侧是静音」。
    ///
    /// 【为什么要按片段给覆盖与峰值，而不是只报 ready】用户真正的疑问是
    /// 「我导出的是不是有声音」。就绪只说明快照发布了，不说明那一段**有内容**。
    pub(super) fn link_diagnostics(
        &self,
        document: &super::document::DocumentSession,
        timeline: &hifishifter_kernel::state::TimelineState,
    ) -> serde_json::Value {
        // 与面板显示的顺序一致：按起点排前 8 条，避免长工程把清单撑大。
        let mut clips = timeline.clips.iter().collect::<Vec<_>>();
        clips.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec));
        let ranges = clips
            .iter()
            .take(8)
            .map(|clip| (clip.id.clone(), clip.start_sec, clip.length_sec))
            .collect::<Vec<_>>();
        let renderers = document
            .renderer_owners()
            .into_iter()
            .filter(|owner| owner.renders_playback())
            .map(|owner| {
                let (busy, error) = owner.local_preparation_state();
                serde_json::json!({
                    "busy": busy,
                    "error": error,
                    "prepared": owner.prepared_snapshot(),
                    "outputs": owner
                        .snapshots
                        .iter()
                        .map(|snapshot| snapshot.diagnose(&ranges))
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({ "renderers": renderers })
    }

    /// 合并一份**回写**的编辑，并返回要落盘的状态。
    ///
    /// 【几何归属判据为什么在这里算】消费者只有宿主几何的一份拷贝，生产者才有权威。
    /// 判据放在生产者侧之后，消费者侧的**活状态**（后台的假立体声折叠扫描、
    /// 波形回填、音高分析都在写它）再也不可能打穿判据 —— 于是那个用于压制后台子系统
    /// 的进程级全局开关也随之消失。
    pub(crate) fn apply_link_edit(
        &self,
        identity: &str,
        edit: &hifishifter_host_link::EditManifest,
    ) -> hifishifter_host_link::LinkStatus {
        use hifishifter_host_link as link;
        let mut status = link::LinkStatus {
            format: link::STATUS_FORMAT.into(),
            format_version: link::FORMAT_VERSION,
            link_id: identity.to_string(),
            ok: false,
            code: None,
            error: None,
            revisions: link::Revisions::default(),
            updated_unix_ms: link::now_ms(),
        };
        let outcome = (|| -> Result<(), (u32, String)> {
            let document = self
                .document
                .lock_or_recover()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
                .ok_or((
                    link::error_code::DOCUMENT_CLOSED,
                    crate::editor::errors::DOCUMENT_CLOSED.error(),
                ))?;
            if edit.link_id != identity {
                return Err((
                    link::error_code::LINK_MISMATCH,
                    format!("this edit belongs to {}, not {identity}", edit.link_id),
                ));
            }
            // ── 捕获：短事务内冻结几何、参数与渲染输入 ─────────────────────
            //
            // 【为什么必须是一个独立的作用域】`transaction` 是**不可重入**的互斥量，
            // 而下面"提交"那一半还要再取一次。第一份守护必须在这里释放 ——
            // 否则第二次获取就是一次必然的死锁（而且它阻塞的是宿主回调）。
            // 同理，渲染也不能在事务里做：那会把宿主挡住整个渲染时长。
            let (base_model, candidate, epoch, scope, inputs) = {
                let _transaction = document.transaction.lock_or_recover();
                self.merge_pending_restore(&document)
                    .map_err(Self::invalid_edit)?;
                if !document.ready.load(Ordering::Acquire) {
                    return Err((
                        link::error_code::MODEL_NOT_READY,
                        format!(
                            "{}; edit something in the host",
                            crate::editor::errors::HOST_MODEL_NOT_READY
                        ),
                    ));
                }
                let base_model = document.revision.load(Ordering::Acquire);
                // 几何归属：宿主拥有的字段变了就拒绝，并**指名字段**。
                // 判据在这里算而不是在消费者侧 —— 只有生产者有权威，
                // 而消费者的活状态（后台的假立体声折叠、波形回填）因此再也打不穿它。
                let current = self
                    .document_timeline(&document)
                    .map_err(Self::invalid_edit)?;
                let differences = hifishifter_kernel::editor::host_timeline::geometry_differences(
                    &edit.host_geometry,
                    &hifishifter_kernel::editor::host_timeline::host_geometry_projection(&current),
                );
                if !differences.is_empty() {
                    return Err((
                        link::error_code::GEOMETRY_CHANGED,
                        format!("ara_host_fields: {}", differences.join("; ")),
                    ));
                }
                if base_model != edit.base_revisions.model {
                    return Err((
                        link::error_code::CONFLICT,
                        format!(
                            "{} the host model is {base_model} but this edit was written against {}",
                            crate::editor::errors::CONFLICT_PREFIX,
                            edit.base_revisions.model
                        ),
                    ));
                }
                let params: std::collections::BTreeMap<
                    String,
                    hifishifter_kernel::state::TrackParamsState,
                > = serde_json::from_value(edit.params.clone()).map_err(Self::invalid_edit)?;
                let tracks: Vec<hifishifter_kernel::state::Track> =
                    serde_json::from_value(edit.tracks.clone()).map_err(Self::invalid_edit)?;
                let mut edits = document.edits.lock_or_recover();
                edits
                    .reconcile(&document.track_bindings.lock_or_recover())
                    .map_err(Self::invalid_edit)?;
                let mut candidate = edits
                    .merge_params(
                        &self
                            .assigned_timeline(&document)
                            .map_err(Self::invalid_edit)?,
                        &params,
                        &tracks,
                        edit.base_revisions.edits,
                    )
                    .map_err(Self::invalid_edit)?;
                let mut curve_timeline = self
                    .assigned_timeline(&document)
                    .map_err(Self::invalid_edit)?;
                candidate.apply(&mut curve_timeline);
                candidate.atlas = edits
                    .atlas
                    .capture(
                        &curve_timeline,
                        &document
                            .parameter_identities_locked(&curve_timeline)
                            .map_err(Self::invalid_edit)?,
                    )
                    .map_err(Self::invalid_edit)?;
                candidate
                    .reconcile(&document.track_bindings.lock_or_recover())
                    .map_err(Self::invalid_edit)?;
                drop(edits);
                let inputs = document
                    .renderer_owners()
                    .into_iter()
                    .filter(|owner| owner.renders_playback())
                    .map(|owner| {
                        let (keys, input) = owner.ara_render_input(&document, &candidate, true)?;
                        Ok((owner, keys, input))
                    })
                    .collect::<Result<Vec<_>, String>>()
                    .map_err(Self::invalid_edit)?;
                (
                    base_model,
                    candidate,
                    document.render_epoch.load(Ordering::Acquire),
                    document.scope_revision.load(Ordering::Acquire),
                    inputs,
                )
            };
            // ── 渲染：**不持事务锁** ──────────────────────────────────────
            let mut prepared = Vec::new();
            for (owner, keys, input) in inputs {
                for publisher in &owner.snapshots {
                    publisher.collect_retired();
                }
                let mut snapshots = input
                    .render(Arc::new(std::sync::atomic::AtomicBool::new(false)))
                    .map_err(Self::invalid_edit)?;
                super::snapshot::compact_prepared(&mut snapshots).map_err(Self::invalid_edit)?;
                prepared.push((owner, keys, snapshots));
            }
            // ── 提交：短事务内复核（渲染期间一切可能都变了）并发布 ──────────
            let _transaction = document.transaction.lock_or_recover();
            if !document.ready.load(Ordering::Acquire)
                || document.revision.load(Ordering::Acquire) != base_model
            {
                return Err((
                    link::error_code::CONFLICT,
                    "host model changed while the edit was being applied; export again".into(),
                ));
            }
            let mut edits = document.edits.lock_or_recover();
            if edits.revision != edit.base_revisions.edits {
                return Err((
                    link::error_code::CONFLICT,
                    "edit revision changed while the edit was being applied; export again".into(),
                ));
            }
            if document.render_epoch.load(Ordering::Acquire) != epoch
                || document.scope_revision.load(Ordering::Acquire) != scope
            {
                return Err((
                    link::error_code::CONFLICT,
                    "host audio access changed while the edit was being applied; try again".into(),
                ));
            }
            for (owner, keys, snapshots) in &prepared {
                if owner.assigned_regions().map_err(Self::invalid_edit)? != *keys {
                    return Err((
                        link::error_code::CONFLICT,
                        "assigned regions changed while the edit was being applied; try again"
                            .into(),
                    ));
                }
                if !owner
                    .snapshots
                    .iter()
                    .zip(snapshots)
                    .all(|(publisher, snapshot)| publisher.has_capacity(snapshot))
                {
                    return Err((
                        link::error_code::CONFLICT,
                        "retired snapshot budget exhausted; reopen the plug-in window".into(),
                    ));
                }
            }
            for (owner, keys, snapshots) in prepared {
                for (publisher, snapshot) in owner.snapshots.iter().zip(snapshots) {
                    publisher
                        .publish(snapshot)
                        .map_err(|error| format!("snapshot publish failed: {error:?}"))
                        .map_err(Self::invalid_edit)?;
                }
                owner.record_prepared(base_model, candidate.revision, epoch, scope, keys);
            }
            *edits = candidate;
            log::info!(
                "[link] applied an edit revision={} model={base_model}",
                edits.revision
            );
            Ok(())
        })();
        match outcome {
            Ok(()) => status.ok = true,
            Err((code, error)) => {
                log::warn!("[link] rejected an edit: {error}");
                status.code = Some(code);
                status.error = Some(error);
            }
        }
        status.revisions = self.link_revisions();
        status
    }

    pub(super) fn link_revisions(&self) -> hifishifter_host_link::Revisions {
        let document = self
            .document
            .lock_or_recover()
            .as_ref()
            .and_then(std::sync::Weak::upgrade);
        match document {
            Some(document) => hifishifter_host_link::Revisions {
                edits: document.edits.lock_or_recover().revision,
                model: document.revision.load(Ordering::Acquire),
            },
            None => hifishifter_host_link::Revisions::default(),
        }
    }

    /// 把一条内部错误规整成「协议码 + 文本」。
    ///
    /// 【为什么统一走 `INVALID_EDIT`】能走到这里的失败都属于「这份编辑本身不合格」：
    /// 曲线非有限、轨道未知、音量越界、JSON 形状不对。消费者据此**不要重试**，
    /// 而是把 `error` 显示给用户 —— 与 `CONFLICT`（重导一次就好）处置相反。
    pub(super) fn invalid_edit(error: impl std::fmt::Display) -> (u32, String) {
        (
            hifishifter_host_link::error_code::INVALID_EDIT,
            error.to_string(),
        )
    }
    /// 调用方在短事务内冻结几何/参数/PCM，只复制本renderer授权源，不执行合成。
    pub(crate) fn ara_render_input(
        &self,
        document: &super::document::DocumentSession,
        edits: &crate::state_channel::EditState,
        edited: bool,
    ) -> Result<(Vec<u64>, super::input::RenderInput), String> {
        if !document.ready.load(Ordering::Acquire) {
            return Err(format!(
                "{}; {}",
                crate::editor::errors::HOST_PCM_UNAVAILABLE,
                crate::editor::errors::HOST_MODEL_NOT_READY
            ));
        }
        let keys = self.assigned_regions().map_err(|e| e.to_string())?;
        let mut timeline = self.assigned_timeline(document)?;
        // 所有授权轨道保留原kernel的全局solo/父链判定，clip仍只有本renderer分配区域。
        timeline.tracks = document.workspace_timeline_locked()?.tracks;
        // 【为什么这里也要播种宿主 take 事实】渲染路径**不经过** `workspace_timeline_locked`
        // 的 clip 过滤（见上一行的注释：只借了 tracks）。而 `assigned_timeline` 读的是
        // `document.timeline`（ARA 映射产物），ARA **没有**方向位/循环源/声道模式
        // （`ara::mapping::LOST_FIELDS`）—— 不播种，倒放片段会被按正放合成（用户报障：
        // "在 HiFiShifter 中，倒放被识别为正放"）。与音阶播种（`document.timeline` 的
        // `project_scale_notes`）是同一条纪律、同一个理由。
        document.project_host_take_facts_locked(&mut timeline);
        let mut resolved = edits.clone();
        resolved.reconcile(&document.track_bindings.lock_or_recover())?;
        resolved.apply(&mut timeline);
        if !resolved.groups.is_empty() {
            // 空父轨也能承载参数组算法；不加入其音频clip，更不混入其它renderer的区域。
            let mut grouped_view = timeline.clone();
            document.project_private_group_view(&mut grouped_view, &resolved)?;
            timeline.tracks = grouped_view.tracks;
            for track in &mut timeline.tracks {
                track.parent_id = None;
            }
            resolved.groups.processing_settings(
                &mut timeline,
                &document.group_aliases(""),
                &std::collections::BTreeSet::new(),
            );
        }
        let geometry = document.regions.lock_or_recover();
        // 【为什么消失的 region 只跳过】此前这里 `ok_or("assigned region disappeared")?`
        // + `collect::<Result<_,_>>()?`：只要**一条**已分配的 key 不在 `regions` 表里
        // （宿主销毁了 region、分配还没撤销的窗口），整份 `ara_render_input` 失败 ⇒
        // `prepare_job` 不发布快照 ⇒ 声音停在旧版本，且界面上只有一条内部错误。
        // 逐条过滤：消失的那条本次不参与渲染，其余照常。`keys` 与 `regions` 必须
        // **同长同序**（它们一起进 `RenderInput`），所以两者一起筛。
        let mut resolved_keys = Vec::with_capacity(keys.len());
        let mut regions = Vec::with_capacity(keys.len());
        let mut vanished = 0_usize;
        for key in &keys {
            match geometry.get(key) {
                Some(region) => {
                    resolved_keys.push(*key);
                    regions.push(region.clone());
                }
                None => vanished += 1,
            }
        }
        drop(geometry);
        if vanished > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.capture.vanished-region",
                format_args!(
                    "[ara] {vanished} assigned region(s) are gone from the model this round; \
                     they are omitted from the render input"
                ),
            );
        }
        let keys = resolved_keys;
        let stretch = regions.iter().any(|region| {
            (region.duration_in_modification_time - region.duration_in_playback_time).abs() > 1e-9
        });
        let owned_fade = regions.iter().any(|region| {
            region.has_content_based_fade_at_head || region.has_content_based_fade_at_tail
        });
        if owned_fade {
            document.project_audio_fades_locked(&mut timeline, &resolved)?;
        }
        // 【倒放也必须走内核】未编辑、无拉伸、无淡化委托的片段本来走
        // `mix_plain_regions` 直通（`RenderInput.timeline = None`）—— 那是"原样放 ARA
        // 授权 PCM"。但倒放片段**不能**直通：ARA 给的是正向 PCM，方向翻转只发生在内核
        // 的 `render_mixdown_internal`（`reverse_interleaved_frames`）。不把倒放片段推进
        // 内核，它就会被原样按正向放出来（实测：输出 = 源窗口正放）。
        let reversed_clip = timeline.clips.iter().any(|clip| clip.reversed);
        // 【为什么"事实还没到"也必须走内核】方向 / 循环源 / 声道模式是 ARA **表达不到**
        // 的输入（`ara::mapping::LOST_FIELDS`），只有宿主清单能给。直通路径
        // （`mix_plain_regions`）不做这些处理 —— 于是在"宿主有原生扩展、清单却还没刷新"
        // 这个窗口里，clip 的 `reversed` 还是默认 false，一个倒放片段会被原样按正放放出来：
        // 波形、按钮都正常，声音是错的，且没有任何线索。
        //
        // 【为什么条件不是"清单为空"】宿主**根本不给**这些概念时（ARA-only），强制走内核
        // 换不来任何信息，只会白白多算一遍、多占一份工作缓冲。真正会出错的是上面那个
        // 中间窗口，所以判据是"有原生扩展 且 清单还没到"。
        let facts_pending =
            self.native_host().is_some() && document.ui_tracks.lock_or_recover().is_empty();
        let kernel_render = edited
            || stretch
            || owned_fade
            || reversed_clip
            || !resolved.groups.is_empty()
            || facts_pending;
        let clip_parameters = if resolved.atlas.is_empty() {
            Default::default()
        } else {
            resolved
                .atlas
                .project_local(&timeline, &document.parameter_identities_locked(&timeline)?)?
        };
        let available = if kernel_render {
            document.edit_sources.lock_or_recover()
        } else {
            document.sources.lock_or_recover()
        };
        let sources = regions
            .iter()
            .map(|region| region.audio_source_persistent_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|id| available.get(&id).map(|pcm| (id.clone(), pcm.clone())))
            .collect();
        Ok((
            keys,
            super::input::RenderInput {
                timeline: kernel_render.then_some(timeline),
                regions,
                sources,
                // 宿主通报的工程速率。没有内核渲染时 `mix_plain_regions` 要靠它把宿主的
                // "工程速率秒"折回源自己的秒（见 `RenderInput::project_sample_rate`）。
                project_sample_rate: Some(self.setup_sample_rate()).filter(|rate| *rate > 0.0),
                clip_parameters,
            },
        ))
    }

    /// 从宿主时间线筛选实际分配的区域，而不是让每个处理器混整张文档。
    pub(super) fn assigned_timeline(
        &self,
        document: &super::document::DocumentSession,
    ) -> Result<hifishifter_kernel::state::TimelineState, String> {
        let keys = self.assigned_regions().map_err(|e| e.to_string())?;
        // 身份缺席时按 region key 派生（见 `clip_id_for`）：region 分配早于
        // `clip_ids` 重算，这个窗口里的渲染不该失败。
        //
        // 【为什么无效键只跳过】`clip_id_for` 只对 key==0 返回 `None`。此前这里的
        // `ok_or(...)?` + `collect::<Result>()` 会让**一个**无效键把整份
        // `assigned_timeline` 打失败 —— 而它是 `ara_render_input` /
        // `merge_pending_restore` / `encode_state` 的共同入口 ⇒ 快照不发布 +
        // 存档保存失败。逐条过滤：那一条 region 本次不参与渲染，其余照常。
        let mut invalid_keys = 0_usize;
        let ids = keys
            .iter()
            .filter_map(|key| match document.clip_id_for(*key) {
                Some(id) => Some(id),
                None => {
                    invalid_keys += 1;
                    None
                }
            })
            .collect::<BTreeSet<_>>();
        if invalid_keys > 0 {
            hifishifter_kernel::log_limiter::emit_limited_key(
                log::Level::Warn,
                "ara.assigned.invalid-region-key",
                format_args!(
                    "[ara] {invalid_keys} assigned region(s) have no usable key this round; \
                     they are omitted from the render input"
                ),
            );
        }
        let mut timeline = document
            .timeline
            .lock_or_recover()
            .clone()
            .ok_or("host timeline unavailable")?;
        if let Some(tempo) = document.clock.tempo() {
            timeline.bpm = tempo;
        }
        // 【为什么这里也要播种音阶】渲染路径**不经过** `workspace_timeline_locked`
        // 的 clip 分支：它克隆原始 ARA 时间线后只借用后者的 `tracks`。所以只在
        // `workspace_timeline_locked` 里播种，渲染拿到的仍是默认 C 大调 ——
        // 正是"界面显示 Gb、内核按 C 渲染"那条缺陷。
        document.project_plugin_musical_context_locked(&mut timeline);
        timeline.clips.retain(|clip| ids.contains(&clip.id));
        // 被分配了 sequence、当前却没有 clip 的轨道仍然留下：空 folder 轨、只有静音
        // item 或 PCM 尚未授权的子轨都属此类，而它们本该是**参数根** —— 与
        // `add_group_view_tracks` 的既有口径一致（空父轨也能承载参数组算法）。
        //
        // 【为什么这不会泄漏无关轨道】这条边只来自宿主 `createRegionSequence` 的真实
        // 键（`sequence_track_index`），且只取**本 owner 已分配**的那些 sequence。
        // 绝不按轨名/位置/序号凭空造轨道。
        let assigned_sequences = self
            .sequences
            .lock_or_recover()
            .values()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let empty_roots = document
            .sequence_track_index
            .lock_or_recover()
            .iter()
            .filter(|(key, _)| assigned_sequences.contains(key))
            .filter_map(|(_, index)| timeline.tracks.get(*index).map(|track| track.id.clone()))
            .collect::<BTreeSet<_>>();
        timeline.tracks.retain(|track| {
            timeline.clips.iter().any(|clip| clip.track_id == track.id)
                || empty_roots.contains(&track.id)
        });
        Ok(timeline)
    }
}
