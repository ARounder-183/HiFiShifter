//! ARA render inputs: geometry and PCM come from the live host document.
use super::*;

impl ExtensionOwner {
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
        // 【几何也必须来自同一份清单】见 `project_host_item_geometry_locked`：移动
        // item 不产生 ARA 收口，所以 `document.timeline` 里的位置会落后一步；而编辑器
        // 读的是 REAPER 清单。不在这里覆盖，就是"GUI 动了、声音没动"。
        document.project_host_item_geometry_locked(&mut timeline);
        let mut resolved = edits.clone();
        resolved.reconcile(&document.track_bindings.lock_or_recover())?;
        resolved.apply(&mut timeline);
        // 【渲染与 GUI 必须同源】派生原始值只住在基线层（`DocumentSession::baseline`）。
        // 少了这一步，`project_local` 会拿到空的 `pitch_orig`：未编辑帧的默认值退化成
        // 0、渲染器也失去"源基频"这条参考 —— 用户看到参数线画好了，却一个采样都没变。
        // 【为什么在 apply 之后】apply 才建立根的参数条目；在它之前安装会被"目标为空"
        // 的条目挡掉。
        document.install_baseline_locked(&mut timeline);
        // 【有编辑但没基线 ⇒ 不许渲染】没有 `pitch_orig` 就没有源基频：渲染器会把
        // 用户曲线原样"应用"到一条没有参考的链路上，输出与原始音频逐采样相同 ——
        // 界面显示已应用、音频一个采样都没变。等分析（actor 会重试），不要静默降级。
        {
            let baseline = document.baseline_snapshot();
            for (root, params) in timeline.params_by_root_track.iter() {
                // 【两条来源都要看】编辑可能在权威里（正常路径），也可能只在显示数组里
                // （旧存档恢复、App 导入）。任一存在而基线缺席 ⇒ 等分析。
                // 【为什么只看这两条】`pitch_edit` 在未编辑时是**分析目标曲线**，
                // 它非空不代表用户改过（测试夹具与"先画后分析"都长这样）。用
                // "曲线非空 + 原始值空"当编辑证据会把正常加载判成"等分析"，实测
                // 会让三条 automatic-apply 用例整条停摆。用户编辑的权威判据只有
                // 权威表里的锚层与 `pitch_edit_user_modified`。
                // 【为什么读冻结的 `edits` 而不是再锁一次 `document.edits`】本函数在
                // 调用方的短事务里工作，`edits` 就是那一刻冻结的快照（各调用点都写作
                // `document.edits.lock_or_recover().clone()`）。回头去锁活表既与本函数
                // 其余输入不是同一代数据，也会在"调用方仍持有那把锁"的写法下**自锁**：
                // `Mutex` 不可重入，整条渲染路径就此停死。
                let edited = edits
                    .authority
                    .root(root)
                    .is_some_and(|authority| !authority.is_empty())
                    || params.pitch_edit_user_modified;
                let baseline_usable = baseline.get(root).is_some_and(|entry| entry.is_ready())
                    || params.pitch_orig.iter().any(|value| *value > 0.0)
                    || params.pitch_edit.iter().any(|value| *value > 0.0);
                if edited && !baseline_usable {
                    return Err(format!(
                        "{}; pitch analysis pending for root {root}",
                        crate::editor::errors::CONFLICT_PREFIX
                    ));
                }
            }
        }
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
            || !resolved.authority.is_pristine()
            || stretch
            || owned_fade
            || reversed_clip
            || !resolved.groups.is_empty()
            || facts_pending;
        let clip_parameters = document.compose_clip_params_locked(&timeline, &resolved)?;
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
                window: None,
                gain_scope: hifishifter_kernel::gain::GainScope::HostOwned,
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
