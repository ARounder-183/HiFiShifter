//! 捕获通路：渲染输入、准备作业、输出仲裁与读数。
use super::*;

/// `[capture] ctx` 在形状不变时补一行的间隔。
///
/// 【为什么要有它】这一行的去重键不含位置（位置一直在走 ⇒ 按整行去重等于每块一行，
/// 现场日志里它占 10162/26617 行）。但位置是用户真正想看的那一栏，所以形状不变时
/// 也要定期补一行 —— 10 秒足够让人看见"宿主把位置钉在 7.375 s"，又不至于刷屏。
const CONTEXT_SUMMARY_MS: std::time::Duration = std::time::Duration::from_secs(10);

impl ExtensionOwner {
    /// Capture 形态的渲染准备（不经过 ARA 文档）。
    ///
    /// 【为什么必须有一份独立的实现】`prepare_job` 的第一件事就是取 ARA 文档，
    /// 取不到直接返回错误。Capture 实例**永远**没有文档，所以那条路对它等于
    /// "永远不准备 ⇒ 永远静音"。
    ///
    /// 【为什么复用 `RenderInput` 而不是另写一条】Capture 的投影产出的正是
    /// `(TimelineState, Vec<AraPlaybackRegion>, sources)` —— 与 ARA 路径同一组类型
    /// （见 `capture::CaptureStore::render_input`）。于是校验、渲染、压缩、双率发布
    /// 全部是同一份代码，Capture 不需要第二条渲染通路。
    /// Capture 形态的渲染输入，以及它的版本 `(take, edit)`。
    ///
    /// 【为什么只有 owner 够得着两边】几何与 PCM 在捕获存储上，用户曲线在编辑器会话上。
    /// **播放与导出共用这一份** —— 于是"导出的音频与听到的不一样"这一类漂移不可能发生。
    ///
    /// 【为什么优先用会话的那份时间线】它带着用户的参数曲线，而曲线就是 Capture 形态下
    /// "处理"的全部内容。用捕获存储那份（只有几何）会让"改了音高却不重算"变成一个静默的
    /// 失效：界面变了，声音还是旧的。
    /// 返回 `(素材代次, 编辑代次, 这份时间线带不带用户曲线, 渲染输入)`。
    ///
    /// 【为什么"带不带曲线"要作为一项返回】会话还没加载时拿到的是**不含用户曲线**的
    /// 几何，而那时 `edit` 恒为 0 —— 若调用方把它当成"已准备"，会话加载之后版本不变、
    /// 这次渲染被跳过，用户画完线永远听不到变化（见 `capture_prepare_job` 的版本闸）。
    /// `require_curves` 决定"会话正在加载"时要不要**放弃这一趟**：
    ///
    ///   * `true`（离线导出 / bounce）：放弃。那是**一次性**的机会，发布一份没有用户
    ///     曲线的结果等于把"编辑没生效"烘焙进产物，用户永远发现不了。
    ///   * `false`（实时播放）：照常发布。播放有无数次机会，而"什么都不发布"意味着
    ///     **永远直通**（用户报的"关着窗口就没有处理结果"）。版本号里的
    ///     `with_curves` 位保证会话一加载就会重算一次。
    pub(crate) fn capture_render_input(
        &self,
        require_curves: bool,
    ) -> Result<Option<(u64, u64, bool, super::input::RenderInput)>, String> {
        if !self.capture.has_material() {
            return Err("no Capture material".into());
        }
        let revision = self.capture.revision();
        if require_curves && self.capture_session_pending() {
            return Ok(None);
        }
        let session = self
            .capture_session
            .get()
            .and_then(|session| session.as_ref().ok())
            .filter(|session| session.is_loaded())
            .cloned();
        let (mut timeline, edit, with_curves) = match &session {
            // 【为什么用"发布代次"而不是"编辑代次"】渲染读的就是这份被发布的时间线，
            // 而 `generation` 只由 `mark_dirty` 推进 —— 有命令改了曲线却不标脏
            // （`convert_mix_param`），只认 `generation` 会让那种改动不推进版本 ⇒
            // 渲染被跳过 ⇒ 曲线永远不生效。
            Some(session) => (
                session.render_timeline()?,
                session.timeline_revision(),
                true,
            ),
            // 会话还没加载（窗口还没开、或前端还没问过时间线）：退回捕获存储的几何。
            // 那时也没有用户在编辑，两者等价 —— 但**必须把这个事实带进版本号**
            // （见 `capture_prepare_job` 的 `with_curves` 位）。
            None => (self.capture.session_timeline().0, 0, false),
        };
        if require_curves && !with_curves {
            log::debug!(
                "[capture] offline render requested curves but no editor session is loaded; deferring"
            );
            return Ok(None);
        }
        // 【内容身份跟着 PCM 走】Capture 的分析缓存键优先读 content_id；这里把
        // 内容哈希也写进 clip 的内容指纹，保证“同 take 号、不同内容”必定 miss，
        // “同内容、不同 take”必定 hit。ARA 的指纹由 refresh_source_fingerprints 负责。
        let fingerprints = self.capture.content_fingerprints();
        for clip in &mut timeline.clips {
            if let Some(content_id) = clip.content_id.as_deref() {
                if let Some(stamp) = fingerprints.get(content_id) {
                    clip.source_file_fingerprint = Some(*stamp);
                }
            }
        }
        let trimmed = crate::capture::timeline::clamp_timeline_to_project(&mut timeline);
        if trimmed > 0 {
            log::warn!(
                "[capture] removed pre-roll from {trimmed} clip(s) before rendering; the material domain starts at project time 0"
            );
        }
        let input = super::input::RenderInput {
            regions: self.capture.regions_for_timeline(&timeline),
            sources: self.capture.material_sources(),
            timeline: Some(timeline),
            project_sample_rate: Some(self.setup_sample_rate()).filter(|rate| *rate > 0.0),
            // 【为什么参数曲线这里为空】Capture 的曲线就在时间线里（它**就是**参数权威），
            // 于是 `mix_interleaved` 走"直接按时间线合成"那一支。ARA 需要图谱投影是因为
            // 它的时间线每次都被宿主重建、clip 身份由 region key 派生 —— Capture 没有
            // 那个问题（见 `render/input.rs::mix_interleaved` 的第一支）。
            clip_parameters: std::collections::BTreeMap::new(),
            window: None,
        };
        Ok(Some((revision, edit, with_curves, input)))
    }

    /// 会话**存在但还没加载完**（窗口刚打开、actor 正在读盘/建模型）。
    ///
    /// 【为什么它不是"没有会话"】"没有会话"= 用户从没开过窗口 ⇒ 确实没有曲线可读，
    /// 照常渲染是对的。"有会话但没加载" = 用户**有**曲线，只是还读不出来 ⇒ 这一趟
    /// 渲染必须**放弃**：发布一份没有曲线的结果会让宿主把它烘焙进产物（离线只有
    /// 一次机会），而下一拍就能拿到带曲线的版本。
    ///
    /// 此前这两种状态被合成一个"没有会话"分支（`with_curves = false`），于是
    /// "窗口开过、actor 正在加载"时渲染会发布一份**没有用户编辑**的产物 —— 用户
    /// 听到的正是"我改的东西没生效"。
    pub(crate) fn capture_session_pending(&self) -> bool {
        match self.capture_session.get() {
            Some(Ok(session)) => !session.is_loaded(),
            // 会话创建失败：那是终态，不是"还在加载"。让它走"没有会话"那一支，
            // 免得离线渲染永远等一个不会到来的加载。
            Some(Err(_)) | None => false,
        }
    }

    pub(crate) fn capture_prepare_job(
        &self,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), String> {
        // 【离线块之后的实时准备也必须带曲线】宿主可能用 setup mode=0 进入离线
        // 预览（VEGAS 的“应用非实时事件 FX”就是一例）。只看 setup mode 会漏掉；
        // `offline_pass_active` 由第一个 mode=2 块置位，随后 actor 的普通准备
        // 就必须走 require_curves 版本，不能再发布“只有几何”的快照。
        let require_curves = self.offline_pass_active() || self.setup_process_mode() == 2;
        if require_curves {
            if let Some(session) = self.capture_session_if_initialized() {
                session.request_reload();
            }
        }
        self.capture_prepare_job_inner(cancel, require_curves)
    }

    /// 离线准备：**要求**这一份结果带着用户曲线（见 `capture_render_input`）。
    pub(crate) fn capture_prepare_job_offline(
        &self,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), String> {
        self.capture_prepare_job_inner(cancel, true)
    }

    /// 离线渲染是否已经拥有“带当前编辑代次”的完整快照。
    ///
    /// 【为什么不能只看 capture_session_pending】它只回答“会话在不在加载”。
    /// 版本闸还必须确认发布的快照属于当前的 (take, edit, with_curves) 组合；
    /// 否则参数线刚改完就点“确定”时，会把上一版曲线烘焙进产物。
    pub(crate) fn capture_offline_ready(&self) -> Result<bool, String> {
        if !self.capture.has_material() {
            return Ok(true);
        }
        let Some((revision, edit, with_curves, _)) = self.capture_render_input(true)? else {
            return Ok(false);
        };
        if !with_curves {
            return Ok(false);
        }
        let version = revision ^ (edit << 32);
        Ok(self.capture_prepared.load(Ordering::Acquire) == version
            && self.snapshots.iter().all(|snapshot| snapshot.is_ready()))
    }

    fn capture_prepare_job_inner(
        &self,
        cancel: Arc<std::sync::atomic::AtomicBool>,
        require_curves: bool,
    ) -> Result<(), String> {
        if !self.capture.has_material() {
            return Ok(());
        }
        // 【为什么这里也要补一次】命令那条路（`request_capture_reload`）只在用户动作时
        // 发生；从组件 state 恢复 take 的会话不走那里。这一份是兜底，且幂等。
        self.sync_raw_tape();
        let Some((revision, edit, with_curves, input)) =
            self.capture_render_input(require_curves)?
        else {
            // 【会话正在加载：这一趟不发布】见 `capture_session_pending`。离线渲染只有
            // 一次机会，发布一份没有用户曲线的结果等于把"编辑没生效"烘焙进产物。
            // 返回 Ok 让调用方继续等（见 `prepare_offline_until` 的重试循环）。
            log::debug!("[capture] the editor session is still loading; the snapshot is deferred");
            return Ok(());
        };
        // 【T-S1-6】发布前把"素材里有多少个洞"变成可读的数（去重打印）。
        self.note_tape_coverage(&input);
        // 【为什么要版本闸】它被 UI 定时器与命令两条路径驱动；没有这道闸，一次 8 秒的
        // 合成会被排队几百次。版本里必须**带上编辑代次**：只认 take 会让参数改动
        // 永远不重算。
        //
        // 【为什么还要带上"这份时间线有没有曲线"】会话没加载时拿到的是**不含用户曲线**
        // 的几何（`edit` 恒为 0）。若把它算作"已准备"，会话加载之后版本不变 ⇒ 这次
        // 渲染被跳过 ⇒ 用户画完线永远听不到变化。把这个事实编进版本号，会话一加载
        // 版本就变，于是恰好补一次渲染，而没加载时也不会反复重算。
        let version = revision ^ (edit << 32) ^ if with_curves { 0 } else { 1 << 63 };
        if self.capture_prepared.load(Ordering::Acquire) == version
            && self.snapshots.iter().all(|snapshot| snapshot.is_ready())
        {
            return Ok(());
        }
        if input
            .timeline
            .as_ref()
            .is_none_or(|timeline| timeline.clips.is_empty())
        {
            return Ok(());
        }
        for publisher in &self.snapshots {
            publisher.collect_retired();
        }
        // 【为什么要量这一步】"起播 0.4–0.7 s 一次咔"的两个候选成因（起播窗口 vs
        // 处理结果晚到）在这一个数上就能分开：若 `render_ms` 与用户听到的时间同量级，
        // 那一段就是"素材在、结果还没算出来"；若它远大于 0.7 s，那 0.4 s 只能来自
        // 起播窗口。没有这个数时，两者在日志里完全同形。
        let render_started = std::time::Instant::now();
        let snapshots = input.render(cancel)?;
        let render_ms = render_started.elapsed().as_millis() as u64;
        if self.is_closed() {
            return Ok(());
        }
        if !self
            .snapshots
            .iter()
            .zip(&snapshots)
            .all(|(publisher, snapshot)| publisher.has_capacity(snapshot))
        {
            return Err("retired snapshot budget exhausted".into());
        }
        for (publisher, snapshot) in self.snapshots.iter().zip(snapshots) {
            publisher
                .publish_shared(snapshot)
                .map_err(|e| format!("snapshot publish failed: {e:?}"))?;
        }
        self.capture_prepared.store(version, Ordering::Release);
        // 处理结果与当前 take 的对应关系：音频线程据此在"处理结果"与"原始磁带"之间选
        // （见 `processed_ready`）。换 take 之后这里才更新，所以中间那段空窗由原始磁带兜底。
        self.capture_ready_revision
            .store(revision, Ordering::Release);
        log::warn!("[capture] prepared material revision {revision} edit {edit} in {render_ms} ms");
        Ok(())
    }

    /// 把这一份渲染输入里"我们没录到"的帧数记成读数（诊断线程读）。
    ///
    /// 【为什么不打印逐源的明细】一条 take 通常只有一份源；有洞时真正想知道的是
    /// "有几个洞、一共多少帧"。源数为 0（全是 ARA 通路那种"整份都算"）时这一行
    /// 不会出现 —— 那正是"没有洞可谈"的正确表达。
    pub(super) fn note_tape_coverage(&self, input: &super::input::RenderInput) {
        let mut holes = 0_u64;
        let mut hole_frames = 0_u64;
        let mut covered_sources = 0_u64;
        for pcm in input.sources.values() {
            if pcm.covered.is_empty() {
                continue;
            }
            covered_sources += 1;
            let total = pcm.planes.first().map_or(0, Vec::len) as i64;
            let covered: i64 = pcm
                .covered
                .iter()
                .map(|range| (range.end - range.start) as i64)
                .sum();
            let gap = total.saturating_sub(covered);
            if gap <= 0 {
                continue;
            }
            hole_frames = hole_frames.saturating_add(gap as u64);
            // 首/尾之外的每一段都算一个洞；段与段之间的空档也算一个。
            holes = holes.saturating_add((pcm.covered.len() as u64).saturating_sub(1));
            if pcm.covered.first().is_some_and(|range| range.start > 0) {
                holes = holes.saturating_add(1);
            }
            if pcm
                .covered
                .last()
                .is_some_and(|range| (range.end as i64) < total)
            {
                holes = holes.saturating_add(1);
            }
        }
        self.tape_holes.store(holes, Ordering::Relaxed);
        self.tape_hole_frames.store(hole_frames, Ordering::Relaxed);
        if covered_sources == 0 {
            return;
        }
        let line = format!(
            "[capture] tape coverage: sources={covered_sources} holes={holes} \
             hole_frames={hole_frames}"
        );
        let mut last = self.last_tape_diag.lock_or_recover();
        if last.as_deref() != Some(line.as_str()) {
            *last = Some(line.clone());
            log::info!("{line}");
        }
    }

    /// 本块在 Capture 通路上该怎么输出。**无锁、无分配**，可从音频线程调用。
    ///
    /// 【为什么要有这个枚举】"没素材时会听到什么"此前要读者把 `capture_stage` 的早退
    /// 与 `vst3` 的后续分支拼起来才能回答。收敛成一个值之后，每个分支都能被单独断言，
    /// 而"静音"只剩下"确实没有素材"这一个含义。
    pub(crate) fn capture_output(&self) -> CaptureOutput {
        self.capture_output_with(None)
    }

    /// 与 [`Self::capture_output`] 相同，但使用**本块**的流/播放证据。
    ///
    /// # 四层仲裁（问题 3 的修复点）
    ///
    /// 旧实现先问“是不是在录”，而那个判据在 Resting 上早退；随后只要有素材就读
    /// **上一张 take** 的 processed 快照。于是“停止 -> 再播”的前 250 ms 永远在放
    /// 上一轮缓存。现在判据是**捕获世代**：
    ///
    /// 1. 本世代、本块有权推进（transport_active / start_window / Start|Rolling）
    ///    ⇒ 一定是这一遍的输入，输出 `Recording`；
    /// 2. 本世代、已停播（Paused）⇒ 这一遍已经录到东西才允许回放它，否则 `Recording`；
    /// 3. 旧世代的迟到块 ⇒ 不能替当前世代作决定；有权推进时仍按 `Recording`；
    /// 4. 没有本块证据 ⇒ 只有“待命”（本世代还没写下任何样本）才允许试听上一张 take。
    pub(crate) fn capture_output_with(&self, evidence: Option<BlockEvidence>) -> CaptureOutput {
        // 【为什么 ARA 实例完全不走这里】它的素材由 ARA 逐个 region 授权，"区间外静音"
        // 是那条通路的既有语义（宿主自己的原声已被 ARA 取代）。这一条必须排在武装之前：
        // 形态一旦是 ARA，任何 Capture 结论都不适用。
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return CaptureOutput::Untouched;
        }
        if !self.capture.is_armed() {
            // 未武装：保留既有“有素材就播结果”的契约。
            if !self.capture.has_material() {
                return CaptureOutput::PassThrough;
            }
            return self.capture_candidate_output(evidence);
        }

        let epoch = self.capture.capture_epoch();
        // 没有本块证据的旧入口（capture_output / 诊断）：保留“位置流在推进
        // ⇒ 正在录”的既有契约。生产路径永远带证据。
        if evidence.is_none() && self.capture_transport_running() {
            return CaptureOutput::Recording;
        }
        if let Some(evidence) = evidence {
            if evidence.capture_epoch == epoch {
                if evidence.processing != 0
                    || evidence.transport_active
                    || evidence.start_window
                    || matches!(
                        evidence.phase,
                        TransportPhase::Start | TransportPhase::Rolling
                    )
                {
                    return CaptureOutput::Recording;
                }
                if evidence.transport_paused {
                    // 【为什么暂停时仍监听输入】当前这一遍的素材还在 Capture 里、
                    // 尚未 promote，没有任何已发布的快照能代表它；此时读 processed
                    // 只会读到上一张 take。监听输入是唯一诚实的输出。
                    if self.capture.capture_epoch_has_samples()
                        && self.capture.current_material_epoch() == epoch
                    {
                        return self.capture_candidate_output(Some(evidence));
                    }
                    return CaptureOutput::Recording;
                }
                // Resting 且没有 transport 证据：这一块不是“这一遍在播”，
                // 但也不能拿旧 take 顶替 —— 继续监听输入。
                if self.capture.capture_epoch_has_samples() {
                    return CaptureOutput::Recording;
                }
            } else {
                // 旧世代的迟到块（prefetch / 补写）：不能替当前世代作决定。
                return if evidence.transport_active {
                    CaptureOutput::Recording
                } else {
                    CaptureOutput::PassThrough
                };
            }
        }

        // 待命：武装之后、本世代第一块有权块之前。此时没有任何新素材在产生，
        // 用户仍应听到上一张 take 的处理结果（既有契约）。
        if !self.capture.capture_epoch_has_samples() && self.capture.has_material() {
            return self.capture_candidate_output(evidence);
        }
        CaptureOutput::Recording
    }

    /// 当前可播的素材档位（Processed / Raw / PendingSynthesis）。
    fn capture_candidate_output(&self, evidence: Option<BlockEvidence>) -> CaptureOutput {
        let candidate = if self.processed_ready() {
            CaptureOutput::Processed
        } else if self.raw_tape_ready() {
            self.render_wanted.store(true, Ordering::Release);
            CaptureOutput::Raw
        } else {
            self.render_wanted.store(true, Ordering::Release);
            CaptureOutput::PendingSynthesis
        };
        self.latch_rung(candidate, evidence)
    }

    /// 撤掉本 pass 锁存的内容源（换 take / 清空之后）。
    ///
    /// 【为什么必须在换 take 时清】锁存的意义是"这一遍只播一个源"，而"这一遍"
    /// 的前提是内容没变。换了 take 之后，上一遍锁存的 `Raw`/`Processed` 指向的是
    /// **另一份素材** —— 继续沿用等于用旧录音盖住新演奏。
    pub(crate) fn clear_rung_latch(&self) {
        self.rung_locked.store(0, Ordering::Release);
        self.rung_pass.store(0, Ordering::Release);
        self.rung_capture_epoch.store(0, Ordering::Release);
        self.rung_take_epoch.store(0, Ordering::Release);
    }

    /// 把本块的候选档位**锁存到本 pass**（I-2：一次播放只有一个内容源）。
    ///
    /// # 为什么必须锁存
    ///
    /// 旧实现每块都重新仲裁，于是 `Raw → Processed` 可以发生在**音乐中途** ——
    /// 而那恰好出现在"刚录完、后台合成刚好跑完"的那一刻，也就是用户起播后的
    /// 0.4–0.7 s。一次换源就是一次咔，而它不是用户做错了什么。
    ///
    /// 锁存之后，一次 pass 只有一个内容源；换源只可能发生在 pass 边界，而那时
    /// 本来就要重新开始（第一块走 `Seam::FirstBlock` / `Seam::FadeIn`）。
    ///
    /// # 代价与它为什么值得
    ///
    /// 若某一遍恰好在"处理结果还没算出来"时起播，那一遍会全程播 `Raw`（原始捕获，
    /// **仍然是这一份 take 的内容**，只是没处理）。用户听到的是"这一遍没生效"，
    /// 而不是"听到一半突然换了"。前者可解释、可重试；后者不可解释。
    /// 真正的解法是让处理结果在播放头之前就绪（见方案 §4.2.8 的水位渲染），
    /// 那一步落地之后这个代价会消失。
    ///
    /// # 哪些档位不锁存
    ///
    /// * `Untouched`：绑定的 ARA 实例，根本不走这条通路（上面已早退）。
    /// * `Recording`：录制中必须直通输入，而且它本身意味着一次新的传输 ——
    ///   顺便把锁存清掉，让录完之后的那一遍重新选。
    pub(super) fn latch_rung(
        &self,
        candidate: CaptureOutput,
        evidence: Option<BlockEvidence>,
    ) -> CaptureOutput {
        // 没有本块证据（离线 / 预取 / 旧测试入口）时不锁存：它们的输出稍后才被播放，
        // "此刻的传输状态"对它们不适用。
        let Some(evidence) = evidence else {
            return candidate;
        };
        if evidence.stream != ProcessStream::Realtime {
            return candidate;
        }
        if candidate == CaptureOutput::Recording {
            self.rung_locked.store(0, Ordering::Release);
            return candidate;
        }
        let pass = self.streams[ProcessStream::Realtime.index()]
            .pass
            .load(Ordering::Acquire);
        let capture_epoch = self.capture.capture_epoch();
        let take_epoch = self.capture.current_material_epoch();
        let locked = self.rung_locked.load(Ordering::Acquire);
        if locked != 0
            && self.rung_pass.load(Ordering::Acquire) == pass
            && self.rung_capture_epoch.load(Ordering::Acquire) == capture_epoch
            && self.rung_take_epoch.load(Ordering::Acquire) == take_epoch
        {
            // 本 pass / 本捕获世代已经选过：沿用，不重新仲裁。
            return CaptureOutput::from_code(locked).unwrap_or(candidate);
        }
        self.rung_locked.store(candidate.code(), Ordering::Release);
        self.rung_pass.store(pass, Ordering::Release);
        self.rung_capture_epoch
            .store(capture_epoch, Ordering::Release);
        self.rung_take_epoch.store(take_epoch, Ordering::Release);
        candidate
    }

    /// 输出源 token：种类 + 快照代次。任一项变化都意味着"用户听到的信号换了源"。
    pub(super) fn output_token_for(&self, output: CaptureOutput) -> u64 {
        let kind = u64::from(output.code());
        let revision = match output {
            // 处理结果换版（曲线编辑 / 换 take）也要去咔。
            //
            // 【为什么 `Untouched` 也要编进代次】bound 实例的输出仲裁恒为 `Untouched`
            // （它走既有快照通路），而 `capture_stage` 存下的 token 只含"种类"。
            // 于是快照第一次发布（此前逐样本写零）或换版时 token 不变 ⇒
            // `apply_output_declick` 认为"没有切换"⇒ 用户听到一次从零到有声的硬切。
            // 把两份发布器的代次编进来，ARA 的接缝也能拿到同样的 1.3 ms 淡化。
            //
            // 【为什么两份都编】真正被读的是 `snapshots[snapshot_index_for_rate(...)]`
            // （48 kHz 及倍率工程读索引 1）。只编索引 0 会让"实际读的是另一份"的宿主
            // 在换版时拿不到去咔。
            // 【为什么用 `content` 而不是 `generation`】token 的用途只有一个：
            // 让 `apply_output_declick` 判断"这一块与上一块是不是同一份输出"。
            // `generation` 回答的是"发布过几次"，于是同一份素材被重新合成一次
            // （曲线没变、只是版本号动了）就会让 token 变、装填一次 64 帧接缝 ——
            // 实测每 0.68 s 一次，正是用户听到的那个周期性"咔"。
            //
            // 【为什么仍然保留 `kind`】种类（Raw / Processed / Passthrough）回答的是
            // "这一侧是什么内容"，与内容指纹正交：两者都必须参与，缺一个就会把
            // 一次真实换源或一次重新合成判反。
            CaptureOutput::Processed | CaptureOutput::Untouched => {
                self.snapshots[0].content() ^ self.snapshots[1].content().rotate_left(32)
            }
            CaptureOutput::Raw => self.raw_tape.content(),
            _ => 0,
        };
        kind | (revision << 8)
    }

    /// 【B0】记下本块分层拷贝的覆盖读数（音频线程）。
    pub(crate) fn note_layered_probe(
        &self,
        stream: ProcessStream,
        probe: crate::render::snapshot::LayeredProbe,
    ) {
        let Some(cell) = self.layered_probe.get(stream.index()) else {
            return;
        };
        cell.uncovered
            .store(probe.uncovered as u32, Ordering::Relaxed);
        cell.input_peak.store(
            probe.input_peak.unwrap_or(f32::NAN).to_bits(),
            Ordering::Relaxed,
        );
        cell.edge.store(probe.edge, Ordering::Relaxed);
        cell.valid.store(true, Ordering::Release);
    }

    /// 【B0】取走本块的覆盖读数（音频线程；取走即失效，避免下一块读到陈旧值）。
    pub(super) fn take_layered_probe(&self, stream: ProcessStream) -> Option<(u32, f32, bool)> {
        let cell = self.layered_probe.get(stream.index())?;
        if !cell.valid.swap(false, Ordering::AcqRel) {
            return None;
        }
        Some((
            cell.uncovered.load(Ordering::Relaxed),
            f32::from_bits(cell.input_peak.load(Ordering::Relaxed)),
            cell.edge.load(Ordering::Relaxed),
        ))
    }

    /// 【B0】量一块输出的形状并记进账本。**音频线程**：只读缓冲、只写原子量。
    ///
    /// 【为什么量 `max_step` 而不是别的】"咔"在波形上就是相邻样本的一次大跳变。
    /// 峰值、RMS、过零率都看不见它 —— 一个 20 ms 的音乐片段可以峰值不变而中间
    /// 跳一下。相邻差分是唯一直接对应听感的读数。
    /// 逐块"咔"账本是否开启（`HIFISHIFTER_CLICK_AUDIT=1`，默认关闭）。
    ///
    /// 【为什么必须可关】`record_click_sample` 每块都要把整个输出缓冲扫一遍
    /// （两声道 × 块长），并且每次都在写 `[click]` 行。现场日志里 `[click]` 占了
    /// 9801/26617 行 —— 一个诊断工具的成本高到会改变被观测的行为，它就不再是诊断。
    /// 与 `HIFISHIFTER_BUFFER_AUDIT` 同一套开关纪律：读一次就缓存，默认关闭。
    /// 预热音频线程会读取的一次性开关（组件创建时调用，避免首块分配）。
    pub(crate) fn warm_audio_caches(&self) {
        let _ = Self::click_audit_enabled();
    }

    fn click_audit_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("HIFISHIFTER_CLICK_AUDIT")
                .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        })
    }

    pub(super) fn record_click_sample(
        &self,
        stream: ProcessStream,
        kind: u8,
        seam: crate::render::click::SeamCode,
        first_of_stream: bool,
        output: &crate::audio_abi::AudioBusBuffers,
        frames: usize,
    ) {
        if !Self::click_audit_enabled() {
            return;
        }
        let channels = (output.num_channels as usize).min(2);
        let mut max_step = 0.0_f32;
        let mut peak = 0.0_f32;
        let mut first = 0.0_f32;
        let mut last = 0.0_f32;
        let mut energy = 0.0_f64;
        let mut energy_frames = 0_u32;
        for channel in 0..channels {
            // SAFETY: 调用方保证总线有 `channels` 个可读平面，且至少 `frames` 帧。
            let plane = unsafe { *output.channel_buffers.add(channel) };
            if plane.is_null() {
                continue;
            }
            let mut previous = 0.0_f32;
            for index in 0..frames {
                // SAFETY: `index < frames`。
                let value = unsafe { *plane.add(index) };
                if !value.is_finite() {
                    continue;
                }
                peak = peak.max(value.abs());
                max_step = max_step.max((value - previous).abs());
                energy += f64::from(value) * f64::from(value);
                energy_frames += 1;
                if channel == 0 {
                    if index == 0 {
                        first = value;
                    }
                    last = value;
                }
                previous = value;
            }
        }
        let (uncovered, input_peak, edge) =
            self.take_layered_probe(stream)
                .unwrap_or((0, f32::NAN, false));
        // 【覆盖边也是一种接缝】没有源切换、但块内有"宿主原声 ↔ 处理结果"的翻转点：
        // 那一下同样是用户听到的一次信号更换。把它标进账本，round 23 那种"假洞"
        // 就不必再靠用户复述才能被发现。
        let seam = if seam == crate::render::click::SeamCode::None && edge {
            crate::render::click::SeamCode::CoverageEdge
        } else {
            seam
        };
        let rms = if energy_frames == 0 {
            0.0
        } else {
            (energy / f64::from(energy_frames)).sqrt() as f32
        };
        self.click.record(crate::render::click::BlockSample {
            stream: stream.index() as u8,
            kind,
            seam,
            fade_len: self.declick[stream.index()]
                .fade_len
                .load(Ordering::Acquire),
            frames: frames as u32,
            first,
            last,
            max_step,
            peak,
            input_peak,
            uncovered,
            rms,
            edge,
            first_of_stream,
        });
    }

    /// 【B0】把尚未报告过的可疑块打印成 `[click]` 行。**非实时线程**（UI 定时器）。
    ///
    /// 【为什么由 UI 定时器驱动】它是唯一稳定存在、且不在音频线程上的周期点；
    /// 音频线程只负责记录。报告本身是"扫 256 个原子 + 格式化"，与一次界面刷新同量级。
    pub(crate) fn log_click_report(&self) {
        for line in self.click.drain_suspicious() {
            log::warn!("[click] {line}");
        }
    }

    /// 把**最近一块**的宿主上下文打成一行（非实时线程；由会话 actor 驱动）。
    ///
    /// 【为什么必须有这一行】排查"起播窗口/位置钉住"时，唯一的硬事实是**宿主到底
    /// 给了什么**：`state` 的哪一位、`projectTimeSamples` 动不动、
    /// `continousTimeSamples` 有没有值、`projectTimeMusic` 有没有值。此前这些字段
    /// 一个都没有被打出来，于是每一轮都只能靠"用户再听一遍"来猜是哪一种形状。
    ///
    /// 【为什么读环形记录而不是新开一组原子】环里已经逐块记下了全部字段（音频线程
    /// 只写原子、不分配），复用它就不必再维护第二份会漂移的读数。
    pub(crate) fn log_capture_context(&self) {
        let Some(ring) = self.block_trace.get() else {
            return;
        };
        let Some((line, key)) = ring.last_context() else {
            return;
        };
        // 【去重键不含位置】见 `BlockTraceRing::last_context`：按整行去重等于每块一行
        // （位置一直在走 ⇒ 整行一直不同）。位置本身由每 `CONTEXT_SUMMARY_MS` 一行的
        // 汇总负责，形状真的变了才值得单独一行。
        let now = std::time::Instant::now();
        let mut last = self.last_ctx_diag.lock_or_recover();
        let unchanged = last.as_ref().is_some_and(|(key_before, at)| {
            key_before == &key && now.duration_since(*at) < CONTEXT_SUMMARY_MS
        });
        if unchanged {
            return;
        }
        *last = Some((key, now));
        log::warn!("{line}");
    }

    /// 【B0】把 I-2 / I-3 的读数打成两行。**非实时线程**（UI 定时器）。
    ///
    /// 【为什么要这两行】本轮之前，排查"咔"只能靠"用户再听一遍"：插件里没有
    /// "这一块换了源"与"这一块位置回退了"的读数，于是同一个症状被反复归因到
    /// 不同机理。这两行把它变成可判定的数字：
    ///
    ///   * `[capture] pass` —— 一次播放里换了几次内容源（I-2，理想为 0）；
    ///
    ///   * `[capture] clock` —— 位置是否逐块续接（I-3，`rewinds` 理想为 0）。
    ///
    /// 【为什么去重】它们由 20 ms 的界面定时器驱动；不去重就是每秒 50 行。判据取
    /// "内容变没变"：所有计数都进了指纹，变了才打印。
    pub(crate) fn log_capture_report(&self) {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            // ARA 实例不走 Capture 仲裁：这两行的数字对它是无意义的零。
            return;
        }
        if !self.capture.has_material() {
            return;
        }
        let switches = self.rung_switches.load(Ordering::Acquire);
        let from = self.rung_last_from.load(Ordering::Acquire);
        let to = self.rung_last_to.load(Ordering::Acquire);
        let pass_switches = self.pass_switches.load(Ordering::Acquire);
        let output = self.capture_output();
        let prepared = self.capture_prepared.load(Ordering::Acquire);
        let ready = self.capture_ready_revision.load(Ordering::Acquire);
        let line = format!(
            "[capture] pass: take={} prepared={prepared} ready_revision={ready} \
             output={} switches={switches} pass_switches={pass_switches} last={}->{}",
            self.capture.revision(),
            output.as_str(),
            CaptureOutput::from_code(from as u8).map_or("none", |value| value.as_str()),
            CaptureOutput::from_code(to as u8).map_or("none", |value| value.as_str()),
        );
        {
            let mut last = self.last_pass_diag.lock_or_recover();
            if last.as_deref() != Some(line.as_str()) {
                *last = Some(line.clone());
                log::warn!("{line}");
            }
        }
        // 【为什么按流遍历而不是挑一个下标】旧实现用 `snapshot_index_for_rate(工程率)`
        // 去索引 `streams[]` —— 那是**快照发布器**的下标（0/1），不是进程流的下标
        // （0=realtime/1=prefetch/2=offline）。48 kHz 工程因此永远读到 prefetch，
        // 而实时流一个块都没被记过 ⇒ 这一行在现场日志里**一行都没打出来**。
        for stream in [
            ProcessStream::Realtime,
            ProcessStream::Prefetch,
            ProcessStream::Offline,
        ] {
            self.log_clock_line(stream);
        }
    }

    /// 一条进程流的位置续接读数（非实时线程）。
    fn log_clock_line(&self, stream: ProcessStream) {
        let geometry = &self.streams[stream.index()];
        let blocks = geometry.blocks.load(Ordering::Acquire);
        if blocks == 0 {
            return;
        }
        // 【这一行的判读方式】稳态下 `delta_min == delta_max == 块长` 且
        // `rewinds == 0`。任何别的形状都说明位置记账漏了一项 —— 而那就是咔。
        //
        // `counted=` 是"位置由计数器推进的块数"，`jumps=` 是"宿主报的位置与计数器
        // 不同"的次数。两者都大而 `rewinds` 为 0 是**健康**的：说明宿主确实钉住过
        // 位置，而每次它恢复报告时我们都跟上了。
        let line = format!(
            "[capture] clock: stream={} blocks={blocks} pass={} counted={} frames={} \
             jumps={} last_jump={} rewinds={} delta_min={} delta_max={} resyncs={} \
             rising={} continuous={} music={}",
            stream.as_str(),
            geometry.pass.load(Ordering::Acquire),
            geometry.counted_blocks.load(Ordering::Acquire),
            geometry.counted_frames.load(Ordering::Acquire),
            geometry.jumps.load(Ordering::Acquire),
            geometry.last_jump.load(Ordering::Acquire),
            geometry.clock_rewinds.load(Ordering::Acquire),
            geometry.clock_delta_min.load(Ordering::Acquire),
            geometry.clock_delta_max.load(Ordering::Acquire),
            geometry.position_resyncs.load(Ordering::Acquire),
            geometry.rising_edges.load(Ordering::Acquire),
            geometry.continuous_used.load(Ordering::Acquire),
            geometry.music_used.load(Ordering::Acquire),
        );
        let mut last = self.last_clock_diag.lock_or_recover();
        if last.as_deref() != Some(line.as_str()) {
            *last = Some(line.clone());
            log::warn!("{line}");
        }
    }
}
