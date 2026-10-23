//! 捕获通路：渲染输入、准备作业、输出仲裁与读数。
use super::*;
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
    pub(crate) fn capture_render_input(
        &self,
    ) -> Result<(u64, u64, bool, super::input::RenderInput), String> {
        if !self.capture.has_material() {
            return Err("no Capture material".into());
        }
        let revision = self.capture.revision();
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
        };
        Ok((revision, edit, with_curves, input))
    }

    pub(crate) fn capture_prepare_job(
        &self,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), String> {
        if !self.capture.has_material() {
            return Ok(());
        }
        // 【为什么这里也要补一次】命令那条路（`request_capture_reload`）只在用户动作时
        // 发生；从组件 state 恢复 take 的会话不走那里。这一份是兜底，且幂等。
        self.sync_raw_tape();
        let (revision, edit, with_curves, input) = self.capture_render_input()?;
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
        let mut snapshots = input.render(cancel)?;
        super::snapshot::compact_prepared(&mut snapshots)?;
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
                .publish(snapshot)
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
    pub(crate) fn capture_output_with(&self, evidence: Option<BlockEvidence>) -> CaptureOutput {
        // 【为什么 ARA 实例完全不走这里】它的素材由 ARA 逐个 region 授权，"区间外静音"
        // 是那条通路的既有语义（宿主自己的原声已被 ARA 取代）。这一条必须排在武装之前：
        // 形态一旦是 ARA，任何 Capture 结论都不适用。
        //
        // 【已知边界】`bound_once` 为真而文档已关的实例（宿主绑定过 ARA、文档后来又关掉）
        // 在这里仍然判成"不介入"。那种实例在界面上本来就是终态（`editor_session` 会以
        // DOCUMENT_CLOSED 收场，整页显示"会话不可用"），所以它不是"有界面却不出声"的
        // 现场；要收掉这条边界得先给"文档是否还活着"一个**无锁**读数（音频线程不能取
        // `document` 那把锁），那是独立一期的事。
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return CaptureOutput::Untouched;
        }
        /*
         * 【为什么"武装"不再直接等于"录制中"】武装之后、宿主开始播放之前是**待命**：
         * 那段时间里没有任何素材在产生，所以"用旧内容盖住新演奏"这个风险根本不存在
         * （那正是录制中要直通的理由）。而此前 `is_armed()` 直接返回 `Recording` ⇒
         * 待命期输出直通宿主原声，用户画完参数线听不到任何变化 —— 表现就是
         * **"参数线不生效"**，而界面只说"等待播放"，两件事对不上。
         *
         * 一旦宿主开始播放就立刻回到 `Recording`：捕获读的是输入总线，而输出可能写进
         * 同一块缓冲（in-place 宿主），直通才不会让捕获录到我们自己的输出。
         * `capture_observe` 在 `capture_stage` 之前更新 `host_playing`，所以起播的
         * 第一块就已经是 `Recording`，捕获读到的是宿主原声。
         */
        // 【为什么把块流证据也算进"录制中"】`kPlaying` 可能在起播第一块才滞后置位；
        // 只按 `capture_started || host_playing` 判断时，那一块会被当成"待命"，
        // 输出上一版 `Processed` —— 旧内容盖在新演奏上。位置流在推进时按录制中直通，
        // 最多在几何变化的瞬间多直通一块；重复块随即回到 `Processed`，待命语义不变。
        // 没有本块证据的调用（旧测试/旧入口）退回实例级实时流读数；生产路径
        // 永远由 `capture_observe` 提供证据，因此不会读到另一条流的结论。
        let recording = if evidence.is_some() {
            self.capture.is_recording_with_evidence(evidence)
        } else {
            self.capture
                .is_recording_with_flow(self.capture_transport_running())
        };
        if recording {
            return CaptureOutput::Recording;
        }
        if !self.capture.has_material() {
            // 【为什么没有素材就一定透传】"不要覆盖一个正常渲染的 ARA 实例的输出"
            // 这条契约由上面的 `is_bound()` 守卫负责：有活的文档时这里根本走不到。
            // 此前这里还额外要求"形态已落定为 Capture"，于是窗口没开过、宽限期未过的
            // 会话（`Undetermined`）在**任何素材之前**都是静音 —— 用户把插件插到一条
            // 有声音的轨道上，得到的是"整条轨哑了"。
            // 插件的诚实默认是**透明**：直到它真的录到了东西，才对声音负责。
            return CaptureOutput::PassThrough;
        }
        let candidate = if self.processed_ready() {
            CaptureOutput::Processed
        } else if self.raw_tape_ready() {
            // 【为什么在这里置位】有素材、但处理结果不是当前代次 ⇒ 该有一次合成在跑。
            // 音频线程只记一个原子；调度仍由会话 actor 完成（见 `render_wanted`）。
            self.render_wanted.store(true, Ordering::Release);
            CaptureOutput::Raw
        } else {
            // 素材存在但两份都还没发布（刚录完、后台还没跑完）：透传宿主原声，但**如实**
            // 报成"等待合成"，而不是与"没有素材"共用同一个名字。
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
        let locked = self.rung_locked.load(Ordering::Acquire);
        if locked != 0 && self.rung_pass.load(Ordering::Acquire) == pass {
            // 本 pass 已经选过：沿用，不重新仲裁。
            return CaptureOutput::from_code(locked).unwrap_or(candidate);
        }
        self.rung_locked.store(candidate.code(), Ordering::Release);
        self.rung_pass.store(pass, Ordering::Release);
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
    pub(super) fn record_click_sample(
        &self,
        stream: ProcessStream,
        kind: u8,
        seam: crate::render::click::SeamCode,
        first_of_stream: bool,
        output: &crate::audio_abi::AudioBusBuffers,
        frames: usize,
    ) {
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
        let Some(line) = ring.last_context_line() else {
            return;
        };
        let mut last = self.last_ctx_diag.lock_or_recover();
        if last.as_deref() != Some(line.as_str()) {
            *last = Some(line.clone());
            log::warn!("{line}");
        }
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
        let index = self.snapshot_index_for_rate(self.setup_sample_rate());
        let geometry = &self.streams[index.min(2)];
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
            index.min(2),
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
