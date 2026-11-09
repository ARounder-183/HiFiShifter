//! 播放记账：流位置、门控、块分割与输出放行，以及它自己的两个切片助手。
use super::host_clock::{ClockDomain, ClockInput};
use super::timebase::TimebaseKind;
use super::transport_session::{SessionInput, SessionPhase, TransportSession};
use super::*;
impl ExtensionOwner {
    /// 取走"该重新合成一次"的意图（非音频线程调用；见 [`Self::render_wanted`]）。
    pub(crate) fn take_render_wanted(&self) -> bool {
        self.render_wanted.swap(false, Ordering::AcqRel)
    }

    /// 一次性诊断：有素材，但处理结果不是当前代次（非音频线程调用）。
    ///
    /// 【为什么需要它】"合成从来没被调度过"此前在界面上与"没有素材"共用同一个
    /// `passthrough` 读数，用户和我们都看不出区别。这条日志把结论直接写下来：
    /// 素材代次是多少、处理结果代次停在哪。合成成功后它自然不再出现。
    pub(crate) fn report_pending_synthesis_once(&self) {
        if !self.capture.has_material() {
            return;
        }
        let revision = self.capture.revision();
        let ready = self.capture_ready_revision.load(Ordering::Acquire);
        if ready == revision {
            return;
        }
        // 按代次去重：同一份素材只报一次。
        if self
            .pending_reported_revision
            .swap(revision, Ordering::AcqRel)
            == revision
        {
            return;
        }
        log::warn!(
            "[capture] material revision {revision} has no processed result yet \
             (last published revision {ready}); the render scheduler has not caught up"
        );
    }

    /// 记录本块的几何，并回答"位置流是否在推进"（音频线程，无锁无分配）。
    ///
    /// 【判据：块首变化，或同头块变长】块长由宿主决定（SDK 只承诺
    /// `numSamples <= maxSamplesPerBlock`），起播第一个块可以很大，也可以与停播块
    /// 同头。因此：
    ///
    ///   * 块首移动 ⇒ 时间轴在走（seek / 循环回卷也算，代价是至多一块瞬态）；
    ///
    ///   * 块首没动但块变长 ⇒ 起播给了更大的块，也是在走；
    ///
    ///   * 块首没动且块不变长/变短 ⇒ 钉住/重复（或宿主在停播时缩小了块）。
    ///
    /// 【为什么不比较块尾/历史高水位】那会让停播/预渲染留下的旧位置把起播块
    /// 压住：播放头还没越过旧块尾时，每一块都被判成"没前进"（用户报障：开头约
    /// 0.7 秒静音/卡顿）。当前块与上一块的关系不会跨停播残留。
    ///
    /// 【首块】`flow == true`（本流第一次出现）。门控允许它出音一次；重复抑制由
    /// [`Self::rt_last_emitted`] 负责，不再靠“首块保守静音”。
    #[cfg(test)]
    pub(crate) fn note_block_position(&self, start: i64, frames: usize) -> bool {
        self.note_stream_position(ProcessStream::Realtime, start, frames)
            .0
    }

    /// 只更新**本流**的几何并回答 `(flow, had_previous)`。
    ///
    /// 【为什么必须带 stream】`kPrefetch` 与 `kRealtime` 的位置节奏不同；跨流比较
    /// 会让 prefetch 的移动给实时块的重复投递作“在前进”的伪证。
    #[cfg(test)]
    pub(crate) fn note_stream_position(
        &self,
        stream: ProcessStream,
        start: i64,
        frames: usize,
    ) -> (bool, bool) {
        let block = self.note_stream_geometry(
            stream,
            start,
            frames,
            48_000.0,
            StreamTime::default(),
            false,
            false,
            // 【这个入口没有 transport 状态】它只回答"这一块的几何形状变了没有"
            // （`(flow, had_previous)`），调用点全在测试里。没有 `ProcessContext`
            // 就没有 `kPlaying`，取"transport 在走"是与它原有契约一致的假设 ——
            // 生产路径走 `note_stream_block`，那里的 `playing` 来自真实上下文。
            true,
        );
        (block.phase != TransportPhase::Resting, block.had_previous)
    }

    /// 生产入口：从完整 `ProcessContext` 计算有效位置与相位。
    #[cfg(test)]
    pub(super) fn note_stream_block(
        &self,
        stream: ProcessStream,
        context: impl Into<crate::block_context::BlockContext>,
        frames: usize,
        playing_rising: bool,
        recording_rising: bool,
    ) -> StreamBlock {
        // 旧测试入口模拟的是"宿主在走"，没有播放位；生产路径走 `_at`，不受这里影响。
        // 给测试上下文补上播放位，保持原有 3 秒计数器语义。
        let mut forced: crate::block_context::BlockContext = context.into();
        forced.playing = true;
        self.note_stream_block_at(stream, forced, frames, playing_rising, recording_rising, None)
    }

    /// 与 note_stream_block 相同，但允许调用方给出**本地有效位置**（离线 cursor）。
    pub(super) fn note_stream_block_at(
        &self,
        stream: ProcessStream,
        context: impl Into<crate::block_context::BlockContext>,
        frames: usize,
        playing_rising: bool,
        recording_rising: bool,
        effective_position: Option<i64>,
    ) -> StreamBlock {
        let context = context.into();
        let rate = if context.sample_rate.is_finite() && context.sample_rate > 0.0 {
            context.sample_rate
        } else {
            self.setup_sample_rate()
        };
        self.note_stream_geometry(
            stream,
            effective_position.unwrap_or(context.project_time_samples),
            frames,
            rate,
            StreamTime::from_context(&context),
            playing_rising,
            recording_rising,
            // kPlaying 是 SDK 对"transport 是否在走"的权威回答，与位置、采样率、
            // 样本同属**同一个块**。计数器的推进权必须以它为准（见 may_advance）。
            context.playing,
        )
    }

    /// 处理 pass（离线 / 冻结预览）的流块：位置完全来自 pass 的本地 cursor。
    ///
    /// 【为什么不能走 HostClock】处理 pass 的“时间”不是宿主传输时间：mode=0 预览的
    /// projectTimeSamples 是固定的工程光标，mode=2 的更是渲染进度。让 HostClock
    /// 参与只会把 cursor 与宿主读数搅在一起。
    pub(super) fn note_processing_block(
        &self,
        stream: ProcessStream,
        context: impl Into<crate::block_context::BlockContext>,
        frames: usize,
        processing: ProcessingBlock,
    ) -> StreamBlock {
        let context = context.into();
        let offline = processing.kind == ProcessingKind::Offline;
        let timebase = if offline {
            TimebaseKind::OfflineLocal
        } else {
            TimebaseKind::PreviewLocal
        };
        // 【处理 pass 也必须推进本流的几何】旧实现直接构造 StreamBlock，不更新
        // StreamGeometry 的 pass / session_phase / last_position：读游标跳变判据仍拿
        // 实时流的 pass 比较，预览确认时的 raw→cursor 跳变只能靠 force_declick 兜，
        // 而 `latch_rung` 的 pass 不前进，预览 pass 切换时旧档位可能继续。
        let geometry = &self.streams[stream.index()];
        if processing.new_pass {
            geometry.pass.fetch_add(1, Ordering::AcqRel);
        }
        let end = processing.local_start.saturating_add(frames as i64);
        let frames32 = u32::try_from(frames).unwrap_or(u32::MAX);
        geometry.seen.store(true, Ordering::Release);
        geometry
            .last_start
            .store(processing.local_start, Ordering::Release);
        geometry.last_frames.store(frames32, Ordering::Release);
        geometry.last_end.store(end, Ordering::Release);
        geometry
            .last_position
            .store(processing.local_start, Ordering::Release);
        geometry
            .last_raw_start
            .store(context.project_time_samples, Ordering::Release);
        geometry
            .timebase_kind
            .store(timebase.code(), Ordering::Release);
        geometry.timebase_offset.store(0, Ordering::Release);
        geometry
            .session_epoch
            .store(processing.pass_id, Ordering::Release);
        geometry.session_phase.store(
            if offline {
                SessionPhase::OfflineRolling.code()
            } else {
                SessionPhase::Rolling.code()
            },
            Ordering::Release,
        );
        geometry.session_frames.store(
            geometry
                .session_frames
                .load(Ordering::Acquire)
                .saturating_add(frames as u64),
            Ordering::Release,
        );
        StreamBlock {
            position: processing.local_start,
            raw_start: context.project_time_samples,
            phase: if processing.new_pass {
                TransportPhase::Start
            } else {
                TransportPhase::Rolling
            },
            had_previous: !processing.new_pass,
            cycle_wrap: false,
            new_pass: processing.new_pass,
            activity_moving: true,
            resync: false,
            resync_limit_frames: 0,
            preceding_silent_frames: 0,
            counted: true,
            discard_pre_roll: false,
            start_window: false,
            repeated_nonzero: false,
            position_source: if offline {
                PositionSource::Offline
            } else {
                PositionSource::Preview
            },
            epoch: processing.pass_id,
            transport_active: true,
            new_epoch: processing.new_pass,
            rebase_frames: processing.rebase_frames,
            prune_before: 0,
            // 预览 cursor 是本 pass 的本地时间轴，从第一块起就可信。
            position_trusted: true,
            publish_deferred: false,
            transport_generation: self.transport_generation.load(Ordering::Acquire),
            preview: !offline,
            timebase,
        }
    }

    /// 每流几何 + 有效位置 + 相位。音频线程调用：只读写原子，不分配。
    ///
    /// # 三条规则（没有第四条，也没有任何预算）
    ///
    /// ```text
    /// A  锚定：!had / playing_rising / recording_rising / cycle_wrap
    ///         ⇒ domain_offset = 0，anchor = raw_start，counted_run = 0
    /// B  续数：宿主没报新位置（raw_start == last_raw）
    ///         ⇒ position = anchor + 已送来的帧数（**无上限**）
    ///            （若 continuousTimeSamples / projectTimeMusic 在前进，优先用它们）
    /// C  对账：宿主报了新位置（raw_start != last_raw）
    ///         ⇒ position = raw_start + domain_offset（宿主永远赢，永不"纠正"它）
    ///            差值交给捕获侧：落在 counted_run 之内 ⇒ 记账纠正；
    ///            之外 ⇒ 时间基切换（首段平移）或真 seek（另起一段）
    /// ```
    ///
    /// # 位置倒退为什么不再一律是新 pass
    ///
    /// 「位置倒退」有两种成因，处置完全相反：
    ///
    ///   * 用户 seek / 循环回卷 —— 必须另起一段（新内容）；
    ///   * 宿主改了**时间基**（VEGAS 的 Event 级 FX：起播窗口报绝对工程位置，窗口
    ///     结束后改报事件相对位置）—— 音频是连续的，另起一段会把一段连续演奏劈成
    ///     两段，时间线上于是出现"前 0.6 s 在 31 秒开外、其余在 0"的断层。
    ///
    /// 区分它们需要的不是位置，而是"计数器在本 pass 里有没有权威"：`counted_run > 0`
    /// 说明宿主**有一段时间没报位置**，那么它现在报的这个数与旧读数**不同域**，
    /// 不可比较。此时保留音频连续性，并记下两域之差（`domain_offset`）。
    /// `counted_run == 0` 说明宿主一直在报位置，倒退只能是 seek。
    ///
    /// # 为什么删掉了 1.5 s 容差与"锁存偏移"
    ///
    /// 旧实现把"宿主报的位置比我们的累积时钟落后不超过 `POSITION_DRIFT_TOLERANCE_SEC`"
    /// 判成"宿主读数滞后"，于是继续用累积时钟，并把差值**锁存**下来供后续每一块复用。
    ///
    /// 用户回到起点再播 0.7 s 时，那个差恰好是 −0.7 s —— **在容差之内** ⇒ 位置被强制成
    /// "上一遍末尾 + 块长"，整遍内容偏移一个循环长度；而且锁存一生效，`offset_valid`
    /// 就再也不清（只有新起播/循环/一次 >250 ms 的跳变能清它），于是**整场会话**都走
    /// 这条路。这就是用户说的"反复播放 0.7 s，内容完全与它应该输出的对不上"。
    ///
    /// 教训是结构性的：**只要存在两只钟，就必须有一个仲裁，而任何仲裁都会在某一种
    /// 真实操作上判错**。所以这里不仲裁 —— 宿主报了，就听宿主的（R3）；宿主没报，
    /// 才按帧数接着数（R2），而那个数**可核对**：宿主下次报位置时 R3 立刻接管，
    /// 差值记进 `jumps` 并交给捕获侧做记账纠正（桥接/裁剪），不会变成 seek。
    ///
    /// # 为什么 R2 仍然必须存在
    ///
    /// 实测 VEGAS 在 Capture 播放期间会让 `projectTimeSamples` 停住不动。那时
    /// "每块前进 frames"是唯一自洽的换算 —— 它是合法记账，不是猜测。把它删掉不会让
    /// 宿主变好，只会让 take 变成一大片洞。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn note_stream_geometry(
        &self,
        stream: ProcessStream,
        raw_start: i64,
        frames: usize,
        rate: f64,
        time: StreamTime,
        playing_rising: bool,
        recording_rising: bool,
        playing: bool,
    ) -> StreamBlock {
        let geometry = &self.streams[stream.index()];
        geometry.blocks.fetch_add(1, Ordering::Relaxed);
        let had = geometry.seen.load(Ordering::Acquire);
        let rate = if rate.is_finite() && rate > 0.0 {
            rate
        } else {
            48_000.0
        };
        let pre_roll_cap = (rate * PRE_ROLL_CAP_SEC).round().max(1.0) as u64;
        if frames == 0 {
            // 零帧块是参数 flush，不代表时间轴在走。
            return StreamBlock {
                position: raw_start,
                raw_start,
                phase: if had {
                    TransportPhase::Resting
                } else {
                    TransportPhase::Start
                },
                had_previous: had,
                cycle_wrap: false,
                new_pass: false,
                activity_moving: false,
                resync: false,
                resync_limit_frames: 0,
                preceding_silent_frames: geometry.trailing_silent_frames.load(Ordering::Acquire),
                counted: false,
                discard_pre_roll: false,
                start_window: false,
                repeated_nonzero: false,
                // 零帧块不写任何样本，位置来源对它没有意义；沿用上一块的结论即可。
                position_source: PositionSource::from_code(
                    geometry.last_source.load(Ordering::Acquire),
                )
                .unwrap_or(PositionSource::Project),
                epoch: geometry.session_epoch.load(Ordering::Acquire),
                transport_active: false,
                new_epoch: false,
                rebase_frames: 0,
                prune_before: 0,
                // 零帧块不写样本：位置也不该发布（否则一次参数 flush 会把光标推到 raw）。
                position_trusted: false,
                publish_deferred: true,
                transport_generation: self.transport_generation.load(Ordering::Acquire),
                preview: false,
                timebase: TimebaseKind::from_code(geometry.timebase_kind.load(Ordering::Acquire))
                    .unwrap_or(TimebaseKind::Unknown),
            };
        }
        let frames32 = u32::try_from(frames).unwrap_or(u32::MAX);

        let last_position = geometry.last_position.load(Ordering::Acquire);
        let last_raw = geometry.last_raw_start.load(Ordering::Acquire);
        let last_continuous = geometry.last_continuous.load(Ordering::Acquire);
        let last_music = f64::from_bits(geometry.last_music_bits.load(Ordering::Acquire));
        let last_source = PositionSource::from_code(geometry.last_source.load(Ordering::Acquire))
            .unwrap_or(PositionSource::Project);
        let anchor_before = geometry.anchor.load(Ordering::Acquire);
        let advance_before = geometry.frames_since_anchor.load(Ordering::Acquire);
        // 【为什么先读再判】`resync_limit_frames` 的语义是"计数器**可能**错过的范围"，
        // 必须用本块处理**之前**的累积值；本块一写就被清零了。
        // 【为什么是 `counted_run` 而不是"自锚点以来送来的帧数"】后者在任何连续播放里
        // 都等于块长，于是"计数器可能错过多少"退化成"一块" —— 一个与事实无关的常数。
        // `counted_run` 只数**位置真的由计数器给出**的帧。
        let counted_before = geometry.counted_run.load(Ordering::Acquire);

        // 【循环回卷】只有循环播放才有"回卷"这个形状：块首回退，或块首不变而时间前进。
        let cycle_wrap = if time.cycle_active {
            let continuous_advanced = time
                .continuous
                .is_some_and(|continuous| continuous > last_continuous);
            let music_wrapped = time.music.is_some_and(|music| music + 1e-9 < last_music);
            let music_boundary = time.cycle_valid
                && time.continuous.is_none()
                && time.music.is_some_and(|music| {
                    music <= time.cycle_start_music + 1e-6
                        && last_music >= time.cycle_end_music - 1e-6
                });
            had && (raw_start < last_raw
                || (raw_start == last_raw
                    && (continuous_advanced || music_wrapped || music_boundary)))
        } else {
            false
        };

        let host_rewound = had && raw_start < last_raw;
        let host_moved = had && raw_start != last_raw;
        let session_frames_before = geometry.session_frames.load(Ordering::Acquire);
        let is_offline = stream == ProcessStream::Offline;
        let phase_before = SessionPhase::from_code(geometry.session_phase.load(Ordering::Acquire))
            .unwrap_or_default();
        // 【HostClock 先于 TransportSession】域切换必须赶在“新会话”判定之前，
        // 否则 VEGAS Event FX 在 kPlaying 上升沿上的切换会被当成新会话而跳过。
        // 这里用 TransportSession 即将做出的同一判据预测 new_epoch，只用于让
        // HostClock 区分“同一 Starting 会话里的上升沿”和“从 Paused/Stopped 起播”。
        let expected_new_epoch = (playing
            && matches!(
                phase_before,
                SessionPhase::Idle | SessionPhase::Paused | SessionPhase::Stopped
            ))
            || (!playing
                && (playing_rising || recording_rising)
                && phase_before != SessionPhase::Starting);
        // 【本次 take 的第一个有权块必须在 HostClock 之前、用 raw 记录】
        // 它是 Event 域切换唯一可靠的绝对起点（choose_origin）；晚一步记录，
        // HostClock 处理首块时看不到它；用 block.start（有效位置）则会在 R2 记账
        // 推进过的块上得到错的 origin。
        let transport_start = playing || playing_rising || recording_rising || !had;
        // 【只在还没记到时写】预览 pass 会先更新几何（had=true），但捕获侧仍需要
        // 第一个有权块的 raw 作为 Event 起点；用捕获存储自己的“未知”状态判断。
        if self.capture().is_armed()
            && transport_start
            && self.capture().capture_origin_raw().is_none()
        {
            self.capture().note_capture_origin(raw_start);
        }
        let mut clock_state = geometry.load_clock_state();
        let clock = clock_state.observe(ClockInput {
            raw_start,
            frames,
            rate,
            playing,
            playing_rising,
            recording_rising,
            cycle_active: time.cycle_active,
            cycle_wrap,
            continuity: geometry.boundary_continuity(),
            had_previous: had,
            session_phase: phase_before,
            session_frames: clock_state.session_frames,
            session_new_epoch: expected_new_epoch,
            capture_origin_raw: self.capture().capture_origin_raw(),
            persisted_event_origin: self.event_origin_samples(),
            music: time.music,
            tempo: time.tempo,
            capture_armed: self.capture().is_armed(),
            transport_start,
        });
        if let Some(origin) = clock.event_origin {
            // 一次确认过的域切换（switched + Event）比“从持久化状态恢复”更可信。
            let confidence = if clock.switched && clock.domain.is_event() {
                2
            } else {
                1
            };
            self.note_event_origin(origin, confidence);
        }
        let mut session = TransportSession {
            epoch: geometry.session_epoch.load(Ordering::Acquire),
            phase: phase_before,
            session_frames: session_frames_before,
            last_playing: geometry.last_playing.load(Ordering::Acquire),
            max_start_window_frames: pre_roll_cap,
        };
        // 【首块起播容差的唯一正证据】见下面 `playing_rising` 的说明：只有本块真的
        // 有信号，才允许“kPlaying 滞后”的宿主在第一块上打开有界起点窗口。
        let audio_evidence = !geometry.last_block_silent.load(Ordering::Acquire);
        let decision = session.observe(SessionInput {
            // 离线 pass 有自己的 cursor；它不需要 kPlaying 也必须在走。
            playing: playing || is_offline,
            // 【首块不再无条件伪造起播】此前 `!had && !playing` 会在**任何**宿主的第一
            // 块上合成一个起播上升沿，打开 1 秒的有界起点窗口。REAPER 等宿主停播时仍
            // 持续 process 固定光标处的**静音块**，而 `capture_start` 会重置流几何
            //（`had` 归零）⇒ 按下录制、宿主根本没播，那 1 秒静音就被逐块记成 take：
            // 界面上正是“捕获开始后立刻计时、到 1 s 自动停住”，take 里
            // `first_signal_at=none` / `pre_roll_frames≈48000`。
            //
            // 【唯一的正证据】“kPlaying 还来不及置位但音频已经在流动”只在**这一块真的
            // 有信号**时成立（VEGAS Event FX 预卷、VST3 起播窗口、prefetch 都是这种形
            // 状）。全静音块没有任何证据能把它与“宿主停播”区分开，因此不放行。
            // 真正起播时 kPlaying 会升起，`transport_active` 照常由 `playing` 打开 ——
            // 这次容差只影响“没播放”的那段，不会吞掉任何一帧真实音频。
            playing_rising: playing_rising
                || recording_rising
                || (!had && !playing && !is_offline && audio_evidence),
            recording_rising,
            // 域切换不是 seek：不能让它触发 host_reset 的新 epoch。
            host_reset: host_rewound && !clock.switched,
            offline: is_offline,
            frames: frames32,
        });
        // 【预览候选也算有权推进】候选期间宿主 kPlaying=0，但它送来的正是预览
        // 要录的音频；不把候选算进来，起播窗口一过位置就冻结，预览只剩 1 秒。
        let preview_candidate = self.processing_pass.phase() == PreviewPhase::Candidate;
        let transport_active =
            decision.transport_active || playing || is_offline || preview_candidate;
        let new_epoch = decision.new_epoch;
        let start_window_open = decision.start_window_open;
        // A：新一次播放。域切换不是新 pass；真 seek / 新 epoch / 循环回卷才是。
        // 【Starting 里的上升沿不是新 pass】那是 pre-roll -> 正文；从 Rolling 里的
        // 上升沿则是新一次起播（宿主可能没有先送 kPlaying=0 的块）。
        // 【域切换不是新 pass】VEGAS Event FX 的 kPlaying 上升沿同时是域切换；
        // 把它当新 pass 会把一条连续演奏切成两段（第一段留在绝对域）。
        let playing_edge_new_pass =
            playing_rising && phase_before != SessionPhase::Starting && !clock.switched;
        let new_pass = new_epoch || clock.seek || cycle_wrap || playing_edge_new_pass;

        let time_base_switch = clock.switched;
        let rebase_frames = clock.rebase_frames;
        let timebase_after = timebase_of(clock.domain);

        // 本块位置的"锚 + 偏移"。R1/R3 时锚就是宿主刚报的值、偏移为 0；
        // R2 时锚保持不动、偏移是已送来的帧数。两条路径因此共用同一句
        // `anchor + advance`，不存在"两个域"。
        let (anchor, advance) = if new_pass || host_moved || time_base_switch {
            (clock.local_start, 0_u64)
        } else {
            (anchor_before, advance_before)
        };

        // 【停播重复投递的判据】同一份**非零**内容逐块重复 ⇒ 宿主在停播时反复送同一
        // 段缓冲，那不是播放时间。它与 VEGAS 起播窗口的区分点是内容：窗口送的是静音，
        // FL Studio 停播送的是同一份非零内容。
        let silent = geometry.last_block_silent.load(Ordering::Acquire);
        let repeated_nonzero = geometry.unchanged_run.load(Ordering::Acquire) > 0 && !silent;

        let mut position = clock.local_start;
        let mut source = if time.cycle_active {
            PositionSource::Cycle
        } else {
            PositionSource::Project
        };
        let mut counted = false;

        if !transport_active {
            // 【冻结】kPlaying=0 且不在有界起点窗口里：位置不得前进。宿主在停播时
            // 仍会送块，甚至送变化的内容/静音；那些都不是播放时间。位置取最后
            // 一个真实位置，而不是宿主此刻报的读数（VEGAS 停播时会报远处假位置）。
            if had {
                position = last_position;
            }
            geometry.counted_run.store(0, Ordering::Release);
            geometry.frames_since_anchor.store(0, Ordering::Release);
        } else if new_pass || host_moved || time_base_switch {
            // A / C：宿主给了位置，它就是权威。
            if host_moved && !new_pass && !time_base_switch {
                let expected = anchor_before.saturating_add(advance_before as i64);
                let jump = raw_start.saturating_sub(expected);
                if jump != 0 {
                    geometry.jumps.fetch_add(1, Ordering::Relaxed);
                    geometry.last_jump.store(jump, Ordering::Release);
                }
            }
            if new_pass {
                // A：新一次播放 / 真 seek。宿主的域重新成为权威；若 HostClock 已经
                // 解析出 Event 域（例如从组件 state 恢复的 origin），直接采用它的映射。
                geometry.domain_offset.store(0, Ordering::Release);
                geometry.counted_run.store(0, Ordering::Release);
                geometry
                    .timebase_kind
                    .store(timebase_after.code(), Ordering::Release);
                geometry
                    .timebase_offset
                    .store(clock.timebase_offset, Ordering::Release);
                position = if new_epoch || playing_edge_new_pass || cycle_wrap {
                    // 新一次演奏 / 循环回卷：宿主的原始读数重新成为权威（R2 的
                    // “上一块终点”属于上一遍，不能拿来当起点）。
                    raw_start.saturating_add(clock.timebase_offset)
                } else {
                    clock.local_start
                };
                geometry.pass.fetch_add(1, Ordering::Relaxed);
            } else if time_base_switch {
                // 【时间基切换：把整条流挪到新域里】不是 seek，不能新开段。
                geometry
                    .timebase_kind
                    .store(timebase_after.code(), Ordering::Release);
                geometry
                    .timebase_offset
                    .store(clock.timebase_offset, Ordering::Release);
                position = clock.local_start;
                geometry.last_rebase.store(rebase_frames, Ordering::Release);
                geometry.rebase_count.fetch_add(1, Ordering::Relaxed);
                geometry.counted_run.store(0, Ordering::Release);
                geometry.position_resyncs.fetch_add(1, Ordering::Relaxed);
                self.position_resyncs.fetch_add(1, Ordering::Relaxed);
            } else {
                // C：宿主报了新位置（同域）。
                position = clock.local_start;
                geometry.counted_run.store(0, Ordering::Release);
            }
            geometry.anchor.store(position, Ordering::Release);
            geometry.continuous_anchor.store(
                time.continuous.unwrap_or(last_continuous),
                Ordering::Release,
            );
            geometry.music_anchor.store(
                time.music.unwrap_or(last_music).to_bits(),
                Ordering::Release,
            );
            geometry
                .frames_since_anchor
                .store(frames as u64, Ordering::Release);
        } else {
            // B：宿主没报新位置。用最好的宿主时钟，最后才是计数器（**无上限**）。
            let sane = (frames as i64).saturating_mul(4).max(64);
            let mut offset = advance as i64;
            let mut advanced = false;
            if !time.cycle_active {
                if let Some(continuous) = time.continuous {
                    let from_anchor = continuous
                        .saturating_sub(geometry.continuous_anchor.load(Ordering::Acquire));
                    let step = from_anchor.saturating_sub(offset);
                    if from_anchor >= 0 && step >= 0 && step <= sane {
                        offset = from_anchor;
                        position = anchor.saturating_add(from_anchor);
                        source = PositionSource::Continuous;
                        advanced = true;
                        self.continuous_used.fetch_add(1, Ordering::Relaxed);
                        geometry.continuous_used.fetch_add(1, Ordering::Relaxed);
                    }
                }
                if !advanced {
                    if let (Some(music), Some(tempo)) = (time.music, time.tempo) {
                        let anchor_music =
                            f64::from_bits(geometry.music_anchor.load(Ordering::Acquire));
                        let beats = music - anchor_music;
                        if beats > 0.0 {
                            let from_anchor = (beats * 60.0 / tempo * rate).round() as i64;
                            let step = from_anchor.saturating_sub(offset);
                            if step >= 0 && step <= sane {
                                offset = from_anchor;
                                position = anchor.saturating_add(from_anchor);
                                source = PositionSource::Music;
                                advanced = true;
                                self.music_used.fetch_add(1, Ordering::Relaxed);
                                geometry.music_used.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
            if advanced {
                geometry.counted_run.store(0, Ordering::Release);
            } else if transport_active && geometry.flowing() {
                // 计数器：只在 transport_active 时才有权推进。停播块永远走不到这里。
                position = anchor.saturating_add(offset);
                counted = true;
                advanced = true;
                geometry.counted_blocks.fetch_add(1, Ordering::Relaxed);
                geometry
                    .counted_frames
                    .fetch_add(frames as u64, Ordering::Relaxed);
                geometry
                    .counted_run
                    .fetch_add(frames as u64, Ordering::Relaxed);
            }
            if !advanced {
                position = anchor;
                offset = 0;
                // 停播期间送来的帧不是播放时间；下一次真正流动时必须从当前锚点
                // 重新起算，而不是把停播时长一次性加到位置上。
                geometry.frames_since_anchor.store(0, Ordering::Release);
                geometry.counted_run.store(0, Ordering::Release);
            }
            geometry.frames_since_anchor.store(
                offset.saturating_add(frames as i64).max(0) as u64,
                Ordering::Release,
            );
        }

        // ── 前导静音（诊断 + 门控）────────────────────────────────────
        let mut preceding_silent_frames = geometry.trailing_silent_frames.load(Ordering::Acquire);
        let mut discard_pre_roll = false;
        // 【停播静音 vs 起播窗口静音】两者都是"全零块"，但只有后者是播放时间。
        //
        // 真正能区分它们的是**上升沿之前那一块在做什么**：
        //   * 起播窗口：宿主先送**静音**，内容出现时才把播放位置置 1；
        //   * 停播重复投递：宿主在**停播**状态下把同一段有内容的音频反复送。
        // 旧判据是"计数器已经累积超过一个窗口"—— 那是一个额度而不是事实（而且它在
        // 非实时预览 / 离线渲染里恒为真）。现在直接问上一块的 kPlaying。
        if (playing_rising || recording_rising)
            && !geometry.last_playing.load(Ordering::Acquire)
            && !silent
            && preceding_silent_frames > 0
        {
            preceding_silent_frames = 0;
            discard_pre_roll = true;
        }
        // 宿主没报新位置时，静音块才是"播放中的静音"；停播重复投递不累积。
        let pinned_since_last = !had || raw_start == last_raw;
        if silent && pinned_since_last {
            geometry.trailing_silent_frames.store(
                preceding_silent_frames
                    .saturating_add(frames as u64)
                    .min(pre_roll_cap),
                Ordering::Release,
            );
        } else if !silent {
            geometry.trailing_silent_frames.store(0, Ordering::Release);
        }

        // ── 相位 ──────────────────────────────────────────────────────
        let activity_moving = geometry.activity_moving();
        let phase = if !transport_active {
            geometry.same_start.fetch_add(1, Ordering::AcqRel);
            TransportPhase::Resting
        } else if !had {
            geometry.same_start.store(0, Ordering::Release);
            geometry.extension_used.store(false, Ordering::Release);
            TransportPhase::Start
        } else if cycle_wrap || position != last_position || activity_moving {
            // 【为什么音频在流动也算 Rolling】起播窗口里宿主可以把位置钉住，而音频
            // 已经在按块长流动。此时"位置没变"不成立 —— 我们刚刚按块长推进了它。
            geometry.same_start.store(0, Ordering::Release);
            geometry.extension_used.store(false, Ordering::Release);
            TransportPhase::Rolling
        } else if playing_rising || recording_rising {
            geometry.same_start.store(0, Ordering::Release);
            geometry.extension_used.store(false, Ordering::Release);
            TransportPhase::Start
        } else {
            geometry.same_start.fetch_add(1, Ordering::AcqRel);
            let extension_min = (rate / 20.0).round().max(1.0) as u32;
            let last_frames = geometry.last_frames.load(Ordering::Acquire);
            if frames32 > last_frames.saturating_add(extension_min)
                && !geometry.extension_used.swap(true, Ordering::AcqRel)
            {
                TransportPhase::Rolling
            } else {
                TransportPhase::Resting
            }
        };

        if playing_rising || recording_rising {
            geometry.rising_edges.fetch_add(1, Ordering::Relaxed);
        }
        match phase {
            TransportPhase::Resting => {
                geometry.resting_blocks.fetch_add(1, Ordering::Relaxed);
                self.stream_resting_blocks[stream.index()].fetch_add(1, Ordering::Relaxed);
            }
            TransportPhase::Start | TransportPhase::Rolling => {
                geometry.rolling_blocks.fetch_add(1, Ordering::Relaxed);
                self.stream_rolling_blocks[stream.index()].fetch_add(1, Ordering::Relaxed);
            }
        }
        if cycle_wrap {
            self.cycle_wraps.fetch_add(1, Ordering::Relaxed);
            geometry.cycle_wraps.fetch_add(1, Ordering::Relaxed);
        } else if time.cycle_active && had && raw_start == last_raw && !playing_rising {
            // 同一块首、cycle 有效但没有可用的回卷证据：可能是短循环，也可能是停播。
            // 不能据此写入；只记一个读数，交给捕获侧的判据解释。
            geometry.cycle_wrap_unknown.fetch_add(1, Ordering::Relaxed);
        }

        // ── I-3 的读数：位置是否逐块续接 ─────────────────────────────
        //
        // 稳态下 `position - last_position` 恒等于块长。它不等于块长只有三种解释：
        // ① 宿主真的 seek（用户动作，可接受）；② pass 边界（可接受）；
        // ③ **我们的记账漏了一项**（不可接受，且它就是咔）。
        if had && !new_pass && !cycle_wrap {
            let step = position.saturating_sub(last_position);
            if step < 0 {
                geometry.clock_rewinds.fetch_add(1, Ordering::Relaxed);
            }
            let magnitude = step.unsigned_abs();
            if geometry.clock_deltas.load(Ordering::Acquire) == 0 {
                geometry.clock_delta_min.store(magnitude, Ordering::Release);
            } else {
                geometry
                    .clock_delta_min
                    .fetch_min(magnitude, Ordering::Relaxed);
            }
            geometry
                .clock_delta_max
                .fetch_max(magnitude, Ordering::Relaxed);
            geometry.clock_deltas.fetch_add(1, Ordering::Relaxed);
        }

        geometry.seen.store(true, Ordering::Release);
        geometry.last_start.store(position, Ordering::Release);
        geometry.last_frames.store(frames32, Ordering::Release);
        geometry
            .last_end
            .store(position.saturating_add(frames as i64), Ordering::Release);
        geometry.last_position.store(position, Ordering::Release);
        geometry.last_raw_start.store(raw_start, Ordering::Release);
        geometry.last_continuous.store(
            time.continuous.unwrap_or(last_continuous),
            Ordering::Release,
        );
        geometry.last_music_bits.store(
            time.music.unwrap_or(last_music).to_bits(),
            Ordering::Release,
        );
        geometry.last_source.store(source.code(), Ordering::Release);
        geometry.last_playing.store(playing, Ordering::Release);
        geometry
            .session_phase
            .store(decision.phase.code(), Ordering::Release);
        geometry
            .session_epoch
            .store(session.epoch, Ordering::Release);
        geometry
            .session_frames
            .store(session.session_frames, Ordering::Release);
        if stream == ProcessStream::Realtime {
            self.realtime_flow
                .store(phase != TransportPhase::Resting, Ordering::Release);
        }
        // 第一次真正有权的块把 Unknown 落成 ProjectAbsolute：Event FX 的首段在
        // 切换前本来就报绝对位置，落成 project 不会改变映射，只让诊断可读。
        // 【为什么由 HostClock 决定】没有事件证据时它只能诚实地保持工程绝对域；
        // 绝不能靠“猜一个 0..5 s 的候选”伪造事件起点。
        if transport_active && is_offline {
            geometry
                .timebase_kind
                .store(TimebaseKind::OfflineLocal.code(), Ordering::Release);
        } else if transport_active && clock_state.domain == ClockDomain::Unknown {
            clock_state.confirm_project();
            geometry
                .timebase_kind
                .store(timebase_of(clock_state.domain).code(), Ordering::Release);
        }
        // 把本块最终采用的位置写回 HostClock：R2 的 continuous/music 校正可能
        // 与它自己的 last_local_end 不同，下一块必须从真实终点继续。
        clock_state.note_final_local(position, frames as i64);
        geometry.store_clock_state(&clock_state);

        // 【resync：宿主纠正了我们数出来的位置】这不是 seek，是记账纠正 —— 真正的
        // 差值判定在捕获侧（它才持有 `expected_next`），这里只给上限与意图。
        let resync = had
            && !time.cycle_active
            && !new_pass
            && !time_base_switch
            && host_moved
            && (counted_before > 0
                || matches!(
                    last_source,
                    PositionSource::Continuous | PositionSource::Music
                ));
        StreamBlock {
            position,
            raw_start,
            phase,
            had_previous: had,
            cycle_wrap,
            new_pass,
            activity_moving,
            resync,
            resync_limit_frames: counted_before,
            preceding_silent_frames,
            counted,
            discard_pre_roll,
            // 【起播窗口 = 计数器此刻有权威，且它还没数满一个前导窗口】它只服务
            // 门控的"窗口内静音也要放行"，**不约束位置推进**（见 PRE_ROLL_CAP_SEC）。
            start_window: start_window_open,
            repeated_nonzero,
            position_source: source,
            epoch: session.epoch,
            transport_active,
            new_epoch,
            rebase_frames,
            prune_before: clock.prune_before,
            position_trusted: clock.position_trusted,
            publish_deferred: clock.publish_deferred,
            transport_generation: self.transport_generation.load(Ordering::Acquire),
            preview: false,
            timebase: TimebaseKind::from_code(geometry.timebase_kind.load(Ordering::Acquire))
                .unwrap_or(TimebaseKind::Unknown),
        }
    }

    /// 会话边界：清空“上一遍”的流几何与输出锁存。
    ///
    /// 【为什么必须同时清四样东西】问题 3 的完整触发链是：
    /// 1. 停止时只暂停 capture，StreamGeometry.seen/last_position/last_playing
    ///    与 stream_playing/stream_recording 留在“上一遍”；
    /// 2. 再播第一块 raw 与停止前相同，没有 kPlaying 上升沿 ⇒ phase=Resting；
    /// 3. 输出仲裁据此回到上一张 take 的 processed 快照；
    /// 4. latch_rung 按 pass 锁存，pass 号不变 ⇒ 整个新一遍都继续读旧档位。
    ///
    /// 四样东西缺一不可：流几何、每流播放/录音位、rung latch、捕获时钟。
    /// 本函数只写原子量，可从音频线程调用（setProcessing(false) 就是音频线程）。
    pub(crate) fn reset_transport_session(&self, reason: &str) {
        let _ = reason;
        // 【输出权威】同一 capture 会话里停止再播时 capture epoch 不变；只有这个
        // 代次能告诉输出仲裁“这一块属于上一遍”。
        self.transport_generation.fetch_add(1, Ordering::AcqRel);
        for stream in [
            ProcessStream::Realtime,
            ProcessStream::Prefetch,
            ProcessStream::Offline,
        ] {
            let geometry = &self.streams[stream.index()];
            geometry.seen.store(false, Ordering::Release);
            geometry.last_start.store(0, Ordering::Release);
            geometry.last_end.store(0, Ordering::Release);
            geometry.last_frames.store(0, Ordering::Release);
            geometry.last_position.store(0, Ordering::Release);
            geometry.last_raw_start.store(0, Ordering::Release);
            geometry.last_playing.store(false, Ordering::Release);
            geometry.anchor.store(0, Ordering::Release);
            geometry.frames_since_anchor.store(0, Ordering::Release);
            geometry.counted_run.store(0, Ordering::Release);
            geometry.same_start.store(0, Ordering::Release);
            geometry.extension_used.store(false, Ordering::Release);
            geometry.session_frames.store(0, Ordering::Release);
            geometry
                .session_phase
                .store(SessionPhase::Idle.code(), Ordering::Release);
            geometry
                .timebase_kind
                .store(TimebaseKind::Unknown.code(), Ordering::Release);
            geometry.timebase_offset.store(0, Ordering::Release);
            geometry.reset_clock_session();
        }
        // pass 号必须推进：所有按 pass 锁存的状态（rung / read_position_jump）都以它
        // 为“是不是同一次播放”的判据。
        self.streams[ProcessStream::Realtime.index()]
            .pass
            .fetch_add(1, Ordering::AcqRel);
        for flag in &self.stream_recording {
            flag.store(false, Ordering::Release);
        }
        self.capture().clear_stream_flags();
        self.clear_rung_latch();
        self.rt_last_emitted_valid.store(false, Ordering::Release);
        self.rt_last_emitted_frames.store(0, Ordering::Release);
        self.rt_last_read_position.store(0, Ordering::Release);
        if let Some(clock) = self.clock.get() {
            clock.clear_capture();
        }
    }

    /// 宿主此刻**确实在推进工程位置**（无锁、无分配，可从音频线程调用）。
    ///
    /// 【为什么要这个判据而不是 `kPlaying`】`kPlaying` 与 `projectTimeSamples` 不是原子
    /// 更新的：宿主从"停止"切到"播放"的那几块里，位置已经在走、播放位可能还没置上。
    /// 而"位置在前进"是宿主无关的硬事实 —— 真在播就必然前进，钉住不动就是在停。
    ///
    /// 它服务的唯一决定是"停播时要不要清输出"：清是对的（固定光标处不清就会把同一小段
    /// 反复送出去，听起来是个嗡嗡声），但**不能拿播放位当唯一的判据** —— 那会让每次
    /// 播放都吞掉开头（用户报障：播放后约 1 秒没有输出）。
    ///
    /// 【判据的粒度是"当前块 vs 上一块"】见 [`Self::note_block_position`]：
    /// 块首移动或同头块变长即算在走。门控**不会**被停播/预渲染留下的历史位置卡住。
    ///
    /// 【为什么 ARA 实例恒为假】它的输出语义由 ARA 通路决定，这个守卫必须对它完全无效
    /// （见 `a_bound_instance_keeps_its_output`）。返回假 ⇒ 保持"停播即静音"的既有行为。
    #[allow(dead_code)]
    pub(crate) fn capture_transport_running(&self) -> bool {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return false;
        }
        self.realtime_flow.load(Ordering::Acquire)
    }

    /// 门控问“本块是否可以不清空”。
    ///
    /// 【真值表】ARA 实例：`false`（保持“停播即静音”的既有语义，由调用方的
    /// `!kPlaying` 决定）。未绑定实例：offline/prefetch 恒真（它们的输出稍后才被
    /// 播放，不能用此刻的传输状态挖洞）；实时流看本块证据。
    pub(crate) fn capture_should_emit(
        &self,
        evidence: Option<BlockEvidence>,
        playing: bool,
    ) -> bool {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            // 【为什么 ARA 实例也要区分流】REAPER 在预取 / 擦洗 / 循环折返时会
            // **合法地**以 `kPlaying == 0` 调用 `process`，而预取块会被它缓存并真的
            // 播出来 —— 按 `kPlaying` 清零等于往缓存里写静音（"有概率完全没声音"）。
            // offline / prefetch 与 Capture 实例同一判据：输出稍后才被播放，此刻的
            // 传输状态不适用。实时流仍保持 ARA 的"停播即静音"语义：光标处的重复块
            // 不该循环出声，而真实起播由 `kPlaying` 与捕获侧证据驱动。
            return match evidence {
                Some(evidence) => match evidence.stream {
                    ProcessStream::Offline | ProcessStream::Prefetch => true,
                    ProcessStream::Realtime => playing,
                },
                None => playing,
            };
        }
        match evidence {
            Some(_) => Self::emit_allowed(evidence),
            // 没有上下文证据时退回宿主播放位（旧测试/退化路径）。
            None => playing,
        }
    }

    /// 本块是否允许写出内容。**三道门共用这一个谓词**。
    ///
    /// 【为什么必须只有一个】此前三道门各写各的：`capture_should_emit` 的第一道放行
    /// 起播窗口（`start_window && !repeated_nonzero`），而 `capture_stage` 的第二道只看
    /// `allow_output`，于是前者放行的每一块都被后者淡成零 —— "窗口内输出应当连续"
    /// 这条承诺从未生效，用户听到的仍然是起播那一下咔。
    ///
    /// 【真值表】离线 / prefetch：永远允许（它们的输出稍后才被播放，此刻的传输状态
    /// 与内容正确性无关，且重复请求可能是参数变化后的重算）。实时：`allow_output`
    /// 为真，或本 pass 的计数器窗口还开着且内容不是"逐块相同的非零重复"。
    pub(super) fn emit_allowed(evidence: Option<BlockEvidence>) -> bool {
        match evidence {
            Some(evidence) => match evidence.stream {
                ProcessStream::Offline | ProcessStream::Prefetch => true,
                ProcessStream::Realtime => {
                    // 【为什么起播窗口要放行】窗口内的前段可能是静音；按
                    // `allow_output == false` 清掉它，内容出现时就是一次零→内容的
                    // 阶跃（"咔"）。窗口本身是"宿主刚开始播放"的证据，输出应当
                    // 连续。`repeated_nonzero` 仍然阻止 FL 停播继续出声。
                    evidence.allow_output || (evidence.start_window && !evidence.repeated_nonzero)
                }
            },
            None => true,
        }
    }

    /// 已发布的**处理结果**是否对应当前活动 take。
    ///
    /// 【为什么只比 take、不比编辑代次】编辑代次在会话那一侧，音频线程取不到。而"曲线
    /// 改了但合成还没跑完"这一档若回落到原始磁带，用户刚调完音高就会听到**未处理的原声**
    /// —— 那是倒退。只比 take 号：曲线编辑不改 take，于是仍然读处理结果（听到上一版处理，
    /// 与既有行为一致）；换 take 才回落。
    /// 此刻音频线程**能不能**服务一份已发布的快照。
    ///
    /// 【为什么需要它】「渲染中」此前绑在"有没有一个渲染任务"上，而那回答的是
    /// "后台在做一件事"，不是"你听到的是不是旧的"。一次**刷新**（旧快照仍可服务）
    /// 在听感上什么都没发生，把它显示成「渲染中」等于把「后台在忙」说成「你听到的是错的」。
    ///
    /// capture：有素材且（处理结果或原始磁带）能对上当前 take；
    /// ARA：发布过一份 prepared 快照。
    pub(crate) fn has_serviceable_snapshot(&self) -> bool {
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return self.prepared.lock_or_recover().is_some();
        }
        self.capture.has_material() && (self.processed_ready() || self.raw_tape_ready())
    }

    pub(super) fn processed_ready(&self) -> bool {
        if !self.capture.has_material() {
            return false;
        }
        let revision = self.capture.revision();
        self.capture_ready_revision.load(Ordering::Acquire) == revision
            // 【为什么不能 any()】实际读取的是
            // `snapshots[snapshot_index_for_rate(...)]`；只就绪另一份发布器时
            // `any()` 会误判"就绪"，随后在空发布器上逐样本写零 ⇒ 从有声阶跃到零。
            && self.processed_snapshot_ready()
    }

    /// 已发布的**原始磁带**是否对应当前活动 take。
    ///
    /// 【为什么要比 take 号】否则换 take 之后会继续播上一份原始磁带 —— 用户听到的是
    /// 上一次录制的内容，而界面上已经是新 take。
    pub(super) fn raw_tape_ready(&self) -> bool {
        if !self.capture.has_material() {
            return false;
        }
        self.raw_tape_revision.load(Ordering::Acquire) == self.capture.revision()
            && self.raw_tape.is_ready()
    }

    /// 每个音频块的**观察与捕获**：只依赖输入总线与工程位置，**不依赖输出形状**。
    ///
    /// 【为什么必须与输出阶段分开 —— 这是"录制与输出全静音"的一个直接原因】
    /// 这两件事此前都长在 [`Self::capture_stage`] 里，而后者被 `process` 在
    /// `num_samples == 0 || num_outputs == 0` 的早退**之后**才调用。于是"宿主没接我们的
    /// 输出总线"这一个与录制**无关**的事实，会让插件：
    ///
    ///   * 永远捕获不到任何东西（录制静音）；
    ///   * `saw_context` 永不置位 ⇒ 界面上的录制键一直是灰的（连录都录不了）。
    ///
    /// 录制只关心两件事：宿主给的**工程位置**、宿主给的**输入**。
    ///
    /// 捕获观察的**唯一**实现：吃一个中立块与中立时间事实。
    ///
    /// 【为什么块与时间分开】缓冲（`InputBlock`）描述"这一段样本是什么"，时间事实
    /// （`BlockContext`）描述"它在宿主时间轴上处于哪一刻"。两者来源不同：VST3 把它们
    /// 分别放在 `ProcessData` 与 `ProcessContext`，CLAP 放在 `clap_process_t` 与
    /// `clap_event_transport_t`。合成一个结构只会逼每种格式再造一次。
    pub(crate) fn capture_observe_block(
        &self,
        block: crate::capture::recorder::InputBlock<'_>,
        stream: ProcessStream,
        context: impl Into<crate::block_context::BlockContext>,
        effective_position: Option<i64>,
    ) -> Option<BlockEvidence> {
        let context = context.into();
        let capture = self.capture();
        // 【为什么离线也要记】它决定前端显示哪一条引导：走过 bounce 的宿主可以
        // "点一次 Capture 然后导出"，而实时宿主必须真的播一遍。
        if stream == ProcessStream::Offline {
            capture.note_offline();
        }
        // 【为什么采样率有兜底】宿主不填采样率的情形真实出现过，而没有率就无法把
        // 工程帧换算成秒，捕获会把每一块都当成"没有位置"丢掉（用户看到"录了 0.0 秒"）。
        // 声明过的率是同一件事的权威来源，用它兜底不引入新语义。
        let rate = if block.sample_rate.is_finite() && block.sample_rate > 0.0 {
            block.sample_rate
        } else {
            self.setup_sample_rate()
        };
        let playing = context.playing;
        let recording = context.recording;
        let requested_position = effective_position;
        // 块的起点就是调用方已经算好的有效位置（适配层用同一个值构造 `InputBlock`）。
        let effective_position = effective_position.unwrap_or(block.start);
        // 诊断环里的 `mode` 与宿主声明的处理模式同值（`ProcessStream` 的 index 就是它）。
        let mode = stream.index() as i32;
        // 【为什么未武装也要观察】观察位描述"宿主实际做了什么"，而录制按钮的门
        // （`canCapture`）读的正是 `saw_context`。此前观察发生在武装**之后**，于是
        // "必须先录一次才能录"——按钮永远是灰的，Capture 在所有宿主上都用不了。
        // 代价是几个原子写 + 一次缓冲形状检查（不分配、不读样本）。
        let playing_rising = capture.observe_with_stream(&block, stream);
        let was_recording = self.stream_recording[stream.index()].swap(recording, Ordering::AcqRel);
        let recording_rising = recording && !was_recording;
        // 【为什么要在相位判定之前记】`note_stream_geometry` 的"音频在流动"证据
        // 来自本块的指纹与上一块的比较；晚一步记录就永远读到上一块的结论。
        //
        // 【为什么零帧块不记】它是参数 flush，不代表时间轴在走。把它记成一次"内容
        // 变化"会让紧随其后的真实块被当成新素材（多出一块本不该有的位移）。
        let mut block_fingerprint_value = 0_u64;
        let mut block_silent = true;
        if block.frames > 0 {
            // 【一次扫描给出两个结论】内容指纹（重复投递判定）与静音（休止判定）
            // 都要遍历整块；分开写就是两遍。见 `block_fingerprint`。
            let (fingerprint, silent) = block_fingerprint(&block);
            block_fingerprint_value = fingerprint;
            block_silent = silent;
            self.streams[stream.index()].note_activity(fingerprint);
            // 【边界连续性探针】相邻块在时间上并不重叠，尾/头点积没有意义；
            // 这里量的是边界处的一阶差分（见 `note_boundary`）。
            self.streams[stream.index()].note_boundary(block.planes[0]);
            // 【T12：不再要求 `has_audio()`】宿主在 pre-roll 期间标 `silence_flags` /
            // 给空输入平面是**正常**的（`capture.rs` 的 `let silent = !block.has_audio();`
            // 就在用同一语义判定"这一块是静音"）。把它当成"没有证据"，等于告诉自己
            // "宿主标了静音 ⇒ 不是静音" —— 整段起播窗口的静音一次都记不下来，
            // `take_pre_roll` 永远是 0。空平面与全零平面对本判据是同一件事。
            //
            // 注意：`has_audio()` 的其它读点（"这条总线接没接"）保持原样，本次只改
            // 沉默累积这一个调用点。
            self.streams[stream.index()].note_block_silence(silent);
        }
        // 【为什么位置是否推进要按流记】它决定"停播时要不要清输出"与捕获的
        // lookbehind。两条流的块会交替到达；跨流比较会把 prefetch 的移动当成
        // 实时流“在前进”的证据（用户听到的开头小段循环）。
        // 【处理 pass 必须先于 HostClock】mode=0 的冻结预览 raw 固定、kPlaying=0；
        // 若先走实时会话，它会在 1 秒起播窗口后被判成 Paused，预览永远只录 1 秒。
        let stream_time = StreamTime::from_context(&context);
        let phase_before = SessionPhase::from_code(
            self.streams[stream.index()]
                .session_phase
                .load(Ordering::Acquire),
        )
        .unwrap_or_default();
        let processing = if let Some(position) = requested_position {
            Some(ProcessingBlock {
                kind: ProcessingKind::Offline,
                pass_id: self.processing_pass.pass_id(),
                local_start: position,
                rebase_frames: 0,
                new_pass: position == 0,
            })
        } else if stream != ProcessStream::Realtime {
            // prefetch 的 raw 是渲染进度，不是播放头；从它进入预览会把 cursor 建在
            // 错误的位置上。预览只认实时流。
            None
        } else {
            // 预览确认要知道 HostClock 此刻在哪个域：Event 域里候选块写的是
            // 本地位置（raw - origin），确认时不能再平移一次。
            let clock_domain = self.streams[stream.index()].clock_domain();
            let clock_is_local = clock_domain.is_event();
            let clock_offset = clock_domain.local_offset();
            let clock_pending = clock_domain == ClockDomain::PendingAbsolute;
            match self.processing_pass.note_preview_block(
                mode,
                context.project_time_samples,
                block.frames,
                rate,
                block_fingerprint_value,
                block_silent,
                playing,
                self.capture_preview_allowed(),
                phase_before,
                capture.capture_started(),
                self.setup_process_mode(),
                self.event_origin_samples(),
                stream_time.music,
                stream_time.tempo,
                clock_is_local,
                clock_offset,
                clock_pending,
                &self.pass_sequence,
            ) {
                PreviewDecision::Block(processing) => Some(processing),
                PreviewDecision::None => None,
            }
        };
        let mut stream_block = if let Some(processing) = processing {
            self.capture()
                .set_preview_active(processing.kind == ProcessingKind::FrozenPreview);
            self.note_processing_block(stream, context, block.frames, processing)
        } else {
            self.capture().set_preview_active(false);
            self.note_stream_block_at(
                stream,
                context,
                block.frames,
                playing_rising,
                recording_rising,
                Some(effective_position),
            )
        };
        // 【候选期间位置不可发布】HostClock 可能已经把 PendingAbsolute 探针走完
        // 并落成 Project；但预览还没确认，把宿主光标（现场 70.375 s）画出去就是
        // 用户看到的那次跳变。候选确认后由处理 pass 的 cursor 接管发布。
        if self.processing_pass.phase() == PreviewPhase::Candidate {
            // 候选块由 HostClock 按 raw / R2 记账写入（transport_active 已在
            // note_stream_geometry 里由“候选”这一事实打开）；位置在确认前不可发布。
            stream_block.position_trusted = false;
            stream_block.publish_deferred = true;
        }
        if stream_block.new_pass {
            self.clear_rung_latch();
        }
        // 新播放会话：允许同一区间再出音一次。
        if stream == ProcessStream::Realtime && (playing_rising || stream_block.new_epoch) {
            self.rt_last_emitted_valid.store(false, Ordering::Release);
            // 【为什么帧数也要清】起播第一块与停播最后一块可能同起点、同长度；
            // 只清 `valid` 已经足够，但把帧数一并归零让"下一次比较"不依赖旧值。
            self.rt_last_emitted_frames.store(0, Ordering::Release);
            // 【T3：新的播放会话从零开始】读位置也必须清：上一轮停在任意位置，
            // 新会话第一块的位置天然与它不同 —— 那不是读游标跳变，是一次新的起播。
            self.rt_last_read_position.store(0, Ordering::Release);
        }
        let split = self.block_split(context, block.frames, rate, stream_block.position);
        let cycle_start = self.cycle_start_position(context, rate, stream_block.position);
        let allow_output = self.allow_output_for(
            stream,
            &block,
            stream_block.position,
            stream_block.phase,
            playing_rising,
            recording_rising,
        );
        // 【为什么捕获不再需要外面的门】`on_block` 自己先读一次 `is_armed` 就早退，
        // 代价与这里再判一次相同；而少一层门就少一处"两个判据漂移"的机会。
        let mut outcome = crate::capture::CaptureBlockOutcome::Idle;
        for (index, part) in split.iter().enumerate() {
            let planes = [
                slice_plane(block.planes[0], part),
                slice_plane(block.planes[1], part),
            ];
            let sub = crate::capture::recorder::InputBlock {
                start: part.position,
                frames: part.frames,
                sample_rate: block.sample_rate,
                planes,
                playing,
            };
            let part_outcome = capture.on_block_with_context(
                sub,
                BlockContext {
                    stream,
                    capture_epoch: self.capture().capture_epoch(),
                    playing,
                    recording,
                    phase: stream_block.phase,
                    playing_rising,
                    recording_rising,
                    cycle_wrap: stream_block.cycle_wrap,
                    cycle_start,
                    new_pass: stream_block.new_pass,
                    transport_active: stream_block.transport_active,
                    start_window: stream_block.start_window,
                    epoch: stream_block.epoch,
                    new_epoch: index == 0 && stream_block.new_epoch,
                    rebase_frames: if index == 0 {
                        stream_block.rebase_frames
                    } else {
                        0
                    },
                    part: index.min(u8::MAX as usize) as u8,
                    advancing: stream_block.activity_moving,
                    resync: stream_block.resync,
                    resync_limit_frames: stream_block.resync_limit_frames,
                    // 循环切分的第二段不是起播起点：前导静音/钉住纠正只属于第一段。
                    preceding_silent_frames: if index == 0 {
                        stream_block.preceding_silent_frames
                    } else {
                        0
                    },
                    counted: index == 0 && stream_block.counted,
                    position_source: stream_block.position_source,
                    discard_pre_roll: index == 0 && stream_block.discard_pre_roll,
                    // 记账域位置：起播窗口里宿主位置被钉住，`part.position` 已经是插件
                    // 按块长推进后的值，与宿主管道域不同（见 `BlockContext` 的说明）。
                    raw_start: stream_block.raw_start,
                    position_trusted: stream_block.position_trusted,
                    publish_deferred: stream_block.publish_deferred,
                    prune_before: if index == 0 {
                        stream_block.prune_before
                    } else {
                        0
                    },
                    pass_sequence: self.pass_sequence.load(Ordering::Acquire),
                    transport_generation: stream_block.transport_generation,
                    preview: stream_block.preview,
                },
            );
            outcome = if index == 0 {
                part_outcome
            } else {
                combine_capture_outcomes(outcome, part_outcome)
            };
        }
        let _ = stream_block.timebase;
        // 【Capture / 离线本地时钟】Event FX 捕获时 kPlaying 可能全程为 0，
        // 而捕获仍在按块推进；把这条本地位置发布给 TransportClock，前端才能
        // 用 60 Hz 视觉播放头画带子。停播/未录制时 playing=false，不贡献播放位。
        if let Some(clock) = self.clock.get() {
            let recording = matches!(outcome, crate::capture::CaptureBlockOutcome::Capturing);
            if stream == ProcessStream::Prefetch {
                // prefetch 的位置是渲染进度，不是播放头；不能发布到 UI。
            } else if !self.capture.is_armed() && stream != ProcessStream::Offline {
                // 未武装的实时块不贡献 Capture 时钟；否则每个宿主块都会推进
                // sample_seq，前端会把空采样当成新读数。
            } else if !stream_block.position_trusted {
                // 【证据不足不发布】PendingAbsolute 期间宿主报的是绝对域预卷
                // （现场 60.000 s）；把它画出去就是用户看到的“光标跳到 60 s 再跳回”。
                // 保持上一位置，等域解析完成再发布。
                clock.hold_capture();
            } else {
                clock.publish_capture_block(
                    stream_block.position as f64 / rate.max(1.0),
                    recording || stream == ProcessStream::Offline || stream_block.preview,
                    stream == ProcessStream::Offline,
                    stream_block.preview,
                    self.capture().capture_epoch(),
                );
            }
        }
        self.note_block_trace(
            context.state_bits(),
            mode,
            stream,
            stream_block.raw_start,
            stream_block.position,
            block.frames,
            playing,
            stream_block.phase,
            playing_rising,
            stream_block.cycle_wrap,
            split.count,
            stream_block.had_previous,
            allow_output,
            outcome.code(),
            capture.capture_started(),
            stream_time,
            rate,
            stream_block.counted,
        );
        Some(BlockEvidence {
            stream,
            position: stream_block.position,
            frames: block.frames,
            playing,
            recording,
            playing_rising,
            recording_rising,
            phase: stream_block.phase,
            had_previous: stream_block.had_previous,
            allow_output,
            start_window: stream_block.start_window,
            transport_active: stream_block.transport_active,
            transport_paused: SessionPhase::from_code(
                self.streams[stream.index()]
                    .session_phase
                    .load(Ordering::Acquire),
            ) == Some(SessionPhase::Paused),
            repeated_nonzero: stream_block.repeated_nonzero,
            cycle_wrap: stream_block.cycle_wrap,
            rebase_frames: stream_block.rebase_frames,
            split,
            capture_epoch: self.capture().capture_epoch(),
            processing: processing.map_or(0, |block| block.kind.code()),
            position_trusted: stream_block.position_trusted,
            publish_deferred: stream_block.publish_deferred,
            prune_before: stream_block.prune_before,
            pass_sequence: self.pass_sequence.load(Ordering::Acquire),
            transport_generation: stream_block.transport_generation,
            preview: stream_block.preview,
        })
    }

    /// VST3 适配层：把宿主 `ProcessData` 变成中立块，再走同一条捕获逻辑。
    ///
    /// 【为什么留着它】它是 VST3 音频回调唯一认识的形状；真正的观察在
    /// [`Self::capture_observe_block`] 里。CLAP 侧从 `clap_audio_buffer_t` 直接造
    /// `InputBlock`，两边之后走的是同一段代码。
    pub(crate) fn capture_observe(
        &self,
        data: *mut crate::audio_abi::ProcessData,
        context: crate::block_context::BlockContext,
    ) -> Option<BlockEvidence> {
        // SAFETY: 调用方保证 data 在本回调期间可读。
        let mode = unsafe { (*data).process_mode };
        let stream = ProcessStream::from_mode(mode)?;
        let rate = if context.sample_rate.is_finite() && context.sample_rate > 0.0 {
            context.sample_rate
        } else {
            self.setup_sample_rate()
        };
        // 【为什么起点先按工程位置给】离线 pass 的本地起点由 `observe_input_block`
        // 算出来并覆盖（它必须先记账再观察）。这里传工程位置只是让非离线块有一个
        // 与 `capture_view` 契约一致的值。
        // SAFETY: 与 `audio_process_inner` 同一份宿主提供的 ProcessData；切片只在本次
        // 回调内有效，而消费方同步读完就返回。
        let block = unsafe {
            crate::audio_abi::capture_view(
                data,
                context.project_time_samples,
                rate,
                context.playing,
            )
        };
        self.observe_input_block(block, context, stream)
    }

    /// 把宿主音频喂进共享捕获路径；返回"本块位置是否可信"。
    ///
    /// 【为什么返回 bool 而不是证据结构】证据里的相位、切分与门控是 VST3 音频通路的
    /// 输入；别的格式要的只是"这一块的位置能不能当绝对位置用"。
    pub fn capture_block(
        &self,
        block: crate::capture::recorder::InputBlock<'_>,
        stream: ProcessStream,
        context: impl Into<crate::block_context::BlockContext>,
        effective_position: Option<i64>,
    ) -> bool {
        self.capture_observe_block(block, stream, context, effective_position)
            .is_some_and(|evidence| evidence.position_trusted)
    }

    /// 循环起点在工程采样域中的位置（`None` = 元数据不足）。
    ///
    /// 只用于把同一圈的多个段标记成同一组；不参与音频定位（音频定位由
    /// `block_split` 的逐段位置负责）。
    pub(super) fn cycle_start_position(
        &self,
        context: impl Into<crate::block_context::BlockContext>,
        rate: f64,
        position: i64,
    ) -> Option<i64> {
        let context = context.into();
        let time = StreamTime::from_context(&context);
        if !time.cycle_active || !time.cycle_valid {
            return None;
        }
        let (Some(tempo), Some(music)) = (time.tempo, time.music) else {
            return None;
        };
        let beats = music - time.cycle_start_music;
        let frames = (beats * 60.0 / tempo * rate).round() as i64;
        Some(position.saturating_sub(frames).max(0))
    }

    /// 计算循环边界切分。只有在 cycle 元数据完整、且本块确实横跨 loop 终点时才切。
    ///
    /// 【为什么用音乐时间算】`ProcessContext` 不提供“循环终点对应的采样位置”，但
    /// 给了 `projectTimeMusic` / `cycleEndMusic` / `tempo`。三者有效时，
    /// `(cycle_end − project_time_music) × 60/tempo × rate` 就是本块到终点的帧数。
    pub(super) fn block_split(
        &self,
        context: impl Into<crate::block_context::BlockContext>,
        frames: usize,
        rate: f64,
        position: i64,
    ) -> BlockSplit {
        let context = context.into();
        let time = StreamTime::from_context(&context);
        let (Some(tempo), Some(music)) = (time.tempo, time.music) else {
            return BlockSplit::whole(position, frames);
        };
        if !time.cycle_active || !time.cycle_valid {
            return BlockSplit::whole(position, frames);
        }
        let loop_beats = time.cycle_end_music - time.cycle_start_music;
        // 【为什么写成 `<= 0.0` 而不是 `!(x > 0.0)`】两者在有限数上等价，但否定的
        // 比较让"这里到底要不要 NaN"变得要读者自己想。这里要的就是"循环长度必须是正的"
        // 这一个陈述。
        if loop_beats <= 0.0 {
            return BlockSplit::whole(position, frames);
        }
        let beats_to_end = time.cycle_end_music - music;
        let to_end = (beats_to_end * 60.0 / tempo * rate).round();
        // 同上：直接陈述"到循环尾的距离必须是正的、且必须在本块之内"。
        if to_end <= 0.0 || to_end >= frames as f64 {
            return BlockSplit::whole(position, frames);
        }
        let length = (loop_beats * 60.0 / tempo * rate).round() as i64;
        if length <= 0 {
            return BlockSplit::whole(position, frames);
        }
        let to_end = to_end as usize;
        let wrapped = position
            .saturating_add(to_end as i64)
            .saturating_sub(length)
            .max(0);
        self.cycle_splits.fetch_add(1, Ordering::Relaxed);
        BlockSplit::split(
            SubBlock {
                position,
                input_offset: 0,
                frames: to_end,
            },
            SubBlock {
                position: wrapped,
                input_offset: to_end,
                frames: frames - to_end,
            },
        )
    }

    /// 本块是否允许写出内容。
    ///
    /// 【真值表】离线/prefetch：永远允许（它们的输出稍后才被播放，此刻的传输状态
    /// 与内容正确性无关）。实时：`kPlaying`、本流推进、或“这个区间还没出过音”
    /// 允许；同一区间在停播光标处重复到达则抑制（避免小段循环/嗡嗡声）。
    pub(super) fn allow_output_for(
        &self,
        stream: ProcessStream,
        block: &crate::capture::recorder::InputBlock<'_>,
        position: i64,
        phase: TransportPhase,
        playing_rising: bool,
        recording_rising: bool,
    ) -> bool {
        // 【为什么 bound 实例不参与这里的判据】它的"停播即静音"由
        // `capture_should_emit` 直接读 `kPlaying` 决定（见那里的说明：预取/offline
        // 恒真、实时流看播放位）。若这里也参与，`evidence.allow_output` 会在"停播
        // 光标处的重复块"上翻假，把 ARA 的既有语义改掉。
        if self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            return true;
        }
        // 【零帧块不参与门控】它是参数 flush，不代表时间轴在走；把它算进
        // "连续多少块没推进"会让紧随其后的真实块被吞掉 10–20 ms。
        if block.frames == 0 {
            return true;
        }
        match stream {
            ProcessStream::Offline => true,
            // prefetch 的输出稍后才被播放；重复请求可能是参数变化后的重算，
            // 抑制会留下旧结果。
            ProcessStream::Prefetch => true,
            ProcessStream::Realtime => {
                if playing_rising || recording_rising {
                    self.gate_resting_run.store(0, Ordering::Release);
                    return true;
                }
                // 【重复的判据只有"起点相同"】块长**不参与**：宿主的块长会变（实测
                // VEGAS 在 1200/848 之间交替），而同一段内容被重复投递时起点不变。
                // 把块长拌进来，重复抑制在块长交替的宿主上就完全失效 —— 停播时每块
                // 都被当成新内容放行，用户听到的是固定光标处那一小段在循环。
                let same = self.rt_last_emitted_valid.load(Ordering::Acquire)
                    && self.rt_last_emitted_start.load(Ordering::Acquire) == position;
                let quiet = phase == TransportPhase::Resting || same;
                if !quiet {
                    self.gate_resting_run.store(0, Ordering::Release);
                    return true;
                }
                // 【T20：单块抖动不静音】见 `gate_resting_run` 的字段说明。
                //
                // 【为什么不用 `kPlaying` 当分界】试过，会把 FL Studio 的回归打破：
                // 那边停播时 `kPlaying` 常置 1，而同一位置的重复块**必须**被淡出，
                // 否则固定光标处同一小段会被反复重写（用户听到嗡嗡声）。因此判据
                // 只能是"连续多少块不推进"——`a_repeated_realtime_block_is_silenced`
                // 与 `a_pinned_block_with_playing_one_is_silenced` 两条用例把这个
                // 契约钉在两侧：单块抖动必须出声，连续不推进必须淡出。
                let run = self.gate_resting_run.fetch_add(1, Ordering::AcqRel) + 1;
                if run < GATE_SETTLE_BLOCKS {
                    return true;
                }
                self.repeats_suppressed.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// 记录“本块确实写出了内容”（由 [`Self::capture_stage`] 在写入分支调用）。
    pub(super) fn note_emitted(&self, evidence: Option<BlockEvidence>) {
        let Some(evidence) = evidence else {
            return;
        };
        if evidence.stream != ProcessStream::Realtime
            || self.is_bound()
            || self.bound_once.load(Ordering::Acquire)
        {
            return;
        }
        // 【T3：读位置单独记一份】输出侧按 `position` 读快照；跳变就是读游标阶跃。
        // 【为什么连帧数一起记】下一块的"预期续接"是 `位置 + 帧数`；只记位置会把
        // 每一次正常推进都算成跳变（见 `read_position_jump`）。
        self.rt_last_read_position
            .store(evidence.position, Ordering::Release);
        self.rt_last_read_frames.store(
            u32::try_from(evidence.frames).unwrap_or(u32::MAX),
            Ordering::Release,
        );
        self.rt_last_read_pass.store(
            self.streams[ProcessStream::Realtime.index()]
                .pass
                .load(Ordering::Acquire),
            Ordering::Release,
        );
        self.rt_last_emitted_start
            .store(evidence.position, Ordering::Release);
        self.rt_last_emitted_frames.store(
            u32::try_from(evidence.frames).unwrap_or(u32::MAX),
            Ordering::Release,
        );
        self.rt_last_emitted_valid.store(true, Ordering::Release);
    }

    /// 测试专用：直接写"上一块读到哪个位置"（T3 判据的唯一输入）。
    #[cfg(test)]
    pub(crate) fn note_read_position_for_test(&self, position: i64) {
        self.rt_last_read_position
            .store(position, Ordering::Release);
    }

    /// 测试专用：写"上一块读了多少帧"（预期续接的第二个输入）。
    #[cfg(test)]
    pub(crate) fn note_read_frames_for_test(&self, frames: u32) {
        self.rt_last_read_frames.store(frames, Ordering::Release);
    }
    /// 测试专用：标记"上一块写出过内容"（T3 判据的第二个输入）。
    #[cfg(test)]
    pub(crate) fn mark_emitted_for_test(&self) {
        self.rt_last_emitted_valid.store(true, Ordering::Release);
    }
    /// 【T3】这一块的读位置与上一块差了多少帧（`None` = 不存在跳变）。
    ///
    /// 四条纪律（都是纯函数，可单测）：
    /// 1. **没有上一块就不存在跳变**：`rt_last_read_position <= 0` 或上一块没写出过
    ///    内容 ⇒ `None`。否则会话刚开始的两次不同位置会被当成跳变。
    /// 2. **跨 pass 不是跳变**：那是"重新开始"，由起播接缝处理。
    /// 3. **位置相同不是跳变**。
    /// 4. **小于一个淡化带的差被淡化吸收**：那种场合读快照的那一小段与上一块只差几帧，
    ///    64 帧的交叉淡化足够；再淡一次反而多一次能量塌陷。
    pub(super) fn read_position_jump(&self, evidence: Option<BlockEvidence>) -> Option<u64> {
        let evidence = evidence?;
        // 【循环回卷不是跳变】回卷让块首回到循环起点，而本函数的判据是"本块起点与
        // 上一块终点的差"。不排除它，每一圈都会多装一次接缝淡化（每圈开头一下咔）。
        if evidence.cycle_wrap || evidence.rebase_frames != 0 {
            // 循环回卷与时间基切换都让块首变化，但音频是连续的；那不是读游标跳变。
            return None;
        }
        // 【不可信的位置不参与判据】PendingAbsolute 的预卷块位置可能仍在绝对域，
        // 拿它和上一块比只会制造一次假跳变。预览 cursor 是另一条本地时间轴，
        // 与实时流的读游标无关。
        if !evidence.position_trusted || evidence.preview {
            return None;
        }
        // 【起播窗口里的位置纠正不是 seek】宿主可能在 pre-roll 里先报一个陈旧位置，
        // 再回到真实播放头；那不是读游标跳变，不该装一次起播淡入。
        if evidence.start_window {
            return None;
        }
        let previous = self.rt_last_read_position.load(Ordering::Acquire);
        let previous_frames = i64::from(self.rt_last_read_frames.load(Ordering::Acquire));
        if previous <= 0 || !self.rt_last_emitted_valid.load(Ordering::Acquire) {
            return None;
        }
        // 【跨 pass 不是跳变】pass 边界是"重新开始"，不是"读游标跳了"。上一块输出的
        // 尾巴属于另一个位置的内容，拿它做交叉淡化只会叠出两段不相干信号。
        // 起播的接缝由 `Seam::FirstBlock` / `Seam::FadeIn` 在种类变化时处理。
        let pass = self.streams[ProcessStream::Realtime.index()]
            .pass
            .load(Ordering::Acquire);
        if pass != self.rt_last_read_pass.load(Ordering::Acquire) {
            return None;
        }
        // 【核心判据：预期续接，而不是"上一块起点"】上一块读了 `previous_frames` 帧，
        // 那么本块的起点**应当**是 `previous + previous_frames`。旧实现直接拿本块起点
        // 减上一块起点，于是任何连续播放（差值 = 块长 > 64）都被判成跳变 ⇒ 整块直通
        // 宿主原声 ⇒ 每一块都在干声与合成结果之间硬接一次。那就是"咔"。
        let expected = previous.saturating_add(previous_frames);
        let jump = evidence.position.saturating_sub(expected).unsigned_abs();
        (jump > u64::try_from(OUTPUT_DECLICK_FRAMES).unwrap_or(u64::MAX)).then_some(jump)
    }
    /// 诊断（基准测试 / 预热）进行中时，**实时**输出必须静音。
    ///
    /// 【为什么必须静音】诊断与实时音频线程抢同一批核（ORT 的 CPU EP 默认自旋
    /// 等待），线程预算只能降低概率，给不出"一定不出声"的保证。与独立 App 的音频
    /// 回调同一判据（`hifishifter_kernel::diagnostics_busy`，`src-tauri` 的 `mix.rs`
    /// 有同款注释）—— 此前只有 App 侧消费它，模块文档承诺的插件读数并不存在。
    ///
    /// 【为什么只压实时流】导出/bounce 的离线渲染要的是**产物**；诊断静音只对
    /// "现在正在听"负责，把离线产物也写成静音是最难发现的一类错误。
    ///
    /// 【为什么不动采集与时钟】`capture_observe` 是独立阶段照常录制，位置照常推进
    /// —— 静音的含义是"这段时间的输出是零"，不是"这块没发生过"。
    pub(super) fn diagnostic_silence_applies(evidence: Option<BlockEvidence>) -> bool {
        matches!(evidence, Some(evidence) if evidence.stream == ProcessStream::Realtime)
            && hifishifter_kernel::diagnostics_busy::active()
    }

    /// 音频线程上的 Capture **输出**阶段。
    ///
    /// 【为什么未录制时要早退而不是统一处理】本函数在**每个音频块**上被调用。
    /// 未录制时它只读几个原子就返回，代价与"没写这段代码"相同 —— 这是它能被放在
    /// 所有既有逻辑之前的前提。
    ///
    /// 【为什么录制中不走快照输出】正在录的那一段，插件手上还没有对应的素材；
    /// 输出上一次的快照会让用户听到**旧内容盖在新演奏上**。所以录制中按选项输出
    /// 直通（默认，让用户听到宿主原声）或静音。
    ///
    /// 【为什么捕获不在这里】见 [`Self::capture_observe`]：它必须发生在任何输出形状
    /// 判定之前，否则"宿主没接输出"会让录制一起失效。
    ///
    /// # 诊断静音
    ///
    /// 基准测试 / 预热进行中时，实时输出由 [`Self::diagnostic_silence_applies`] 压成
    /// 静音（见该判据的说明）—— 这是模块文档承诺的"诊断期间音频通路静音"在插件侧
    /// 的落点，与独立 App 音频回调同判据。
    /// # Safety
    /// `block` 的平面由宿主提供，非空平面至少含 `frames` 帧；本函数只在本回调内使用
    /// 它们，不保留任何指针。
    pub(crate) unsafe fn capture_stage(
        &self,
        block: &mut crate::audio_block::AudioBlock<'_>,
        evidence: Option<BlockEvidence>,
    ) -> CaptureStage {
        let capture = self.capture();
        let offline = block.is_offline() || self.processing_pass.is_offline();
        // 选项是无锁快照（见 `CaptureStore::options`）。
        let options = capture.options();
        // 去咔状态按进程流隔离；块长取宿主本回调声明的帧数。
        let stream = evidence.map_or(ProcessStream::Realtime, |evidence| evidence.stream);
        let frames = block.frames;
        // 【为什么先仲裁、再做防御性清空】录制中的输出是**宿主正在送进来的原声**
        // （用户监听自己），它不该被"停播重复抑制"影响 —— 那个判据描述的是"同一段
        // 内容又来了"，而录制中来的每一块都是要录的新内容。此前防御清空排在仲裁之前，
        // 于是一次误判就能把监听整块抹掉。
        let output = self.capture_output_with(evidence);
        if output == CaptureOutput::Recording {
            // 录制中不锁存：录完之后的那一遍必须重新仲裁内容源。
            self.rung_locked.store(0, Ordering::Release);
        }
        // 【T3：读游标跳变 ⇒ 这一块做一次去咔】种类与代次都没变，但这一块读的位置与
        // 上一块不同（pass 边界、宿主纠正记账、流交接）。那时读快照等于把**另一段**
        // 内容写进总线 —— 一次单样本阶跃。判据在 `read_position_jump`，纯函数、可单测。
        //
        // 【为什么不再"整块直通"】直通本身就是**换了一个信号源**：一个块最长可达 6 s，
        // 把它整块换成宿主原声，前后各是一次硬切换 —— 用户听到的那一下"咔"其实是这个
        // 保护动作自己（代码自己的注释早就这么写过）。现在位置只有一个权威
        // （`note_stream_geometry` 的 R1–R4），跳变只可能发生在 pass 边界，而 pass
        // 边界本来就要重新开始，一次交叉淡化就够。
        if let Some(jump) = self.read_position_jump(evidence) {
            self.position_corrections.fetch_add(1, Ordering::Relaxed);
            self.max_correction_frames
                .fetch_max(jump, Ordering::Relaxed);
            self.force_declick[stream.index()].store(true, Ordering::Release);
        }
        self.output_token[stream.index()].store(self.output_token_for(output), Ordering::Release);
        self.note_output_rung(output);
        // 【防御】门控已经在 `vst3.rs` 里判过一次；这里再判一次实时流，避免未来
        // 改动绕过门控时把停播重复块写出去。ARA 实例保持原语义（不参与）。
        if output != CaptureOutput::Recording {
            if let Some(evidence) = evidence {
                if evidence.stream == ProcessStream::Realtime
                    && !Self::emit_allowed(Some(evidence))
                    && !self.is_bound()
                    && !self.bound_once.load(Ordering::Acquire)
                {
                    // 从上一块尾巴淡到零。本函数写满整块，**不能**再调 `clear_outputs`
                    // —— 那会把刚写好的淡出一起抹掉（见其文档）。
                    self.apply_gate_fade_out(stream, &mut block.output, frames);
                    return CaptureStage::Handled;
                }
            }
        }
        // 【诊断期间必须静音】见 [`Self::diagnostic_silence_applies`]。同一套淡出
        // 写法：淡到零并写满整块，不用再清（也**不能**再清）。
        if Self::diagnostic_silence_applies(evidence) {
            self.apply_gate_fade_out(stream, &mut block.output, frames);
            return CaptureStage::Handled;
        }
        match output {
            // 不介入：与改动前的 `return false` 完全等价，继续走既有快照通路。
            //
            // 【为什么快照没就绪时要 layered】这是 ARA 实例（`is_bound()` / `bound_once`）
            // 唯一会走到的输出分支。快照被 `clear_renderers()` / `stop_editor()` /
            // `document_closed()` 清空后，`copy_block_at` 的读闭包逐样本返回 `0.0`，
            // 而 `layered == false` 时 `audio_process_inner` 既不预清也不兜底 ——
            // 于是每个音频块都是纯零，用户听到的就是"有概率完全没声音"。
            // layered 让 `copy_block_layered_at` 在快照缺失处回退取宿主输入
            // （`snapshot.rs` 的 `input_at`），至少保住宿主原声而不是静音。
            CaptureOutput::Untouched => {
                let ready = self.processed_snapshot_ready();
                self.note_bound_snapshot_fallback(!ready);
                CaptureStage::Snapshot {
                    source: SnapshotSource::Processed,
                    layered: !ready,
                }
            }
            CaptureOutput::Recording => {
                // 【为什么离线那一趟不监听】离线渲染（bounce / 导出）若只有一趟，直通会把
                // **输入原声**写进产物 —— 而用户会把它当成"处理后的结果"，那是最难发现的一类
                // 错误。静音是诚实的：这一趟的唯一作用是把素材拿进来；第二趟（解除武装后再
                // 导出）才是结果。实时播放则照旧监听，因为用户要听到自己正在录的东西。
                if options.monitor && !offline {
                    // SAFETY: 与上面同一份数据；`pass_through` 自己会校验形状。
                    let _ = unsafe { block.pass_through() };
                    self.note_emitted(evidence);
                } else {
                    // 【为什么"不监听"必须显式清空】本分支**整块都不写**，而调用方拿到
                    // `Handled` 就立刻返回（见 `audio_process_inner`）—— 于是输出缓冲里
                    // 留着上一次的内容：in-place 宿主下那是**输入原声**（关掉监听反而
                    // 听得见），非 in-place 宿主下是未定义字节。本文件头那条纪律正是
                    // "只有本通路确实不写整块时才清"，这里就是那种情形。
                    block.clear_output();
                }
                self.apply_output_declick(stream, &mut block.output, frames);
                CaptureStage::Handled
            }
            // 【为什么区间外要直通】只录了 10 秒的素材不该把整条轨变成静音：快照覆盖的
            // 那段被替换成处理结果，其余保持宿主原声。见 `copy_block_layered`。
            CaptureOutput::Processed => {
                self.note_emitted(evidence);
                CaptureStage::Snapshot {
                    source: SnapshotSource::Processed,
                    layered: options.outside_passthrough,
                }
            }
            CaptureOutput::Raw => {
                self.note_emitted(evidence);
                CaptureStage::Snapshot {
                    source: SnapshotSource::Raw,
                    layered: options.outside_passthrough,
                }
            }
            // 【为什么与 `PassThrough` 输出相同】素材在，但还没有任何可播的版本；
            // 唯一诚实的输出仍是宿主原声（静音会让整条轨"哑掉"）。区别只在**读数**：
            // 这一档让界面能说"素材已就绪，等待合成"，从而把"合成没跑"暴露出来。
            // `capture_output` 已置好 `render_wanted`，actor 会补排一次合成。
            CaptureOutput::PendingSynthesis | CaptureOutput::PassThrough => {
                // SAFETY: 与上面同一份数据；`pass_through` 自己会校验形状。
                let _ = unsafe { block.pass_through() };
                self.note_emitted(evidence);
                self.apply_output_declick(stream, &mut block.output, frames);
                CaptureStage::Handled
            }
        }
    }
}
/// HostClock 域 -> 诊断时基编码。
fn timebase_of(domain: ClockDomain) -> TimebaseKind {
    match domain {
        ClockDomain::Unknown => TimebaseKind::Unknown,
        ClockDomain::Project => TimebaseKind::ProjectAbsolute,
        ClockDomain::PendingAbsolute => TimebaseKind::Unknown,
        ClockDomain::Event {
            raw_domain: super::host_clock::RawDomain::Absolute,
            ..
        } => TimebaseKind::EventAbsolute,
        ClockDomain::Event {
            raw_domain: super::host_clock::RawDomain::Relative,
            ..
        } => TimebaseKind::EventRelative,
    }
}

/// 取宿主缓冲中属于某个循环切分段的那一片。
pub(super) fn slice_plane<'a>(plane: Option<&'a [f32]>, part: &SubBlock) -> Option<&'a [f32]> {
    plane.and_then(|samples| {
        let end = part.input_offset.checked_add(part.frames)?;
        samples.get(part.input_offset..end)
    })
}

/// 合并循环切分两段的捕获结论：只要有一段真的写了，就报 `Capturing`。
pub(super) fn combine_capture_outcomes(
    first: crate::capture::CaptureBlockOutcome,
    second: crate::capture::CaptureBlockOutcome,
) -> crate::capture::CaptureBlockOutcome {
    use crate::capture::CaptureBlockOutcome;
    if first == CaptureBlockOutcome::Capturing || second == CaptureBlockOutcome::Capturing {
        return CaptureBlockOutcome::Capturing;
    }
    if first != CaptureBlockOutcome::Idle {
        first
    } else {
        second
    }
}
