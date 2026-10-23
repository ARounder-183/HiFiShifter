//! 诊断读数：块级环、缓冲审计与点击报告。
use super::*;
impl ExtensionOwner {
    /// 预分配块级诊断环（非音频线程；幂等）。
    pub(crate) fn init_block_trace(&self) {
        let _ = self
            .block_trace
            .get_or_init(|| Box::new(BlockTraceRing::new()));
    }

    /// 记录 `setupProcessing` 声明的单块上限（诊断用）。
    pub(crate) fn note_max_samples_per_block(&self, value: i32) {
        self.max_samples_per_block
            .store(value.max(0), Ordering::Release);
    }

    /// 记一次**输出写满审计**（`HIFISHIFTER_BUFFER_AUDIT=1` 时由 vst3 层调用）。
    ///
    /// 【它回答什么】"这一块有没有哪一帧根本没被写过"。立体声下每条分支都写满整块，
    /// 所以这个数应当恒为 0；非零只有三种来源：>2 声道的总线（`pass_through` 只写前
    /// 两个平面）、短块、或者某条新分支漏写。三种都是"用户听到上一次的内容"。
    ///
    /// 【为什么按档位去重打印】它由音频线程记原子，格式化留给 actor；一次播放里同一档位
    /// 只报一次，避免每块一行。
    pub(crate) fn note_buffer_audit(&self, channels: usize, unwritten: usize, frames: usize) {
        self.buffer_audit_channels
            .store(channels as u32, Ordering::Relaxed);
        self.buffer_audit_frames
            .store(frames as u32, Ordering::Relaxed);
        self.buffer_audit_unwritten
            .fetch_max(unwritten as u64, Ordering::Relaxed);
    }

    /// 把写满审计的结论打成一行（非实时线程）。
    pub(crate) fn log_buffer_audit(&self) {
        if !self.buffer_audit_active.swap(true, Ordering::AcqRel) {
            // 第一次调用只置位：没有读数可报。
            return;
        }
        let unwritten = self.buffer_audit_unwritten.load(Ordering::Acquire);
        let line = format!(
            "[buf] audit: channels={} frames={} unwritten={unwritten} output={}",
            self.buffer_audit_channels.load(Ordering::Acquire),
            self.buffer_audit_frames.load(Ordering::Acquire),
            self.capture_output().as_str(),
        );
        let mut last = self.last_buffer_diag.lock_or_recover();
        if last.as_deref() != Some(line.as_str()) {
            *last = Some(line.clone());
            if unwritten > 0 {
                log::warn!("{line}");
            } else {
                log::info!("{line}");
            }
        }
    }

    /// 门控清掉一块（诊断用）。按流分开计数：prefetch/offline 的清零必须是 0。
    pub(crate) fn note_gate_cleared(&self, stream: ProcessStream) {
        self.gate_cleared.fetch_add(1, Ordering::Relaxed);
        self.stream_gate_cleared[stream.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// ARA 实例因快照未就绪而退回宿主原声（见 `capture_stage` 的 `Untouched` 分支）。
    pub(crate) fn note_bound_snapshot_fallback(&self, fell_back: bool) {
        if fell_back {
            self.bound_snapshot_fallback.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 本块最终走了哪一档输出仲裁（由 `capture_stage` 在决定后调用）。
    ///
    /// 【I-2 的读数在这里攒】"一次播放只有一个内容源"是可以直接数出来的：只要
    /// 相邻两块之间的档位不同，那一次切换就可能在听感上表现为一次咔。音频线程只做
    /// 一次整数比较与两次原子写；报告留给非实时线程（见 `log_capture_report`）。
    pub(crate) fn note_output_rung(&self, output: CaptureOutput) {
        if let Some(ring) = self.block_trace.get() {
            ring.update_last_rung(output.code());
        }
        let code = u32::from(output.code());
        let previous = self.rung_previous.swap(code, Ordering::AcqRel);
        // `0` 是 `Untouched` 的编码，也是"从未记录"的初值 —— 用独立的 valid 位区分，
        // 否则首块会被算成一次切换（与 `apply_output_declick` 里 `primed` 的教训相同）。
        if !self.rung_valid.swap(true, Ordering::AcqRel) {
            return;
        }
        if previous != code {
            self.rung_switches.fetch_add(1, Ordering::Relaxed);
            self.rung_last_from.store(previous, Ordering::Release);
            self.rung_last_to.store(code, Ordering::Release);
            // 起播/换 take 会让"这一次播放"重新开始计数：切换数归零由非实时线程
            // 依据 `pass_epoch` 完成（音频线程不能分配、也不该做字符串格式化）。
            self.pass_switches.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 取走“该落一次块级 trace”的意图（非音频线程）。
    pub(crate) fn take_trace_wanted(&self) -> bool {
        let wanted = self.trace_wanted.swap(false, Ordering::AcqRel);
        if wanted {
            self.trace_logged.store(true, Ordering::Release);
        }
        wanted
    }

    /// 音频线程：计数并写入一条块级记录。无锁、无分配。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn note_block_trace(
        &self,
        state: u32,
        mode: i32,
        stream: ProcessStream,
        raw_start: i64,
        position: i64,
        frames: usize,
        playing: bool,
        phase: TransportPhase,
        playing_rising: bool,
        cycle_wrap: bool,
        split: u8,
        had_previous: bool,
        allow_output: bool,
        capture: u8,
        capture_started: bool,
        time: StreamTime,
        rate: f64,
        counted: bool,
    ) {
        self.blocks_seen.fetch_add(1, Ordering::Relaxed);
        self.stream_blocks[stream.index()].fetch_add(1, Ordering::Relaxed);
        if playing {
            self.blocks_playing.fetch_add(1, Ordering::Relaxed);
            self.stream_playing_blocks[stream.index()].fetch_add(1, Ordering::Relaxed);
        }
        let flow = phase != TransportPhase::Resting;
        if flow && !playing {
            self.blocks_flow_without_playing
                .fetch_add(1, Ordering::Relaxed);
            self.stream_flow_without_playing[stream.index()].fetch_add(1, Ordering::Relaxed);
        }
        // 只在确实与 Capture 有关的实例上留证，避免给纯 ARA 实例刷日志。
        // 同一次可疑窗口只报一次：`trace_logged` 由 actor 在 dump 后置位，
        // 直到宿主回到“钉住/停播”才复位，下一次起播再报。
        let suspicious = (!playing && flow) || raw_start < 0;
        if (self.capture.is_armed() || self.capture.has_material())
            && suspicious
            && !self.trace_logged.load(Ordering::Relaxed)
        {
            self.trace_wanted.store(true, Ordering::Release);
        }
        if !playing && !flow && (self.capture.is_armed() || self.capture.has_material()) {
            self.trace_logged.store(false, Ordering::Release);
        }
        if let Some(ring) = self.block_trace.get() {
            let source = self
                .streams
                .get(stream.index())
                .and_then(|geometry| {
                    PositionSource::from_code(geometry.last_source.load(Ordering::Relaxed))
                })
                .unwrap_or(PositionSource::Project);
            ring.write(
                state,
                mode,
                stream,
                raw_start,
                position,
                source,
                frames,
                playing,
                playing_rising,
                phase,
                cycle_wrap,
                split,
                had_previous,
                allow_output,
                capture,
                capture_started,
                time,
                rate,
                counted,
            );
        }
    }

    /// 块级诊断的 JSON（非音频线程）。`trace_limit` 控制环形记录条数。
    pub(crate) fn block_diagnostics_json(&self, trace_limit: usize) -> serde_json::Value {
        let capture = self.capture.capture_diagnostics().map(|capture| {
            serde_json::json!({
                "preroll_frames_dropped": capture.preroll_frames_dropped,
                "pending_stashed": capture.pending_stashed,
                "pending_committed": capture.pending_committed,
                "pending_discarded": capture.pending_discarded,
                "pending_ready": capture.pending_ready,
                "pending_available": capture.pending_available,
                "gaps": capture.gaps,
                "backwards": capture.backwards,
                "dropouts": capture.dropouts,
                "oversized_blocks": capture.oversized_blocks,
                "silent_blocks": capture.silent_blocks,
                "spares": capture.spared,
                "elected": capture.elected.map(|stream| stream.as_str()),
                "stream_switches": capture.stream_switches,
                "repeated_blocks": capture.repeated_blocks,
                "overlapped_frames": capture.overlapped_frames,
                "resting_blocks": capture.resting_blocks,
                "resting_frames": capture.resting_frames,
                "overwritten_blocks": capture.overwritten_blocks,
                "overwritten_frames": capture.overwritten_frames,
                "overhang_trimmed_frames": capture.overhang_trimmed_frames,
                "cycle_splits": capture.cycle_splits,
                "resync_bridged_frames": capture.resync_bridged_frames,
                // 起播窗口整段都没写下、由 resync 兜底补出一个段的次数。
                // T1 让窗口内记账连续后这个数应当长期为 0；只要它 > 0，说明暂存/
                // 提交链路上仍有漏，但 T2 保证了时间线不断。
                "bridged_window_blocks": capture.bridged_window_blocks,
                // 起播前导静音：捕获起点晚于播放起点时，这一段被合成进段首。
                "pre_roll_frames": capture.pre_roll_frames,
                // T19：旁路队列。竞争时暂存、下一次进锁时补写；`dropped > 0` 说明
                // 槽全满而竞争又在持续 —— 那时快照里会有洞。
                "bypass_pending": self.capture.bypass_pending(),
            })
        });
        let stream = |index: usize| {
            serde_json::json!({
                "blocks": self.stream_blocks[index].load(Ordering::Acquire),
                "playing": self.stream_playing_blocks[index].load(Ordering::Acquire),
                "flow_without_playing": self.stream_flow_without_playing[index]
                    .load(Ordering::Acquire),
                "gate_cleared": self.stream_gate_cleared[index].load(Ordering::Acquire),
                "resting": self.stream_resting_blocks[index].load(Ordering::Acquire),
                "rolling": self.stream_rolling_blocks[index].load(Ordering::Acquire),
                // 【位置记账的四个数】`counted` = 位置由计数器推进的块数；
                // `jumps` = 宿主报的位置与计数器不同的次数；`rewinds` = 位置倒退的
                // 块数（稳态恒为 0）；`passes` = 这一流经历过几次"新一次播放"。
                "counted": self.streams[index]
                    .counted_blocks
                    .load(Ordering::Acquire),
                "counted_frames": self.streams[index]
                    .counted_frames
                    .load(Ordering::Acquire),
                "jumps": self.streams[index].jumps.load(Ordering::Acquire),
                "last_jump": self.streams[index].last_jump.load(Ordering::Acquire),
                "rewinds": self.streams[index].clock_rewinds.load(Ordering::Acquire),
                "passes": self.streams[index].pass.load(Ordering::Acquire),
            })
        };
        // 【位置记账的汇总读数】三条流加起来：位置由计数器推进的块数、宿主纠正的次数、
        // 以及"这一流经历过几次新一次播放"。
        let (counted_blocks, jumps, passes) =
            self.streams
                .iter()
                .fold((0u64, 0u64, 0u64), |(counted, jumps, passes), geometry| {
                    (
                        counted.saturating_add(geometry.counted_blocks.load(Ordering::Acquire)),
                        jumps.saturating_add(geometry.jumps.load(Ordering::Acquire)),
                        passes.saturating_add(geometry.pass.load(Ordering::Acquire)),
                    )
                });
        serde_json::json!({
            "blocks_seen": self.blocks_seen.load(Ordering::Acquire),
            "blocks_playing": self.blocks_playing.load(Ordering::Acquire),
            "blocks_flow_without_playing": self.blocks_flow_without_playing.load(Ordering::Acquire),
            "gate_cleared": self.gate_cleared.load(Ordering::Acquire),
            "bound_snapshot_fallback": self.bound_snapshot_fallback.load(Ordering::Acquire),
            "continuous_used": self.continuous_used.load(Ordering::Acquire),
            "music_used": self.music_used.load(Ordering::Acquire),
            "counted_blocks": counted_blocks,
            "jumps": jumps,
            "passes": passes,
            "position_resyncs": self.position_resyncs.load(Ordering::Acquire),
            "bound_fallback": {
                "snapshot_missing_blocks": self.bound_snapshot_fallback.load(Ordering::Acquire),
                "promoted_to_renderer": self
                    .promoted_to_renderer
                    .load(Ordering::Acquire),
                "rebind_count": self.rebind_generation.load(Ordering::Acquire),
                // companion 的一次性绑定被复位重建了几次。重开工程正常是 1；
                // 持续增长说明宿主在反复重绑。
                "companion_resets": self.companion_reset_count(),
            },
            // 【为什么恢复读数要进诊断包】恢复失败此前只有一行 `log::warn!`，
            // 用户无法回答"参数到底读回来没有"。这里的三个字段就是那个答案：
            // 0 = 从未尝试、1 = 已应用、2 = 待文档就绪/重试中、3 = 失败。
            "restore": {
                "status": self.restore_status.load(Ordering::Acquire),
                "error": self.restore_error.lock_or_recover().clone(),
                "applied_generation": self
                    .restore_applied_generation
                    .load(Ordering::Relaxed),
            },
            "clock": {
                "counted_blocks": counted_blocks,
                "jumps": jumps,
                "passes": passes,
            },
            "cycle_wraps": self.cycle_wraps.load(Ordering::Acquire),
            "cycle_wrap_unknown": self.cycle_wrap_unknown.load(Ordering::Acquire),
            "cycle_splits": self.cycle_splits.load(Ordering::Acquire),
            "streams": {
                "realtime": stream(ProcessStream::Realtime.index()),
                "prefetch": stream(ProcessStream::Prefetch.index()),
                "offline": stream(ProcessStream::Offline.index()),
            },
            "repeats_suppressed": self.repeats_suppressed.load(Ordering::Acquire),
        // T3：读游标纠正。跳变幅度是"起播 0.4~0.7 s 一次咔"最直接的量。
        "position_corrections": self.position_corrections.load(Ordering::Acquire),
        "max_correction_frames": self.max_correction_frames.load(Ordering::Acquire),
            "max_samples_per_block": self.max_samples_per_block.load(Ordering::Acquire),
            "capture": capture,
            "trace": self
                .block_trace
                .get()
                .map(|ring| ring.snapshot(trace_limit))
                .unwrap_or_default(),
        })
    }
}
