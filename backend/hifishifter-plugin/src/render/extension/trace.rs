//! 块级诊断环：最近若干块的 seqlock 快照。
use super::*;
/// 块级诊断环的容量。256 块在 48 kHz/512 帧下约 2.7 秒的实时历史。
pub(super) const BLOCK_TRACE_CAPACITY: usize = 256;

/// 一条块级记录。字段全部是原子，读者用 seqlock 校验一致性。
#[derive(Default)]
pub(super) struct BlockTraceRecord {
    pub(super) seq: AtomicU64,
    pub(super) state: AtomicU32,
    pub(super) mode: AtomicI32,
    pub(super) stream: AtomicU8,
    pub(super) start: AtomicI64,
    pub(super) position: AtomicI64,
    pub(super) position_source: AtomicU8,
    pub(super) frames: AtomicU32,
    pub(super) playing: AtomicBool,
    pub(super) playing_rising: AtomicBool,
    pub(super) phase: AtomicU8,
    pub(super) flow: AtomicBool,
    pub(super) cycle_wrap: AtomicBool,
    pub(super) split: AtomicU8,
    pub(super) had_previous: AtomicBool,
    pub(super) emitted: AtomicBool,
    pub(super) capture: AtomicU8,
    pub(super) capture_started: AtomicBool,
    pub(super) rung: AtomicU8,
    // ── 宿主时间字段（`[capture] ctx` 用它回答"宿主到底给了什么"）──
    pub(super) continuous: AtomicI64,
    pub(super) continuous_valid: AtomicBool,
    pub(super) music_bits: AtomicU64,
    pub(super) music_valid: AtomicBool,
    /// 【为什么要 tempo】`[capture] ctx` 只打 `music=` 时，读者无法判断
    /// "Music 分支为什么没生效"：那一条要求 `music` 与 `tempo` **同时**有效。
    /// 现场日志里 `music=841.5(valid)` 连着几块不变，而 `counted=1` —— 只有
    /// 把 tempo 也打出来，才能一眼区分"宿主没给 tempo"与"music 冻结了"。
    pub(super) tempo_bits: AtomicU64,
    pub(super) tempo_valid: AtomicBool,
    /// `0` = 非循环；`1` = 循环且元数据完整；`2` = 循环但元数据不完整。
    pub(super) cycle: AtomicU8,
    pub(super) rate_bits: AtomicU64,
    /// 本块位置是否由**计数器**推进（见 `StreamBlock::counted`）。
    pub(super) counted: AtomicBool,
}

/// 预分配的块级环形记录（SPSC：音频线程写、actor/诊断读）。
pub(super) struct BlockTraceRing {
    pub(super) head: AtomicU64,
    pub(super) records: Vec<BlockTraceRecord>,
}

impl BlockTraceRing {
    pub(super) fn new() -> Self {
        Self {
            head: AtomicU64::new(0),
            records: (0..BLOCK_TRACE_CAPACITY)
                .map(|_| BlockTraceRecord::default())
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn write(
        &self,
        state: u32,
        mode: i32,
        stream: ProcessStream,
        raw_start: i64,
        position: i64,
        position_source: PositionSource,
        frames: usize,
        playing: bool,
        playing_rising: bool,
        phase: TransportPhase,
        cycle_wrap: bool,
        split: u8,
        had_previous: bool,
        emitted: bool,
        capture: u8,
        capture_started: bool,
        time: StreamTime,
        rate: f64,
        counted: bool,
    ) {
        let index = self.head.fetch_add(1, Ordering::Relaxed) as usize % self.records.len();
        let record = &self.records[index];
        record.seq.fetch_add(1, Ordering::Relaxed);
        record.state.store(state, Ordering::Relaxed);
        record.mode.store(mode, Ordering::Relaxed);
        record.stream.store(stream.code(), Ordering::Relaxed);
        record.start.store(raw_start, Ordering::Relaxed);
        record.position.store(position, Ordering::Relaxed);
        record
            .position_source
            .store(position_source.code(), Ordering::Relaxed);
        record
            .frames
            .store(u32::try_from(frames).unwrap_or(u32::MAX), Ordering::Relaxed);
        record.playing.store(playing, Ordering::Relaxed);
        record
            .playing_rising
            .store(playing_rising, Ordering::Relaxed);
        record.phase.store(phase.code(), Ordering::Relaxed);
        record
            .flow
            .store(phase != TransportPhase::Resting, Ordering::Relaxed);
        record.cycle_wrap.store(cycle_wrap, Ordering::Relaxed);
        record.split.store(split, Ordering::Relaxed);
        record.had_previous.store(had_previous, Ordering::Relaxed);
        record.emitted.store(emitted, Ordering::Relaxed);
        record.capture.store(capture, Ordering::Relaxed);
        record
            .capture_started
            .store(capture_started, Ordering::Relaxed);
        record.rung.store(0, Ordering::Relaxed);
        record
            .continuous
            .store(time.continuous.unwrap_or(i64::MIN), Ordering::Relaxed);
        record
            .continuous_valid
            .store(time.continuous.is_some(), Ordering::Relaxed);
        record
            .music_bits
            .store(time.music.unwrap_or(f64::NAN).to_bits(), Ordering::Relaxed);
        record
            .music_valid
            .store(time.music.is_some(), Ordering::Relaxed);
        record
            .tempo_bits
            .store(time.tempo.unwrap_or(f64::NAN).to_bits(), Ordering::Relaxed);
        record
            .tempo_valid
            .store(time.tempo.is_some(), Ordering::Relaxed);
        record.cycle.store(
            if time.cycle_active {
                1 + u8::from(!time.cycle_valid)
            } else {
                0
            },
            Ordering::Relaxed,
        );
        record.rate_bits.store(rate.to_bits(), Ordering::Relaxed);
        record.counted.store(counted, Ordering::Relaxed);
        record.seq.fetch_add(1, Ordering::Release);
    }

    pub(super) fn update_last_rung(&self, rung: u8) {
        let head = self.head.load(Ordering::Relaxed);
        if head == 0 {
            return;
        }
        let index = (head - 1) as usize % self.records.len();
        let record = &self.records[index];
        record.seq.fetch_add(1, Ordering::Relaxed);
        record.rung.store(rung, Ordering::Relaxed);
        record.seq.fetch_add(1, Ordering::Release);
    }

    /// 最近一块的宿主上下文，格式化成 `[capture] ctx: …` 一行。
    ///
    /// 【为什么只有一块】它回答的是"宿主此刻给的是什么形状"，不是历史。需要历史时
    /// 用 `snapshot`（诊断包里的块级 trace）。
    pub(super) fn last_context_line(&self) -> Option<String> {
        let head = self.head.load(Ordering::Acquire);
        if head == 0 {
            return None;
        }
        let index = (head - 1) as usize % self.records.len();
        let record = &self.records[index];
        // seqlock：读两遍一致才算稳定（音频线程可能正在写同一个槽）。
        let first = record.seq.load(Ordering::Acquire);
        if !first.is_multiple_of(2) {
            return None;
        }
        let read = || {
            (
                record.state.load(Ordering::Relaxed),
                record.mode.load(Ordering::Relaxed),
                record.stream.load(Ordering::Relaxed),
                record.start.load(Ordering::Relaxed),
                record.position.load(Ordering::Relaxed),
                record.frames.load(Ordering::Relaxed),
                record.phase.load(Ordering::Relaxed),
                record.playing.load(Ordering::Relaxed),
                record.playing_rising.load(Ordering::Relaxed),
                record.cycle_wrap.load(Ordering::Relaxed),
                record.had_previous.load(Ordering::Relaxed),
                record.continuous.load(Ordering::Relaxed),
                record.continuous_valid.load(Ordering::Relaxed),
                record.music_bits.load(Ordering::Relaxed),
                record.music_valid.load(Ordering::Relaxed),
                record.tempo_bits.load(Ordering::Relaxed),
                record.tempo_valid.load(Ordering::Relaxed),
                record.cycle.load(Ordering::Relaxed),
                record.rate_bits.load(Ordering::Relaxed),
                record.counted.load(Ordering::Relaxed),
                record.emitted.load(Ordering::Relaxed),
                record.capture.load(Ordering::Relaxed),
            )
        };
        let values = read();
        if record.seq.load(Ordering::Acquire) != first {
            return None;
        }
        let (
            state,
            mode,
            stream,
            raw_start,
            position,
            frames,
            phase,
            playing,
            playing_rising,
            cycle_wrap,
            had_previous,
            continuous,
            continuous_valid,
            music_bits,
            music_valid,
            tempo_bits,
            tempo_valid,
            cycle,
            rate_bits,
            counted,
            emitted,
            capture,
        ) = values;
        let rate = f64::from_bits(rate_bits);
        let music = f64::from_bits(music_bits);
        let tempo = f64::from_bits(tempo_bits);
        Some(format!(
            "[capture] ctx: stream={} mode={mode} state=0x{state:x} \
             raw={raw_start} pos={position} delta={} frames={frames} phase={phase} \
             playing={} rising={} cycle={} had_prev={} \
             cycle_meta={} cont={}{} music={}{} tempo={}{} rate={rate:.0} counted={} capture={} \
             emitted={}",
            ProcessStream::from_code(stream).map_or("?", |value| value.as_str()),
            position.saturating_sub(raw_start),
            u8::from(playing),
            u8::from(playing_rising),
            cycle_wrap,
            u8::from(had_previous),
            cycle,
            if continuous_valid { continuous } else { 0 },
            if continuous_valid {
                "(valid)"
            } else {
                "(absent)"
            },
            if music_valid { music } else { 0.0 },
            if music_valid { "(valid)" } else { "(absent)" },
            if tempo_valid { tempo } else { 0.0 },
            if tempo_valid { "(valid)" } else { "(absent)" },
            u8::from(counted),
            emitted,
            capture,
        ))
    }

    /// 最近的 `limit` 条记录（按写入顺序）。撕裂的记录直接跳过。
    pub(super) fn snapshot(&self, limit: usize) -> Vec<serde_json::Value> {
        let head = self.head.load(Ordering::Acquire);
        let count = usize::try_from(head)
            .unwrap_or(usize::MAX)
            .min(self.records.len())
            .min(limit);
        let mut out = Vec::with_capacity(count);
        for offset in 0..count {
            let index = (head.saturating_sub(1 + offset as u64)) as usize % self.records.len();
            let record = &self.records[index];
            let first = record.seq.load(Ordering::Acquire);
            if first & 1 != 0 {
                continue;
            }
            let state = record.state.load(Ordering::Relaxed);
            let mode = record.mode.load(Ordering::Relaxed);
            let stream = record.stream.load(Ordering::Relaxed);
            let start = record.start.load(Ordering::Relaxed);
            let position = record.position.load(Ordering::Relaxed);
            let position_source = record.position_source.load(Ordering::Relaxed);
            let frames = record.frames.load(Ordering::Relaxed);
            let playing = record.playing.load(Ordering::Relaxed);
            let playing_rising = record.playing_rising.load(Ordering::Relaxed);
            let phase = record.phase.load(Ordering::Relaxed);
            let flow = record.flow.load(Ordering::Relaxed);
            let cycle_wrap = record.cycle_wrap.load(Ordering::Relaxed);
            let split = record.split.load(Ordering::Relaxed);
            let had_previous = record.had_previous.load(Ordering::Relaxed);
            let emitted = record.emitted.load(Ordering::Relaxed);
            let capture = record.capture.load(Ordering::Relaxed);
            let capture_started = record.capture_started.load(Ordering::Relaxed);
            let rung = record.rung.load(Ordering::Relaxed);
            if record.seq.load(Ordering::Acquire) != first {
                continue;
            }
            out.push(serde_json::json!({
                "seq": first / 2,
                "state": state,
                "mode": mode,
                "stream": stream,
                "start": start,
                "position": position,
                "position_source": position_source,
                "frames": frames,
                "playing": playing,
                "playing_rising": playing_rising,
                "phase": phase,
                "flow": flow,
                "cycle_wrap": cycle_wrap,
                "split": split,
                "had_previous": had_previous,
                "emitted": emitted,
                "capture": capture,
                "capture_started": capture_started,
                "rung": rung,
            }));
        }
        out.reverse();
        out
    }
}
