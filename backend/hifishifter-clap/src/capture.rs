//! CLAP 外壳的捕获宿主：把 CLAP 的音频块喂给共享捕获运行时。
//!
//! 【为什么这里没有第二份捕获逻辑】起播滞后、停播钉住、循环回绕、段封口、sidecar
//! 恢复、时间线投影全在 `hifishifter-capture` 里（与 VST3 同一份）。本文件只做三件
//! 格式专属的事：
//!
//! 1. 把 `clap_process_t` 变成 `InputBlock`（工程位置、帧数、采样率、两个平面、
//!    是否在播放）；位置与播放位都来自 `clap_event_transport_t`，绝不用
//!    `steady_time` 推断播放；
//! 2. 把 `CaptureBlockOutcome` 丢掉 —— CLAP 外壳是直通（本阶段没有渲染），
//!    捕获只写自己的时间线，不改输出；
//! 3. 把 `capture_*` 命令接到 `CaptureStore`（状态形状由运行时的
//!    `status::status_json` 给出，与 VST3 逐字段相同）。

use crate::abi::clap_process_t;
use crate::audio::AudioState;
use hifishifter_capture::recorder::{
    BlockContext, InputBlock, PositionSource, ProcessStream, TransportPhase,
};
use hifishifter_capture::status::{status_json, StatusExtras};
use hifishifter_capture::store::{CaptureOptions, CaptureStore};
use hifishifter_capture::transport::{SessionInput, TransportSession};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;

/// 一个 CLAP 实例的捕获宿主。
pub struct CaptureHost {
    store: CaptureStore,
    /// 传输会话状态机（与 VST3 共用同一份实现）。
    session: Mutex<TransportSession>,
    /// 上一块的工程位置（判 `Resting` / 回卷）。
    last_position: AtomicI64,
    last_playing: AtomicBool,
    last_recording: AtomicBool,
    pass: AtomicU64,
    /// 宿主已经激活并开始处理本实例（`activate` + `start_processing`）。
    ///
    /// 【为什么它也算"能录"】录制的操作顺序是"先按捕获、再按播放"。宿主在停播时
    /// 可能一个块都不送（REAPER 的 CLAP 通路就是这样），于是"等到看见块才允许按"
    /// 会让按钮在用户做任何事之前**永远是灰的**——用户看到的正是"当前不可用"。
    /// 激活过 != 一定能拿到位置，但"能不能拿到位置"由 `sawContext` 如实回答，
    /// 而"能不能先武装"只需要"这个宿主确实在跑我们"。
    activated: AtomicBool,
    /// 处理过的块数（诊断；与 `AudioState` 的计数互相印证）。
    blocks: AtomicU64,
    /// 见过的 transport 标志位并集（诊断：宿主给的是拍还是秒）。
    transport_flags: AtomicU64,
    /// 宿主给过**可信的工程位置**（不是"给过 transport 事件"）。
    ///
    /// 【为什么不能复用运行时的 `saw_context`】那个读数只要求"块里有有效采样率"，
    /// 而录制的硬前提是**位置**：没有位置的块写不进时间线。CLAP 的 transport 事件
    /// 可以存在而 `song_pos_seconds` 不可用（宿主只给拍号/速度），此时界面必须说
    /// "这个宿主没有把播放位置交给插件"，不能让录制键看起来能用。
    saw_position: AtomicBool,
}

impl Default for CaptureHost {
    fn default() -> Self {
        Self {
            store: CaptureStore::default(),
            session: Mutex::new(TransportSession::default()),
            last_position: AtomicI64::new(i64::MIN),
            last_playing: AtomicBool::new(false),
            last_recording: AtomicBool::new(false),
            pass: AtomicU64::new(0),
            activated: AtomicBool::new(false),
            blocks: AtomicU64::new(0),
            transport_flags: AtomicU64::new(0),
            saw_position: AtomicBool::new(false),
        }
    }
}

impl CaptureHost {
    /// 激活时登记工程采样率（捕获拒绝在没有采样率的实例上武装）。
    pub fn note_activate(&self, rate: f64) {
        self.store.note_project_rate(rate);
        self.activated.store(true, Ordering::Release);
    }

    /// 音频线程：喂一块。
    ///
    /// # Safety
    /// `process` 必须是宿主在本次 `process` 回调里给出的有效指针；其音频缓冲在
    /// 调用期间有效。本函数在音频线程上执行，只做常量时间的读写，不加阻塞锁以外的
    /// 事（`CaptureStore` 内部已按"拿不到锁就旁路暂存"实现）。
    pub unsafe fn on_process(&self, process: &clap_process_t, audio: &AudioState) {
        self.blocks.fetch_add(1, Ordering::AcqRel);
        if let Some(transport) = unsafe { process.transport.as_ref() } {
            self.transport_flags
                .fetch_or(u64::from(transport.flags), Ordering::AcqRel);
        }
        let frame = audio.to_transport_frame(process);
        if frame.sample_rate > 0.0 {
            self.store.note_project_rate(frame.sample_rate);
        }
        if frame.offline {
            self.store.note_offline();
        }
        if frame.position_trusted {
            self.saw_position.store(true, Ordering::Release);
        }
        let playing = frame.playing.unwrap_or(false) && frame.position_trusted;
        let recording = frame.recording.unwrap_or(false);
        let was_playing = self.last_playing.swap(playing, Ordering::AcqRel);
        let was_recording = self.last_recording.swap(recording, Ordering::AcqRel);
        let decision = {
            let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
            if frame.sample_rate > 0.0 {
                session.max_start_window_frames = (frame.sample_rate.round().max(1.0)) as u64;
            }
            session.observe(SessionInput {
                playing,
                playing_rising: playing && !was_playing,
                recording_rising: recording && !was_recording,
                host_reset: false,
                offline: frame.offline,
                frames: process.frames_count,
            })
        };
        let frames = process.frames_count as usize;
        let position = frame.project_time_samples;
        let previous = self.last_position.swap(position, Ordering::AcqRel);
        let repeated = previous == position;
        let cycle_wrap = frame.cycle_active == Some(true) && playing && position < previous;
        let phase = if decision.new_epoch {
            TransportPhase::Start
        } else if repeated && !cycle_wrap {
            TransportPhase::Resting
        } else {
            TransportPhase::Rolling
        };
        let planes = unsafe { input_planes(process, frames) };
        let stream = if frame.offline {
            ProcessStream::Offline
        } else {
            ProcessStream::Realtime
        };
        let block = InputBlock {
            start: position,
            frames,
            sample_rate: frame.sample_rate,
            planes,
            playing,
        };
        // 先记观察位（sawContext / sawInput）：它决定界面说"这个宿主没给位置"还是
        // "插件没接在这条轨上"，与录制是否武装无关。
        self.store.observe_with_stream(&block, stream);
        if !frame.position_trusted || frames == 0 {
            return;
        }
        let pass = self.pass.fetch_add(1, Ordering::AcqRel) + 1;
        let ctx = BlockContext {
            stream,
            capture_epoch: self.store.capture_epoch(),
            position_source: PositionSource::Project,
            playing,
            recording,
            phase,
            playing_rising: playing && !was_playing,
            recording_rising: recording && !was_recording,
            cycle_wrap,
            cycle_start: None,
            new_pass: decision.new_epoch || cycle_wrap,
            transport_active: decision.transport_active,
            start_window: decision.start_window_open,
            epoch: decision.phase.code() as u64,
            new_epoch: decision.new_epoch,
            rebase_frames: 0,
            part: 0,
            // 【为什么直接用 transport_active】CLAP 没有 VST3 的 prefetch 双流，也没有
            // 内容比较；"宿主在推进"就是"音频在流动"的唯一证据。
            advancing: decision.transport_active,
            resync: false,
            resync_limit_frames: 0,
            preceding_silent_frames: 0,
            counted: false,
            discard_pre_roll: false,
            raw_start: position,
            position_trusted: true,
            publish_deferred: false,
            prune_before: 0,
            pass_sequence: pass,
            transport_generation: 0,
            preview: false,
        };
        let _ = self.store.on_block_with_context(block, ctx);
    }

    /// 非音频线程：收走封口段、把"自己停下"的捕获提升成 take。
    pub fn service(&self) {
        self.store.service();
        self.store.report_missing_input_once();
    }

    /// 处理一条 `capture_*` 命令。返回与 VST3 完全相同的 JSON 形状。
    pub fn handle(&self, command: &str, args: &Value) -> Result<Value, String> {
        match command {
            "capture_status" => Ok(self.status()),
            "capture_block_trace" => Ok(json!([])),
            "capture_start" => {
                self.store.start()?;
                Ok(self.status())
            }
            "capture_stop" => {
                let take = self.store.stop()?;
                Ok(json!({ "take": take, "status": self.status() }))
            }
            "capture_cancel" => {
                self.store.cancel();
                Ok(self.status())
            }
            "capture_clear" => {
                self.store.clear();
                Ok(self.status())
            }
            "capture_delete_take" => {
                let id = args["takeId"]
                    .as_u64()
                    .ok_or("capture_delete_take needs a numeric takeId")?;
                self.store.delete_take(id)?;
                Ok(self.status())
            }
            "capture_set_options" => {
                let current = self.store.options();
                self.store.set_options(CaptureOptions {
                    max_duration_sec: args["maxDurationSec"]
                        .as_f64()
                        .unwrap_or(current.max_duration_sec),
                    segment_sec: args["segmentSec"].as_f64().unwrap_or(current.segment_sec),
                    monitor: args["monitor"].as_bool().unwrap_or(current.monitor),
                    outside_passthrough: args["outsidePassthrough"]
                        .as_bool()
                        .unwrap_or(current.outside_passthrough),
                });
                Ok(self.status())
            }
            other => Err(format!("unknown capture command: {other}")),
        }
    }

    /// `capture_status` 的 JSON（形状由运行时给出，与 VST3 逐字段相同）。
    pub fn status(&self) -> Value {
        self.service();
        let mut status = status_json(
            &self.store,
            StatusExtras {
                // CLAP 阶段 1 不暴露 ARA：Capture 永远适用。
                ara_bound: false,
                // 本阶段没有渲染：听到的永远是宿主原声。
                output: "passthrough",
                block_diagnostics: json!({}),
            },
        );
        // 【CLAP 上这两个读数的判据是"位置可信"】运行时里的 `sawContext` 只要求
        // "块里有有效采样率"，那是 VST3 的 `ProcessContext` 语义。CLAP 的界面要回答的
        // 是"这个宿主把播放位置交给我了吗" —— 用 `saw_context && saw_position` 才不会
        // 在"只给速度/拍号、不给位置"的宿主上说假话。
        let position = self.saw_position.load(Ordering::Acquire);
        let saw_context = status["sawContext"].as_bool().unwrap_or(false) && position;
        status["sawContext"] = json!(saw_context);
        // 能先武装 != 已经有位置：后者由 sawContext 如实回答（设置窗口据此解释）。
        status["canCapture"] = json!(self.activated.load(Ordering::Acquire) || saw_context);
        self.log_evidence_once();
        status
    }

    /// 第一次被问状态时把"宿主到底做了什么"写进日志（只写一次）。
    ///
    /// 【为什么必须有】"按钮一直是灰的"在宿主里完全同形：可能是宿主没在跑我们、
    /// 可能是跑了但没给时间轴、也可能是给了时间轴但我们没认出标志位。三个读数
    /// （块数 / transport 标志并集 / 是否推出过位置）一写，就不必再猜。
    fn log_evidence_once(&self) {
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if LOGGED.swap(true, Ordering::AcqRel) {
            return;
        }
        let status = self.store.status();
        log::info!(
            "[clap][capture] status requested: blocks={} transport_flags=0x{:x} saw_position={}              saw_context={} saw_input={} can_capture={}",
            self.blocks.load(Ordering::Acquire),
            self.transport_flags.load(Ordering::Acquire),
            self.saw_position.load(Ordering::Acquire),
            status.saw_context,
            status.saw_input,
            self.activated.load(Ordering::Acquire)
        );
    }

    /// 捕获清单（随宿主工程存取的字节块里那一段）。
    pub fn manifest(&self) -> Value {
        self.store.manifest()
    }

    /// 从工程存档里恢复捕获清单（sidecar 缺失的 take 会被如实报为 missingTakes）。
    pub fn restore_manifest(&self, manifest: &Value) -> Result<(), String> {
        self.store.restore_manifest(manifest)
    }

    /// 编辑器要的时间线（含捕获出来的轨道与片段）。
    pub fn timeline_payload(&self, playhead_sec: f64) -> Value {
        let (timeline, _sources) = self.store.session_timeline();
        let mut payload = serde_json::to_value(timeline.to_payload()).unwrap_or_else(|_| json!({}));
        payload["playhead_sec"] = json!(playhead_sec.max(0.0));
        payload["undo_depth"] = json!(0);
        payload["redo_depth"] = json!(0);
        payload
    }

    /// bootstrap 的 `captureCaps`：四个读数**都是观察到的**。
    ///
    /// 【为什么 sawContext 取自存储而不是 AudioState】"宿主给过可用的工程位置"是捕获
    /// 自己的判定（它同时要求采样率有效），而 `AudioState` 只知道"收到过 transport 事件"。
    /// 两者混用会让界面在"宿主给了 transport 但位置不可用"时说"能录"。
    pub fn capture_caps(&self, saw_offline: bool) -> Value {
        let status = self.store.status();
        json!({
            "sawContext": status.saw_context,
            "sawInput": status.saw_input,
            "sawOffline": saw_offline,
            "canCapture": self.activated.load(Ordering::Acquire) || status.saw_context,
        })
    }
}

/// 取至多两个输入平面（CLAP 的 `audio_inputs[0]`）。
unsafe fn input_planes(process: &clap_process_t, frames: usize) -> [Option<&[f32]>; 2] {
    let mut planes: [Option<&[f32]>; 2] = [None, None];
    if frames == 0 || process.audio_inputs.is_null() || process.audio_inputs_count == 0 {
        return planes;
    }
    let buffer = unsafe { &*process.audio_inputs };
    if buffer.data32.is_null() {
        return planes;
    }
    for channel in 0..(buffer.channel_count.min(2) as usize) {
        let data = unsafe { *buffer.data32.add(channel) };
        if data.is_null() {
            continue;
        }
        planes[channel] = Some(unsafe { std::slice::from_raw_parts(data, frames) });
    }
    planes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{
        clap_audio_buffer_t, clap_event_transport_t, transport_flags, CLAP_SECTIME_FACTOR,
    };

    /// 一个"宿主在播放"的 transport 事件（秒 → CLAP 定点）。
    fn playing_at(seconds: f64) -> clap_event_transport_t {
        clap_event_transport_t {
            header: crate::abi::clap_event_header_t {
                size: std::mem::size_of::<clap_event_transport_t>() as u32,
                time: 0,
                space_id: 0,
                type_: crate::abi::CLAP_EVENT_TRANSPORT,
                flags: 0,
            },
            flags: transport_flags::HAS_SECONDS_TIMELINE | transport_flags::IS_PLAYING,
            song_pos_beats: 0,
            song_pos_seconds: (seconds * CLAP_SECTIME_FACTOR).round() as i64,
            tempo: 120.0,
            tempo_inc: 0.0,
            loop_start_beats: 0,
            loop_end_beats: 0,
            loop_start_seconds: 0,
            loop_end_seconds: 0,
            bar_start: 0,
            bar_number: 0,
            tsig_num: 4,
            tsig_denom: 4,
        }
    }

    /// 造一块立体声音频。返回值活到 `audio` 离开作用域。
    struct Block {
        left: Vec<f32>,
        right: Vec<f32>,
        planes: [*mut f32; 2],
        buffer: clap_audio_buffer_t,
        process: clap_process_t,
    }

    impl Block {
        fn new(transport: &clap_event_transport_t, frames: usize, offset: f32) -> Self {
            let left: Vec<f32> = (0..frames)
                .map(|index| ((index as f32 + offset) * 0.01).sin())
                .collect();
            let right = left.clone();
            let mut block = Self {
                planes: [std::ptr::null_mut(); 2],
                left,
                right,
                buffer: clap_audio_buffer_t {
                    data32: std::ptr::null_mut(),
                    data64: std::ptr::null_mut(),
                    channel_count: 2,
                    latency: 0,
                    constant_mask: 0,
                },
                process: clap_process_t {
                    steady_time: 0,
                    frames_count: frames as u32,
                    transport,
                    audio_inputs: std::ptr::null(),
                    audio_outputs: std::ptr::null_mut(),
                    audio_inputs_count: 1,
                    audio_outputs_count: 0,
                    in_events: std::ptr::null(),
                    out_events: std::ptr::null(),
                },
            };
            block.planes = [block.left.as_mut_ptr(), block.right.as_mut_ptr()];
            block.buffer.data32 = block.planes.as_mut_ptr();
            block.process.audio_inputs = &block.buffer;
            block
        }
    }

    /// 未激活的实例（宿主还没调用 activate）。
    fn host() -> CaptureHost {
        CaptureHost::default()
    }

    /// 已激活的实例：宿主声明了工程率，可以武装。
    fn activated_host() -> CaptureHost {
        let host = host();
        host.note_activate(48_000.0);
        host
    }

    fn audio() -> AudioState {
        let audio = AudioState::default();
        audio.set_sample_rate(48_000.0);
        audio
    }

    /// 录制按钮必须**在用户按播放之前**就能按：激活即"能武装"，位置另算。
    ///
    /// 【为什么这两件事要分开】REAPER 的 CLAP 通路在停播时可能一个块都不送；若
    /// "能录"要等到看见位置，按钮就永远是灰的（用户报的"当前不可用"）。
    #[test]
    fn activation_is_enough_to_arm_and_a_playhead_is_reported_separately() {
        let host = host();
        let audio = audio();
        assert_eq!(
            host.status()["canCapture"],
            json!(false),
            "not activated yet"
        );
        host.note_activate(48_000.0);
        assert_eq!(
            host.status()["canCapture"],
            json!(true),
            "activation is enough"
        );
        assert_eq!(host.status()["sawContext"], json!(false), "no position yet");
        let transport = playing_at(0.0);
        let mut block = Block::new(&transport, 64, 0.0);
        unsafe { host.on_process(&block.process, &audio) };
        assert_eq!(
            host.status()["sawContext"],
            json!(true),
            "one block with a trusted position proves the host supplies one"
        );
        block.left.clear();
        block.right.clear();
    }

    /// 只有拍时间轴的宿主也必须能算出工程位置。
    ///
    /// 【为什么这条必须有】CLAP 允许宿主只声明 `HAS_BEATS_TIMELINE + HAS_TEMPO`。
    /// 只认 seconds 的实现会让 `position_trusted` 永远是 false —— 表现为"捕获按钮
    /// 永远灰着"，而插件其实被正常处理了。2 拍 @120 BPM = 1 秒 = 48000 帧。
    #[test]
    fn a_beats_only_transport_still_yields_a_position() {
        let host = activated_host();
        let audio = audio();
        let mut transport = playing_at(0.0);
        transport.flags = crate::abi::transport_flags::HAS_BEATS_TIMELINE
            | crate::abi::transport_flags::HAS_TEMPO
            | crate::abi::transport_flags::IS_PLAYING;
        transport.tempo = 120.0;
        transport.song_pos_beats = 2 * (1_i64 << 31);
        let mut block = Block::new(&transport, 64, 0.0);
        unsafe { host.on_process(&block.process, &audio) };
        block.left.clear();
        block.right.clear();
        assert_eq!(
            host.status()["sawContext"],
            json!(true),
            "beats + tempo is a usable playhead"
        );
    }

    /// 武装后喂一段"宿主在播放"的块：状态进入录制中，且捕获帧数在涨。
    #[test]
    fn arming_then_feeding_playing_blocks_records_audio() {
        let host = activated_host();
        let audio = audio();
        host.handle("capture_start", &json!({})).unwrap();
        for index in 0..20 {
            let transport = playing_at(index as f64 * 480.0 / 48_000.0);
            let mut block = Block::new(&transport, 480, index as f32);
            unsafe { host.on_process(&block.process, &audio) };
            block.left.clear();
            block.right.clear();
        }
        let status = host.status();
        assert_eq!(status["state"], json!("recording"));
        assert_eq!(status["hostPlaying"], json!(true));
        let captured = status["recording"]["capturedFrames"].as_u64().unwrap_or(0);
        assert!(captured > 0, "no frames recorded: {status}");
    }

    /// 取消之后不能留下任何 take（录制中的素材也一并丢弃）。
    #[test]
    fn cancelling_a_recording_leaves_no_take() {
        let host = activated_host();
        let audio = audio();
        host.handle("capture_start", &json!({})).unwrap();
        for index in 0..4 {
            let transport = playing_at(index as f64 * 0.01);
            let mut block = Block::new(&transport, 480, index as f32);
            unsafe { host.on_process(&block.process, &audio) };
            block.left.clear();
            block.right.clear();
        }
        let status = host.handle("capture_cancel", &json!({})).unwrap();
        assert_eq!(status["state"], json!("empty"));
        assert_eq!(status["takeCount"], json!(0));
    }

    /// 没有位置信息的宿主必须**如实**报"不能录"，而不是让按钮看起来能用。
    #[test]
    fn a_host_without_a_playhead_reports_no_context_but_still_arms() {
        let host = activated_host();
        let audio = audio();
        let mut transport = playing_at(1.0);
        transport.flags = transport_flags::IS_PLAYING;
        let mut block = Block::new(&transport, 64, 0.0);
        unsafe { host.on_process(&block.process, &audio) };
        block.left.clear();
        block.right.clear();
        assert_eq!(
            host.status()["sawContext"],
            json!(false),
            "a host that only sends tempo/time-signature never proves a playhead"
        );
        assert_eq!(
            host.status()["canCapture"],
            json!(true),
            "armed is still allowed; the panel explains the missing playhead"
        );
    }
}
