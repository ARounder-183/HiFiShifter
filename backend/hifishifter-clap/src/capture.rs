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
    /// 最近一次**可信**的宿主位置（秒）与它被观测到的时刻（Unix 毫秒）。
    ///
    /// 【为什么不在读的时候现算】CLAP 的 transport 只在 `process` 回调里有效，而编辑器
    /// 的轮询在宿主主线程上。位置必须由音频线程抄进原子量，读侧只能读那份快照 ——
    /// 反过来"读的时候去问宿主"根本没有可问的入口。
    position_bits: AtomicU64,
    position_at_ms: AtomicU64,
    /// 宿主此刻是否在播放（与 `position_bits` 同一时刻的读数）。
    ///
    /// 【为什么与位置分开】停播时宿主**仍然**给一个有效位置（编辑光标）。把它当成
    /// "在播"会让插件的光标自己往前跑；把它丢掉又会让停止后的光标跳回 0。
    host_playing: AtomicBool,
    /// 读数序号：前端按它丢弃重复与乱序的采样（与 VST3 的 `sample_seq` 同一语义）。
    sample_seq: AtomicU64,
    /// 编辑器时间线（捕获投影 + 用户画的参数曲线）。
    ///
    /// 【为什么不是每次请求现场投影】参数曲线是**编辑状态**：现场投影会把用户刚画的
    /// 线连同对象一起丢掉。详见 `editor::timeline_host` 的模块说明。
    editor: std::sync::Arc<crate::editor::timeline_host::TimelineHost>,
    /// 音频输出：时间线的离线渲染结果（直通之外的唯一输出来源）。
    ///
    /// 【为什么与编辑器分开】编辑器时间线是"用户看到/编辑的东西"，渲染引擎是
    /// "用户听到的东西"。两者同源（渲染输入就是编辑器时间线的快照），但一个是
    /// 可变文档、一个是不可变的已发布快照 —— 音频线程只读后者。
    render: std::sync::Arc<crate::render::RenderEngine>,
    /// 已编码好的波形 mipmap（`(源 id, 级别)` → Base64）。
    ///
    /// 【为什么缓存】编码一次 L0 要遍历整条素材（60 秒立体声 ≈ 580 万样本）。前端每次
    /// 缩放都会问同一级，重算等于把音频线程旁边的 UI 线程按在 CPU 上。源 id 唯一对应
    /// 一份不变的 PCM，因此缓存不会被"改了内容"作废。
    mipmaps: Mutex<std::collections::HashMap<(String, u8), String>>,
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
            position_bits: AtomicU64::new(0),
            position_at_ms: AtomicU64::new(0),
            host_playing: AtomicBool::new(false),
            sample_seq: AtomicU64::new(0),
            editor: std::sync::Arc::new(crate::editor::timeline_host::TimelineHost::default()),
            render: std::sync::Arc::new(crate::render::RenderEngine::default()),
            mipmaps: Mutex::new(std::collections::HashMap::new()),
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
        log::trace!(
            "[clap] block frames={} trusted={} playing={:?} offline={}",
            process.frames_count,
            frame.position_trusted,
            frame.playing,
            frame.offline
        );
        // 【先把渲染结果盖上去】直通已经由 `audio::process_block` 写进输出；
        // 这里只覆盖属于素材的帧，其余保持宿主原声。
        if let Some(tape) = self.render.current() {
            if (tape.sample_rate as f64 - frame.sample_rate).abs() < 0.5 {
                unsafe {
                    crate::audio::overwrite_output(
                        process,
                        &tape,
                        frame.project_time_samples,
                        self.store.options().outside_passthrough,
                    )
                };
            }
        }
        if frame.sample_rate > 0.0 {
            self.store.note_project_rate(frame.sample_rate);
        }
        if frame.offline {
            self.store.note_offline();
        }
        if frame.position_trusted && frame.sample_rate > 0.0 {
            // 【为什么位置用"秒"存】编辑器要的就是秒；存帧还要连带存采样率，而
            // 采样率已经由 `note_activate` 记过一份 —— 两份会让"改了采样率"变成
            // 两个各自更新的值。
            let seconds = frame.project_time_samples as f64 / frame.sample_rate;
            self.position_bits
                .store(seconds.to_bits(), Ordering::Release);
            self.position_at_ms.store(now_ms(), Ordering::Release);
            self.sample_seq.fetch_add(1, Ordering::AcqRel);
            self.saw_position.store(true, Ordering::Release);
        }
        let playing = frame.playing.unwrap_or(false) && frame.position_trusted;
        // 【为什么连"没在播"也要记】停播时宿主给的位置是编辑光标：界面要把它画出来，
        // 而"要不要让光标继续走"由这一个布尔回答。
        self.host_playing.store(playing, Ordering::Release);
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

    /// 宿主此刻的传输读数（`get_playback_state` 的载荷）。
    ///
    /// 【为什么外壳必须回答这一条】前端的光标与捕获带右缘都读它：编辑器每 400 ms
    /// 问一次（插件形态不用 33 ms 档，见 `App.tsx` 的常驻轮询），视觉层再按 1x
    /// 插值。这一条没人回答时，页面上的光标永远停在 0 —— 而捕获带的右缘又只能
    /// 跟着"视觉播放头"延伸，于是带子与光标一起冻住。用户看到的就是"捕获在计时，
    /// 但界面什么都不动"。
    ///
    /// 【为什么形状与 VST3 逐字段相同】前端只有**一个** reducer（`applyTransportSample`），
    /// 两条通道共用。少一个 `ok` 就会让整条采样被丢弃（VST3 曾经就是这样静默失效的）。
    pub fn playback_state(&self) -> Value {
        let seen = self.saw_position.load(Ordering::Acquire);
        let seconds = f64::from_bits(self.position_bits.load(Ordering::Acquire));
        let at = self.position_at_ms.load(Ordering::Acquire);
        let now = now_ms();
        let age_ms = if at == 0 {
            u32::MAX
        } else {
            now.saturating_sub(at).min(u64::from(u32::MAX)) as u32
        };
        // 【为什么播放位要有寿命】宿主停播后可能直接不再调用 `process`（REAPER 的
        // 空闲通路就是这样）。没有这道闸门，最后一次 `kPlaying=1` 会永远生效 ——
        // 光标在静止的画面上继续外推，正是"停止后光标还在走"。
        let playing = seen && self.host_playing.load(Ordering::Acquire) && age_ms <= PLAYING_TTL_MS;
        let duration_sec = self
            .store
            .material_span()
            .map(|span| span.end_frame as f64 / f64::from(span.sample_rate.max(1)))
            .unwrap_or(0.0);
        json!({
            "ok": true,
            "is_playing": playing,
            "waiting_for_render": false,
            "target": Value::Null,
            "base_sec": 0.0,
            "position_sec": if seen { seconds } else { 0.0 },
            "duration_sec": duration_sec,
            // 【为什么恒真】这里的每一个位置都来自宿主的 transport 事件，没有一格是
            // 插件自己推的。前端据此关掉"按往返时延外推"（那会把宿主的读数推过头）。
            "host_authoritative": true,
            "position_source": "host",
            "transport_source": "host",
            "position_age_ms": age_ms,
            "seek_pending": false,
            "sample_seq": self.sample_seq.load(Ordering::Acquire),
        })
    }

    /// `get_param_frames`：参数编辑器读一条曲线（与 VST3 同一份载荷）。
    ///
    /// 【为什么外壳必须回答】这是参数编辑器的**唯一**取数入口：没有它，piano roll
    /// 既画不出原线也画不出用户画的线，整块区域是空的（"参数编辑器里什么都没有"）。
    pub fn param_frames(&self, args: &Value) -> Result<Value, String> {
        let track_id = args["trackId"]
            .as_str()
            .ok_or("get_param_frames needs trackId")?
            .to_owned();
        let param = args["param"]
            .as_str()
            .ok_or("get_param_frames needs param")?
            .to_owned();
        let start_frame = args["startFrame"].as_u64().unwrap_or(0) as u32;
        let frame_count = args["frameCount"].as_u64().unwrap_or(0) as u32;
        let stride = args["stride"].as_u64().map(|value| value as u32);
        let binary = args["binary"].as_bool();
        let with_sentinel = args["withSentinel"].as_bool();
        self.editor.refresh(&self.store);
        if !self.editor.track_exists(&track_id) {
            return Err(format!("unknown track: {track_id}"));
        }
        // 音高先确保基线：第一次问会**在后台起一轮分析**并如实回 analysis_pending，
        // 前端据此继续轮询（与 VST3 的"分析中"同一个字段）。
        if param == "pitch" {
            let _ = self.editor.ensure_pitch_baseline(&self.store);
        }
        let mut payload = hifishifter_kernel::editor::params::get_param_frames(
            self.editor.as_ref(),
            track_id,
            param,
            start_frame,
            frame_count,
            stride,
            binary,
            with_sentinel,
        );
        // 【为什么是"或"而不是"没有才写"】内核自己会回 `Some(false)`（它调度的是
        // FCPE 批处理）。我们这条 WORLD 基线不在它的调度里，必须并进来 —— 否则前端
        // 拿到"分析完成"而曲线还是空的。
        payload.analysis_pending =
            Some(payload.analysis_pending.unwrap_or(false) || self.editor.pitch_pending());
        serde_json::to_value(payload).map_err(|error| error.to_string())
    }

    /// `set_param_frames`：把用户画的曲线写回时间线。
    ///
    /// 【为什么 present 必须逐帧传下去】它是"这一帧用户真的画过"的显式位图；丢了它，
    /// 整段窗口里没碰过的帧会被烘成硬编辑，之后基线与几何变化都不再跟随。
    pub fn set_param_frames(&self, args: &Value) -> Result<Value, String> {
        let track_id = args["trackId"]
            .as_str()
            .ok_or("set_param_frames needs trackId")?
            .to_owned();
        let param = args["param"]
            .as_str()
            .ok_or("set_param_frames needs param")?
            .to_owned();
        let start_frame = args["startFrame"].as_u64().unwrap_or(0) as u32;
        let values: Vec<f32> = args["values"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_f64)
                    .map(|value| value as f32)
                    .collect()
            })
            .unwrap_or_default();
        if values.is_empty() {
            return Err("set_param_frames needs values".into());
        }
        let present = args["present"].as_array().map(|items| {
            items
                .iter()
                .map(|value| value.as_bool().unwrap_or(true))
                .collect::<Vec<bool>>()
        });
        let checkpoint = args["checkpoint"].as_bool();
        self.editor.refresh(&self.store);
        if !self.editor.track_exists(&track_id) {
            return Err(format!("unknown track: {track_id}"));
        }
        if param == "pitch" {
            // 先让基线就位：没有 orig 的 edit 就是"一条凭空画的线"，回读时会被
            // 0 哨兵规则解释错。
            let _ = self.editor.ensure_pitch_baseline(&self.store);
        }
        let result = hifishifter_kernel::editor::params::set_param_frames_with_present(
            self.editor.as_ref(),
            track_id,
            param,
            start_frame,
            values,
            present,
            checkpoint,
        );
        // 【为什么每次写入都请求重渲染】曲线是 Capture 形态下"处理"的全部内容：
        // 写完不重渲染，用户画完听到的还是上一版（"改了没反应"）。防抖在引擎里，
        // 拖拽期间几十条写入只会触发一次真正的渲染。
        self.schedule_render();
        Ok(result)
    }

    /// `get_static_param` / `set_static_param`：静态枚举参数（算法下拉）。
    pub fn get_static_param(&self, args: &Value) -> Result<Value, String> {
        let track_id = args["trackId"]
            .as_str()
            .ok_or("get_static_param needs trackId")?
            .to_owned();
        let param = args["param"]
            .as_str()
            .ok_or("get_static_param needs param")?
            .to_owned();
        self.editor.refresh(&self.store);
        if !self.editor.track_exists(&track_id) {
            return Err(format!("unknown track: {track_id}"));
        }
        serde_json::to_value(hifishifter_kernel::editor::params::get_static_param(
            self.editor.as_ref(),
            track_id,
            param,
        ))
        .map_err(|error| error.to_string())
    }

    /// 写一个静态参数（值来自算法自己的枚举描述符）。
    pub fn set_static_param(&self, args: &Value) -> Result<Value, String> {
        let track_id = args["trackId"]
            .as_str()
            .ok_or("set_static_param needs trackId")?
            .to_owned();
        let param = args["param"]
            .as_str()
            .ok_or("set_static_param needs param")?
            .to_owned();
        let value = args["value"]
            .as_f64()
            .ok_or("set_static_param needs value")?;
        let checkpoint = args["checkpoint"].as_bool();
        self.editor.refresh(&self.store);
        if !self.editor.track_exists(&track_id) {
            return Err(format!("unknown track: {track_id}"));
        }
        let result = hifishifter_kernel::editor::params::set_static_param(
            self.editor.as_ref(),
            track_id,
            param,
            value,
            checkpoint,
        );
        // 静态参数（算法下拉等）会改变渲染链：写完必须重渲染。
        self.schedule_render();
        self.schedule_render();
        Ok(result)
    }

    /// 一条源在某一级的波形 mipmap（Base64，WFPK v2 布局）。
    ///
    /// 【为什么由外壳回答】前端的波形层按 `source_path` 取峰值
    /// （`get_waveform_mipmap_binary`）。CLAP 的源是内存里的 `capture://blake3:<hash>`，
    /// 磁盘上根本没有文件 —— 不回答这一条，时间轴与参数编辑器里的片段就是一块
    /// **没有内容的色块**（用户说的"没有预览"）。
    ///
    /// 未知源返回 `None`（协议层回空串，前端按"尚未就绪"冷却重试）。
    pub fn waveform_mipmap(&self, source_path: &str, level: u8) -> Option<String> {
        let key = (source_path.to_owned(), level);
        if let Some(cached) = self
            .mipmaps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Some(cached.clone());
        }
        let sources = self.store.material_sources();
        let pcm = sources.get(source_path)?;
        let bytes = mipmap_bytes(pcm, level)?;
        let encoded = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
        let mut cache = self.mipmaps.lock().unwrap_or_else(|e| e.into_inner());
        // 【为什么有上限】每个级别是一份新分配的编码串；素材很多时不设界就是内存泄漏。
        // 淘汰最老的一条（HashMap 的插入顺序在 Rust 里不保证，因此按"随便删一个"处理：
        // 缓存只是加速，删错不会错结果）。
        if cache.len() >= MAX_MIPMAP_CACHE {
            if let Some(stale) = cache.keys().next().cloned() {
                cache.remove(&stale);
            }
        }
        cache.insert(key, encoded.clone());
        Some(encoded)
    }

    /// 捕获存储的修订号：时间线上的素材每变一次就 +1。
    ///
    /// 【为什么要暴露它】编辑器靠对比它来决定"要不要重取整份时间线"
    /// （`plugin_get_apply_state.host_version`）。没有它，录完之后时间线永远不会
    /// 重新取 —— take 已经落盘、界面却还是空的（用户报的"录完没有片段"）。
    pub fn revision(&self) -> u64 {
        self.store.revision()
    }

    /// 请求重渲染（素材或曲线变了之后调用）。
    ///
    /// 【为什么所有入口都调同一个函数】渲染输入只有一来源：编辑器时间线 + 捕获 PCM。
    /// 三处（录制结束、清空/删 take、参数写入）各拼一份输入，迟早会有一处漏掉某个字段
    /// —— 表现就是"这次改动没生效"。
    fn schedule_render(&self) {
        self.editor.refresh(&self.store);
        if !self.store.has_material() {
            self.render.clear();
            return;
        }
        let timeline = self.editor.snapshot();
        let sources = self.store.material_sources();
        let rate = self.store.project_rate().round().max(0.0) as u32;
        self.render.request(timeline, sources, rate);
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
                log::info!(
                    "[clap][capture] armed: blocks={} saw_position={} activated={}",
                    self.blocks.load(Ordering::Acquire),
                    self.saw_position.load(Ordering::Acquire),
                    self.activated.load(Ordering::Acquire)
                );
                Ok(self.status())
            }
            "capture_stop" => {
                let take = self.store.stop()?;
                self.schedule_render();
                let status = self.status();
                // 【为什么把这条写成 info 而不是 trace】"录完没有片段"在宿主里只有三种
                // 形状：没录到帧、录到了但整段静音、录到了但时间线没被重取。三个读数
                // 一行写完，下一次现场就不必再猜。
                log::info!(
                    "[clap][capture] stopped: take={} frames={} clips={} silent={} span={} takes={}",
                    take,
                    status["material"]["capturedFrames"].as_u64().unwrap_or(0),
                    status["material"]["clips"].as_u64().unwrap_or(0),
                    status["material"]["silent"].as_bool().unwrap_or(true),
                    status["span"],
                    status["takeCount"].as_u64().unwrap_or(0)
                );
                Ok(json!({ "take": take, "status": status }))
            }
            "capture_cancel" => {
                self.store.cancel();
                Ok(self.status())
            }
            "capture_clear" => {
                self.store.clear();
                self.schedule_render();
                Ok(self.status())
            }
            "capture_delete_take" => {
                let id = args["takeId"]
                    .as_u64()
                    .ok_or("capture_delete_take needs a numeric takeId")?;
                self.store.delete_take(id)?;
                self.schedule_render();
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
                // 【如实报】有渲染结果就是 processed（素材之外的帧仍是宿主原声，
                // 那是 `outside_passthrough` 的语义，不改变这个读数）。
                output: if self.render.current().is_some() {
                    "processed"
                } else {
                    "passthrough"
                },
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
        let result = self.store.restore_manifest(manifest);
        self.schedule_render();
        result
    }

    /// 编辑器要的时间线（含捕获出来的轨道与片段）。
    pub fn timeline_payload(&self, playhead_sec: f64) -> Value {
        self.editor.refresh(&self.store);
        self.editor.payload(playhead_sec)
    }

    /// 编辑器时间线（参数命令的前置校验与音高基线都在这里）。
    pub fn editor(&self) -> &std::sync::Arc<crate::editor::timeline_host::TimelineHost> {
        &self.editor
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

/// 波形 mipmap 缓存的最多条数（见 `waveform_mipmap`）。
const MAX_MIPMAP_CACHE: usize = 16;

/// 把一份源 PCM 编码成 `WFPK` v2 二进制（前端 `waveformBinaryCodec` 的协议）。
///
/// 布局：`[WFPK][version=2][sample_rate][division][peak_count][level][channels]`
/// 之后按声道分块 `ch0_min / ch0_max / ch1_min / ch1_max`，每块 `peak_count` 个 f32。
fn mipmap_bytes(pcm: &hifishifter_capture::types::CapturePcm, level: u8) -> Option<Vec<u8>> {
    let divisor: u32 = match level {
        0 => 16,
        1 => 512,
        2 => 4096,
        _ => return None,
    };
    let frames = pcm.planes.iter().map(Vec::len).max().unwrap_or(0);
    if frames == 0 || pcm.sample_rate == 0 {
        return None;
    }
    let channels = pcm.planes.len().clamp(1, 2);
    let divisor = divisor as usize;
    let peak_count = frames.div_ceil(divisor).max(1);
    let mut out = Vec::with_capacity(28 + peak_count * channels * 8);
    out.extend_from_slice(b"WFPK");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&pcm.sample_rate.to_le_bytes());
    out.extend_from_slice(&(divisor as u32).to_le_bytes());
    out.extend_from_slice(&(peak_count as u32).to_le_bytes());
    out.extend_from_slice(&(u32::from(level)).to_le_bytes());
    out.extend_from_slice(&(channels as u32).to_le_bytes());
    for channel in 0..channels {
        let plane = &pcm.planes[channel.min(pcm.planes.len() - 1)];
        for peak in 0..peak_count {
            let lo = peak * divisor;
            let hi = (lo + divisor).min(plane.len());
            let mut min = 0.0_f32;
            if lo < hi {
                min = plane[lo];
                for sample in &plane[lo + 1..hi] {
                    if *sample < min {
                        min = *sample;
                    }
                }
            }
            out.extend_from_slice(&min.to_le_bytes());
        }
        for peak in 0..peak_count {
            let lo = peak * divisor;
            let hi = (lo + divisor).min(plane.len());
            let mut max = 0.0_f32;
            if lo < hi {
                max = plane[lo];
                for sample in &plane[lo + 1..hi] {
                    if *sample > max {
                        max = *sample;
                    }
                }
            }
            out.extend_from_slice(&max.to_le_bytes());
        }
    }
    Some(out)
}

/// 播放位允许的最大读数年龄（毫秒）：超过它，最后一次 `kPlaying=1` 不再算数。
///
/// 【为什么是 250】与前端 `TRANSPORT_ADVANCE_MAX_AGE_MS` 同一个量级：44.1/48 kHz 下
/// 一块是 21~46 ms，250 ms 足以吸收调度抖动，又短到用户看不见"停止后还在走"。
const PLAYING_TTL_MS: u32 = 250;

/// 当前 Unix 毫秒（读侧与音频侧共用；只做差值，不做绝对时间语义）。
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
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

    /// 等到渲染引擎发布**第 `at_least` 代**输出（后台线程 + 防抖，需要轮询）。
    fn wait_for_tape(
        host: &CaptureHost,
        at_least: u64,
    ) -> std::sync::Arc<crate::render::RenderTape> {
        let mut last = None;
        for _ in 0..600 {
            if let Some(tape) = host.render.current() {
                // 【为什么看代数而不是"有没有"】第一次渲染的结果在改完曲线后仍然在，
                // 直接读会拿到上一版 —— 那条断言就永远成立、也永远测不到东西。
                if host.render.generation() >= at_least {
                    return tape;
                }
                last = Some(tape);
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        panic!(
            "no render generation >= {at_least} (last tape: {:?}, generation: {}, error: {:?})",
            last.map(|t| t.frames),
            host.render.generation(),
            host.render.last_error()
        );
    }

    /// 某个频率上的能量（Goertzel）。用于判断"这一版里 440 Hz 是不是变强了"。
    fn tone_energy(tape: &crate::render::RenderTape, hz: f64) -> f64 {
        let channels = tape.channels.max(1);
        let lo = tape.frames / 4;
        let hi = tape.frames * 3 / 4;
        let rate = f64::from(tape.sample_rate.max(1));
        let omega = 2.0 * std::f64::consts::PI * hz / rate;
        let coeff = 2.0 * omega.cos();
        let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
        for index in lo..hi {
            let s0 = f64::from(tape.interleaved[index * channels]) + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0)
    }

    /// 用**过零率**估计主频（Hz）。
    ///
    /// 【为什么不用 FFT】这里只比较"同一个信号被移调前后"的频率比，过零率对纯音
    /// 足够准确，而且不引入新依赖。取中段 50% 以避开首尾的淡化与静音。
    fn dominant_hz(tape: &crate::render::RenderTape) -> f64 {
        let channels = tape.channels.max(1);
        let lo = tape.frames / 4;
        let hi = tape.frames * 3 / 4;
        if hi <= lo + 1 {
            return 0.0;
        }
        let mut crossings = 0_usize;
        let mut previous = tape.interleaved[lo * channels];
        for index in lo + 1..hi {
            let value = tape.interleaved[index * channels];
            if (previous < 0.0 && value >= 0.0) || (previous >= 0.0 && value < 0.0) {
                crossings += 1;
            }
            previous = value;
        }
        let seconds = (hi - lo) as f64 / f64::from(tape.sample_rate.max(1));
        crossings as f64 / (2.0 * seconds)
    }

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
            let block = Self {
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
            // 【为什么指针要由调用方在**最终位置**上绑】`Block` 在 `new` 里只是
            // 一个临时值：返回时整个结构体会被移动，而 `audio_inputs` / `data32`
            // 里存的是指向**结构体内部**的裸指针。移动之后它们指向已被释放的栈槽，
            // 测试于是读到垃圾（表现为"录到了帧数但全是静音"）。`attach` 在
            // 调用方拿到最终位置之后再绑一次。
            block
        }

        /// 把内部指针绑到**当前位置**。必须在 `Block` 落到最终位置之后调用。
        fn attach(&mut self) {
            self.planes = [self.left.as_mut_ptr(), self.right.as_mut_ptr()];
            self.buffer.data32 = self.planes.as_mut_ptr();
            self.process.audio_inputs = &self.buffer;
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

        block.attach();
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

        block.attach();
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

            block.attach();
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

    /// 录制中必须给出**正在长出来的捕获带**（前端画预览与光标全靠它）。
    ///
    /// 【为什么是这条的回归测试】用户报"捕获时没有预览"时，界面读的是
    /// `status.capture`；它为 null 就等于"没有任何东西在长"。这里直接钉住
    /// 线上形状：区间列表非空、位置在推进、采样率有效。
    #[test]
    fn a_recording_publishes_the_growing_capture_band() {
        let host = activated_host();
        let audio = audio();
        host.handle("capture_start", &json!({})).unwrap();
        for index in 0..40 {
            let transport = playing_at(index as f64 * 480.0 / 48_000.0);
            let mut block = Block::new(&transport, 480, index as f32);

            block.attach();
            unsafe { host.on_process(&block.process, &audio) };
            block.left.clear();
            block.right.clear();
        }
        let status = host.status();
        let capture = &status["capture"];
        assert!(capture.is_object(), "no capture band in {status}");
        assert_eq!(
            capture["sampleRate"].as_f64(),
            Some(48_000.0),
            "the band must carry the project rate: {status}"
        );
        let runs = capture["runs"].as_array().expect("runs must be a list");
        assert!(!runs.is_empty(), "no covered run in {status}");
        assert!(
            capture["positionFrame"].as_u64().unwrap_or(0) > 0,
            "the capture position must advance: {status}"
        );
    }

    /// 停止之后必须真的留下一个 take 和一个时间轴片段。
    ///
    /// 【为什么用 `timeline_payload` 断言】它是编辑器窗口真正渲染的那份数据。
    /// take 存在但投影里没有片段时，用户看到的就是"录完了什么都没有"。
    #[test]
    fn stopping_produces_a_take_and_a_clip_on_the_timeline() {
        let host = activated_host();
        let audio = audio();
        host.handle("capture_start", &json!({})).unwrap();
        for index in 0..40 {
            let transport = playing_at(index as f64 * 480.0 / 48_000.0);
            let mut block = Block::new(&transport, 480, index as f32);

            block.attach();
            unsafe { host.on_process(&block.process, &audio) };
            block.left.clear();
            block.right.clear();
        }
        let reply = host.handle("capture_stop", &json!({})).unwrap();
        assert_eq!(reply["status"]["takeCount"], json!(1), "no take: {reply}");
        let timeline = host.timeline_payload(0.5);
        let clips = timeline["clips"].as_array().expect("clips must be a list");
        assert!(!clips.is_empty(), "no clip on the timeline: {timeline}");
    }

    /// 宿主的**播放位置与播放位**必须能被编辑器读到（`get_playback_state`）。
    ///
    /// 【为什么这条是用户报障的回归测试】CLAP 外壳此前根本不回答这条命令，于是
    /// 前端的光标永远停在 0：捕获带右缘只跟着"视觉播放头"延伸，两者一起冻住。
    /// 这里钉住三件事：位置来自宿主、播放位如实、停播时位置是编辑光标而不是"在播"。
    #[test]
    fn the_editor_can_read_the_host_playhead() {
        let host = activated_host();
        let audio = audio();
        let transport = playing_at(0.5);
        let mut block = Block::new(&transport, 480, 0.0);
        block.attach();
        unsafe { host.on_process(&block.process, &audio) };
        let state = host.playback_state();
        assert_eq!(state["ok"], json!(true));
        assert_eq!(state["is_playing"], json!(true));
        assert_eq!(state["host_authoritative"], json!(true));
        let position = state["position_sec"].as_f64().unwrap_or(-1.0);
        assert!((position - 0.5).abs() < 1e-6, "position was {position}");

        // 宿主停播：位置仍然有效（那是编辑光标），但绝不能再报"在播"。
        let mut stopped = playing_at(3.0);
        stopped.flags = crate::abi::transport_flags::HAS_SECONDS_TIMELINE;
        let mut block = Block::new(&stopped, 480, 0.0);
        block.attach();
        unsafe { host.on_process(&block.process, &audio) };
        let state = host.playback_state();
        assert_eq!(state["is_playing"], json!(false));
        let position = state["position_sec"].as_f64().unwrap_or(-1.0);
        assert!((position - 3.0).abs() < 1e-6, "position was {position}");
    }

    /// 时间线与参数编辑器的波形层按源取峰值：外壳必须给出 `WFPK` v2 二进制。
    ///
    /// 【为什么这条属于"预览"的验收】片段没有峰值就只是一块色块。这里解出真二进制，
    /// 逐字段核对头部与"确实有非零振幅"。
    #[test]
    fn a_captured_take_serves_a_waveform_mipmap() {
        let host = activated_host();
        let audio = audio();
        host.handle("capture_start", &json!({})).unwrap();
        for index in 0..40 {
            let transport = playing_at(index as f64 * 480.0 / 48_000.0);
            let mut block = Block::new(&transport, 480, index as f32);
            block.attach();
            unsafe { host.on_process(&block.process, &audio) };
            block.left.clear();
            block.right.clear();
        }
        let reply = host.handle("capture_stop", &json!({})).unwrap();
        assert_eq!(reply["status"]["takeCount"], json!(1));
        // 源 id 是内容寻址的（`capture://blake3:<hex>`），必须从时间线上读回来 ——
        // 波形层正是拿片段的 `source_path` 去问峰值的。
        let timeline = host.timeline_payload(0.0);
        let source = timeline["clips"][0]["source_path"]
            .as_str()
            .expect("the clip must name its source")
            .to_owned();
        let encoded = host
            .waveform_mipmap(&source, 0)
            .expect("a captured take must serve a waveform mipmap");
        let bytes = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            encoded.as_bytes(),
        )
        .expect("the mipmap must be valid base64");
        assert_eq!(&bytes[..4], b"WFPK", "wrong magic");
        let read =
            |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        assert_eq!(read(4), 2, "format version");
        assert_eq!(read(8), 48_000, "sample rate");
        assert_eq!(read(12), 16, "L0 divisor");
        assert_eq!(read(24), 2, "stereo");
        let peaks = read(16) as usize;
        assert_eq!(bytes.len(), 28 + peaks * 2 * 8, "payload size");
        let floats: Vec<f32> = bytes[28..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert!(
            floats.iter().any(|value| value.abs() > 0.0),
            "the recorded sine must appear in the peaks"
        );
        // 未知源必须是"没有"，而不是一份静音波形 —— 后者会让前端画出一条假直线。
        assert!(host.waveform_mipmap("capture://999", 0).is_none());
    }

    /// 参数曲线必须能**写进去再读回来**：这是参数编辑器"能编辑"的全部含义。
    #[test]
    fn parameter_edits_round_trip_on_the_capture_timeline() {
        let host = activated_host();
        let audio = audio();
        host.handle("capture_start", &json!({})).unwrap();
        for index in 0..40 {
            let transport = playing_at(index as f64 * 480.0 / 48_000.0);
            let mut block = Block::new(&transport, 480, index as f32);
            block.attach();
            unsafe { host.on_process(&block.process, &audio) };
            block.left.clear();
            block.right.clear();
        }
        host.handle("capture_stop", &json!({})).unwrap();
        let track = crate::editor::timeline_host::CAPTURE_TRACK_ID;
        let written = host
            .set_param_frames(&json!({
                "trackId": track,
                "param": "tension",
                "startFrame": 3,
                "values": [0.5, 0.75],
                "present": [true, true],
                "checkpoint": false,
            }))
            .expect("set_param_frames");
        assert_eq!(written["ok"], json!(true), "{written}");
        let read = host
            .param_frames(&json!({
                "trackId": track,
                "param": "tension",
                "startFrame": 3,
                "frameCount": 2,
                "binary": false,
            }))
            .expect("get_param_frames");
        assert_eq!(read["ok"], json!(true), "{read}");
        assert_eq!(read["edit"], json!([0.5, 0.75]), "{read}");
        assert_eq!(read["edited"], json!([true, true]), "{read}");
    }

    /// 捕获后的音高**原线**必须真的出现（WORLD 基线），否则参数编辑器是一片空白。
    ///
    /// 【为什么用 220 Hz 正弦】它对应 A3 = MIDI 57，落在人声范围内；检测器若把整段
    /// 判成无声（全 0）或判错八度（45 / 69），这条就会失败。
    #[test]
    fn a_captured_tone_gets_a_pitch_baseline() {
        let host = activated_host();
        let audio = audio();
        host.handle("capture_start", &json!({})).unwrap();
        let mut phase = 0.0_f64;
        for index in 0..100 {
            let transport = playing_at(index as f64 * 480.0 / 48_000.0);
            let mut block = Block::new(&transport, 480, 0.0);
            block.attach();
            let samples: Vec<f32> = (0..480)
                .map(|_| {
                    phase += 2.0 * std::f64::consts::PI * 220.0 / 48_000.0;
                    (phase.sin() * 0.5) as f32
                })
                .collect();
            block.left.copy_from_slice(&samples);
            block.right.copy_from_slice(&samples);
            unsafe { host.on_process(&block.process, &audio) };
        }
        host.handle("capture_stop", &json!({})).unwrap();
        let track = crate::editor::timeline_host::CAPTURE_TRACK_ID;
        let mut last = json!(null);
        for _ in 0..200 {
            let read = host
                .param_frames(&json!({
                    "trackId": track,
                    "param": "pitch",
                    "startFrame": 0,
                    "frameCount": 80,
                    "binary": false,
                }))
                .expect("get_param_frames");
            last = read.clone();
            let pending = read["analysis_pending"].as_bool().unwrap_or(false);
            if !pending {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert_eq!(last["ok"], json!(true), "{last}");
        let orig: Vec<f64> = last["orig"]
            .as_array()
            .map(|items| items.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        assert!(!orig.is_empty(), "no frames in {last}");
        let voiced = orig.iter().filter(|value| **value > 0.0).count();
        assert!(voiced > orig.len() / 2, "too few voiced frames: {orig:?}");
        let median = {
            let mut values: Vec<f64> = orig.iter().copied().filter(|value| *value > 0.0).collect();
            values.sort_by(|a, b| a.total_cmp(b));
            values[values.len() / 2]
        };
        assert!(
            (median - 57.0).abs() < 2.0,
            "expected A3 (57) but the baseline says {median}: {orig:?}"
        );
    }

    /// 捕获结束后必须真的**发布一份渲染输出**：这是"音频输出接上"的验收。
    ///
    /// 【为什么这条要等待】渲染在后台线程上跑（防抖 600 ms + WORLD 声码器），
    /// 因此这里轮询到超时为止；发布出来的磁带必须覆盖素材区间且有非零样本。
    #[test]
    fn a_captured_take_is_rendered_into_a_playable_tape() {
        let host = activated_host();
        let audio = audio();
        host.handle("capture_start", &json!({})).unwrap();
        let mut phase = 0.0_f64;
        for index in 0..100 {
            let transport = playing_at(index as f64 * 480.0 / 48_000.0);
            let mut block = Block::new(&transport, 480, 0.0);
            block.attach();
            let samples: Vec<f32> = (0..480)
                .map(|_| {
                    phase += 2.0 * std::f64::consts::PI * 220.0 / 48_000.0;
                    (phase.sin() * 0.4) as f32
                })
                .collect();
            block.left.copy_from_slice(&samples);
            block.right.copy_from_slice(&samples);
            unsafe { host.on_process(&block.process, &audio) };
        }
        host.handle("capture_stop", &json!({})).unwrap();
        let mut tape = None;
        for _ in 0..400 {
            if let Some(current) = host.render.current() {
                tape = Some(current);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let tape = tape.expect("the capture must be rendered into a playable tape");
        assert_eq!(tape.sample_rate, 48_000);
        assert!(
            tape.end_frame > tape.start_frame,
            "the tape must cover the captured material ({}..{})",
            tape.start_frame,
            tape.end_frame
        );
        assert!(tape.frames > 0, "no frames in the rendered tape");
        assert!(
            tape.interleaved.iter().any(|sample| sample.abs() > 1e-6),
            "the rendered tape is silent"
        );
        // 状态里的输出仲裁要如实说"现在听到的是处理结果"。
        assert_eq!(host.status()["output"], json!("processed"));
    }

    /// **输出真的接上了**：把音高改高一个八度，渲染结果的主频必须翻倍。
    ///
    /// 【为什么这是唯一有说服力的验收】"输出接上了"可以有很多弱证据（有样本、
    /// 非静音、状态说 processed），但它们全都无法区分"播放的是我们的渲染"与
    /// "播放的是宿主原声"。一个八度的音高变化只有真正走了声码器才可能出现。
    #[test]
    fn a_pitch_edit_changes_the_rendered_audio() {
        let host = activated_host();
        let audio = audio();
        host.handle("capture_start", &json!({})).unwrap();
        let mut phase = 0.0_f64;
        for index in 0..100 {
            let transport = playing_at(index as f64 * 480.0 / 48_000.0);
            let mut block = Block::new(&transport, 480, 0.0);
            block.attach();
            let samples: Vec<f32> = (0..480)
                .map(|_| {
                    phase += 2.0 * std::f64::consts::PI * 220.0 / 48_000.0;
                    (phase.sin() * 0.4) as f32
                })
                .collect();
            block.left.copy_from_slice(&samples);
            block.right.copy_from_slice(&samples);
            unsafe { host.on_process(&block.process, &audio) };
        }
        host.handle("capture_stop", &json!({})).unwrap();
        let original = wait_for_tape(&host, 1);
        let base_hz = dominant_hz(&original);

        // 等音高基线（WORLD）就绪，然后把它整体抬高 12 个半音。
        let track = crate::editor::timeline_host::CAPTURE_TRACK_ID;
        let mut orig: Vec<f32> = Vec::new();
        for _ in 0..200 {
            let read = host
                .param_frames(&json!({
                    "trackId": track, "param": "pitch",
                    "startFrame": 0, "frameCount": 200, "binary": false,
                }))
                .expect("get_param_frames");
            let pending = read["analysis_pending"].as_bool().unwrap_or(false);
            orig = read["orig"]
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                        .collect()
                })
                .unwrap_or_default();
            if !pending && orig.iter().any(|value| *value > 0.0) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert!(
            orig.iter().any(|value| *value > 0.0),
            "no pitch baseline to edit"
        );
        let shifted: Vec<f32> = orig
            .iter()
            .map(|value| if *value > 0.0 { value + 12.0 } else { 0.0 })
            .collect();
        let present = vec![true; shifted.len()];
        host.set_param_frames(&json!({
            "trackId": track, "param": "pitch",
            "startFrame": 0, "values": shifted,
            "present": present, "checkpoint": false,
        }))
        .expect("set_param_frames");

        let edited = wait_for_tape(&host, 2);
        let edited_hz = dominant_hz(&edited);
        let energy = |tape: &crate::render::RenderTape| -> f64 {
            tape.interleaved
                .iter()
                .map(|v| f64::from(*v).abs())
                .sum::<f64>()
        };
        eprintln!(
            "[test] 220Hz base={:.1} edited={:.1} | 440Hz base={:.1} edited={:.1}",
            tone_energy(&original, 220.0),
            tone_energy(&edited, 220.0),
            tone_energy(&original, 440.0),
            tone_energy(&edited, 440.0)
        );
        eprintln!(
            "[test] base frames={} energy={:.3} hz={:.1} | edited frames={} energy={:.3} hz={:.1}",
            original.frames,
            energy(&original),
            base_hz,
            edited.frames,
            energy(&edited),
            edited_hz
        );
        let ratio = edited_hz / base_hz.max(1.0);
        assert!(
            (1.7..2.3).contains(&ratio),
            "an octave up must roughly double the pitch: {base_hz:.1} Hz -> {edited_hz:.1} Hz"
        );
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

            block.attach();
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

        block.attach();
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
