use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use super::byte_budget_cache::ByteBudgetCache;
use arc_swap::ArcSwap;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::state::TimelineState;
use crate::time_stretch::time_stretch_interleaved;
use hifishifter_kernel::rt_retirement::RetirementKeeper;

/// Maximum stretch cache entries to prevent unbounded RAM growth.
const MAX_STRETCH_CACHE_ENTRIES: usize = 128;

/// 总线停滞多久后把瞬时电平归零回落。播放时音频块每 ~10-20ms 前进一次
/// 总线 generation，此阈值远大于块周期、远小于人的感知；暂停 / 流停止后
/// 用它触发电平条回落（见 meter 线程内的注释）。
const METER_STALL_FALLBACK: Duration = Duration::from_millis(150);

/// 电平表总线槽位数 = 电平表可覆盖的轨道上限（见 TrackMeterBus）。
/// 每槽 4 字节，取宽裕值让"死表"远离现实工程规模；越界时 meter 线程告警一次。
const METER_BUS_CAPACITY: usize = 512;

use super::device_watch::{
    output_device_changed, spawn_default_output_watcher, OutputDeviceIdentity,
};
use super::metronome;
use super::mix::TrackMeterBus;
use super::output_stream::{build_output_stream, OutputStreamArgs};
use super::resource_manager::ResourceManager;
use super::snapshot::{
    build_snapshot, build_snapshot_for_file, schedule_stretch_jobs, source_bounds_frames,
};
use super::types::{
    AudioEngineStateSnapshot, EngineCommand, EngineSnapshot, ResampledStereo, StretchJob,
    StretchKey, TrackMeterValue,
};
use crate::pitch_clip::schedule_clip_pitch_jobs;

use crate::pitch_clip::get_clip_pitch_midi_global;
use tauri::{Emitter, Manager};

// debug_eprintln! 宏现定义于 crate 根（lib.rs），全 crate 热路径可用。

/// 拉伸进度推送给前端的事件 payload。
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StretchProgressPayload {
    /// 是否正在拉伸
    pub active: bool,
    /// 当前正在拉伸的 clip 名称
    pub clip_name: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackMeterEntryPayload {
    pub track_id: String,
    pub peak_linear: f32,
    pub max_peak_linear: f32,
    pub clipped: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackMeterPayload {
    pub tracks: Vec<TrackMeterEntryPayload>,
}

/// 音高检测完成后推送给前端的事件 payload。
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClipPitchDataPayload {
    /// clip 的唯一 ID
    pub clip_id: String,
    /// MIDI 曲线第 0 帧对应的 timeline 绝对时间（秒）
    pub curve_start_sec: f64,
    /// MIDI 音高曲线（每帧一个 MIDI 值，0 表示无声）
    pub midi_curve: Vec<f32>,
    /// 帧周期（毫秒），用于前端将帧索引转换为时间
    pub frame_period_ms: f64,
}

pub struct AudioEngine {
    tx: mpsc::Sender<EngineCommand>,

    snapshot: Arc<ArcSwap<EngineSnapshot>>,

    is_playing: Arc<AtomicBool>,
    /// "传输层原地等待渲染"标志（RT 回调动态判定）：后台渲染进行中遇到
    /// 未渲染窗口（起播位置落在其中，或播放途中推进到其开头）时置位 ——
    /// 保持播放态、位置冻结、静音输出，直到覆盖该窗口的 Clip 全部就绪
    /// （后台渲染入库 + 快照重建）后自动开始/继续播放。就绪块 / 停止 /
    /// 暂停判定清除。后台渲染线程据它决定是否重建快照解锁等待。
    play_start_wait: Arc<AtomicBool>,
    target: Arc<Mutex<Option<String>>>,
    base_frames: Arc<AtomicU64>,
    position_frames: Arc<AtomicU64>,
    duration_frames: Arc<AtomicU64>,
    sample_rate: Arc<AtomicU32>,
    /// worker 线程已把 sample_rate 初始化为输出设备真实值的标志。
    /// 初始 AtomicU32 为 44100：真实设备就是 44100 时无法用“值变化”判断
    /// 就绪（此前 play_original 由此固定白等 200ms）。
    worker_ready: Arc<AtomicBool>,
    /// Shutdown flag shared with the meter thread.
    meter_shutdown: Arc<AtomicBool>,
    /// Shutdown flag for the default-output watcher.
    device_watch_shutdown: Arc<AtomicBool>,
}

impl Clone for AudioEngine {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            snapshot: self.snapshot.clone(),
            is_playing: self.is_playing.clone(),
            play_start_wait: self.play_start_wait.clone(),
            target: self.target.clone(),
            base_frames: self.base_frames.clone(),
            position_frames: self.position_frames.clone(),
            duration_frames: self.duration_frames.clone(),
            sample_rate: self.sample_rate.clone(),
            worker_ready: self.worker_ready.clone(),
            meter_shutdown: self.meter_shutdown.clone(),
            device_watch_shutdown: self.device_watch_shutdown.clone(),
        }
    }
}

impl AudioEngine {
    pub fn new() -> Self {
        Self::with_app_handle(None)
    }

    /// 把一条引擎命令投递给 worker。
    ///
    /// 【为什么需要它】内核侧的宿主回调（`app_events::AppHostCallbacks`）拿不到
    /// `AudioEngine` 的私有 channel，只能经由这个入口 —— 它取代了原来的
    /// `set_app_handle`：句柄不再经命令通道传递（那条通道现在是纯数据的内核类型）。
    pub fn send(&self, command: EngineCommand) {
        let _ = self.tx.send(command);
    }

    fn with_app_handle(app_handle: Option<tauri::AppHandle>) -> Self {
        let (tx, rx) = mpsc::channel::<EngineCommand>();
        let tx_for_worker = tx.clone();

        let is_playing = Arc::new(AtomicBool::new(false));
        let play_start_wait = Arc::new(AtomicBool::new(false));
        let target = Arc::new(Mutex::new(None));
        let base_frames = Arc::new(AtomicU64::new(0));
        let position_frames = Arc::new(AtomicU64::new(0));
        let duration_frames = Arc::new(AtomicU64::new(0));
        let sample_rate = Arc::new(AtomicU32::new(44100));
        let worker_ready = Arc::new(AtomicBool::new(false));

        // Shared snapshot store for both the audio callback and command-side status queries.
        // This is updated by the engine worker thread.
        let snapshot: Arc<ArcSwap<EngineSnapshot>> =
            Arc::new(ArcSwap::from_pointee(EngineSnapshot::empty(44100)));
        let meter_state = Arc::new(Mutex::new(HashMap::<String, TrackMeterValue>::new()));
        let meter_generation = Arc::new(AtomicU64::new(0));
        let meter_app_handle = Arc::new(Mutex::new(app_handle.clone()));
        let meter_shutdown = Arc::new(AtomicBool::new(false));
        let meter_shutdown_for_thread = meter_shutdown.clone();
        let device_watch_shutdown = Arc::new(AtomicBool::new(false));
        // Lock-free per-track peak handoff between the audio callback and the
        // meter thread (see mix::TrackMeterBus). 槽位数即电平表可覆盖的轨道上限：
        // 超出容量的轨道混音与播放完全正常，但电平表恒为 0（静默失效），因此容量
        // 取宽裕值（每槽 4 字节，512 槽仅 2 KB），并在 meter 线程对越界做一次性告警
        // （见下方 meter loop）。
        let meter_bus = Arc::new(TrackMeterBus::with_capacity(METER_BUS_CAPACITY));
        let meter_bus_for_meter = meter_bus.clone();

        // 节拍器：命令线程写配置/响点表，音频回调无锁读取（见 metronome.rs）。
        let metro = Arc::new(metronome::MetronomeRt::new());
        let metro_for_worker = metro.clone();
        let metro_for_cb = metro.clone();

        let metro_for_thread = metro_for_worker;
        let is_playing_thread = is_playing.clone();
        let play_start_wait_thread = play_start_wait.clone();
        let target_thread = target.clone();
        let base_frames_thread = base_frames.clone();
        let position_frames_thread = position_frames.clone();
        let duration_frames_thread = duration_frames.clone();
        let sample_rate_thread = sample_rate.clone();
        let worker_ready_thread = worker_ready.clone();

        let snapshot_for_thread = snapshot.clone();
        {
            let meter_state = meter_state.clone();
            let meter_generation = meter_generation.clone();
            let meter_app_handle = meter_app_handle.clone();
            let meter_shutdown = meter_shutdown_for_thread.clone();
            let snapshot_for_meter = snapshot_for_thread.clone();
            let meter_bus = meter_bus_for_meter.clone();
            // 传输层心跳所需的原子量（只读观测，用于让日志能验证"播放是否在推进、
            // 是否反复进入等待"——光标是否应当前进由此可判定）。
            let hb_is_playing = is_playing_thread.clone();
            let hb_position = position_frames_thread.clone();
            let hb_waiting = play_start_wait_thread.clone();
            let hb_sample_rate = sample_rate_thread.clone();
            thread::spawn(move || {
                // stderr logging, map rebuilds and event emission all live on
                // this thread so the audio callback stays lock-free.
                //
                // 本线程**只做观测**（日志 / 电平发布）：等待的解除由渲染线程
                // 推送 `RenderedClipsChanged` 驱动，不依赖此处的采样或判定，
                // 因而没有任何观测窗口或时序竞态。
                let debug_pending = hifishifter_kernel::log_policy::probes_enabled();
                let mut last_bus_generation = u64::MAX;
                let mut last_map_generation = u64::MAX;
                let mut last_pending_log_sec = u64::MAX;
                // 上一次记录的等待位置（去重等待日志：回调每个块都上报，
                // 同一位置只记一次；位置变化 / 新一轮等待重新记录）。
                let mut last_wait_log_pos = u64::MAX;
                // 传输层心跳计时（见下方 heartbeat）。
                let mut last_transport_heartbeat = Instant::now();
                // 总线最后一次前进的时刻（播放中每块都前进）。
                let mut last_bus_advance = Instant::now();
                // 越界告警只发一次的标志（meter 线程私有，无需跨线程）。
                let meter_over_capacity_warned = AtomicBool::new(false);
                loop {
                    if meter_shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(33));

                    // Drain RT-side reports and log them here, not in the callback.
                    // 仅用于诊断：等待中的音频回调每块上报一次冻结位置。
                    let transport_wait_pos = meter_bus.take_transport_wait_pos();
                    if transport_wait_pos > 0 && transport_wait_pos != last_wait_log_pos {
                        log::warn!(
                            "[mix] transport waiting for render at pos={} — frozen until the covering clips are ready",
                            transport_wait_pos
                        );
                    }
                    last_wait_log_pos = transport_wait_pos;

                    // ── 传输层心跳（每 5s，仅播放/等待期间）──────────────────
                    // 排查"音频在响但光标不动"时唯一可靠的观测：能直接看出
                    // 播放是否在推进、是否在反复进入原地等待（音频被静音冻结）。
                    if last_transport_heartbeat.elapsed() >= Duration::from_secs(5) {
                        let playing = hb_is_playing.load(Ordering::Relaxed);
                        let waiting = hb_waiting.load(Ordering::Relaxed);
                        if playing || waiting {
                            let sr = hb_sample_rate.load(Ordering::Relaxed).max(1) as f64;
                            let pos = hb_position.load(Ordering::Relaxed) as f64 / sr;
                            log::warn!(
                                "[transport] heartbeat pos={:.2}s playing={} waiting={}",
                                pos,
                                playing,
                                waiting
                            );
                        }
                        last_transport_heartbeat = Instant::now();
                    }
                    let pending_pos = meter_bus.take_pending_pos();
                    if debug_pending && pending_pos > 0 {
                        let snap = snapshot_for_meter.load_full();
                        let sec = pending_pos / u64::from(snap.sample_rate.max(1));
                        if sec != last_pending_log_sec {
                            last_pending_log_sec = sec;
                            for clip in snap.clips.iter() {
                                if clip.needs_synthesis && clip.rendered_pcm.is_none() {
                                    let clip_end =
                                        clip.start_frame.saturating_add(clip.length_frames);
                                    if clip.start_frame < pending_pos && clip_end > pending_pos {
                                        log::warn!(
                                            "[mix] PENDING clip_id={} needs_synthesis=true rendered_pcm=None pos={}",
                                            clip.clip_id, pending_pos
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // RT 侧登记的音频线程故障（panic / 流错误）在这里落地成日志：
                    // 回调里只做了一次原子写（见 `mix::RtFault`）。
                    if let Some(fault) = meter_bus.take_rt_fault() {
                        log::error!("{}", fault.log_message());
                    }

                    let bus_generation = meter_bus.generation();
                    let map_generation = meter_generation.load(Ordering::Relaxed);
                    let bus_advanced = bus_generation != last_bus_generation;
                    let map_changed = map_generation != last_map_generation;
                    if !bus_advanced && !map_changed {
                        // ── 总线停滞回落（暂停 / 流停止）──────────────────
                        // RT 回调在 is_playing=false 时静音返回、不再
                        // publish，总线 generation 冻结。若冻结前最后一块的
                        // 峰值尚未被本线程消费，上面的 bus_advanced 合并会把
                        // slot 里保留的非零块峰值重新写进刚被 idle 清零的
                        // meter_state，随后 generation 永不再前进 —— 电平计
                        // 就冻结在最后一次峰值上（“暂停不回落”）。这里在
                        // 停滞超过 METER_STALL_FALLBACK 后把瞬时峰值归零并
                        // 重发一次：无论最后一块与 idle 的先后时序如何都会
                        // 自愈；峰值保持（max）与削波标志保留不动。正常播放
                        // 时总线每块都前进，此分支不会触发。
                        if last_bus_advance.elapsed() < METER_STALL_FALLBACK {
                            continue;
                        }
                        let Ok(mut state) = meter_state.lock() else {
                            continue;
                        };
                        let mut changed = false;
                        for value in state.values_mut() {
                            if value.peak_linear != 0.0 {
                                value.peak_linear = 0.0;
                                changed = true;
                            }
                        }
                        if !changed {
                            continue;
                        }
                    }

                    // Merge fresh RT peaks into the shared meter map. When only
                    // map_generation changed (worker reset/idle), re-emit as-is.
                    if bus_advanced {
                        last_bus_advance = Instant::now();
                        let snap = snapshot_for_meter.load_full();
                        // 轨道数超出总线容量：电平表对越界轨道静默失效（slot_peak
                        // 返回 0）。这是"看不见的半失效"，必须至少说一次——否则
                        // 用户只会报告"轨道 65+ 的电平表不动了"而无人能归因。
                        if snap.track_ids.len() > METER_BUS_CAPACITY
                            && !meter_over_capacity_warned.swap(true, Ordering::Relaxed)
                        {
                            log::warn!(
                                "[engine] track count {} exceeds meter bus capacity {METER_BUS_CAPACITY}: meters dead beyond capacity (audio unaffected)",
                                snap.track_ids.len()
                            );
                        }
                        let Ok(mut state) = meter_state.lock() else {
                            continue;
                        };
                        let mut next: HashMap<String, TrackMeterValue> =
                            HashMap::with_capacity(snap.track_ids.len());
                        for (slot, track_id) in snap.track_ids.iter().enumerate() {
                            let block_peak = meter_bus.slot_peak(slot);
                            let prev = state.get(track_id).copied().unwrap_or_default();
                            next.insert(
                                track_id.clone(),
                                TrackMeterValue {
                                    peak_linear: block_peak,
                                    max_peak_linear: prev.max_peak_linear.max(block_peak),
                                    clipped: prev.clipped || block_peak >= 1.0,
                                },
                            );
                        }
                        *state = next;
                    }
                    last_bus_generation = bus_generation;
                    last_map_generation = map_generation;

                    let payload = {
                        let Ok(state) = meter_state.lock() else {
                            continue;
                        };
                        TrackMeterPayload {
                            tracks: state
                                .iter()
                                .map(|(track_id, value)| TrackMeterEntryPayload {
                                    track_id: track_id.clone(),
                                    peak_linear: value.peak_linear,
                                    max_peak_linear: value.max_peak_linear,
                                    clipped: value.clipped,
                                })
                                .collect(),
                        }
                    };

                    let Some(app) = meter_app_handle.lock().ok().and_then(|guard| guard.clone())
                    else {
                        continue;
                    };
                    let _ = app.emit("track_meter", payload);
                }
            });
        }
        // watcher 的通道与关停标志要在 worker 线程里用，先把句柄带进闭包。
        let tx_for_device_watch = tx.clone();
        let device_watch_shutdown_for_thread = device_watch_shutdown.clone();
        thread::spawn(move || {
            // 【顺序】先等出一个真正能播的 stream，再拿它的身份启动 watcher。
            // 反过来（watcher 先自己猜基线）会让两者基线不同步：worker 用了
            // 另一个设备建流，而 watcher 认为"没变"，真正的切换永远被漏掉。
            let startup =
                match wait_for_output_stream(&rx, &worker_ready_thread, |device, config| {
                    let channels = config.channels() as usize;
                    let sample_format = config.sample_format();
                    let stream_config: cpal::StreamConfig = config.into();
                    build_output_stream(OutputStreamArgs {
                        device,
                        config: stream_config,
                        sample_format,
                        channels,
                        snapshot: snapshot_for_thread.clone(),
                        is_playing: is_playing_thread.clone(),
                        play_start_wait: play_start_wait_thread.clone(),
                        position_frames: position_frames_thread.clone(),
                        duration_frames: duration_frames_thread.clone(),
                        meter_bus: meter_bus.clone(),
                        metro: metro_for_cb.clone(),
                        tx: tx_for_worker.clone(),
                    })
                }) {
                    Some(value) => value,
                    None => return,
                };

            let mut stream = Some(startup.stream);
            let mut sr = startup.sample_rate;
            let mut current_output = Some(startup.identity.clone());
            // 等待期缓冲下来的命令（按原序）在进入主循环后先处理。
            let mut pending_commands: VecDeque<EngineCommand> = startup.buffered;
            sample_rate_thread.store(sr, Ordering::Relaxed);
            worker_ready_thread.store(true, Ordering::Release);
            spawn_default_output_watcher(
                tx_for_device_watch.clone(),
                device_watch_shutdown_for_thread.clone(),
                Some(startup.identity),
            );

            // Re-initialize the shared snapshot to the actual output sample rate.
            //
            // 用 `swap` 取回旧值而不是 `store`：此刻音频回调可能已经在跑并持有
            // 旧快照，若在这里就地丢弃，回调稍后的释放就成了最后一次 —— 析构会
            // 落到实时线程上（见 `crate::rt_retirement`）。
            //
            // 【为什么是 8 而不是 3】后台渲染会**按 clip** 换入快照（现已节流到
            // 200ms 一次，见 `SNAPSHOT_REFRESH_MIN_MS`），但一次编辑风暴仍可能在
            // 两个音频回调之间产生多次换代。`keep` 小于"两次回调之间的换代数"时，
            // 最旧的已退休值会在发布线程上被丢弃 —— 而实时线程可能仍持有它，
            // 于是析构重新落回实时线程，正是这套机制要避免的爆音。
            // 8 给实时线程留出一个宽裕的释放窗口；代价很小：退休快照与
            // `RenderedClipCache` **共享**同一批 PCM 的 `Arc`，多留几代并不复制音频。
            let mut retired_snapshots = RetirementKeeper::<Arc<EngineSnapshot>>::new(8);
            retired_snapshots.retire(snapshot_for_thread.swap(Arc::new(EngineSnapshot::empty(sr))));

            // Async resource manager for decoded/resampled PCM.
            let resources = ResourceManager::new(tx_for_worker.clone());
            let cache_for_cmd = resources.cache().clone();

            // Time-stretch cache: computed, pitch-preserving loop buffers.
            let stretch_cache: Arc<Mutex<ByteBudgetCache<StretchKey, ResampledStereo>>> = Arc::new(
                Mutex::new(ByteBudgetCache::from_env(MAX_STRETCH_CACHE_ENTRIES)),
            );
            let stretch_cache_for_cmd = stretch_cache.clone();
            let stretch_cache_for_worker = stretch_cache.clone();

            let stretch_inflight: Arc<Mutex<HashSet<StretchKey>>> =
                Arc::new(Mutex::new(HashSet::new()));
            let stretch_inflight_for_cmd = stretch_inflight.clone();
            let stretch_inflight_for_worker = stretch_inflight.clone();

            let (stretch_tx, stretch_rx) = mpsc::channel::<StretchJob>();

            // Worker that computes SoundTouch stretch off the command thread.
            // Keep it small to avoid CPU spikes.
            {
                let cache = cache_for_cmd.clone();
                let stretch_cache = stretch_cache_for_worker.clone();
                let inflight = stretch_inflight_for_worker.clone();
                let tx_ready = tx_for_worker.clone();
                thread::spawn(move || {
                    while let Ok(job) = stretch_rx.recv() {
                        let runtime_available = match job.algorithm {
                            crate::time_stretch::StretchAlgorithm::SoundTouchDll => {
                                crate::soundtouch::is_available()
                            }
                            crate::time_stretch::StretchAlgorithm::SignalsmithStretch
                            | crate::time_stretch::StretchAlgorithm::LinearResample => true,
                            crate::time_stretch::StretchAlgorithm::ElastiqueSoloist => false,
                        };
                        if !runtime_available {
                            if let Ok(mut s) = inflight.lock() {
                                s.remove(&job.key);
                            }
                            continue;
                        }

                        let src = match super::io::get_resampled_stereo_cached(
                            &job.key.path,
                            job.key.out_rate,
                            &cache,
                        ) {
                            Some(v) => v,
                            None => {
                                if let Ok(mut s) = inflight.lock() {
                                    s.remove(&job.key);
                                }
                                continue;
                            }
                        };

                        let (src_start, src_end) = source_bounds_frames(
                            job.source_start_sec,
                            job.source_end_sec,
                            src.frames,
                            job.key.out_rate,
                        );
                        let loop_in_frames = src_end.saturating_sub(src_start) as usize;
                        if loop_in_frames < 2 {
                            if let Ok(mut s) = inflight.lock() {
                                s.remove(&job.key);
                            }
                            continue;
                        }

                        let playback_rate =
                            if job.playback_rate.is_finite() && job.playback_rate > 0.0 {
                                job.playback_rate
                            } else {
                                1.0
                            };
                        if (playback_rate - 1.0).abs() <= 1e-6 {
                            if let Ok(mut s) = inflight.lock() {
                                s.remove(&job.key);
                            }
                            continue;
                        }

                        let i0 = (src_start as usize) * 2;
                        let i1 = (src_end as usize) * 2;
                        if i1 > src.pcm.len() || i0 + 4 > i1 {
                            if let Ok(mut s) = inflight.lock() {
                                s.remove(&job.key);
                            }
                            continue;
                        }

                        let loop_pcm: Vec<f32> = src.pcm[i0..i1].to_vec();
                        let loop_out_frames =
                            ((loop_in_frames as f64) / playback_rate).round().max(2.0) as usize;

                        // 向前端推送拉伸开始事件
                        if let Some(ref app) = job.app_handle {
                            let _ = app.emit(
                                "stretch_progress",
                                StretchProgressPayload {
                                    active: true,
                                    clip_name: Some(job.clip_name.clone()),
                                },
                            );
                        }

                        let stretched = time_stretch_interleaved(
                            &loop_pcm,
                            2,
                            job.key.out_rate,
                            loop_out_frames,
                            job.algorithm,
                        );

                        let stretched_src = ResampledStereo {
                            sample_rate: job.key.out_rate,
                            frames: loop_out_frames,
                            pcm: Arc::new(stretched),
                        };

                        if let Ok(mut m) = stretch_cache.lock() {
                            // Byte-budgeted LRU: evicts the least-recently-used
                            // entry beyond the entry cap or the byte budget.
                            let pcm_bytes = stretched_src.pcm_bytes();
                            m.insert(job.key.clone(), stretched_src, pcm_bytes);
                        }

                        // 向前端推送拉伸完成事件
                        if let Some(ref app) = job.app_handle {
                            let _ = app.emit(
                                "stretch_progress",
                                StretchProgressPayload {
                                    active: false,
                                    clip_name: None,
                                },
                            );
                        }

                        let _ = tx_ready.send(EngineCommand::StretchReady { key: job.key });
                    }
                });
            }

            let mut last_timeline: Option<Arc<TimelineState>> = None;
            let mut last_play_file: Option<(PathBuf, f64, String)> = None;
            let mut app_handle_for_worker: Option<tauri::AppHandle> = app_handle;
            // 合并连发推送时被 try_recv 取出的下一条命令：渲染线程可能以远快
            // 于本线程消费的速度逐 clip 推送 RenderedClipsChanged，排水时取到
            // 的其他命令必须原序保留、下一轮优先处理（FIFO 语义不变）。
            // 启动期缓冲下来的命令同样排在这个队列的前面。
            let mut stashed_cmd: Option<EngineCommand> = None;
            // 设备重建失败后的重试时刻（见 `next_command`）。
            let mut device_retry: Option<Instant> = None;
            let mut device_retry_backoff = Duration::from_millis(250);
            // 最近一次**成功**的强制重建时刻（风暴节流判据，见下方 force 分支）。
            let mut last_forced_rebuild_ok: Option<Instant> = None;

            loop {
                // 宿主句柄改为**惰性**获取：worker 可能早于 Tauri setup 启动
                // （`AudioEngine::new()` 在 `AppState::default()` 里就跑起来了），
                // 而句柄是 setup 时注册到 `app_events` 的进程级出口。
                // 之前它靠 `EngineCommand::SetAppHandle` 送进来，而那条命令通道
                // 现在是内核的纯数据类型，不能塞 `AppHandle`。
                if app_handle_for_worker.is_none() {
                    app_handle_for_worker = crate::app_events::app_handle().cloned();
                }
                let cmd = match stashed_cmd.take() {
                    Some(cmd) => cmd,
                    None => match pending_commands.pop_front() {
                        Some(cmd) => cmd,
                        None => match next_command(&rx, &mut device_retry) {
                            Some(cmd) => cmd,
                            None => break,
                        },
                    },
                };
                // 合并连发的 RenderedClipsChanged：每条推送的处理都是一次全量
                // 快照重建（O(clips)），而重建按当前 last_timeline 整体换入、
                // 天然幂等 —— 排空队列中紧随其后的重复推送，一次重建即可。
                // 不合并时，缓存命中的渲染 pass 会以远快于消费的速度把队列
                // 塞满冗余重建，排在后面的用户命令（stop/seek）被拖延。
                if matches!(cmd, EngineCommand::RenderedClipsChanged) {
                    loop {
                        match rx.try_recv() {
                            Ok(EngineCommand::RenderedClipsChanged) => continue,
                            Ok(other) => {
                                stashed_cmd = Some(other);
                                break;
                            }
                            Err(_) => break,
                        }
                    }
                }
                // Shutdown 无论直接收到还是经排水暂存，都统一在此退出；
                // 下方 match 的 Shutdown 分支因此保持 unreachable 不变。
                if matches!(cmd, EngineCommand::Shutdown) {
                    break;
                }
                let mut state = EngineWorkerState {
                    sr,
                    is_playing: &is_playing_thread,
                    play_start_wait: &play_start_wait_thread,
                    target: &target_thread,
                    base_frames: &base_frames_thread,
                    position_frames: &position_frames_thread,
                    duration_frames: &duration_frames_thread,
                    snapshot: &snapshot_for_thread,
                    cache: &cache_for_cmd,
                    stretch_cache: &stretch_cache_for_cmd,
                    stretch_inflight: &stretch_inflight_for_cmd,
                    stretch_tx: &stretch_tx,
                    resources: &resources,
                    tx: &tx_for_worker,
                    last_timeline: &mut last_timeline,
                    last_play_file: &mut last_play_file,
                    retired_snapshots: &mut retired_snapshots,
                    app_handle: app_handle_for_worker.clone(),
                    meter_state: &meter_state,
                    meter_generation: &meter_generation,
                };
                match cmd {
                    EngineCommand::SetMetronome { config } => {
                        metro_for_thread.store_config(&config);
                    }
                    EngineCommand::SetMetronomeSchedule { clicks } => {
                        metro_for_thread.store_schedule(clicks);
                    }
                    EngineCommand::EvictSourcePath { path } => {
                        handle_evict_source_path(&mut state, &path);
                    }
                    EngineCommand::Stop => handle_stop(&mut state),
                    EngineCommand::SeekSec { sec } => handle_seek_sec(&mut state, sec),
                    EngineCommand::PlayFrom { sec } => handle_play_from(&mut state, sec),
                    EngineCommand::DefaultOutputDeviceChanged { force } => {
                        // 【风暴节流】err_fn 对每一次流错误都发一条 force，而一次
                        // 成功的重建并不证明故障结束 —— 驱动重协商会"建好又立刻
                        // 坏"循环，每圈都是一次全量 build_snapshot + 流churn，退避
                        // 却在每次成功时被复位成 250ms，形成无节流的热循环。
                        // 判据只有一条：上一次成功的强制重建**存活满
                        // STREAM_STORM_QUIET** 才算故障结束、退避才复位。
                        // 风暴内到达的 force：已有排定重试就是冗余，丢弃；否则按
                        // 当前退避排一次 —— 与失败重试共用同一节奏。
                        if force
                            && last_forced_rebuild_ok
                                .is_some_and(|at| at.elapsed() < STREAM_STORM_QUIET)
                        {
                            if device_retry.is_none() {
                                device_retry = Some(Instant::now() + device_retry_backoff);
                                device_retry_backoff =
                                    (device_retry_backoff * 2).min(Duration::from_secs(2));
                                log::warn!(
                                    "AudioEngine: repeated stream fault; forced rebuild deferred by {:?}",
                                    device_retry_backoff
                                );
                            }
                            continue;
                        }
                        match handle_default_output_device_changed(
                            &mut state,
                            &mut stream,
                            &mut sr,
                            &mut current_output,
                            is_playing_thread.clone(),
                            play_start_wait_thread.clone(),
                            position_frames_thread.clone(),
                            duration_frames_thread.clone(),
                            sample_rate_thread.clone(),
                            meter_bus.clone(),
                            metro_for_thread.clone(),
                            &tx_for_worker,
                            force,
                        ) {
                            Ok(()) => {
                                // 成功：回到最短重试间隔，下次失败仍能快速恢复。
                                device_retry = None;
                                device_retry_backoff = Duration::from_millis(250);
                                // 强制重建的成功时间戳是风暴判据；非强制的
                                // watcher 切换不改变风暴状态。
                                if force {
                                    last_forced_rebuild_ok = Some(Instant::now());
                                }
                            }
                            Err(error) => {
                                // 【失败必须可重试】旧实现只打一条日志就 return，
                                // 此时旧 stream 已丢、新 stream 为 None、身份已提交
                                // 成新设备 —— watcher 之后永远判"没变"，用户永久
                                // 静音且无自愈路径。这里安排一次强制重试（指数退避
                                // 上限 2s），期间所有命令照常处理。
                                log::warn!(
                                    "AudioEngine: output device switch failed: {error}; retrying in {:?}",
                                    device_retry_backoff
                                );
                                device_retry = Some(Instant::now() + device_retry_backoff);
                                device_retry_backoff =
                                    (device_retry_backoff * 2).min(Duration::from_secs(2));
                            }
                        }
                    }
                    EngineCommand::SetPlaying { playing, target } => {
                        handle_set_playing(&mut state, playing, target)
                    }
                    EngineCommand::UpdateTimeline(tl) => handle_update_timeline(&mut state, tl),
                    EngineCommand::RenderedClipsChanged => {
                        handle_rendered_clips_changed(&mut state)
                    }
                    EngineCommand::StretchReady { key } => handle_stretch_ready(&mut state, key),
                    EngineCommand::ClipPitchReady { clip_id } => {
                        handle_clip_pitch_ready(&mut state, clip_id)
                    }
                    // 动态（DYN）的后台电平分析：不受 compose_enabled 门控，
                    // 由参数面板请求 dyn 参数时驱动（见 commands::params）。
                    EngineCommand::ScheduleDynLevelAnalysis => {
                        if let Some(tl) = state.last_timeline.clone() {
                            schedule_clip_pitch_jobs(&tl, state.tx, state.sr);
                        }
                    }
                    EngineCommand::AudioReady { key } => handle_audio_ready(&mut state, key),
                    EngineCommand::PlayFile {
                        path,
                        offset_sec,
                        target,
                    } => handle_play_file(&mut state, path, offset_sec, target),
                    // 两个入口（recv 与排水暂存）都已在上方对 Shutdown break；
                    // 若未来新增路径漏掉预过滤，这里降级为跳过而不是 abort。
                    EngineCommand::Shutdown => {
                        log::error!(
                            "engine command loop received an unfiltered Shutdown; skipping it"
                        );
                    }
                }
            }
        });

        Self {
            tx,
            snapshot,
            is_playing,
            play_start_wait,
            target,
            base_frames,
            position_frames,
            duration_frames,
            sample_rate,
            worker_ready,
            meter_shutdown,
            device_watch_shutdown,
        }
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate.load(Ordering::Relaxed).max(1)
    }

    /// 更新节拍器配置（开关 / 音量 / 细分模式 / 重音 / 音色）。
    pub fn set_metronome(&self, config: metronome::MetronomeConfig) {
        let _ = self.tx.send(EngineCommand::SetMetronome { config });
    }

    /// 换入节拍器响点表（升序，输出设备采样率域）。
    pub fn set_metronome_schedule(&self, clicks: Arc<Vec<metronome::MetronomeClick>>) {
        let _ = self.tx.send(EngineCommand::SetMetronomeSchedule { clicks });
    }

    pub fn shutdown(&self) {
        self.device_watch_shutdown.store(true, Ordering::Relaxed);
        self.meter_shutdown.store(true, Ordering::Relaxed);
        let _ = self.tx.send(EngineCommand::Shutdown);
    }

    /// 使指定源路径的解码缓存和拉伸缓存失效。
    ///
    /// 在 `replace_clip_source` 中被调用，早于 `update_timeline` 发送，
    /// 确保引擎在重建 snapshot 时不会使用旧文件的解码/拉伸数据。
    pub fn evict_source_path(&self, path: &str) {
        let _ = self.tx.send(EngineCommand::EvictSourcePath {
            path: path.to_string(),
        });
    }

    pub fn update_timeline(&self, timeline: TimelineState) {
        let _ = self.tx.send(EngineCommand::UpdateTimeline(timeline));
    }

    pub fn seek_sec(&self, sec: f64) {
        let _ = self.tx.send(EngineCommand::SeekSec { sec });
    }

    /// 引擎当前的**绝对**播放位置（秒）。
    ///
    /// 【为什么它是唯一权威】播放位置本来有两个候选来源：引擎的
    /// `base_frames + position_frames` 与前端 Redux 里的 `playheadSec`。后者
    /// 由 30Hz 轮询外推得到（含 IPC 往返时延），与真实可听位置天然差一拍；
    /// 把它再传回后端当起播锚点，等于把这份误差固化进每次播放。因此
    /// `play_original` 一律读这里，前端不再参与起播决策。
    ///
    /// 时间线播放的 `base_frames` 恒为 0（位置一律用绝对帧表达），文件预览
    /// 的 `base_frames` 才是 offset，两者相加即"当前听到的那一刻"。
    pub fn position_sec(&self) -> f64 {
        let sr = self.sample_rate.load(Ordering::Relaxed).max(1) as f64;
        let base = self.base_frames.load(Ordering::Relaxed);
        let pos = self.position_frames.load(Ordering::Relaxed);
        (base.saturating_add(pos)) as f64 / sr
    }

    /// 原子地从绝对秒开始播放时间线。
    /// 从引擎**在命令序里的真实位置**起播。
    ///
    /// 【为什么不用 position_sec() 的读数】那个读数绕过命令序：worker 队列里
    /// 还没应用的 SeekSec / UpdateTimeline 对它不可见，按它起播会覆盖在途的
    /// seek 并把光标拖回去。`PlayFrom { sec: None }` 让位置在 worker 应用完
    /// 此前所有命令之后才解析 —— 起播点因此只来自引擎事实（见
    /// `handle_play_from`，`Some` 分支由 worker 级测试覆盖）。
    /// 调用方负责确认当前会话目标是时间线；文件预览的 base 是文件域偏移。
    pub fn play_from_current(&self) {
        let _ = self.tx.send(EngineCommand::PlayFrom { sec: None });
    }

    #[allow(dead_code)]
    pub fn set_playing(&self, playing: bool, target: Option<&str>) {
        let _ = self.tx.send(EngineCommand::SetPlaying {
            playing,
            target: target.map(|s| s.to_string()),
        });
    }

    /// 引擎侧"单文件预览播放"入口。
    ///
    /// 【为什么保留 allow(dead_code)】整条 `PlayFile` 命令通路仍完整保留 —— 变体
    /// `EngineCommand::PlayFile` 定义在**内核的命令词汇表**里、`handle_play_file` 与
    /// 分派臂都在位，只差这一个 UI 调用方。删掉入口而不动词汇表会让分派 `match` 变成
    /// 非穷尽；而删词汇表是内核 API 变更，超出"清理死代码"的范围。等预览功能重新接上
    /// 时无需重建这条通路。
    #[allow(dead_code)]
    pub fn play_file(&self, path: &Path, offset_sec: f64, target: &str) {
        let _ = self.tx.send(EngineCommand::PlayFile {
            path: path.to_path_buf(),
            offset_sec,
            target: target.to_string(),
        });
    }

    pub fn stop(&self) {
        let _ = self.tx.send(EngineCommand::Stop);
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::Relaxed)
    }

    /// Returns true if the current snapshot contains clips that need synthesis
    /// but have no rendered PCM yet (i.e. a render is in progress or pending).
    pub fn snapshot_has_pending_clips(&self) -> bool {
        let snap = self.snapshot.load();
        snap.clips
            .iter()
            .any(|c| c.needs_synthesis && c.rendered_pcm.is_none())
    }

    pub fn snapshot_state(&self) -> AudioEngineStateSnapshot {
        let sr = self.sample_rate.load(Ordering::Relaxed).max(1);
        let base = self.base_frames.load(Ordering::Relaxed);
        let pos = self.position_frames.load(Ordering::Relaxed);
        let dur = self.duration_frames.load(Ordering::Relaxed);
        AudioEngineStateSnapshot {
            is_playing: self.is_playing(),
            // 原地等待渲染：is_playing=true 但引擎未推进（等待后台渲染完成
            // 后自动开始/继续播放），前端轮询据此冻结光标、不做时延外推。
            waiting_for_render: self.play_start_wait.load(Ordering::Relaxed),
            target: self
                .target
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            base_sec: base as f64 / sr as f64,
            position_sec: pos as f64 / sr as f64,
            duration_sec: dur as f64 / sr as f64,
        }
    }

    /// 渲染线程在完成一个 Clip 的渲染（或命中缓存并注册 key）后调用：
    /// 通知 worker 换入反映最新渲染结果的快照。
    ///
    /// 这是"原地等待渲染"解除的唯一入口 —— 由**产出者**主动推送，
    /// 无需任何轮询、版本号比对或等待状态上报。
    pub fn refresh_rendered_snapshot(&self) {
        let _ = self.tx.send(EngineCommand::RenderedClipsChanged);
    }
}

// ─── Worker 状态结构体 ────────────────────────────────────────────────────────

/// 启动时无输出设备：保持命令通道存活并每秒重试，设备插入后直接返回。
/// 启动成功的产物：可播放的 stream + 它的采样率/身份 + 等待期缓冲的命令。
struct StartupStream {
    stream: cpal::Stream,
    sample_rate: u32,
    identity: OutputDeviceIdentity,
    buffered: VecDeque<EngineCommand>,
}

/// 一次成功的强制重建要"存活"多久才承认故障结束（风暴节流判据）。
///
/// 【为什么是 5 秒】驱动重协商风暴里，坏流会在重建后的几百毫秒内再次报错；
/// 正常使用中一次重建之后 5 秒内又坏的场景几乎不存在。5 秒让风暴期的重建
/// 节奏收敛到约 2s 上限的退避链，而不是每个错误一次全量重建。
const STREAM_STORM_QUIET: std::time::Duration = std::time::Duration::from_secs(5);

/// 取下一条要处理的命令；设备重建的重试到点时**合成**一条强制重建命令。
///
/// 【为什么重试要合成命令，而不是在失败处 sleep】worker 是命令通道的唯一
/// 消费者，阻塞在那里会连用户按下的 Stop 都收不到（表现为"卡住几秒才有反应"）。
/// 用 `recv_timeout` 把等待让给通道：到期前到达的命令照常处理，到期后再重试。
fn next_command(
    rx: &mpsc::Receiver<EngineCommand>,
    device_retry: &mut Option<Instant>,
) -> Option<EngineCommand> {
    if let Some(at) = *device_retry {
        let now = Instant::now();
        if now >= at {
            *device_retry = None;
            return Some(EngineCommand::DefaultOutputDeviceChanged { force: true });
        }
        match rx.recv_timeout(at - now) {
            Ok(EngineCommand::Shutdown) | Err(mpsc::RecvTimeoutError::Disconnected) => return None,
            Ok(cmd) => return Some(cmd),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                *device_retry = None;
                return Some(EngineCommand::DefaultOutputDeviceChanged { force: true });
            }
        }
    }
    match rx.recv() {
        Ok(EngineCommand::Shutdown) | Err(_) => None,
        Ok(cmd) => Some(cmd),
    }
}

/// 启动期：等到"设备 + 配置 + 可播放的 stream"三者齐备，期间命令全部缓冲。
///
/// 【为什么不是"等到设备就返回"】设备枚举成功不等于能建流：被独占、格式协商
/// 失败、USB 瞬态都可能在 `build_output_stream` / `play` 这一步失败。旧实现
/// 在那种情况下直接 `return`（线程结束、命令通道断开、整个音频子系统到重启
/// 前不可用）。这里把"建流成功"也并进等待条件。
///
/// 【为什么命令必须缓冲而不是丢弃】冷启动时默认设备可能还没出现（HDMI 握手、
/// 蓝牙连接），此时前端已经在发 `UpdateTimeline` / `PlayFrom`/`SeekSec`。
/// 旧实现把它们统统吃掉并把 `is_playing` 复位，设备就绪后引擎既没有时间线也
/// 不在播放位置 —— 用户按播放没声，再按一次才有声。
fn wait_for_output_stream<F>(
    rx: &mpsc::Receiver<EngineCommand>,
    worker_ready: &AtomicBool,
    mut build: F,
) -> Option<StartupStream>
where
    F: FnMut(cpal::Device, cpal::SupportedStreamConfig) -> Result<cpal::Stream, String>,
{
    // 无输出路径也能"就绪"：调用方要等的是 worker 起来，而不是有声音。
    worker_ready.store(true, Ordering::Release);
    let mut buffered: VecDeque<EngineCommand> = VecDeque::new();
    let mut backoff = Duration::from_millis(250);
    loop {
        let host = cpal::default_host();
        if let Some(device) = host.default_output_device() {
            if let Ok(config) = device.default_output_config() {
                let sample_rate = config.sample_rate().max(1);
                let identity = OutputDeviceIdentity::from_device(&device, &config);
                match build(device, config) {
                    Ok(stream) => match stream.play() {
                        Ok(()) => {
                            return Some(StartupStream {
                                stream,
                                sample_rate,
                                identity,
                                buffered,
                            })
                        }
                        Err(error) => {
                            log::error!("AudioEngine: stream.play failed: {error}; will retry")
                        }
                    },
                    Err(error) => log::error!("AudioEngine: {error}; will retry"),
                }
            }
        }
        // 失败（或暂时没设备）：把等待时间让给命令通道，期间命令进缓冲，Shutdown 退出。
        match rx.recv_timeout(backoff) {
            Ok(EngineCommand::Shutdown) | Err(mpsc::RecvTimeoutError::Disconnected) => return None,
            Ok(cmd) => {
                if !matches!(cmd, EngineCommand::RenderedClipsChanged) {
                    buffered.push_back(cmd);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        backoff = (backoff * 2).min(Duration::from_secs(2));
    }
}

/// Worker 线程的所有可变状态，按命令处理函数传递。
fn reset_track_meter_state(
    meter_state: &Arc<Mutex<HashMap<String, TrackMeterValue>>>,
    meter_generation: &Arc<AtomicU64>,
) {
    if let Ok(mut state) = meter_state.lock() {
        state.clear();
        meter_generation.fetch_add(1, Ordering::Relaxed);
    }
}

fn idle_track_meter_state(
    meter_state: &Arc<Mutex<HashMap<String, TrackMeterValue>>>,
    meter_generation: &Arc<AtomicU64>,
) {
    if let Ok(mut state) = meter_state.lock() {
        for value in state.values_mut() {
            value.peak_linear = 0.0;
        }
        meter_generation.fetch_add(1, Ordering::Relaxed);
    }
}

struct EngineWorkerState<'a> {
    sr: u32,
    is_playing: &'a Arc<AtomicBool>,
    /// 播放起点等待渲染标志（RT 原子的 worker 侧引用，见
    /// [`AudioEngine::play_start_wait`]）。
    play_start_wait: &'a Arc<AtomicBool>,
    target: &'a Arc<Mutex<Option<String>>>,
    base_frames: &'a Arc<AtomicU64>,
    position_frames: &'a Arc<AtomicU64>,
    duration_frames: &'a Arc<AtomicU64>,
    snapshot: &'a Arc<ArcSwap<EngineSnapshot>>,
    cache: &'a Arc<Mutex<ByteBudgetCache<(PathBuf, u32), ResampledStereo>>>,
    stretch_cache: &'a Arc<Mutex<ByteBudgetCache<StretchKey, ResampledStereo>>>,
    stretch_inflight: &'a Arc<Mutex<HashSet<StretchKey>>>,
    stretch_tx: &'a mpsc::Sender<StretchJob>,
    resources: &'a ResourceManager,
    tx: &'a mpsc::Sender<EngineCommand>,
    /// 上一份时间轴。
    ///
    /// 【为什么是 `Arc`】它要被整份克隆的场合有两处，都是热路径：
    /// `handle_rendered_clips_changed`（渲染线程**每完成一个 clip** 就推一次）
    /// 与 `ScheduleDynLevelAnalysis`。存 `Arc` 后这些克隆退化成引用计数 +1，
    /// 而 `TimelineState` 含全部 clips/takes 与 `params_by_root_track` 的
    /// 音高/曲线向量，可达 MB 级 —— 原来每次都要深拷贝一份。
    last_timeline: &'a mut Option<Arc<TimelineState>>,
    last_play_file: &'a mut Option<(PathBuf, f64, String)>,
    /// 被换下的快照暂存区（见 [`crate::rt_retirement`]）。换快照时必须把旧值交给
    /// 它，而不是让 `ArcSwap::store` 就地丢弃 —— 否则析构会落到音频回调上。
    retired_snapshots: &'a mut RetirementKeeper<Arc<EngineSnapshot>>,
    /// 可选的 Tauri app handle，用于向前端推送事件
    app_handle: Option<tauri::AppHandle>,
    meter_state: &'a Arc<Mutex<HashMap<String, TrackMeterValue>>>,
    meter_generation: &'a Arc<AtomicU64>,
}

// ─── 命令处理函数 ─────────────────────────────────────────────────────────────

fn handle_stop(s: &mut EngineWorkerState) {
    s.is_playing.store(false, Ordering::Relaxed);
    // 清除起点等待：等待中的用户停止立即生效，不再自动起播。
    s.play_start_wait.store(false, Ordering::Relaxed);
    *s.target.lock().unwrap_or_else(|e| e.into_inner()) = None;
    // 【停止 ≠ 归零】`position_frames` 保留真实停止点，下一次 `play_original`
    // 直接从这里起播。此前这里把位置清零，而自然结束（`advance_playback_position`）
    // 冻结位置 —— 同一个"停止"在引擎里于是有了两种互斥状态：一种答得出真实
    // 停止点，一种答 0。引擎因此不可能是可靠权威，前端只好另带一个锚点，
    // 于是锚点与实际起播点继续分裂。
    //
    // 文件预览是唯一例外：它的位置是"文件内偏移 + 已播放"（`base_frames` 非 0），
    // 停止后没有"下次从哪继续"的语义，归零更诚实。
    if s.last_play_file.is_some() {
        s.base_frames.store(0, Ordering::Relaxed);
        s.position_frames.store(0, Ordering::Relaxed);
    } else {
        // 时间线播放：base 恒为 0（绝对帧口径），位置保持不动。
        s.base_frames.store(0, Ordering::Relaxed);
    }
    *s.last_play_file = None;
    idle_track_meter_state(s.meter_state, s.meter_generation);
    // ★ 起播等待期的垫音抑制随播放会话结束而清空：抑制集合表达的是"本次
    // 起播时哪些 clip 未就绪"（见 synth_clip_cache 的垫音抑制说明），跨会话
    // 残留会让下一次播放中段的参数编辑被错误地禁止垫音。下一次 play_original
    // 会按当时的渲染状态整体重新登记。
    crate::synth_clip_cache::clear_pad_suppressed_clips();
    // ★ 不再在停止时清空 pending_rendered_keys：key 在渲染失效处按需移除
    //（见 invalidate_clip_all_caches），始终与缓存条目一致；清空只会让下一次
    // 播放的首个快照把已渲染 clip 判为未渲染 —— 无谓的"起播即静音等待"。
}

/// 原子起播：写入绝对位置后立即置 playing，避免“is_playing=true 但位置还是旧值”的窗口。
///
/// `sec = None`（`PlayFromCurrent` 语义）在**这里**才解析位置：worker 已把
/// 此前排队的 SeekSec / UpdateTimeline 全部应用完，`base + position` 就是
/// 真实的起播点。IPC 线程直接读原子做不到这一点 —— 它读到的是命令序之外的
/// 旧值。
fn handle_play_from(s: &mut EngineWorkerState, sec: Option<f64>) {
    let sec = match sec {
        Some(sec) => sec,
        None => {
            let sr = s.sr.max(1) as f64;
            s.base_frames
                .load(Ordering::Relaxed)
                .saturating_add(s.position_frames.load(Ordering::Relaxed)) as f64
                / sr
        }
    };
    let sec = sec.max(0.0);
    let frame = (sec * s.sr as f64).round().max(0.0) as u64;
    s.base_frames.store(0, Ordering::Relaxed);
    s.position_frames.store(frame, Ordering::Relaxed);
    s.is_playing.store(true, Ordering::Relaxed);
    s.play_start_wait.store(false, Ordering::Relaxed);
    *s.target.lock().unwrap_or_else(|e| e.into_inner()) = Some("original".to_string());
    *s.last_play_file = None;
    reset_track_meter_state(s.meter_state, s.meter_generation);
}

/// 采样率变化后必须整体失效的资源。
///
/// 【为什么必须全部清空，而不是"让它们自然过期"】
/// - 解码/重采样缓存按 `(path, out_rate)` 索引：旧 sr 的条目永远命不中，只会
///   白占预算；
/// - 拉伸缓存同理（`StretchKey.out_rate`）；
/// - 整 clip 渲染缓存按 `clip_id` 索引，但内容是**按输出采样率渲染的 PCM**，
///   留着旧条目就会被当成命中 —— 播放出变速变调的结果。
fn invalidate_sample_rate_bound_caches(s: &EngineWorkerState, old_sr: u32, new_sr: u32) {
    if old_sr == new_sr {
        return;
    }
    // 中毒锁也要清：跳过清理只会把旧采样率的 PCM 留在缓存里 —— 正是这条
    // 函数要消灭的"切换设备后变速变调"。与全仓的 lock 约定一致。
    s.cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
    s.stretch_cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    s.stretch_inflight
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    hifishifter_kernel::synth_clip_cache::global_rendered_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    hifishifter_kernel::synth_clip_cache::clear_pending_rendered_keys();
}

/// 切换默认输出设备：**事务式** —— 新 stream 建好且播放成功之前，旧 stream 一直活着。
///
/// 【为什么必须事务式】旧实现先 `stream.take()` 丢掉旧流、再提交新采样率与
/// 新身份、最后才建流。任何一步失败（USB 枚举瞬态、设备被独占、格式协商失败）
/// 都会留下"旧流已丢、新流为 None、身份已指向新设备"的状态：watcher 之后恒判
/// "没变"，没有任何自愈路径 —— 用户看到的是界面在播、声音永久消失。
///
/// 顺序（任何失败都保持现状并让调用方安排重试）：
/// 1. 预构建新采样率下的快照（不发布）；
/// 2. 预构建新 stream（旧 stream 仍在出声）；
/// 3. 暂停旧流 → 播放新流；失败则恢复旧流并返回错误；
/// 4. 全部成功后才提交原子量 / 快照 / 身份，并丢弃旧流。
#[allow(clippy::too_many_arguments)]
fn handle_default_output_device_changed(
    s: &mut EngineWorkerState,
    stream: &mut Option<cpal::Stream>,
    sr: &mut u32,
    current_output: &mut Option<OutputDeviceIdentity>,
    is_playing: Arc<AtomicBool>,
    play_start_wait: Arc<AtomicBool>,
    position_frames: Arc<AtomicU64>,
    duration_frames: Arc<AtomicU64>,
    sample_rate: Arc<AtomicU32>,
    meter_bus: Arc<TrackMeterBus>,
    metro: Arc<metronome::MetronomeRt>,
    tx: &mpsc::Sender<EngineCommand>,
    force: bool,
) -> Result<(), String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default output device".to_string())?;
    let supported = device
        .default_output_config()
        .map_err(|error| format!("default_output_config failed: {error}"))?;
    let identity = OutputDeviceIdentity::from_device(&device, &supported);
    // `force` 跳过身份判等：RT 侧流错误与失败重试都必须能重建，
    // 即使新设备身份与旧值完全相同（回退到同一台设备是常态）。
    if !force && !output_device_changed(current_output.as_ref(), Some(&identity)) {
        return Ok(());
    }

    let was_playing = is_playing.load(Ordering::Relaxed);
    let old_sr = (*sr).max(1);
    let old_base = s.base_frames.load(Ordering::Relaxed);
    let old_position = s.position_frames.load(Ordering::Relaxed);
    let audible_sec = (old_base + old_position) as f64 / old_sr as f64;
    let new_sr = supported.sample_rate().max(1);

    // 1) 预构建快照（尚未发布）。
    let new_snapshot = if let Some(timeline) = s.last_timeline.as_ref() {
        build_snapshot(timeline, new_sr, s.cache, s.stretch_cache)
    } else if let Some((path, _offset, _target)) = s.last_play_file.as_ref() {
        // 【偏移不能重复累加】`handle_play_file` 把 `base_frames` 设成
        // `offset * sr`，因此 `audible_sec` 里已经包含 offset；这里再 `+offset`
        // 就是 2×，表现为"切设备后预览位置跳到两倍处"。
        build_snapshot_for_file(path, new_sr, audible_sec.max(0.0), s.cache)
    } else {
        EngineSnapshot::empty(new_sr)
    };
    let new_duration_frames = new_snapshot.duration_frames;

    // 2) 预构建 stream（旧 stream 未动）。
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let new_stream = build_output_stream(OutputStreamArgs {
        device,
        config,
        sample_format,
        channels,
        snapshot: s.snapshot.clone(),
        is_playing: is_playing.clone(),
        play_start_wait: play_start_wait.clone(),
        position_frames: position_frames.clone(),
        duration_frames: duration_frames.clone(),
        meter_bus,
        metro,
        tx: tx.clone(),
    })?;

    // 3) 切换播放：先停旧流；新流起不来就把它恢复，保持"切换失败等于没发生"。
    if let Some(old_stream) = stream.as_ref() {
        let _ = old_stream.pause();
    }
    if let Err(error) = new_stream.play() {
        if let Some(old_stream) = stream.as_ref() {
            let _ = old_stream.play();
        }
        return Err(format!("new output stream play failed: {error}"));
    }

    // 4) 提交。
    let (new_base, new_position) =
        super::device_watch::rescale_position_frames(old_base, old_position, old_sr, new_sr);
    *sr = new_sr;
    sample_rate.store(new_sr, Ordering::Relaxed);
    s.base_frames.store(new_base, Ordering::Relaxed);
    s.position_frames.store(new_position, Ordering::Relaxed);
    duration_frames.store(new_duration_frames, Ordering::Relaxed);
    s.retired_snapshots
        .retire(s.snapshot.swap(Arc::new(new_snapshot)));
    s.play_start_wait.store(false, Ordering::Relaxed);
    if let Some((_, offset, _)) = s.last_play_file.as_mut() {
        *offset = audible_sec.max(0.0);
    }
    *current_output = Some(identity);
    if let Some(old_stream) = stream.replace(new_stream) {
        let _ = old_stream.pause();
        drop(old_stream);
    }
    is_playing.store(was_playing, Ordering::Relaxed);

    // 5) 采样率变了：sr 绑定资源全部失效（节拍器响点表由命令层按新 sr 重建）。
    invalidate_sample_rate_bound_caches(s, old_sr, new_sr);
    // 整 clip 渲染结果被清空后必须重新渲染，否则本次播放会停在"等待渲染"。
    if old_sr != new_sr {
        crate::synth_clip_cache::clear_pad_suppressed_clips();
    }
    Ok(())
}

fn handle_seek_sec(s: &mut EngineWorkerState, sec: f64) {
    let sec = sec.max(0.0);
    let frame = (sec * s.sr as f64).round().max(0.0) as u64;
    // Timeline playback reports absolute position via position_frames.
    s.base_frames.store(0, Ordering::Relaxed);
    s.position_frames.store(frame, Ordering::Relaxed);
}

fn handle_set_playing(s: &mut EngineWorkerState, playing: bool, target: Option<String>) {
    s.is_playing.store(playing, Ordering::Relaxed);
    // 等待标志由音频回调动态判定（未就绪块置位 / 就绪块清除），命令侧
    // 一律复位：新播放会话 / 停止都从"非等待"开始，避免残留等待态。
    s.play_start_wait.store(false, Ordering::Relaxed);
    *s.target.lock().unwrap_or_else(|e| e.into_inner()) = target;
    if playing {
        reset_track_meter_state(s.meter_state, s.meter_generation);
    } else {
        idle_track_meter_state(s.meter_state, s.meter_generation);
    }
}

/// 渲染结果已变更 → 按当前 last_timeline 重建快照并换入。
///
/// 这是"原地等待渲染"得以解除的唯一机制（见
/// [`EngineCommand::RenderedClipsChanged`]）：
/// 渲染线程每处理完一个 Clip 就推送一次，worker 换入包含最新 rendered_pcm
/// 的快照；音频回调在下一个块重新判定窗口就绪性 —— 就绪即自动开始/继续播放。
///
/// 刻意保持**最小路径**：不执行 `handle_update_timeline` 的播放暂停判定与
/// pitch 分析调度副作用 —— 本命令在渲染 pass 期间会被高频发送（每 Clip 一次），
/// 必须幂等、无副作用、无反馈循环。
fn handle_rendered_clips_changed(s: &mut EngineWorkerState) {
    let Some(tl) = s.last_timeline.clone() else {
        return;
    };
    let snap = build_snapshot(&tl, s.sr, s.cache, s.stretch_cache);
    s.duration_frames
        .store(snap.duration_frames, Ordering::Relaxed);
    s.retired_snapshots.retire(s.snapshot.swap(Arc::new(snap)));
    idle_track_meter_state(s.meter_state, s.meter_generation);
}

/// `TimelineState::resolve_root_track_id` 的**查表版**。
///
/// 【为什么需要】原函数每次调用都要沿 parent 链走一遍，每跳一次分配一个
/// `String` 并对 `tracks` 线性扫描；而它在**按 clip** 的循环里被反复调用。
/// 预先建好 `root_track_map` 之后查询是 O(1)。
///
/// 【为什么这样写才是等价】`resolve_root_track_id` 对未知 track_id 返回的是
/// 它自己，因此这里用 `unwrap_or(track_id)` 补齐；空 id 返回 `None`。等价性由
/// `hifishifter-kernel/tests/root_track_map.rs` 逐例对拍守住。
fn root_of(roots: &HashMap<&str, &str>, track_id: &str) -> Option<String> {
    if track_id.trim().is_empty() {
        return None;
    }
    Some(roots.get(track_id).copied().unwrap_or(track_id).to_string())
}

/// 轨道参数变化是否会影响**底层合成输出**。
///
/// 共通混音级曲线（volume / pan / dyn，含旧 `hifigan_volume`）被显式排除：
/// 它们由 mix 阶段实时应用，改动只应换入新快照，**不得**失效合成缓存、
/// 不得触发后台重渲染 —— 否则拖动音量/动态会持续打断渲染线程并让播放卡顿。
fn track_params_affect_render(
    old: &crate::state::TrackParamsState,
    new: &crate::state::TrackParamsState,
) -> bool {
    old.frame_period_ms != new.frame_period_ms
        || old.pitch_orig != new.pitch_orig
        || old.pitch_edit != new.pitch_edit
        || old.tension_edit != new.tension_edit
        || extra_curves_affect_render(&old.extra_curves, &new.extra_curves)
        || old.extra_params != new.extra_params
}

/// 比较两组 extra_curves，忽略只在混音阶段消费的共通曲线。
fn extra_curves_affect_render(
    old: &std::collections::HashMap<String, Vec<f32>>,
    new: &std::collections::HashMap<String, Vec<f32>>,
) -> bool {
    let mut keys: std::collections::HashSet<&str> = std::collections::HashSet::new();
    keys.extend(old.keys().map(|k| k.as_str()));
    keys.extend(new.keys().map(|k| k.as_str()));
    keys.into_iter().any(|key| {
        if crate::renderer::common_params::is_common_mix_param(key) {
            return false;
        }
        old.get(key) != new.get(key)
    })
}

/// 轨道级**渲染设置**是否变化 —— 合成算法与 Compose 都直接进入渲染键
/// （见 `build_rendered_hash_input`：算法决定 `renderer_id`，Compose 决定是否
/// 走声码器链路），任一项变化都必须让该轨道组重新渲染。
///
/// 【为什么单独成函数】这两项是 `Track` 上的字段，**不在** clip 级比较口径里
/// （`render_shape_changed` 只看 clip，`track_params_affect_render` 只看
/// `TrackParamsState`）。判定一旦漏项，切换算法就会被当成"无变化"：缓存不失效、
/// 后台预渲染不启动，用户感知为"切了合成算法音频没变"。
fn track_render_settings_differ(old: &crate::state::Track, new: &crate::state::Track) -> bool {
    old.pitch_analysis_algo != new.pitch_analysis_algo || old.compose_enabled != new.compose_enabled
}

/// 使某个根轨道组下**所有** clip 的合成缓存失效。
///
/// 返回是否确有 clip 被失效：空轨道组返回 `false`，调用方据此决定是否需要请求
/// 后台重渲染 —— 没有 clip 就没有可渲染的工作，不该空转一轮渲染 pass。
fn invalidate_root_group_clips(tl: &TimelineState, root_id: Option<&str>) -> bool {
    let mut invalidated = false;
    for clip in &tl.clips {
        if tl.resolve_root_track_id(&clip.track_id).as_deref() == root_id {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
            invalidated = true;
        }
    }
    invalidated
}

fn handle_update_timeline(s: &mut EngineWorkerState, tl: TimelineState) {
    // DYN 的原声电平基线组装需要对 timeline 的可变借用（见下方"动态"段）。
    let mut tl = tl;
    debug_eprintln!(
        "[engine] handle_update_timeline: tracks={}, clips={}",
        tl.tracks.len(),
        tl.clips.len()
    );

    // ── 1. 合成缓存失效 ──────────────────────────────────────────────────
    // 当某个 root_track 的 pitch_edit 曲线发生变化时，
    // 使该 track 上所有 clip 的合成缓存失效，触发下次播放时重新合成。
    // WORLD 和 ONNX 共享同一个 synth_clip_cache。
    // 用于后台预渲染自动触发：仅当本次 update 确实使缓存失效时才触发。
    let mut any_cache_invalidated = false;

    // 轨道级渲染设置（合成算法 / Compose）是否变化。见 `track_render_settings_differ`
    // 与下方"轨道级渲染设置变更"段：这两项决定渲染键，但不在 clip 级比较口径里。
    // 无 last_timeline（首次更新）时按"全变"处理，与下方 `needs_pitch_schedule`
    // 原先的 `is_none_or` 语义一致 —— 该结论在函数末尾复用，两处判定不会漂移。
    let mut track_render_settings_changed = s.last_timeline.is_none();

    // ── 0. 旧时间轴的索引 ────────────────────────────────────────────────
    //
    // 【为什么必须在最前面建】下面三处都要按 id 找旧值：删除 clip 的判定、
    // clip 级参数比较、轨道级渲染设置比较。原实现用 `old_tl.clips.iter().find()`
    // / `old_tl.tracks.iter().find()` 就地线性扫描，于是整个函数是 O(clips²) +
    // O(tracks²)；而函数中段其实**已经**建过一份 `old_clips_map` 并注明"把后续
    // 查询从 O(N²) 降维到 O(N)"，只是没覆盖到那两个循环。这里一次建好、全程复用。
    let old_clips_map: HashMap<&str, &crate::state::Clip> = s
        .last_timeline
        .as_ref()
        .map(|old_tl| old_tl.clips.iter().map(|c| (c.id.as_str(), c)).collect())
        .unwrap_or_default();
    let old_tracks_map: HashMap<&str, &crate::state::Track> = s
        .last_timeline
        .as_ref()
        .map(|old_tl| old_tl.tracks.iter().map(|t| (t.id.as_str(), t)).collect())
        .unwrap_or_default();
    // `track_id -> root_track_id` 的两份表（旧 / 新），供下面的按 clip 循环查表。
    let old_root_by_track: HashMap<&str, &str> = s
        .last_timeline
        .as_ref()
        .map(|old_tl| old_tl.root_track_map())
        .unwrap_or_default();
    // 只供下面这个"合成缓存失效"循环使用。它的借用必须在 DYN 段对 `tl` 的
    // 可变借用之前结束（NLL 按最后一次使用判定），因此 DYN 段之后会另建一份
    // 绑定到 `Arc` 上的同名表。
    let new_root_by_track: HashMap<&str, &str> = tl.root_track_map();

    if let Some(old_tl) = s.last_timeline.as_ref() {
        use std::collections::HashSet;

        let new_clip_ids: HashSet<&str> = tl.clips.iter().map(|c| c.id.as_str()).collect();

        // 删除的 clip 立即全量失效缓存，避免残留 pending key / 旧渲染结果。
        for old_clip in &old_tl.clips {
            if !new_clip_ids.contains(old_clip.id.as_str()) {
                crate::synth_clip_cache::invalidate_clip_all_caches(&old_clip.id);
                any_cache_invalidated = true;
            }
        }

        for clip in &tl.clips {
            let old_clip = old_clips_map.get(clip.id.as_str()).copied();

            // 检查该 clip 所在 track 的 pitch_edit 是否发生变化
            let pitch_changed = {
                // 先解析 root_track_id，否则子轨道中的 clip 无法正确失效缓存
                let old_root = root_of(&old_root_by_track, &clip.track_id);
                let new_root = root_of(&new_root_by_track, &clip.track_id);

                let old_params = old_root.and_then(|r| old_tl.params_by_root_track.get(&r));
                let new_params = new_root.and_then(|r| tl.params_by_root_track.get(&r));

                match (old_params, new_params) {
                    (Some(old), Some(new)) => track_params_affect_render(old, new),
                    (None, Some(new)) => track_params_affect_render(&Default::default(), new),
                    (Some(_), None) => true, // track params 被删除
                    (None, None) => false,
                }
            };

            let render_shape_changed = old_clip
                .map(|old| {
                    old.source_path != clip.source_path
                        || old.track_id != clip.track_id
                        || (old.source_start_sec - clip.source_start_sec).abs() > 1e-6
                        || (old.source_end_sec - clip.source_end_sec).abs() > 1e-6
                        || (old.playback_rate - clip.playback_rate).abs() > 1e-6
                        || old.reversed != clip.reversed
                        || old.loop_enabled != clip.loop_enabled
                        // 声道模式改变条件化语义（渲染输入/扇出/引擎读取映射），
                        // 与 reversed 同级 —— 不进比较会让引擎层感知不到变化，
                        // 不触发主动重渲染（用户感知：改了模式音频没变）。
                        || old.channel_mode != clip.channel_mode
                        || (old.length_sec - clip.length_sec).abs() > 1e-6
                        // 检测同路径文件替换：
                        // - duration_frames / source_sample_rate：文件长度或采样率变化
                        // - source_file_mtime：文件修改时间变化（最可靠的检测信号，
                        //   replace_clip_sources 每次都会更新此字段，即使文件长度未变也能检测到替换）
                        // clip 在时间轴上的位置改变同样改变渲染窗口：`start_frame`
                        // 参与按键、且 pitch 曲线窗口随之平移。漏掉这一项会让
                        // 移动后的精确键 miss，而 miss 又会退回"同 clip_id 的最近
                        // 一次渲染"垫音 —— 播的正是移动前那个窗口的 PCM。
                        || (old.start_sec - clip.start_sec).abs() > 1e-9
                        || old.duration_frames != clip.duration_frames
                        || old.source_sample_rate != clip.source_sample_rate
                        || old.source_file_mtime != clip.source_file_mtime
                        || old.active_take_id != clip.active_take_id
                        || old.source_file_fingerprint != clip.source_file_fingerprint
                        // Clip 级覆盖率（extra_curves / extra_params）走
                        // `unwrap_or(&track)` 的二选一语义：它的出现/消失会让
                        // 整份 track 级参数被替换，而 track 参数本身毫无变化 ——
                        // 不比对就会拿着旧的有效参数继续命中缓存。
                        || old.extra_curves != clip.extra_curves
                        || old.extra_params != clip.extra_params
                        || old.formant_morph != clip.formant_morph
                })
                .unwrap_or(false);

            // 收集需要清理旧源文件缓存的路径（源文件发生变更时）
            let mut stale_source_paths: Vec<String> = Vec::new();
            if let Some(old) = old_clip {
                let source_changed = old.source_path != clip.source_path
                    || old.duration_frames != clip.duration_frames
                    || old.source_sample_rate != clip.source_sample_rate
                    || old.source_file_mtime != clip.source_file_mtime;
                if source_changed {
                    // 旧的源路径（如果与新路径不同，旧的也需要清理）
                    if let Some(ref old_path) = old.source_path {
                        if old.source_path != clip.source_path {
                            stale_source_paths.push(old_path.clone());
                        }
                    }
                    // 当前源路径（即使是同一路径，文件内容也可能已变）
                    if let Some(ref cur_path) = clip.source_path {
                        stale_source_paths.push(cur_path.clone());
                    }
                }
            }

            if render_shape_changed {
                // 片段源范围/轨道归属/速率/长度等变化后，旧渲染结果不可安全复用。
                crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
                // 同时使 formant 缓存失效（formant 缓存按 clip_id + source_path 双重 key）
                crate::formant_cache::invalidate_formant_cache_for_clip(&clip.id);
                any_cache_invalidated = true;
            } else if pitch_changed {
                // 仅 pitch 曲线变化时保留最近一次完整渲染，允许短时无缝垫音。
                crate::synth_clip_cache::invalidate_clip_for_pitch_edit(&clip.id);
                any_cache_invalidated = true;
            }

            // ── 清理路径级缓存（解码缓存 + stretch 缓存 + inflight）──────────
            // 当源文件变更时（包括同路径替换），必须使所有以文件路径为 key 的缓存失效。
            // 使用字符串比较（与 handle_evict_source_path 保持一致），兼容不同 PathBuf 表示形式。
            for stale_path in &stale_source_paths {
                // 1. 解码缓存（解码 LRU，按字节预算）
                if let Ok(mut cache) = s.cache.lock() {
                    cache.invalidate_where(|(p, _)| {
                        p.to_string_lossy().as_ref() == stale_path.as_str()
                    });
                }

                // 2. Stretch 缓存
                if let Ok(mut sc) = s.stretch_cache.lock() {
                    sc.invalidate_where(|k| {
                        k.path.to_string_lossy().as_ref() == stale_path.as_str()
                    });
                }

                // 3. Stretch inflight
                if let Ok(mut si) = s.stretch_inflight.lock() {
                    let inflight_keys: Vec<StretchKey> = si
                        .iter()
                        .filter(|k| k.path.to_string_lossy().as_ref() == stale_path.as_str())
                        .cloned()
                        .collect();
                    for k in inflight_keys {
                        si.remove(&k);
                    }
                }
            }
        }

        // ── extra_params 变更检测 ──────────────────────────────────────────
        // 当 track 级别的 extra_params（如 breath_enabled、synth_mode 等）发生变化时，
        // 使该 root track 及其子轨道上所有 clip 的合成缓存失效，并触发后台预渲染。
        // 场景：用户切换"气声"功能开关，或更改其他静态声码器参数。
        // 此检测必须放在 clip 循环之后，以确保每个 root track 只检查一次。
        for (root_id, new_params) in &tl.params_by_root_track {
            let old_extra_params = old_tl
                .params_by_root_track
                .get(root_id)
                .map(|p| &p.extra_params);
            if old_extra_params != Some(&new_params.extra_params) {
                debug_eprintln!(
                    "[engine] extra_params changed for root_track={}, invalidating all clips on this track group",
                    root_id
                );
                any_cache_invalidated |= invalidate_root_group_clips(&tl, Some(root_id.as_str()));
            }
        }

        // ── 轨道级渲染设置变更（合成算法 / Compose）────────────────────────
        // 与 extra_params 同属"轨道级、影响底层合成输出"的变更：使整组缓存失效，
        // 让本次编辑立刻进入后台渲染。场景：用户切换轨道的合成算法
        // （WORLD ↔ NSF-HiFiGAN ↔ vslib），或开关 Compose。
        //
        // 判据见 `track_render_settings_differ`：这两项都参与渲染键，但不在
        // clip 级比较口径里 —— 不在此单独判定，切换算法时 `any_cache_invalidated`
        // 会保持 false，后台预渲染根本不会启动。
        for track in &tl.tracks {
            let changed = match old_tracks_map.get(track.id.as_str()) {
                Some(old_track) => track_render_settings_differ(old_track, track),
                // 新增轨道没有旧值可比，按"已变化"处理（与旧判定一致）。
                None => true,
            };
            if !changed {
                continue;
            }
            debug_eprintln!(
                "[engine] Track '{}' render settings changed (algo/compose), invalidating all clips on this track group",
                track.id
            );
            track_render_settings_changed = true;
            let root_id = tl.resolve_root_track_id(&track.id);
            any_cache_invalidated |= invalidate_root_group_clips(&tl, root_id.as_deref());
        }
    }

    // ── 后台预渲染触发 ──────────────────────────────────────────────────────
    // 仅当本次 update_timeline 确实使缓存失效时才触发（避免形成反馈循环）。
    // any_cache_invalidated 由上方的 clip 变化检测代码设置。
    //
    // 触发时机与后台预渲染开关的关系（开关只决定"编辑是否自动触发"）：
    // - 已有渲染 pass 在跑 → 无论开关如何都**合流重启**：失效必须反映进
    //   重渲染，否则等待中的传输层永远拿不到正确参数的结果；
    // - 没有 pass 在跑 → 仅当开关启用，**或传输层正在播放**（播放中的等待
    //   需要渲染结果来解除，见原地等待契约）时启动。
    if any_cache_invalidated {
        use std::sync::atomic::Ordering;
        let enabled = crate::commands::playback::AUTO_BG_RENDER_ENABLED.load(Ordering::Relaxed);
        let running = crate::commands::playback::BG_RENDER_ACTIVE.load(Ordering::Relaxed);
        let transport_playing = s.is_playing.load(Ordering::Relaxed);
        if running {
            // 渲染已在运行 → 合流地请求重启（不再立即取消在途渲染，
            // 避免加载 / 连续编辑期间的失效风暴反复打断渲染）。
            // 渲染循环会在 clip 边界按静默窗口决定是否升级为取消。
            debug_eprintln!("[engine] bg render already running, requesting coalesced restart ({} clips invalidated)", tl.clips.len());
            crate::commands::playback::request_bg_render_restart();
        } else if (enabled || transport_playing) && !tl.clips.is_empty() {
            debug_eprintln!(
                "[engine] auto-triggering background render ({} clips invalidated, transport_playing={})",
                tl.clips.len(),
                transport_playing
            );
            if let Some(app) = s.app_handle.as_ref() {
                crate::commands::playback::request_background_render(app);
            }
        }
    }

    // 旧 Clip 的 Hash 表已在函数开头建好（见"0. 旧时间轴的索引"），全程复用。

    // ── 动态（DYN）原声电平基线的组装/调度 ────────────────────────────────
    // 动态是**混音级**参数：即使未开 Compose 也要生效，因此它的分析触发条件
    // 与音高不同 —— 只要根轨道组存在有效的 dyn 曲线，就必须保证基线就绪
    // （否则增益会一直退化为 1.0，用户画的曲线"没声音"）。
    //
    // 这里改的是即将换入的 `tl`，因此本次 snapshot 直接带上新基线；
    // 缓存未命中时并入下面的 `needs_pitch_schedule` 一起提交后台分析。
    //
    // 【位置】必须在 `moved_clip_ids` 等共享借用建立之前完成对 `tl` 的可变借用。
    let mut dyn_needs_analysis = false;
    {
        // 收集需要基线的根轨道：**已画过 dyn**，或**动态面板正打开**（前端登记，
        // 见 `pitch_clip::dyn_needs_level_analysis`）。两者都不看 compose_enabled
        // —— 动态是混音级参数，未开启合成的轨道同样要算。
        let panel_open = match crate::pitch_clip::dyn_panel_open_roots().lock() {
            Ok(roots) => roots.clone(),
            Err(e) => e.into_inner().clone(),
        };
        let roots_using_dyn: Vec<String> = tl
            .params_by_root_track
            .keys()
            .filter(|root| crate::pitch_clip::dyn_needs_level_analysis(&tl, root))
            .cloned()
            .chain(panel_open)
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();
        for root in roots_using_dyn {
            if !crate::pitch_analysis::assemble_dyn_orig_for_engine(&mut tl, &root) {
                dyn_needs_analysis = true;
            }
        }
    }

    // `tl` 的可变借用到此结束（上方 DYN 段是最后一位写者）。包进 `Arc` 之后，
    // "存入 last_timeline" 与后面所有 `&tl` 的读取共享同一份数据，`tl.clone()`
    // 退化成引用计数 +1；而 `handle_rendered_clips_changed` 里那次
    // `last_timeline.clone()` 也从"每个渲染完成的 clip 深拷贝整份时间轴"
    // 变成同样的 +1。见字段说明。
    let tl = Arc::new(tl);
    // 绑定到 `Arc` 上的 root 表：后面的按 clip 循环查它（借用 `Arc` 的目标，
    // 与 `&tl.tracks` 等读取共存无碍）。
    let new_root_by_track: HashMap<&str, &str> = tl.root_track_map();

    // ── 3. 需要重发 clip_pitch_data 的 clip ────────────────────────────────
    //
    // 判据收口在 `clip_pitch_data_needs_emit`：曲线是按 `[start_sec, start_sec + length_sec)`
    // 截取 + resample 后推送的，所以**任何**改变"播放哪段内容 / 在哪一刻播放"的编辑
    // 都必须重发（含**新建**）。旧实现分三路内联判据，漏了两类：新建 clip（分割出的
    // 右半）与"只改 `length_sec` 的编辑"（Loop 分割）。漏发就会在参数编辑器里留下
    // 一条过期的旧曲线 —— 用户看到的"音高线残影"。
    let pitch_emit_clip_ids: std::collections::HashSet<&str> = tl
        .clips
        .iter()
        .filter(|clip| {
            clip_pitch_data_needs_emit(old_clips_map.get(clip.id.as_str()).copied(), clip)
        })
        .map(|clip| clip.id.as_str()) // 零拷贝
        .collect();

    // ── 4. 检测音高相关变化（必须在 last_timeline 更新之前）──────────────────
    // Detect changes at both clip level AND track level (pitch_edit lives in params_by_root_track).
    let track_pitch_edit_changed = s.last_timeline.as_ref().is_some_and(|old_tl| {
        tl.params_by_root_track.iter().any(|(root_id, new_params)| {
            old_tl
                .params_by_root_track
                .get(root_id)
                .is_none_or(|old_params| old_params.pitch_edit != new_params.pitch_edit)
        })
    });

    let clip_changed = track_pitch_edit_changed
        || tl.clips.iter().any(|clip| {
            match old_clips_map.get(clip.id.as_str()) {
                None => true, // 新增 clip
                Some(old) => clip_pitch_params_changed(old, clip),
            }
        });

    // 检测 track 级别的变化。
    //
    // 复用函数开头算出的 `track_render_settings_changed`（同一份"算法 / Compose
    // 是否变化"的结论）：那里已经遍历过 tracks 并据此失效了缓存，这里再算一遍
    // 只会多一次 O(tracks²) 查找，且两处判定可能漂移。
    let has_last_timeline = s.last_timeline.is_some();

    if !has_last_timeline {
        debug_eprintln!(
            "[engine] First timeline update (no last_timeline), forcing pitch schedule"
        );
    }

    // ── 动态（DYN）原声电平基线的组装/调度 ────────────────────────────────
    // 实际组装已在上方（`pitch_emit_clip_ids` 建立之前）完成，此处只并入调度判定。
    //
    // 【它只回答"要不要重跑音频分析"】与"要不要重发曲线"是两件事：移动 / 裁剪 /
    // 分割都不需要重新分析（内容缓存命中），但都必须重发曲线（见下方的唯一发射点）。
    let needs_pitch_schedule = clip_changed || track_render_settings_changed || dyn_needs_analysis;

    *s.last_timeline = Some(tl.clone());
    *s.last_play_file = None;

    // Pre-request decoded PCM for all audible clips (async, non-blocking).
    {
        let track_gain = super::snapshot::compute_track_gains(&tl.tracks);
        // 消除高频循环中的 String 堆分配，直接使用 &str
        let mut audible_tracks: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (&tid, (_gain, muted, solo_ok)) in &track_gain {
            if !*muted && *solo_ok {
                audible_tracks.insert(tid);
            }
        }
        for clip in &tl.clips {
            if clip.muted {
                continue;
            }
            if !audible_tracks.contains(clip.track_id.as_str()) {
                continue;
            }
            let Some(source_path) = clip.source_path.as_ref() else {
                continue;
            };
            let path = Path::new(source_path);
            if !super::io::is_audio_path(path) {
                continue;
            }
            let _ = s.resources.get_or_request(path, s.sr);
        }
    }

    // Schedule stretch work in background (do not block snapshot build).
    if crate::sstretch::is_available() {
        schedule_stretch_jobs(
            &tl,
            s.sr,
            s.stretch_tx,
            s.stretch_inflight.as_ref(),
            s.stretch_cache,
            s.app_handle.as_ref(),
        );
    }

    // 异步预计算所有可见 clip 的 pitch MIDI（缓存未命中时后台计算，
    // 完成后发送 ClipPitchReady 触发 snapshot rebuild，不阻塞当前构建）。
    // 若 clip 已有拉伸后 PCM，优先使用拉伸后 PCM 作为分析输入。
    //
    // 优化：只有当存在 pitch-relevant 参数变化的 clip 或新增 clip 时，
    // 才调用 schedule_clip_pitch_jobs。clip 仅移动（start_sec 变化）时跳过，
    // 避免对所有 clip 做不必要的文件系统 I/O 和缓存查询。
    // 注意：检测逻辑已在 last_timeline 更新之前完成（见上方）
    if needs_pitch_schedule {
        schedule_clip_pitch_jobs(&tl, s.tx, s.sr);
    }

    // 曲线推送（唯一发射点）。
    //
    // 【为什么独立于 `needs_pitch_schedule`】那个标志回答的是"要不要重跑音频分析"，
    // 与"要不要重发曲线"是两件事：移动 / 裁剪 / 分割都不需要重新分析（内容缓存
    // 命中），但都必须重发曲线。此前移动走一条支路、参数变化走另一条、MIDI 再走
    // 第三条，新建 clip 与"只改长度"两类正好落在三条路的缝里 —— 那正是残影。
    if !pitch_emit_clip_ids.is_empty() {
        if let Some(app) = s.app_handle.as_ref() {
            for clip in tl
                .clips
                .iter()
                .filter(|c| pitch_emit_clip_ids.contains(c.id.as_str()))
            {
                emit_clip_pitch_data_for_clip(app, &tl, clip);
            }
        }
    }

    // ── pitch_orig 组装收敛卫兵 ────────────────────────────────────────────
    // 任何 clip 编辑都会使根曲线组装键（`pitch_orig_key`）失效；但把它重新
    // 组装收敛的触发点只有两个：① 某个 clip 的音高分析完成回调 —— 同源文件
    // 已分析过时不会再来；② 前端拉取「音高」参数帧时顺带调度 —— 参数面板
    // 停留在音量/声像等其他参数时不会发生。两条路都断时组装永不收敛，后台
    // 渲染的"输入稳定性卫兵"会把该根所有 clip 判为未就绪跳过，播放中的
    // 传输层原地冻结、永不解除（用户感知：切走参数后编辑不触发渲染，切回
    // 音高才恢复）。此处作为 UpdateTimeline 的必经收口补一次调度：未收敛的
    // 根当场重组，已收敛的根是纯键比较，开销可忽略。
    if let Some(app) = s.app_handle.as_ref() {
        let state = app.state::<crate::state::AppState>();
        let mut guarded_roots: std::collections::HashSet<String> = std::collections::HashSet::new();
        for clip in &tl.clips {
            if let Some(root) = root_of(&new_root_by_track, &clip.track_id) {
                if guarded_roots.insert(root.clone()) {
                    crate::pitch_analysis::maybe_schedule_pitch_orig(&state.timeline, &root);
                }
            }
        }
    }

    let snap = build_snapshot(&tl, s.sr, s.cache, s.stretch_cache);
    s.duration_frames
        .store(snap.duration_frames, Ordering::Relaxed);
    s.retired_snapshots.retire(s.snapshot.swap(Arc::new(snap)));
    idle_track_meter_state(s.meter_state, s.meter_generation);
}

fn handle_stretch_ready(s: &mut EngineWorkerState, key: StretchKey) {
    if let Ok(mut inflight) = s.stretch_inflight.lock() {
        inflight.remove(&key);
    }

    // 拉伸完成后，pitch 分析不再依赖拉伸 PCM（始终分析原始源音频），
    // 所以不需要 invalidate pitch 缓存或重新调度分析。
    // 但需要从全量缓存重新截取+resample 推送 clip_pitch_data（因为 rate 可能变了），
    // 并重新触发 pitch_orig 组装（因为组装也需要根据 rate 做 resample）。
    if let Some(tl) = s.last_timeline.as_ref() {
        if let Some(app) = s.app_handle.as_ref() {
            let state = app.state::<crate::state::AppState>();
            let mut root_track_ids: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            let root_by_track = tl.root_track_map();
            for clip in &tl.clips {
                if let Some(src) = clip.source_path.as_deref() {
                    if std::path::Path::new(src) == key.path.as_path() {
                        // 重新推送该 clip 的 pitch data（使用全量缓存截取+resample）
                        emit_clip_pitch_data_for_clip(app, tl, clip);
                        if let Some(rt) = root_of(&root_by_track, &clip.track_id) {
                            root_track_ids.insert(rt);
                        }
                    }
                }
            }
            // 重新触发 pitch_orig 组装推送
            for rt in &root_track_ids {
                crate::pitch_analysis::maybe_schedule_pitch_orig(&state.timeline, rt);
            }
        }
    }

    if let Some(tl) = s.last_timeline.as_ref() {
        let snap = build_snapshot(tl, s.sr, s.cache, s.stretch_cache);
        s.duration_frames
            .store(snap.duration_frames, Ordering::Relaxed);
        s.retired_snapshots.retire(s.snapshot.swap(Arc::new(snap)));
        idle_track_meter_state(s.meter_state, s.meter_generation);
    }
}

fn handle_clip_pitch_ready(s: &mut EngineWorkerState, clip_id: String) {
    debug_eprintln!("[engine] handle_clip_pitch_ready: clip_id={}", clip_id);
    // clip pitch MIDI 异步预计算完成（全量源音频），缓存已就绪。
    // 向前端推送 ClipPitchData 事件（rate==1 全量传送，rate!=1 截取+resample）。
    if let Some(app) = s.app_handle.as_ref() {
        if let Some(tl) = s.last_timeline.as_ref() {
            if let Some(clip) = tl.clips.iter().find(|c| c.id == clip_id) {
                debug_eprintln!("[engine] Found clip in timeline, emitting clip_pitch_data");
                emit_clip_pitch_data_for_clip(app, tl, clip);
                debug_eprintln!("[engine] clip_pitch_data emitted successfully");
            } else {
                debug_eprintln!("[engine] Clip not found in timeline");
            }
        }
    }

    // 每当一个 clip 分析完成，尝试重新组装整体音高线（pitch_orig）
    // 使用"最上方 clip 为准"的覆盖策略，直接从缓存读取，不重新分析音频
    debug_eprintln!("[engine] Calling maybe_schedule_pitch_orig");
    if let Some(app) = s.app_handle.as_ref() {
        if let Some(tl) = s.last_timeline.as_ref() {
            if let Some(clip) = tl.clips.iter().find(|c| c.id == clip_id) {
                let root_track_id = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();
                if !root_track_id.is_empty() {
                    let state = app.state::<crate::state::AppState>();
                    crate::pitch_analysis::maybe_schedule_pitch_orig(
                        &state.timeline,
                        &root_track_id,
                    );
                    debug_eprintln!("[engine] maybe_schedule_pitch_orig called");
                }
            }
        }
    }

    debug_eprintln!("[engine] Building snapshot...");
    if let Some(tl) = s.last_timeline.as_ref() {
        let snap = build_snapshot(tl, s.sr, s.cache, s.stretch_cache);
        debug_eprintln!(
            "[engine] Snapshot built successfully, duration_frames={}",
            snap.duration_frames
        );
        s.duration_frames
            .store(snap.duration_frames, Ordering::Relaxed);
        s.retired_snapshots.retire(s.snapshot.swap(Arc::new(snap)));
        idle_track_meter_state(s.meter_state, s.meter_generation);
        debug_eprintln!("[engine] Snapshot stored, handle_clip_pitch_ready done");

        // 若此前后台预渲染因为音高分析未完成而跳过了一些 clip，
        // 现在缓存已经就绪，补触发一次渲染，避免用户等待进度结束后
        // 首次播放时仍要重新渲染。
        // 触发规则与 handle_update_timeline 的"开关启用 || 传输层播放"一致：
        // 后台预渲染关闭时，播放触发的按需渲染同样依赖这些 clip 的结果 ——
        // 分析完成前被跳过的 clip 若不补渲染，原地等待中的传输层会一直冻结
        //（直到用户再按一次播放）。标志先消费再分流：无动作可做时不复原
        // 旧标记，避免后续开启 AUTO 时被陈旧标记触发多余渲染。
        if crate::commands::playback::BG_RENDER_PITCH_PENDING
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            if let Some(app) = s.app_handle.as_ref() {
                let auto_enabled = crate::commands::playback::AUTO_BG_RENDER_ENABLED
                    .load(std::sync::atomic::Ordering::Relaxed);
                if auto_enabled {
                    let _ = crate::commands::playback::request_background_render(app);
                } else if s.is_playing.load(std::sync::atomic::Ordering::Relaxed) {
                    crate::commands::playback::ensure_render_pass_running(app);
                }
            }
        }
    }
}

/// 使指定源路径的所有解码缓存和拉伸缓存条目失效。
///
/// 在源文件被替换（`replace_clip_source`）时由 command handler 直接发送到引擎，
/// 确保在 `UpdateTimeline` 处理之前缓存已被清空，避免 `build_snapshot` 使用旧数据。
fn handle_evict_source_path(s: &mut EngineWorkerState, path: &str) {
    // 1. 解码缓存（解码 LRU，按字节预算）—— 使用字符串比较以兼容不同 PathBuf 表示
    if let Ok(mut cache) = s.cache.lock() {
        let before = cache.len();
        cache.invalidate_where(|(p, _)| p.to_string_lossy().as_ref() == path);
        let evicted = before - cache.len();
        if evicted > 0 {
            debug_eprintln!(
                "[engine:evict] decode cache: evicted {} entries for path={}",
                evicted,
                path
            );
        }
    }

    // 2. 拉伸缓存
    if let Ok(mut sc) = s.stretch_cache.lock() {
        let before = sc.len();
        sc.invalidate_where(|k| k.path.to_string_lossy().as_ref() == path);
        let evicted = before - sc.len();
        if evicted > 0 {
            debug_eprintln!(
                "[engine:evict] stretch cache: evicted {} entries for path={}",
                evicted,
                path
            );
        }
    }

    // 3. 拉伸 inflight
    if let Ok(mut si) = s.stretch_inflight.lock() {
        let inflight_keys: Vec<StretchKey> = si
            .iter()
            .filter(|k| k.path.to_string_lossy().as_ref() == path)
            .cloned()
            .collect();
        for k in &inflight_keys {
            si.remove(k);
        }
        if !inflight_keys.is_empty() {
            log::warn!(
                "[engine:evict] stretch inflight: removed {} entries for path={}",
                inflight_keys.len(),
                path
            );
        }
    }
}

fn handle_audio_ready(s: &mut EngineWorkerState, _key: super::types::AudioKey) {
    // A decoded/resampled buffer became available.
    // Requeue stretch work now that the source PCM cache exists, then rebuild the snapshot so
    // plain-audio clips can swap from playback_rate fallback to the selected external stretcher.
    if let Some(tl) = s.last_timeline.as_ref() {
        schedule_stretch_jobs(
            tl,
            s.sr,
            s.stretch_tx,
            s.stretch_inflight,
            s.stretch_cache,
            s.app_handle.as_ref(),
        );
        let snap = build_snapshot(tl, s.sr, s.cache, s.stretch_cache);
        s.duration_frames
            .store(snap.duration_frames, Ordering::Relaxed);
        s.retired_snapshots.retire(s.snapshot.swap(Arc::new(snap)));
        idle_track_meter_state(s.meter_state, s.meter_generation);
    } else if let Some((path, offset_sec, _target)) = s.last_play_file.as_ref() {
        let snap = build_snapshot_for_file(path.as_path(), s.sr, *offset_sec, s.cache);
        s.duration_frames
            .store(snap.duration_frames, Ordering::Relaxed);
        s.retired_snapshots.retire(s.snapshot.swap(Arc::new(snap)));
        idle_track_meter_state(s.meter_state, s.meter_generation);
    }
}

fn handle_play_file(s: &mut EngineWorkerState, path: PathBuf, offset_sec: f64, target: String) {
    *s.last_timeline = None;
    *s.last_play_file = Some((path.clone(), offset_sec, target.clone()));

    // Request decode asynchronously (snapshot building is cache-only).
    let _ = s.resources.get_or_request(path.as_path(), s.sr);

    // Represent the file as a single clip in a snapshot.
    let snap = build_snapshot_for_file(&path, s.sr, offset_sec, s.cache);
    s.duration_frames
        .store(snap.duration_frames, Ordering::Relaxed);
    s.retired_snapshots.retire(s.snapshot.swap(Arc::new(snap)));
    // File playback reports absolute position via base_sec + position_sec.
    let base = (offset_sec.max(0.0) * s.sr as f64).round().max(0.0) as u64;
    s.base_frames.store(base, Ordering::Relaxed);
    s.position_frames.store(0, Ordering::Relaxed);
    // 经 handle_set_playing 进入播放：同步复位停止意图标志（直接 store
    // is_playing 会绕过它，文件播放之后的前台预渲染完成会被历史停止误拒）。
    handle_set_playing(s, true, Some(target));
}

// ─── 辅助函数 ───────────────────────────────────────────────────────────────

/// 计算曲线第 0 帧对应的 timeline 绝对时间（秒）。
///
/// 无论 playback_rate 是多少，emit 的都是「已按 clip 窗口截取 / 重采样」的曲线，
/// 第 0 帧一律对应 clip 在时间线上的起点，因此这里恒为 `start_sec`。
///
/// 【与帧网格的关系】这个秒值是**未量化**的浮点起点；前端落笔时必须把它换算成
/// 工程帧 `round(start_sec·1000/fp)`（与 `pv.orig`、参数块投影同一把尺子），
/// 否则每条 Clip 会各偏一个亚帧相位（≤2.5 ms），放大后与原始音高线错开数像素。
fn compute_pitch_curve_start_sec(clip: &crate::state::Clip) -> f64 {
    // 统一截取后传送：curve 起点 = clip 在时间线上的起始位置
    clip.start_sec.max(0.0)
}

/// 统一的 clip pitch data 推送辅助函数。
/// 从全量缓存中读取 MIDI 曲线，根据 playback_rate 决定是否做 source range 截取 + resample。
/// - rate==1：全量传送，curveStartSec = start_sec - source_start_sec，前端 ctx.clip() 裁剪
/// - rate!=1：source range 截取 → resample 到 clip timeline 长度，curveStartSec = start_sec
pub(crate) fn build_clip_pitch_data_payload(
    tl: &crate::state::TimelineState,
    clip: &crate::state::Clip,
) -> Option<ClipPitchDataPayload> {
    let frame_period_ms = 5.0f64;
    let root = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();

    debug_eprintln!(
        "[pitch:emit] clip_id={} start={:.3}s len={:.3}s src_start={:.3}s src_end={:.3}s pr={:.3} root={}",
        clip.id, clip.start_sec, clip.length_sec,
        clip.source_start_sec, clip.source_end_sec,
        clip.playback_rate, root,
    );

    // ── MIDI clip 路径：直接从 midi_note_data 生成音高曲线 ──
    if let Some(ref notes) = clip.midi_note_data {
        let fp = frame_period_ms.max(0.1);
        let clip_len_sec = clip.length_sec.max(0.0);
        let target_frames = ((clip_len_sec * 1000.0) / fp).round().max(1.0) as usize;
        let mut midi_curve = vec![0.0f32; target_frames];

        let pr = clip.playback_rate as f64;
        let pr_valid = if pr.is_finite() && pr > 0.0 { pr } else { 1.0 };
        // 消费窗口（非 Loop 倒放锚定 se：win=[se−len·r, se]，可为负/超界，
        // 域外为静音；正放/Loop 保持原字段）。音符可见性与镜像全部以该窗口
        // 为基准 —— 否则延伸过的倒放 Clip 曲线整体错位。
        let src_end = if clip.source_end_sec > 0.0 {
            clip.source_end_sec
        } else {
            // 估算源范围终点：用最后一个音符的结束时间
            notes
                .iter()
                .map(|n| n.end_sec)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(clip_len_sec)
                .max(clip_len_sec)
        };
        let src_start = if !clip.loop_enabled && clip.reversed {
            src_end - clip_len_sec * pr_valid
        } else {
            clip.source_start_sec
        };
        let src_total_len = src_end - src_start;

        let clip_visible_frames = target_frames;

        for note in notes {
            // Loop（循环源）：按媒体时长 D 的锚点回绕放置音符（与音频渲染的
            // floor_mod 映射逐帧一致；纯 MIDI clip 无媒体 → 周期退化为窗口
            // 跨度）。不能用窗口比较做可见性过滤 —— split 产生的"环绕窗口"
            // （start > end）会把所有音符误判为越界而全部丢弃。
            if let Some(placement) =
                crate::state::place_note_occurrence_in_loop(clip, note.start_sec, note.end_sec, fp)
            {
                let note_value = note.note;
                let mut cycle_offset = 0usize;
                while cycle_offset < clip_visible_frames {
                    let write_start = cycle_offset + placement.first_start_frame;
                    let write_end =
                        (cycle_offset + placement.first_start_frame + placement.len_frames)
                            .min(clip_visible_frames);
                    if write_start >= write_end {
                        break;
                    }
                    for frame in write_start..write_end {
                        let current = midi_curve[frame];
                        if note_value > current || current <= 0.0 {
                            midi_curve[frame] = note_value;
                        }
                    }
                    cycle_offset += placement.cycle_frames;
                }
                continue;
            }

            if note.end_sec <= src_start || note.start_sec >= src_end {
                continue;
            }
            let rel_start = (note.start_sec - src_start).max(0.0);
            let rel_end = (note.end_sec - src_start).min(src_total_len);
            if rel_end <= rel_start {
                continue;
            }
            // 倒放时镜像音符位置
            let (eff_start, eff_end) = if clip.reversed {
                (
                    (src_total_len - rel_end).max(0.0),
                    (src_total_len - rel_start).min(src_total_len),
                )
            } else {
                (rel_start, rel_end)
            };
            if eff_end <= eff_start {
                continue;
            }
            let note_start_frame = ((eff_start / pr_valid * 1000.0) / fp).round() as usize;
            let note_end_frame = ((eff_end / pr_valid * 1000.0) / fp).round() as usize;
            let note_value = note.note;
            // 非 Loop：单次写入（Loop 已在上方 placement 分支处理）。
            {
                let write_end = note_end_frame.min(target_frames);
                if note_start_frame < write_end {
                    for frame in note_start_frame..write_end {
                        let current = midi_curve[frame];
                        if note_value > current || current <= 0.0 {
                            midi_curve[frame] = note_value;
                        }
                    }
                }
            }
        }

        // 填补音符之间的空隙
        if clip.midi_fill_gaps && target_frames > 0 {
            let first = midi_curve.iter().position(|&v| v > 0.0);
            let last = midi_curve.iter().rposition(|&v| v > 0.0);
            if let (Some(f), Some(l)) = (first, last) {
                if f < l {
                    let mut last_pitch: f32 = 0.0;
                    for i in f..=l {
                        let current = midi_curve[i];
                        if current > 0.0 {
                            last_pitch = current;
                        } else if last_pitch > 0.0 {
                            midi_curve[i] = last_pitch;
                        }
                    }
                }
            }
        }

        let curve_start_sec = compute_pitch_curve_start_sec(clip);
        debug_eprintln!(
            "[pitch:emit] clip_id={} (MIDI) → curve_start={:.3}s curve_len={} fp={:.1}ms total_dur={:.3}s",
            clip.id,
            curve_start_sec,
            midi_curve.len(),
            frame_period_ms,
            midi_curve.len() as f64 * frame_period_ms / 1000.0,
        );
        return Some(ClipPitchDataPayload {
            clip_id: clip.id.clone(),
            curve_start_sec,
            midi_curve,
            frame_period_ms,
        });
    }

    // ── 音频 clip 路径：从 FCPE 缓存获取音高曲线 ──
    let Some(cached_midi) = get_clip_pitch_midi_global(tl, clip, &root, frame_period_ms) else {
        debug_eprintln!("[pitch:emit] clip_id={} → cache miss, skipping", clip.id);
        return None;
    };

    debug_eprintln!(
        "[pitch:emit] clip_id={} cached_midi_len={}",
        clip.id,
        cached_midi.len(),
    );

    // 统一截取 + resample（rate==1 时 resample 实际为无损复制）
    let pr = clip.playback_rate as f64;
    let pr = if pr.is_finite() && pr > 0.0 { pr } else { 1.0 };
    // 非 Loop 倒放：传入真实消费窗口 [se−len·r, se]。
    let (trim_src_start, trim_src_end) = crate::state::clip_pitch_trim_window_sec(clip);
    let mut midi_curve = crate::pitch_clip::trim_and_resample_midi(
        &cached_midi,
        frame_period_ms,
        trim_src_start,
        trim_src_end,
        pr,
        clip.length_sec.max(0.0),
        clip.loop_enabled,
        crate::state::clip_source_media_duration_sec(clip),
        clip.reversed && clip.loop_enabled,
    );
    // 非 Loop 倒放：trim_and_resample_midi 按升序窗口映射（文档契约
    // "调用方在输出后整体翻转"）。前端按时间线顺序直接绘制本曲线，
    // 必须在此翻转 —— 否则倒放 Clip 的音高曲线相对音频整体镜像
    // （与 midi_export / assemble_pitch_orig 的处理保持一致）。
    if clip.reversed && !clip.loop_enabled {
        midi_curve.reverse();
    }
    let curve_start_sec = compute_pitch_curve_start_sec(clip);

    debug_eprintln!(
        "[pitch:emit] clip_id={} → curve_start={:.3}s curve_len={} fp={:.1}ms total_dur={:.3}s",
        clip.id,
        curve_start_sec,
        midi_curve.len(),
        frame_period_ms,
        midi_curve.len() as f64 * frame_period_ms / 1000.0,
    );

    Some(ClipPitchDataPayload {
        clip_id: clip.id.clone(),
        curve_start_sec,
        midi_curve,
        frame_period_ms,
    })
}

fn emit_clip_pitch_data_for_clip(
    app: &tauri::AppHandle,
    tl: &crate::state::TimelineState,
    clip: &crate::state::Clip,
) {
    use tauri::Emitter;
    if let Some(payload) = build_clip_pitch_data_payload(tl, clip) {
        let _ = app.emit("clip_pitch_data", payload);
    }
}

/// 判断 clip 的 pitch 分析相关参数是否发生变化。
/// 全量分析策略：只有源文件变化才需要重新分析。
/// trim/rate 变化时不触发重新分析（全量缓存不会 miss），
/// 而是由 `clip_pitch_data_needs_emit` 重新推送 trim+resample 后的 pitch 数据。
fn clip_pitch_params_changed(old: &crate::state::Clip, new: &crate::state::Clip) -> bool {
    old.source_path != new.source_path
        // 同路径文件替换：mtime 变化说明文件内容已变，必须重新分析 pitch
        || old.source_file_mtime != new.source_file_mtime
}

/// 这个 clip 的 `clip_pitch_data` 是否必须重发。
///
/// 【为什么判据是"可见窗口"而不是"源文件"】曲线按
/// `[start_sec, start_sec + length_sec)` 截取 + resample 后推送（见
/// `emit_clip_pitch_data_for_clip`）；窗口任何一项变了，旧曲线就不再对应这个 clip。
///
/// 旧实现把这件事拆成三处内联判据，漏了两类：
///
/// - **新建**（分割出的右半）：`old_clips_map.get(..)` 为 `None` 时一律跳过，
///   而内容缓存命中又不会走 `ClipPitchReady` 回调 —— 它永远拿不到曲线；
/// - **只改 `length_sec`**（Loop 分割：左侧"不修改 source 窗口"）：四个比较项里
///   没有它 —— 左半保留着分割前那条**比 clip 更长**的曲线。
///
/// 两类漏发都会让前端缓存里留着一条过长的旧曲线；参数编辑器不按 clip 裁剪，于是把
/// 旧内容画到了 clip 外面（用户报的"音高线残影"）。
///
/// 【为什么用位相等而不是容差】这里回答的是"用户有没有动它"，不是"数值是否可视为
/// 相同"：落库的值是精确的，中间帧的乐观值不经过这条路。
fn clip_pitch_data_needs_emit(old: Option<&crate::state::Clip>, new: &crate::state::Clip) -> bool {
    let Some(old) = old else { return true }; // 新建：必须发
    old.start_sec != new.start_sec
        || old.length_sec != new.length_sec
        || old.source_start_sec != new.source_start_sec
        || old.source_end_sec != new.source_end_sec
        || old.playback_rate != new.playback_rate
        || old.reversed != new.reversed
        || old.loop_enabled != new.loop_enabled
        || old.muted != new.muted
        || old.midi_note_data != new.midi_note_data
        || old.midi_fill_gaps != new.midi_fill_gaps
        || old.source_path != new.source_path
        || old.source_file_mtime != new.source_file_mtime
        // 换 active take = 换内容（窗口与速率都可能跟着变）。
        || old.active_take_id != new.active_take_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_engine::resource_manager::ResourceManager;
    use crate::audio_engine::types::{EngineCommand, EngineSnapshot};
    use crate::state::{Clip, TimelineState, TrackParamsState};
    use crate::time_stretch::{update_runtime_stretch_settings, UserStretchAlgorithm};
    use arc_swap::ArcSwap;
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::{mpsc, Arc, Mutex};

    fn make_plain_stretch_timeline() -> TimelineState {
        let mut timeline = TimelineState::default();
        let track_id = timeline.tracks[0].id.clone();
        timeline.clips.push(Clip {
            content_id: None,
            loop_period_sec: None,
            takes: vec![],
            active_take_id: None,
            clip_playback_rate: 1.0,
            id: "clip-1".to_string(),
            track_id,
            name: "clip".to_string(),
            start_sec: 0.0,
            length_sec: 1.0,
            color: "#ffffff".to_string(),
            source_path: Some("E:/virtual/test.wav".to_string()),
            source_path_relative: None,
            duration_sec: Some(1.0),
            duration_frames: Some(44_100),
            source_sample_rate: Some(44_100),
            source_file_mtime: None,
            source_file_size: None,
            source_file_fingerprint: None,
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            clip_gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: 1.0,
            playback_rate: 0.75,
            reversed: false,
            channel_mode: 0,
            source_channels: None,
            loop_enabled: false,
            snap_offset_sec: 0.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_curve: "sine".to_string(),
            fade_out_curve: "sine".to_string(),
            fade_in_shape: 0.0,
            fade_out_shape: 0.0,
            fade_in_dir: 0.0,
            fade_out_dir: 0.0,
            fade_in_s: 0.0,
            fade_out_s: 0.0,
            auto_fade_in_sec: 0.0,
            auto_fade_out_sec: 0.0,
            extra_curves: None,
            extra_params: None,
            formant_morph: None,
            group_id: None,
            midi_fill_gaps: false,
            midi_note_data: None,
        });
        timeline
    }

    /// 曲线推送的判据回归。
    ///
    /// 这两类曾经漏发，正是"参数编辑器里留下旧音高线残影"的来源：新建 clip
    /// （分割出的右半）与"只改 `length_sec` 的编辑"（Loop 分割不动 source 窗口）。
    #[test]
    fn clip_pitch_data_needs_emit_covers_new_and_resized_clips() {
        let clip = make_plain_stretch_timeline().clips.remove(0);

        // 新建：必须发（分割出的右半靠这一条拿到曲线）。
        assert!(clip_pitch_data_needs_emit(None, &clip));
        // 什么都没改：不发（否则每次时间线更新都会重发整条曲线）。
        assert!(!clip_pitch_data_needs_emit(Some(&clip), &clip));

        // 移动。
        let mut moved = clip.clone();
        moved.start_sec += 1.0;
        assert!(clip_pitch_data_needs_emit(Some(&clip), &moved));

        // 只改长度（Loop 分割：source 窗口原样不动）。
        let mut shorter = clip.clone();
        shorter.length_sec -= 0.25;
        assert!(clip_pitch_data_needs_emit(Some(&clip), &shorter));

        // 只改源窗口（Slip）。
        let mut slipped = clip.clone();
        slipped.source_start_sec += 0.25;
        slipped.source_end_sec += 0.25;
        assert!(clip_pitch_data_needs_emit(Some(&clip), &slipped));

        // 静音：曲线不该再画出来，必须重发（payload 里的曲线会变空）。
        let mut muted = clip.clone();
        muted.muted = true;
        assert!(clip_pitch_data_needs_emit(Some(&clip), &muted));
    }

    #[test]
    fn track_params_affect_render_detects_curve_and_static_param_changes() {
        let old = TrackParamsState::default();
        let mut new = old.clone();

        assert!(!track_params_affect_render(&old, &new));

        new.tension_edit.push(1.0);
        assert!(track_params_affect_render(&old, &new));

        new = old.clone();
        new.extra_curves
            .insert("breath_gain".to_string(), vec![0.5]);
        assert!(track_params_affect_render(&old, &new));

        new = old.clone();
        new.extra_params.insert("synth_mode".to_string(), 1.0);
        assert!(track_params_affect_render(&old, &new));
    }

    /// 切换合成算法 / Compose 必须被判成"渲染设置变化"。
    ///
    /// 这两项决定渲染键（`renderer_id` / `compose_enabled`），却不在 clip 级比较
    /// 口径里 —— 漏判的后果是缓存不失效、后台预渲染不启动，用户切了算法音频没变。
    #[test]
    fn track_render_settings_differ_covers_algo_and_compose() {
        let base = TimelineState::default().tracks[0].clone();
        assert!(!track_render_settings_differ(&base, &base.clone()));

        let mut algo_changed = base.clone();
        algo_changed.pitch_analysis_algo = crate::state::PitchAnalysisAlgo::WorldDll;
        assert!(
            track_render_settings_differ(&base, &algo_changed),
            "切换合成算法必须判定为渲染设置变化"
        );

        let mut compose_changed = base.clone();
        compose_changed.compose_enabled = !base.compose_enabled;
        assert!(
            track_render_settings_differ(&base, &compose_changed),
            "切换 Compose 必须判定为渲染设置变化"
        );
    }

    /// 缓存失效按**根轨道组**生效：挂在子轨道上的 clip 也必须被根轨道的变化波及。
    #[test]
    fn invalidate_root_group_clips_resolves_child_tracks_to_their_root() {
        let mut timeline = make_plain_stretch_timeline();
        let root_id = timeline.tracks[0].id.clone();
        let mut child = timeline.tracks[0].clone();
        child.id = "child-track".to_string();
        child.parent_id = Some(root_id.clone());
        timeline.tracks.push(child);
        timeline.clips[0].track_id = "child-track".to_string();

        // clip 挂在子轨道上，但根轨道变化必须命中它。
        assert!(invalidate_root_group_clips(
            &timeline,
            Some(root_id.as_str())
        ));

        // 子轨道 id 不是根 id —— 传入子轨道 id 不应命中（解析出的根不等于它）。
        assert!(!invalidate_root_group_clips(&timeline, Some("child-track")));

        // 空轨道组没有可渲染工作，返回 false，调用方据此不空转一轮渲染 pass。
        assert!(!invalidate_root_group_clips(
            &timeline,
            Some("no-such-track")
        ));
    }

    #[test]
    fn audio_ready_requeues_stretch_jobs_for_plain_audio_clips() {
        update_runtime_stretch_settings(UserStretchAlgorithm::Signalsmith, false, None, None);

        let timeline = make_plain_stretch_timeline();
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineCommand>();
        let (stretch_tx, stretch_rx) = mpsc::channel();

        let resources = ResourceManager::new(engine_tx.clone());
        let cache = resources.cache().clone();
        let stretch_cache = Arc::new(Mutex::new(ByteBudgetCache::new(8, u64::MAX)));
        let stretch_inflight = Arc::new(Mutex::new(HashSet::new()));
        let snapshot = Arc::new(ArcSwap::from_pointee(EngineSnapshot {
            sample_rate: 44_100,
            duration_frames: 0,
            track_ids: Arc::new(vec![]),
            clips: Arc::new(vec![]),
        }));
        let is_playing = Arc::new(AtomicBool::new(false));
        let play_start_wait = Arc::new(AtomicBool::new(false));
        let target = Arc::new(Mutex::new(None));
        let base_frames = Arc::new(AtomicU64::new(0));
        let position_frames = Arc::new(AtomicU64::new(0));
        let duration_frames = Arc::new(AtomicU64::new(0));
        let meter_state = Arc::new(Mutex::new(HashMap::new()));
        let meter_generation = Arc::new(AtomicU64::new(0));
        let mut last_timeline = Some(Arc::new(timeline));
        let mut last_play_file = None;
        let mut retired_snapshots = RetirementKeeper::<Arc<EngineSnapshot>>::new(3);

        let mut worker = EngineWorkerState {
            sr: 44_100,
            is_playing: &is_playing,
            play_start_wait: &play_start_wait,
            target: &target,
            base_frames: &base_frames,
            position_frames: &position_frames,
            duration_frames: &duration_frames,
            snapshot: &snapshot,
            cache: &cache,
            stretch_cache: &stretch_cache,
            stretch_inflight: &stretch_inflight,
            stretch_tx: &stretch_tx,
            resources: &resources,
            tx: &engine_tx,
            last_timeline: &mut last_timeline,
            last_play_file: &mut last_play_file,
            retired_snapshots: &mut retired_snapshots,
            app_handle: None,
            meter_state: &meter_state,
            meter_generation: &meter_generation,
        };

        handle_audio_ready(&mut worker, (PathBuf::from("E:/virtual/test.wav"), 44_100));

        let job = stretch_rx
            .try_recv()
            .expect("audio_ready should requeue stretch work once source audio becomes available");
        assert!(matches!(
            job.algorithm,
            crate::time_stretch::StretchAlgorithm::SignalsmithStretch
        ));
    }

    /// 停止必须保留真实停止点：引擎是"从哪里继续播放"的唯一权威。
    ///
    /// 【为什么这条是回归锚点】此前 `handle_stop` 把 `position_frames` 清零，而
    /// 自然结束/自动暂停只把 `is_playing` 翻成 false、位置冻结 —— 同一个"停止"
    /// 有两种互斥的引擎状态。于是"按 Stop 后再播放"只能靠前端另带锚点，
    /// 锚点一错就整体错位（光标跳回旧起播点）。这里钉住两件事：
    /// 1. Stop 之后 `position_frames` 仍等于停止点（`position_sec()` 可读）；
    /// 2. `base_frames` 归零（时间线播放一律绝对帧）。
    ///
    /// 文件预览必须归零：它的位置是"文件内偏移 + 已播放"，没有"下次继续"语义。
    #[test]
    fn stop_preserves_the_real_stop_point_for_timeline_playback() {
        let sample_rate = 48_000u32;
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineCommand>();
        let (stretch_tx, _stretch_rx) = mpsc::channel();

        let resources = ResourceManager::new(engine_tx.clone());
        let cache = resources.cache().clone();
        let stretch_cache = Arc::new(Mutex::new(ByteBudgetCache::new(8, u64::MAX)));
        let stretch_inflight = Arc::new(Mutex::new(HashSet::new()));
        let snapshot = Arc::new(ArcSwap::from_pointee(EngineSnapshot::empty(sample_rate)));
        let is_playing = Arc::new(AtomicBool::new(false));
        let play_start_wait = Arc::new(AtomicBool::new(false));
        let target = Arc::new(Mutex::new(None));
        let base_frames = Arc::new(AtomicU64::new(0));
        let position_frames = Arc::new(AtomicU64::new(0));
        let duration_frames = Arc::new(AtomicU64::new(0));
        let meter_state = Arc::new(Mutex::new(HashMap::new()));
        let meter_generation = Arc::new(AtomicU64::new(0));
        let mut last_timeline = Some(Arc::new(make_plain_stretch_timeline()));
        let mut last_play_file = None;
        let mut retired_snapshots = RetirementKeeper::<Arc<EngineSnapshot>>::new(3);

        let mut worker = EngineWorkerState {
            sr: sample_rate,
            is_playing: &is_playing,
            play_start_wait: &play_start_wait,
            target: &target,
            base_frames: &base_frames,
            position_frames: &position_frames,
            duration_frames: &duration_frames,
            snapshot: &snapshot,
            cache: &cache,
            stretch_cache: &stretch_cache,
            stretch_inflight: &stretch_inflight,
            stretch_tx: &stretch_tx,
            resources: &resources,
            tx: &engine_tx,
            last_timeline: &mut last_timeline,
            last_play_file: &mut last_play_file,
            retired_snapshots: &mut retired_snapshots,
            app_handle: None,
            meter_state: &meter_state,
            meter_generation: &meter_generation,
        };

        handle_play_from(&mut worker, Some(12.5));
        let start_frame = worker.position_frames.load(Ordering::Relaxed);
        assert_eq!(start_frame, 600_000, "12.5s @ 48kHz");
        assert!(worker.is_playing.load(Ordering::Relaxed));

        // 播放推进到停止点。
        worker.position_frames.store(700_000, Ordering::Relaxed);
        handle_stop(&mut worker);

        assert!(
            !worker.is_playing.load(Ordering::Relaxed),
            "stop must pause"
        );
        assert_eq!(
            worker.base_frames.load(Ordering::Relaxed),
            0,
            "timeline playback always reports absolute frames"
        );
        assert_eq!(
            worker.position_frames.load(Ordering::Relaxed),
            700_000,
            "the stop point must survive so the next play resumes exactly there"
        );

        // 再播放：起播点必须就是刚才的停止点（无需任何前端锚点）。
        handle_play_from(&mut worker, Some(700_000.0 / f64::from(sample_rate)));
        assert_eq!(worker.position_frames.load(Ordering::Relaxed), 700_000);
    }

    /// 文件预览的停止仍然归零：它的位置不是工程时间轴上的坐标。
    #[test]
    fn stop_resets_a_file_preview_position() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineCommand>();
        let (stretch_tx, _stretch_rx) = mpsc::channel();
        let resources = ResourceManager::new(engine_tx.clone());
        let cache = resources.cache().clone();
        let stretch_cache = Arc::new(Mutex::new(ByteBudgetCache::new(8, u64::MAX)));
        let stretch_inflight = Arc::new(Mutex::new(HashSet::new()));
        let snapshot = Arc::new(ArcSwap::from_pointee(EngineSnapshot::empty(44_100)));
        let is_playing = Arc::new(AtomicBool::new(false));
        let play_start_wait = Arc::new(AtomicBool::new(false));
        let target = Arc::new(Mutex::new(None));
        let base_frames = Arc::new(AtomicU64::new(44_100));
        let position_frames = Arc::new(AtomicU64::new(22_050));
        let duration_frames = Arc::new(AtomicU64::new(0));
        let meter_state = Arc::new(Mutex::new(HashMap::new()));
        let meter_generation = Arc::new(AtomicU64::new(0));
        let mut last_timeline = None;
        let mut last_play_file = Some((
            PathBuf::from("E:/virtual/example-media.wav"),
            1.0,
            "file".to_string(),
        ));
        let mut retired_snapshots = RetirementKeeper::<Arc<EngineSnapshot>>::new(3);

        let mut worker = EngineWorkerState {
            sr: 44_100,
            is_playing: &is_playing,
            play_start_wait: &play_start_wait,
            target: &target,
            base_frames: &base_frames,
            position_frames: &position_frames,
            duration_frames: &duration_frames,
            snapshot: &snapshot,
            cache: &cache,
            stretch_cache: &stretch_cache,
            stretch_inflight: &stretch_inflight,
            stretch_tx: &stretch_tx,
            resources: &resources,
            tx: &engine_tx,
            last_timeline: &mut last_timeline,
            last_play_file: &mut last_play_file,
            retired_snapshots: &mut retired_snapshots,
            app_handle: None,
            meter_state: &meter_state,
            meter_generation: &meter_generation,
        };

        handle_stop(&mut worker);
        assert_eq!(worker.base_frames.load(Ordering::Relaxed), 0);
        assert_eq!(worker.position_frames.load(Ordering::Relaxed), 0);
        assert!(worker.last_play_file.is_none());
    }

    /// 渲染结果推送（`RenderedClipsChanged`）必须换入新快照 —— 这是"原地
    /// 等待渲染"解除的唯一机制：产出者发布 → worker 换入包含最新 rendered_pcm
    /// 的快照 → 音频回调下一块自动重新判定就绪性并继续播放。
    #[test]
    fn rendered_clips_changed_swaps_in_a_fresh_snapshot() {
        update_runtime_stretch_settings(UserStretchAlgorithm::Signalsmith, false, None, None);

        let timeline = make_plain_stretch_timeline();
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineCommand>();
        let (stretch_tx, _stretch_rx) = mpsc::channel();

        let resources = ResourceManager::new(engine_tx.clone());
        let cache = resources.cache().clone();
        let stretch_cache = Arc::new(Mutex::new(ByteBudgetCache::new(8, u64::MAX)));
        let stretch_inflight = Arc::new(Mutex::new(HashSet::new()));
        let snapshot = Arc::new(ArcSwap::from_pointee(EngineSnapshot::empty(44_100)));
        let is_playing = Arc::new(AtomicBool::new(false));
        let play_start_wait = Arc::new(AtomicBool::new(false));
        let target = Arc::new(Mutex::new(None));
        let base_frames = Arc::new(AtomicU64::new(0));
        let position_frames = Arc::new(AtomicU64::new(0));
        let duration_frames = Arc::new(AtomicU64::new(0));
        let meter_state = Arc::new(Mutex::new(HashMap::new()));
        let meter_generation = Arc::new(AtomicU64::new(0));
        let mut last_timeline = Some(Arc::new(timeline.clone()));
        let mut last_play_file = None;
        let mut retired_snapshots = RetirementKeeper::<Arc<EngineSnapshot>>::new(3);

        let mut worker = EngineWorkerState {
            sr: 44_100,
            is_playing: &is_playing,
            play_start_wait: &play_start_wait,
            target: &target,
            base_frames: &base_frames,
            position_frames: &position_frames,
            duration_frames: &duration_frames,
            snapshot: &snapshot,
            cache: &cache,
            stretch_cache: &stretch_cache,
            stretch_inflight: &stretch_inflight,
            stretch_tx: &stretch_tx,
            resources: &resources,
            tx: &engine_tx,
            last_timeline: &mut last_timeline,
            last_play_file: &mut last_play_file,
            retired_snapshots: &mut retired_snapshots,
            app_handle: None,
            meter_state: &meter_state,
            meter_generation: &meter_generation,
        };

        // 建立基线快照（模拟播放命令的 update_timeline）。
        handle_update_timeline(&mut worker, timeline.clone());
        let before = snapshot.load_full();
        let before_ptr = Arc::as_ptr(&before) as usize;

        // 渲染线程推送 → 快照必须被替换（等待中的传输层据此恢复播放）。
        //
        // 记下推送前的时间轴分配，用来钉住"不得深拷贝"（见下方断言）。
        let stored_timeline = worker
            .last_timeline
            .as_ref()
            .expect("update_timeline must store a timeline")
            .clone();
        handle_rendered_clips_changed(&mut worker);

        // 时间轴必须是**共享**的同一份分配，而不是被深拷贝一份。
        //
        // 【为什么值得一条测试】`handle_rendered_clips_changed` 由渲染线程**每完成
        // 一个 clip** 推送一次；原实现里 `s.last_timeline.clone()` 是整份
        // `TimelineState` 的深拷贝（含全部 clips 与 params_by_clip 曲线，MB 级），
        // 465 个 clip 的一轮渲染就是 465 次深拷贝。存 `Arc` 后退化为引用计数 +1 ——
        // `Arc::ptr_eq` 正是"没有发生深拷贝"的直接证据。
        assert!(
            Arc::ptr_eq(
                &stored_timeline,
                worker
                    .last_timeline
                    .as_ref()
                    .expect("timeline must survive the refresh")
            ),
            "handle_rendered_clips_changed must reuse the stored timeline allocation, not deep-clone it"
        );

        let after = snapshot.load_full();
        let after_ptr = Arc::as_ptr(&after) as usize;
        assert_ne!(
            before_ptr, after_ptr,
            "a render result push must swap in a freshly built snapshot"
        );

        // 被换下的快照必须仍被暂存区持有。
        //
        // 【为什么是回归锚点】`ArcSwap::store` 等价于 `drop(swap(v))`：它会在
        // **本线程**立刻丢掉旧快照。而旧快照此刻仍被音频回调持有（回调要等到
        // 下一个块才发现换代），于是回调成了最后一个持有者 —— `EngineSnapshot`
        // （含 `Arc<Vec<f32>>` PCM）的析构就落到实时线程上。引用计数 ≥ 2 正是
        // "暂存区还留着它"的证据；若退回 `store`，这里会掉到 1。
        // 见 `crate::rt_retirement`。
        assert!(
            Arc::strong_count(&before) >= 2,
            "retired snapshot must stay retained by the worker instead of being dropped on swap"
        );

        // 无时间线时是安全的空操作（不崩溃、不替换快照）。
        *worker.last_timeline = None;
        handle_rendered_clips_changed(&mut worker);
        assert_eq!(
            Arc::as_ptr(&snapshot.load_full()) as usize,
            after_ptr,
            "without a timeline the refresh is a no-op"
        );
    }
}
