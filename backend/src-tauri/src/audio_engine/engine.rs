use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::Duration;

use arc_swap::ArcSwap;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use lru::LruCache;

use crate::state::TimelineState;
use crate::time_stretch::time_stretch_interleaved;

/// Maximum stretch cache entries to prevent unbounded RAM growth.
const MAX_STRETCH_CACHE_ENTRIES: usize = 128;

use super::mix::{
    render_callback_f32, render_callback_i16, render_callback_u16, SnapshotTransitionState,
    TrackMeterScratch,
};
use super::resource_manager::ResourceManager;
use super::snapshot::{
    build_snapshot, build_snapshot_for_file, schedule_stretch_jobs, source_bounds_frames,
};
use super::types::{
    AudioEngineStateSnapshot, EngineCommand, EngineSnapshot, ResampledStereo, StretchJob,
    StretchKey, TrackMeterValue,
};
use crate::pitch_clip::schedule_clip_pitch_jobs;

use crate::pitch_clip::get_or_compute_clip_pitch_midi_global;
use tauri::{Emitter, Manager};

// 仅在 Debug 模式下编译并执行打印
macro_rules! debug_eprintln {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        std::eprintln!($($arg)*);
    }
}

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
    target: Arc<Mutex<Option<String>>>,
    base_frames: Arc<AtomicU64>,
    position_frames: Arc<AtomicU64>,
    duration_frames: Arc<AtomicU64>,
    sample_rate: Arc<AtomicU32>,
    /// Shutdown flag shared with the meter thread.
    meter_shutdown: Arc<AtomicBool>,
}

impl Clone for AudioEngine {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            snapshot: self.snapshot.clone(),
            is_playing: self.is_playing.clone(),
            target: self.target.clone(),
            base_frames: self.base_frames.clone(),
            position_frames: self.position_frames.clone(),
            duration_frames: self.duration_frames.clone(),
            sample_rate: self.sample_rate.clone(),
            meter_shutdown: self.meter_shutdown.clone(),
        }
    }
}

impl AudioEngine {
    pub fn new() -> Self {
        Self::with_app_handle(None)
    }

    /// 在 Tauri setup 完成后调用，将 app_handle 传递给 engine worker，
    /// 使其能够向前端推送事件（如 `clip_pitch_data`）。
    pub fn set_app_handle(&self, handle: tauri::AppHandle) {
        let _ = self.tx.send(EngineCommand::SetAppHandle { handle });
    }

    fn with_app_handle(app_handle: Option<tauri::AppHandle>) -> Self {
        let (tx, rx) = mpsc::channel::<EngineCommand>();
        let tx_for_worker = tx.clone();

        let is_playing = Arc::new(AtomicBool::new(false));
        let target = Arc::new(Mutex::new(None));
        let base_frames = Arc::new(AtomicU64::new(0));
        let position_frames = Arc::new(AtomicU64::new(0));
        let duration_frames = Arc::new(AtomicU64::new(0));
        let sample_rate = Arc::new(AtomicU32::new(44100));

        // Shared snapshot store for both the audio callback and command-side status queries.
        // This is updated by the engine worker thread.
        let snapshot: Arc<ArcSwap<EngineSnapshot>> =
            Arc::new(ArcSwap::from_pointee(EngineSnapshot::empty(44100)));
        let meter_state = Arc::new(Mutex::new(HashMap::<String, TrackMeterValue>::new()));
        let meter_generation = Arc::new(AtomicU64::new(0));
        let meter_app_handle = Arc::new(Mutex::new(app_handle.clone()));
        let meter_shutdown = Arc::new(AtomicBool::new(false));
        let meter_shutdown_for_thread = meter_shutdown.clone();

        let is_playing_thread = is_playing.clone();
        let target_thread = target.clone();
        let base_frames_thread = base_frames.clone();
        let position_frames_thread = position_frames.clone();
        let duration_frames_thread = duration_frames.clone();
        let sample_rate_thread = sample_rate.clone();

        let snapshot_for_thread = snapshot.clone();
        {
            let meter_state = meter_state.clone();
            let meter_generation = meter_generation.clone();
            let meter_app_handle = meter_app_handle.clone();
            let meter_shutdown = meter_shutdown_for_thread.clone();
            thread::spawn(move || {
                let mut last_generation = u64::MAX;
                loop {
                    if meter_shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(33));
                    let generation = meter_generation.load(Ordering::Relaxed);
                    if generation == last_generation {
                        continue;
                    }
                    last_generation = generation;

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
        thread::spawn(move || {
            let host = cpal::default_host();
            let device = match host.default_output_device() {
                Some(d) => d,
                None => {
                    eprintln!("AudioEngine: no default output device");
                    loop {
                        match rx.recv() {
                            Ok(EngineCommand::Shutdown) | Err(_) => break,
                            Ok(_) => {
                                is_playing_thread.store(false, Ordering::Relaxed);
                                *target_thread.lock().unwrap_or_else(|e| e.into_inner()) = None;
                            }
                        }
                    }
                    return;
                }
            };

            let default_config = match device.default_output_config() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("AudioEngine: default_output_config failed: {e}");
                    return;
                }
            };

            let sr = default_config.sample_rate().0;
            sample_rate_thread.store(sr, Ordering::Relaxed);

            // Re-initialize the shared snapshot to the actual output sample rate.
            snapshot_for_thread.store(Arc::new(EngineSnapshot::empty(sr)));
            let snapshot_for_cb = snapshot_for_thread.clone();

            // Async resource manager for decoded/resampled PCM.
            let resources = ResourceManager::new(tx_for_worker.clone());
            let cache_for_cmd = resources.cache().clone();

            // Time-stretch cache: computed, pitch-preserving loop buffers.
            let stretch_cache: Arc<Mutex<HashMap<StretchKey, ResampledStereo>>> =
                Arc::new(Mutex::new(HashMap::new()));
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
                            // Evict oldest entries if cache is full.
                            if m.len() >= MAX_STRETCH_CACHE_ENTRIES {
                                if let Some(evict_key) = m.keys().next().cloned() {
                                    m.remove(&evict_key);
                                }
                            }
                            m.insert(job.key.clone(), stretched_src);
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

            // Helper to (re)build snapshot from timeline.
            let channels = default_config.channels() as usize;
            let sample_format = default_config.sample_format();
            let config: cpal::StreamConfig = default_config.into();

            let mut scratch_mix: Vec<f32> = Vec::new();
            let mut scratch_mix_fade_from: Vec<f32> = Vec::new();
            let mut snapshot_transition = SnapshotTransitionState::default();
            let meter_state_for_cb = meter_state.clone();
            let meter_generation_for_cb = meter_generation.clone();

            // Clone atomics for the audio callback to avoid moving the originals.
            let is_playing_cb = is_playing_thread.clone();
            let position_frames_cb = position_frames_thread.clone();
            let duration_frames_cb = duration_frames_thread.clone();
            let err_fn = |err| eprintln!("AudioEngine stream error: {err}");

            let stream = match sample_format {
                cpal::SampleFormat::F32 => {
                    let meter_state = meter_state_for_cb.clone();
                    let meter_generation = meter_generation_for_cb.clone();
                    let mut meter_scratch = TrackMeterScratch::default();
                    device
                        .build_output_stream(
                            &config,
                            move |data: &mut [f32], _| {
                                let r =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        render_callback_f32(
                                            data,
                                            channels,
                                            &snapshot_for_cb,
                                            is_playing_cb.as_ref(),
                                            position_frames_cb.as_ref(),
                                            duration_frames_cb.as_ref(),
                                            &mut scratch_mix,
                                            &mut scratch_mix_fade_from,
                                            &mut snapshot_transition,
                                            &mut meter_scratch,
                                            &meter_state,
                                            meter_generation.as_ref(),
                                        );
                                    }));
                                if r.is_err() {
                                    eprintln!(
                                        "AudioEngine: panic in audio callback (f32); silencing output"
                                    );
                                    data.fill(0.0);
                                    is_playing_cb.store(false, Ordering::Relaxed);
                                }
                            },
                            err_fn,
                            None,
                        )
                        .ok()
                }
                cpal::SampleFormat::I16 => {
                    let meter_state = meter_state_for_cb.clone();
                    let meter_generation = meter_generation_for_cb.clone();
                    let mut meter_scratch = TrackMeterScratch::default();
                    device
                        .build_output_stream(
                            &config,
                            move |data: &mut [i16], _| {
                                let r =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        render_callback_i16(
                                            data,
                                            channels,
                                            &snapshot_for_cb,
                                            is_playing_cb.as_ref(),
                                            position_frames_cb.as_ref(),
                                            duration_frames_cb.as_ref(),
                                            &mut scratch_mix,
                                            &mut scratch_mix_fade_from,
                                            &mut snapshot_transition,
                                            &mut meter_scratch,
                                            &meter_state,
                                            meter_generation.as_ref(),
                                        );
                                    }));
                                if r.is_err() {
                                    eprintln!(
                                        "AudioEngine: panic in audio callback (i16); silencing output"
                                    );
                                    data.fill(0);
                                    is_playing_cb.store(false, Ordering::Relaxed);
                                }
                            },
                            err_fn,
                            None,
                        )
                        .ok()
                }
                cpal::SampleFormat::U16 => {
                    let meter_state = meter_state_for_cb.clone();
                    let meter_generation = meter_generation_for_cb.clone();
                    let mut meter_scratch = TrackMeterScratch::default();
                    device
                        .build_output_stream(
                            &config,
                            move |data: &mut [u16], _| {
                                let r =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        render_callback_u16(
                                            data,
                                            channels,
                                            &snapshot_for_cb,
                                            is_playing_cb.as_ref(),
                                            position_frames_cb.as_ref(),
                                            duration_frames_cb.as_ref(),
                                            &mut scratch_mix,
                                            &mut scratch_mix_fade_from,
                                            &mut snapshot_transition,
                                            &mut meter_scratch,
                                            &meter_state,
                                            meter_generation.as_ref(),
                                        );
                                    }));
                                if r.is_err() {
                                    eprintln!(
                                        "AudioEngine: panic in audio callback (u16); silencing output"
                                    );
                                    data.fill(u16::MAX / 2);
                                    is_playing_cb.store(false, Ordering::Relaxed);
                                }
                            },
                            err_fn,
                            None,
                        )
                        .ok()
                }
                _ => None,
            };

            let Some(stream) = stream else {
                eprintln!("AudioEngine: failed to build output stream");
                return;
            };

            if let Err(e) = stream.play() {
                eprintln!("AudioEngine: stream.play failed: {e}");
                return;
            }

            let mut last_timeline: Option<TimelineState> = None;
            let mut last_play_file: Option<(PathBuf, f64, String)> = None;
            let mut app_handle_for_worker: Option<tauri::AppHandle> = app_handle;

            loop {
                match rx.recv() {
                    Ok(EngineCommand::Shutdown) | Err(_) => break,
                    Ok(cmd) => {
                        let mut state = EngineWorkerState {
                            sr,
                            is_playing: &is_playing_thread,
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
                            app_handle: app_handle_for_worker.clone(),
                            meter_state: &meter_state,
                            meter_generation: &meter_generation,
                        };
                        match cmd {
                            EngineCommand::EvictSourcePath { path } => {
                                handle_evict_source_path(&mut state, &path);
                            }
                            EngineCommand::Stop => handle_stop(&mut state),
                            EngineCommand::SeekSec { sec } => handle_seek_sec(&mut state, sec),
                            EngineCommand::SetPlaying { playing, target } => {
                                handle_set_playing(&mut state, playing, target)
                            }
                            EngineCommand::UpdateTimeline(tl) => {
                                handle_update_timeline(&mut state, tl)
                            }
                            EngineCommand::StretchReady { key } => {
                                handle_stretch_ready(&mut state, key)
                            }
                            EngineCommand::ClipPitchReady { clip_id } => {
                                handle_clip_pitch_ready(&mut state, clip_id)
                            }
                            EngineCommand::SetAppHandle { handle } => {
                                if let Ok(mut app) = meter_app_handle.lock() {
                                    *app = Some(handle.clone());
                                }
                                app_handle_for_worker = Some(handle);
                            }
                            EngineCommand::AudioReady { key } => {
                                handle_audio_ready(&mut state, key)
                            }
                            EngineCommand::PlayFile {
                                path,
                                offset_sec,
                                target,
                            } => handle_play_file(&mut state, path, offset_sec, target),
                            EngineCommand::Shutdown => unreachable!(),
                        }
                    }
                }
            }
        });

        Self {
            tx,
            snapshot,
            is_playing,
            target,
            base_frames,
            position_frames,
            duration_frames,
            sample_rate,
            meter_shutdown,
        }
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate.load(Ordering::Relaxed).max(1)
    }

    #[allow(dead_code)]
    pub fn position_frames(&self) -> u64 {
        self.position_frames.load(Ordering::Relaxed)
    }

    #[allow(dead_code)]
    pub fn shutdown(&self) {
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

    pub fn set_playing(&self, playing: bool, target: Option<&str>) {
        let _ = self.tx.send(EngineCommand::SetPlaying {
            playing,
            target: target.map(|s| s.to_string()),
        });
    }

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
            target: self
                .target
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            base_sec: base as f64 / sr as f64,
            position_sec: pos as f64 / sr as f64,
            duration_sec: dur as f64 / sr as f64,
            sample_rate: sr,
        }
    }
}

// ─── Worker 状态结构体 ────────────────────────────────────────────────────────

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
    target: &'a Arc<Mutex<Option<String>>>,
    base_frames: &'a Arc<AtomicU64>,
    position_frames: &'a Arc<AtomicU64>,
    duration_frames: &'a Arc<AtomicU64>,
    snapshot: &'a Arc<ArcSwap<EngineSnapshot>>,
    cache: &'a Arc<Mutex<LruCache<(PathBuf, u32), ResampledStereo>>>,
    stretch_cache: &'a Arc<Mutex<HashMap<StretchKey, ResampledStereo>>>,
    stretch_inflight: &'a Arc<Mutex<HashSet<StretchKey>>>,
    stretch_tx: &'a mpsc::Sender<StretchJob>,
    resources: &'a ResourceManager,
    tx: &'a mpsc::Sender<EngineCommand>,
    last_timeline: &'a mut Option<TimelineState>,
    last_play_file: &'a mut Option<(PathBuf, f64, String)>,
    /// 可选的 Tauri app handle，用于向前端推送事件
    app_handle: Option<tauri::AppHandle>,
    meter_state: &'a Arc<Mutex<HashMap<String, TrackMeterValue>>>,
    meter_generation: &'a Arc<AtomicU64>,
}

// ─── 命令处理函数 ─────────────────────────────────────────────────────────────

fn handle_stop(s: &mut EngineWorkerState) {
    s.is_playing.store(false, Ordering::Relaxed);
    *s.target.lock().unwrap_or_else(|e| e.into_inner()) = None;
    s.base_frames.store(0, Ordering::Relaxed);
    *s.last_play_file = None;
    idle_track_meter_state(s.meter_state, s.meter_generation);
    // 播放停止时清空渲染线程传递的 cache_key 映射
    crate::synth_clip_cache::clear_pending_rendered_keys();
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
    *s.target.lock().unwrap_or_else(|e| e.into_inner()) = target;
    if playing {
        reset_track_meter_state(s.meter_state, s.meter_generation);
    } else {
        idle_track_meter_state(s.meter_state, s.meter_generation);
    }
}

fn track_params_affect_render(
    old: &crate::state::TrackParamsState,
    new: &crate::state::TrackParamsState,
) -> bool {
    old.frame_period_ms != new.frame_period_ms
        || old.pitch_orig != new.pitch_orig
        || old.pitch_edit != new.pitch_edit
        || old.tension_edit != new.tension_edit
        || old.extra_curves != new.extra_curves
        || old.extra_params != new.extra_params
}

fn handle_update_timeline(s: &mut EngineWorkerState, tl: TimelineState) {
    eprintln!(
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
            let old_clip = old_tl.clips.iter().find(|c| c.id == clip.id);

            // 检查该 clip 所在 track 的 pitch_edit 是否发生变化
            let pitch_changed = {
                // 先解析 root_track_id，否则子轨道中的 clip 无法正确失效缓存
                let old_root = old_tl.resolve_root_track_id(&clip.track_id);
                let new_root = tl.resolve_root_track_id(&clip.track_id);

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
                        || (old.length_sec - clip.length_sec).abs() > 1e-6
                        // 检测同路径文件替换：
                        // - duration_frames / source_sample_rate：文件长度或采样率变化
                        // - source_file_mtime：文件修改时间变化（最可靠的检测信号，
                        //   replace_clip_sources 每次都会更新此字段，即使文件长度未变也能检测到替换）
                        || old.duration_frames != clip.duration_frames
                        || old.source_sample_rate != clip.source_sample_rate
                        || old.source_file_mtime != clip.source_file_mtime
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
                // 1. 解码缓存（ResourceManager LRU）
                if let Ok(mut cache) = s.cache.lock() {
                    let keys_to_evict: Vec<(std::path::PathBuf, u32)> = cache
                        .iter()
                        .filter(|((p, _), _)| p.to_string_lossy().as_ref() == stale_path.as_str())
                        .map(|(k, _)| k.clone())
                        .collect();
                    for k in keys_to_evict {
                        cache.pop(&k);
                    }
                }

                // 2. Stretch 缓存
                if let Ok(mut sc) = s.stretch_cache.lock() {
                    let stretch_keys: Vec<StretchKey> = sc
                        .keys()
                        .filter(|k| k.path.to_string_lossy().as_ref() == stale_path.as_str())
                        .cloned()
                        .collect();
                    for k in stretch_keys {
                        sc.remove(&k);
                    }
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
                eprintln!(
                    "[engine] extra_params changed for root_track={}, invalidating all clips on this track group",
                    root_id
                );
                for clip in &tl.clips {
                    if tl.resolve_root_track_id(&clip.track_id).as_deref() == Some(root_id.as_str())
                    {
                        crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
                        any_cache_invalidated = true;
                    }
                }
            }
        }
    }

    // ── 后台预渲染自动触发 ──────────────────────────────────────────────────
    // 仅当本次 update_timeline 确实使缓存失效时才触发（避免形成反馈循环）。
    // any_cache_invalidated 由上方的 clip 变化检测代码设置。
    if any_cache_invalidated {
        use std::sync::atomic::Ordering;
        let enabled = crate::commands::playback::AUTO_BG_RENDER_ENABLED.load(Ordering::Relaxed);
        let running = crate::commands::playback::BG_RENDER_ACTIVE.load(Ordering::Relaxed);
        if enabled && !tl.clips.is_empty() {
            if !running {
                // 没有渲染在运行 → 直接启动
                if let Some(app) = s.app_handle.as_ref() {
                    eprintln!(
                        "[engine] auto-triggering background render ({} clips invalidated)",
                        tl.clips.len()
                    );
                    let _ = crate::commands::playback::request_background_render(app);
                }
            } else {
                // 渲染已在运行 → 取消旧渲染，标记需要重启
                // 旧渲染线程检测到取消标志后会退出，然后自动启动新一轮渲染
                eprintln!("[engine] bg render already running, cancelling and requesting restart ({} clips invalidated)", tl.clips.len());
                crate::commands::playback::BG_RENDER_CANCEL.store(true, Ordering::Release);
                crate::commands::playback::BG_RENDER_RESTART_NEEDED.store(true, Ordering::Release);
            }
        }
    }

    // 提前构建旧 Clip 的 Hash 表，将后续所有查询从 O(N^2) 降维到 O(N)
    let old_clips_map: std::collections::HashMap<&str, &crate::state::Clip> = s
        .last_timeline
        .as_ref()
        .map(|old_tl| old_tl.clips.iter().map(|c| (c.id.as_str(), c)).collect())
        .unwrap_or_default();

    // ── 3. 收集需要重新推送 pitch data 的 clip ─────────────────────────────
    let moved_clip_ids: std::collections::HashSet<&str> = tl
        .clips
        .iter()
        .filter(|clip| {
            old_clips_map
                .get(clip.id.as_str())
                .map(|old| {
                    let pos_changed = (old.start_sec - clip.start_sec).abs() > 1e-9;
                    let source_range_changed = (old.source_start_sec - clip.source_start_sec).abs()
                        > 1e-6
                        || (old.source_end_sec - clip.source_end_sec).abs() > 1e-6;
                    let rate_changed = (old.playback_rate - clip.playback_rate).abs() > 1e-6;
                    // Loop 开关会改变 pitch 曲线的目标长度（回绕铺满），
                    // 需要重新截取 + resample 推送。
                    let loop_changed = old.loop_enabled != clip.loop_enabled;
                    pos_changed || source_range_changed || rate_changed || loop_changed
                })
                .unwrap_or(false)
        })
        .map(|clip| clip.id.as_str()) // 零拷贝
        .collect();

    // ── 4. 检测音高相关变化（必须在 last_timeline 更新之前）──────────────────
    // Detect changes at both clip level AND track level (pitch_edit lives in params_by_root_track).
    let track_pitch_edit_changed = s.last_timeline.as_ref().map_or(false, |old_tl| {
        tl.params_by_root_track.iter().any(|(root_id, new_params)| {
            old_tl
                .params_by_root_track
                .get(root_id)
                .map_or(true, |old_params| {
                    old_params.pitch_edit != new_params.pitch_edit
                })
        })
    });

    let clip_changed = track_pitch_edit_changed
        || tl.clips.iter().any(|clip| {
            match old_clips_map.get(clip.id.as_str()) {
                None => true, // 新增 clip
                Some(old) => clip_pitch_params_changed(old, clip),
            }
        });

    let pitch_params_changed_clip_ids: std::collections::HashSet<&str> = tl
        .clips
        .iter()
        .filter_map(|clip| {
            old_clips_map.get(clip.id.as_str()).and_then(|old| {
                if clip_pitch_params_changed(*old, clip) {
                    Some(clip.id.as_str()) // 优化：零拷贝
                } else {
                    None
                }
            })
        })
        .collect();

    // 检测 track 级别的变化
    let has_last_timeline = s.last_timeline.is_some();
    let track_pitch_settings_changed = s.last_timeline.as_ref().map_or(true, |old_tl| {
        tl.tracks.iter().any(|track| {
            old_tl.tracks.iter()
                .find(|t| t.id == track.id)
                .map_or(true, |old_track| {
                    let compose_changed = old_track.compose_enabled != track.compose_enabled;
                    let algo_changed = old_track.pitch_analysis_algo != track.pitch_analysis_algo;
                    if compose_changed || algo_changed {
                        eprintln!("[engine] Track '{}' pitch settings changed: compose {} -> {}, algo {:?} -> {:?}",
                            track.id, old_track.compose_enabled, track.compose_enabled,
                            old_track.pitch_analysis_algo, track.pitch_analysis_algo);
                    }
                    compose_changed || algo_changed
                })
        })
    });

    if !has_last_timeline {
        eprintln!("[engine] First timeline update (no last_timeline), forcing pitch schedule");
    }

    let needs_pitch_schedule = clip_changed || track_pitch_settings_changed;
    eprintln!("[engine] Pitch schedule check: clips={}, clip_changed={}, track_changed={}, pitch_edit_changed={}, needs_schedule={}",
        tl.clips.len(), clip_changed, track_pitch_settings_changed, track_pitch_edit_changed, needs_pitch_schedule);

    // 预计算需要推送 pitch data 的 MIDI clip（必须在 *s.last_timeline 赋值之前，避免 borrow 冲突）
    let midi_clips_needing_emit: std::collections::HashSet<String> = tl
        .clips
        .iter()
        .filter(|c| c.midi_note_data.is_some())
        .filter(|c| {
            old_clips_map.get(c.id.as_str()).map_or(true, |old| {
                (old.start_sec - c.start_sec).abs() > 1e-9
                    || (old.source_start_sec - c.source_start_sec).abs() > 1e-6
                    || (old.source_end_sec - c.source_end_sec).abs() > 1e-6
                    || (old.playback_rate - c.playback_rate).abs() > 1e-6
                    || old.reversed != c.reversed
                    || old.loop_enabled != c.loop_enabled
                    || old.midi_fill_gaps != c.midi_fill_gaps
                    || old.muted != c.muted
                    || old.midi_note_data != c.midi_note_data
            })
        })
        .map(|c| c.id.clone())
        .collect();

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
        if !pitch_params_changed_clip_ids.is_empty() {
            if let Some(app) = s.app_handle.as_ref() {
                for clip in tl
                    .clips
                    .iter()
                    .filter(|c| pitch_params_changed_clip_ids.contains(c.id.as_str()))
                // 适配 &str
                {
                    emit_clip_pitch_data_for_clip(app, &tl, clip);
                }
            }
        }
        schedule_clip_pitch_jobs(&tl, s.tx, s.app_handle.as_ref(), s.sr);
    }

    if !moved_clip_ids.is_empty() {
        if let Some(app) = s.app_handle.as_ref() {
            for clip in tl
                .clips
                .iter()
                .filter(|c| moved_clip_ids.contains(c.id.as_str()))
            {
                // 适配 &str
                emit_clip_pitch_data_for_clip(app, &tl, clip);
            }
        }
    }

    // 新增或参数变更的 MIDI clip 需要推送 pitch data
    // （MIDI clip 没有 source_path，不会进入 pitch_params_changed_clip_ids 和 schedule_clip_pitch_jobs）
    if !midi_clips_needing_emit.is_empty() {
        if let Some(app) = s.app_handle.as_ref() {
            for clip in tl
                .clips
                .iter()
                .filter(|c| midi_clips_needing_emit.contains(c.id.as_str()))
            {
                emit_clip_pitch_data_for_clip(app, &tl, clip);
            }
        }

        // MIDI clip 的变更需要触发 pitch_orig 组装，设置 has_pitch_adjustment_active 标志，
        // 确保渲染管线能正确地将音高参考块的 MIDI 数据应用到同组音频块的渲染中。
        if let Some(app) = s.app_handle.as_ref() {
            let state = app.state::<crate::state::AppState>();
            let mut emitted_roots: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            for clip in tl
                .clips
                .iter()
                .filter(|c| midi_clips_needing_emit.contains(c.id.as_str()))
            {
                if let Some(root) = tl.resolve_root_track_id(&clip.track_id) {
                    if emitted_roots.insert(root.clone()) {
                        crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root);
                    }
                }
            }
        }
    }

    let snap = build_snapshot(&tl, s.sr, s.cache, s.stretch_cache);
    s.duration_frames
        .store(snap.duration_frames, Ordering::Relaxed);

    // Halt playback when re-rendering is needed so the user never hears stale audio.
    // Two triggers:
    //   1. ALL clips need re-rendering (cache miss, no fallback PCM).
    //   2. Pitch edit data changed — the rendered cache was just invalidated by
    //      `invalidate_clip_for_pitch_edit`, but the snapshot may not yet mark
    //      `needs_synthesis` (there is a one-update lag where `user_modified`
    //      hasn't propagated).  Playing during this window produces the old
    //      (un-pitched) audio, which the user perceives as "nothing changed".
    //
    // 如果后台预渲染（Background Pre-render）正在运行，则不暂停播放，
    // 允许音频回调在未渲染 clip 位置输出静音（自然暂停），渲染线程在后台继续推进。
    let bg_render =
        crate::commands::playback::BG_RENDER_ACTIVE.load(std::sync::atomic::Ordering::Relaxed);
    let any_need_render = !snap.clips.is_empty()
        && snap
            .clips
            .iter()
            .any(|c| c.needs_synthesis && c.rendered_pcm.is_none());
    let should_halt = (any_need_render || track_pitch_edit_changed) && !bg_render;
    if should_halt && s.is_playing.load(Ordering::Relaxed) {
        eprintln!(
            "[engine] Halting playback: any_need_render={}, pitch_edit_changed={}, bg_render={}",
            any_need_render, track_pitch_edit_changed, bg_render
        );
        s.is_playing.store(false, Ordering::Relaxed);
        if let Some(app) = s.app_handle.as_ref() {
            #[derive(Clone, serde::Serialize)]
            #[serde(rename_all = "camelCase")]
            struct Evt {
                active: bool,
                progress: Option<f64>,
                target: Option<String>,
            }
            let _ = app.emit(
                "playback_rendering_state",
                Evt {
                    active: false,
                    progress: Some(0.0),
                    target: Some("original".to_string()),
                },
            );
        }
    }

    s.snapshot.store(Arc::new(snap));
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
            for clip in &tl.clips {
                if let Some(src) = clip.source_path.as_deref() {
                    if std::path::Path::new(src) == key.path.as_path() {
                        // 重新推送该 clip 的 pitch data（使用全量缓存截取+resample）
                        emit_clip_pitch_data_for_clip(app, tl, clip);
                        if let Some(rt) = tl.resolve_root_track_id(&clip.track_id) {
                            root_track_ids.insert(rt);
                        }
                    }
                }
            }
            // 重新触发 pitch_orig 组装推送
            for rt in &root_track_ids {
                crate::pitch_analysis::maybe_schedule_pitch_orig(&state, rt);
            }
        }
    }

    if let Some(tl) = s.last_timeline.as_ref() {
        let snap = build_snapshot(tl, s.sr, s.cache, s.stretch_cache);
        s.duration_frames
            .store(snap.duration_frames, Ordering::Relaxed);
        s.snapshot.store(Arc::new(snap));
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
                    crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root_track_id);
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
        s.snapshot.store(Arc::new(snap));
        idle_track_meter_state(s.meter_state, s.meter_generation);
        debug_eprintln!("[engine] Snapshot stored, handle_clip_pitch_ready done");

        // 若此前后台预渲染因为音高分析未完成而跳过了一些 clip，
        // 现在缓存已经就绪，补触发一次后台渲染，避免用户等待进度结束后
        // 首次播放时仍要重新渲染。
        if crate::commands::playback::AUTO_BG_RENDER_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
            && crate::commands::playback::BG_RENDER_PITCH_PENDING.swap(
                false,
                std::sync::atomic::Ordering::AcqRel,
            )
        {
            if let Some(app) = s.app_handle.as_ref() {
                let _ = crate::commands::playback::request_background_render(app);
            }
        }
    }
}

/// 使指定源路径的所有解码缓存和拉伸缓存条目失效。
///
/// 在源文件被替换（`replace_clip_source`）时由 command handler 直接发送到引擎，
/// 确保在 `UpdateTimeline` 处理之前缓存已被清空，避免 `build_snapshot` 使用旧数据。
fn handle_evict_source_path(s: &mut EngineWorkerState, path: &str) {
    // 1. 解码缓存（ResourceManager LRU）—— 使用字符串比较以兼容不同 PathBuf 表示
    if let Ok(mut cache) = s.cache.lock() {
        let keys: Vec<(std::path::PathBuf, u32)> = cache
            .iter()
            .filter(|((p, _), _)| p.to_string_lossy().as_ref() == path)
            .map(|(k, _)| k.clone())
            .collect();
        for k in &keys {
            cache.pop(k);
        }
        if !keys.is_empty() {
            eprintln!(
                "[engine:evict] decode cache: evicted {} entries for path={}",
                keys.len(),
                path
            );
        }
    }

    // 2. 拉伸缓存
    if let Ok(mut sc) = s.stretch_cache.lock() {
        let stretch_keys: Vec<StretchKey> = sc
            .keys()
            .filter(|k| k.path.to_string_lossy().as_ref() == path)
            .cloned()
            .collect();
        for k in &stretch_keys {
            sc.remove(k);
        }
        if !stretch_keys.is_empty() {
            eprintln!(
                "[engine:evict] stretch cache: evicted {} entries for path={}",
                stretch_keys.len(),
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
            eprintln!(
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
        s.snapshot.store(Arc::new(snap));
        idle_track_meter_state(s.meter_state, s.meter_generation);
    } else if let Some((path, offset_sec, _target)) = s.last_play_file.as_ref() {
        let snap = build_snapshot_for_file(path.as_path(), s.sr, *offset_sec, s.cache);
        s.duration_frames
            .store(snap.duration_frames, Ordering::Relaxed);
        s.snapshot.store(Arc::new(snap));
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
    s.snapshot.store(Arc::new(snap));
    // File playback reports absolute position via base_sec + position_sec.
    let base = (offset_sec.max(0.0) * s.sr as f64).round().max(0.0) as u64;
    s.base_frames.store(base, Ordering::Relaxed);
    s.position_frames.store(0, Ordering::Relaxed);
    s.is_playing.store(true, Ordering::Relaxed);
    *s.target.lock().unwrap_or_else(|e| e.into_inner()) = Some(target);
    reset_track_meter_state(s.meter_state, s.meter_generation);
}

// ─── 辅助函数 ───────────────────────────────────────────────────────────────

/// 计算 MIDI 曲线第 0 帧对应的 timeline 绝对时间（秒）。
/// - playback_rate == 1（全量传送）：curve_start_sec = start_sec - source_start_sec
///   全量 MIDI 曲线的第 0 帧对应源音频第 0 秒在时间线上的绝对位置。
/// - playback_rate != 1（截取+resample 后传送）：curve_start_sec = start_sec
///   截取+resample 后的曲线直接对应 clip 的时间线位置。
fn compute_pitch_curve_start_sec(clip: &crate::state::Clip) -> f64 {
    // 统一截取后传送：curve 起点 = clip 在时间线上的起始位置
    clip.start_sec.max(0.0)
}

/// 统一的 clip pitch data 推送辅助函数。
/// 从全量缓存中读取 MIDI 曲线，根据 playback_rate 决定是否做 source range 截取 + resample。
/// - rate==1：全量传送，curveStartSec = start_sec - source_start_sec，前端 ctx.clip() 裁剪
/// - rate!=1：source range 截取 → resample 到 clip timeline 长度，curveStartSec = start_sec
fn emit_clip_pitch_data_for_clip(
    app: &tauri::AppHandle,
    tl: &crate::state::TimelineState,
    clip: &crate::state::Clip,
) {
    use tauri::Emitter;
    let frame_period_ms = 5.0f64;
    let root = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();

    eprintln!(
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
            if let Some(placement) = crate::state::place_note_occurrence_in_loop(
                clip,
                note.start_sec,
                note.end_sec,
                fp,
            ) {
                let note_value = note.note as f32;
                let mut cycle_offset = 0usize;
                while cycle_offset < clip_visible_frames {
                    let write_start = cycle_offset + placement.first_start_frame;
                    let write_end = (cycle_offset + placement.first_start_frame + placement.len_frames)
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
            let note_value = note.note as f32;
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
        eprintln!(
            "[pitch:emit] clip_id={} (MIDI) → curve_start={:.3}s curve_len={} fp={:.1}ms total_dur={:.3}s",
            clip.id,
            curve_start_sec,
            midi_curve.len(),
            frame_period_ms,
            midi_curve.len() as f64 * frame_period_ms / 1000.0,
        );
        let payload = ClipPitchDataPayload {
            clip_id: clip.id.clone(),
            curve_start_sec,
            midi_curve,
            frame_period_ms,
        };
        let _ = app.emit("clip_pitch_data", payload);
        return;
    }

    // ── 音频 clip 路径：从 FCPE 缓存获取音高曲线 ──
    let Some(cached) = get_or_compute_clip_pitch_midi_global(tl, clip, &root, frame_period_ms)
    else {
        eprintln!("[pitch:emit] clip_id={} → 缓存未命中，跳过", clip.id);
        return;
    };

    eprintln!(
        "[pitch:emit] clip_id={} cached_midi_len={}",
        clip.id,
        cached.midi.len(),
    );

    // 统一截取 + resample（rate==1 时 resample 实际为无损复制）
    let pr = clip.playback_rate as f64;
    let pr = if pr.is_finite() && pr > 0.0 { pr } else { 1.0 };
    // 非 Loop 倒放：传入真实消费窗口 [se−len·r, se]。
    let (trim_src_start, trim_src_end) = crate::state::clip_pitch_trim_window_sec(clip);
    let mut midi_curve = crate::pitch_clip::trim_and_resample_midi(
        &cached.midi,
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

    eprintln!(
        "[pitch:emit] clip_id={} → curve_start={:.3}s curve_len={} fp={:.1}ms total_dur={:.3}s",
        clip.id,
        curve_start_sec,
        midi_curve.len(),
        frame_period_ms,
        midi_curve.len() as f64 * frame_period_ms / 1000.0,
    );

    let payload = ClipPitchDataPayload {
        clip_id: clip.id.clone(),
        curve_start_sec,
        midi_curve,
        frame_period_ms,
    };
    let _ = app.emit("clip_pitch_data", payload);
}

/// 判断 clip 的 pitch 分析相关参数是否发生变化。
/// 全量分析策略：只有源文件变化才需要重新分析。
/// trim/rate 变化时不触发重新分析（全量缓存不会 miss），
/// 而是由 moved_clip_ids 分支重新推送 trim+resample 后的 pitch 数据。
fn clip_pitch_params_changed(old: &crate::state::Clip, new: &crate::state::Clip) -> bool {
    old.source_path != new.source_path
        // 同路径文件替换：mtime 变化说明文件内容已变，必须重新分析 pitch
        || old.source_file_mtime != new.source_file_mtime
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
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: 1.0,
            playback_rate: 0.75,
            reversed: false,
            loop_enabled: false,
            snap_offset_sec: 0.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_curve: "sine".to_string(),
            fade_out_curve: "sine".to_string(),
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

    #[test]
    fn audio_ready_requeues_stretch_jobs_for_plain_audio_clips() {
        update_runtime_stretch_settings(UserStretchAlgorithm::Signalsmith, false, None, None);

        let timeline = make_plain_stretch_timeline();
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineCommand>();
        let (stretch_tx, stretch_rx) = mpsc::channel();

        let resources = ResourceManager::new(engine_tx.clone());
        let cache = resources.cache().clone();
        let stretch_cache = Arc::new(Mutex::new(HashMap::new()));
        let stretch_inflight = Arc::new(Mutex::new(HashSet::new()));
        let snapshot = Arc::new(ArcSwap::from_pointee(EngineSnapshot {
            bpm: 120.0,
            sample_rate: 44_100,
            duration_frames: 0,
            track_ids: Arc::new(vec![]),
            clips: Arc::new(vec![]),
        }));
        let is_playing = Arc::new(AtomicBool::new(false));
        let target = Arc::new(Mutex::new(None));
        let base_frames = Arc::new(AtomicU64::new(0));
        let position_frames = Arc::new(AtomicU64::new(0));
        let duration_frames = Arc::new(AtomicU64::new(0));
        let meter_state = Arc::new(Mutex::new(HashMap::new()));
        let meter_generation = Arc::new(AtomicU64::new(0));
        let mut last_timeline = Some(timeline);
        let mut last_play_file = None;

        let mut worker = EngineWorkerState {
            sr: 44_100,
            is_playing: &is_playing,
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
}
