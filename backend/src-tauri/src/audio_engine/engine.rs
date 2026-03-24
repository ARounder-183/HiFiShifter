use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;

use arc_swap::ArcSwap;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::state::TimelineState;
use crate::time_stretch::{time_stretch_interleaved, StretchAlgorithm};

use super::mix::{
    render_callback_f32, render_callback_i16, render_callback_u16, SnapshotTransitionState,
};
use super::resource_manager::ResourceManager;
use super::snapshot::{
    build_snapshot, build_snapshot_for_file, schedule_stretch_jobs, source_bounds_frames,
};
use super::types::{
    AudioEngineStateSnapshot, EngineCommand, EngineSnapshot, ResampledStereo, StretchJob,
    StretchKey,
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

        let is_playing_thread = is_playing.clone();
        let target_thread = target.clone();
        let base_frames_thread = base_frames.clone();
        let position_frames_thread = position_frames.clone();
        let duration_frames_thread = duration_frames.clone();
        let sample_rate_thread = sample_rate.clone();

        let snapshot_for_thread = snapshot.clone();
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

            // Worker that computes Signalsmith Stretch off the command thread.
            // Keep it small to avoid CPU spikes.
            {
                let cache = cache_for_cmd.clone();
                let stretch_cache = stretch_cache_for_worker.clone();
                let inflight = stretch_inflight_for_worker.clone();
                let tx_ready = tx_for_worker.clone();
                thread::spawn(move || {
                    while let Ok(job) = stretch_rx.recv() {
                        // If Signalsmith Stretch isn't available, drop the job.
                        if !crate::sstretch::is_available() {
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
                            StretchAlgorithm::SignalsmithStretch,
                        );

                        let stretched_src = ResampledStereo {
                            sample_rate: job.key.out_rate,
                            frames: loop_out_frames,
                            pcm: Arc::new(stretched),
                        };

                        if let Ok(mut m) = stretch_cache.lock() {
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

            // Clone atomics for the audio callback to avoid moving the originals.
            let is_playing_cb = is_playing_thread.clone();
            let position_frames_cb = position_frames_thread.clone();
            let duration_frames_cb = duration_frames_thread.clone();
            let err_fn = |err| eprintln!("AudioEngine stream error: {err}");

            let stream = match sample_format {
                cpal::SampleFormat::F32 => device
                    .build_output_stream(
                        &config,
                        move |data: &mut [f32], _| {
                            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
                    .ok(),
                cpal::SampleFormat::I16 => device
                    .build_output_stream(
                        &config,
                        move |data: &mut [i16], _| {
                            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
                    .ok(),
                cpal::SampleFormat::U16 => device
                    .build_output_stream(
                        &config,
                        move |data: &mut [u16], _| {
                            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
                    .ok(),
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
                        };
                        match cmd {
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
        let _ = self.tx.send(EngineCommand::Shutdown);
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
struct EngineWorkerState<'a> {
    sr: u32,
    is_playing: &'a Arc<AtomicBool>,
    target: &'a Arc<Mutex<Option<String>>>,
    base_frames: &'a Arc<AtomicU64>,
    position_frames: &'a Arc<AtomicU64>,
    duration_frames: &'a Arc<AtomicU64>,
    snapshot: &'a Arc<ArcSwap<EngineSnapshot>>,
    cache: &'a Arc<Mutex<HashMap<(PathBuf, u32), ResampledStereo>>>,
    stretch_cache: &'a Arc<Mutex<HashMap<StretchKey, ResampledStereo>>>,
    stretch_inflight: &'a Arc<Mutex<HashSet<StretchKey>>>,
    stretch_tx: &'a mpsc::Sender<StretchJob>,
    resources: &'a ResourceManager,
    tx: &'a mpsc::Sender<EngineCommand>,
    last_timeline: &'a mut Option<TimelineState>,
    last_play_file: &'a mut Option<(PathBuf, f64, String)>,
    /// 可选的 Tauri app handle，用于向前端推送事件
    app_handle: Option<tauri::AppHandle>,
}

// ─── 命令处理函数 ─────────────────────────────────────────────────────────────

fn handle_stop(s: &mut EngineWorkerState) {
    s.is_playing.store(false, Ordering::Relaxed);
    *s.target.lock().unwrap_or_else(|e| e.into_inner()) = None;
    s.base_frames.store(0, Ordering::Relaxed);
    *s.last_play_file = None;
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
    for clip in &tl.clips {
        // 检查该 clip 所在 track 的 pitch_edit 是否发生变化
        let pitch_changed = s
            .last_timeline
            .as_ref()
            .map(|old_tl| {
                // 先解析 root_track_id，否则子轨道中的 clip 无法正确失效缓存
                let old_root = old_tl.resolve_root_track_id(&clip.track_id);
                let new_root = tl.resolve_root_track_id(&clip.track_id);

                let old_params = old_root.and_then(|r| old_tl.params_by_root_track.get(&r));
                let new_params = new_root.and_then(|r| tl.params_by_root_track.get(&r));

                match (old_params, new_params) {
                    (Some(old), Some(new)) => old.pitch_edit != new.pitch_edit,
                    (None, Some(new)) => !new.pitch_edit.is_empty(),
                    (Some(_), None) => true, // track params 被删除
                    (None, None) => false,
                }
            })
            .unwrap_or(false); // 首次 timeline，不触发重新合成

        if pitch_changed {
            // 修改：不再暴力清除全部缓存，而是仅清除片段缓存和解绑 pending_key。
            // 这样系统会自然计算出新的 Hash，并在新 Hash 未渲染完成时，
            // 利用留存下来的 RenderedClipCache 进行无缝垫音播放
            crate::synth_clip_cache::invalidate_clip_for_pitch_edit(&clip.id);
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
                    pos_changed || source_range_changed || rate_changed
                })
                .unwrap_or(false)
        })
        .map(|clip| clip.id.as_str()) // 零拷贝
        .collect();

    // ── 4. 检测音高相关变化（必须在 last_timeline 更新之前）──────────────────
    let clip_changed = tl.clips.iter().any(|clip| {
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
    eprintln!("[engine] Pitch schedule check: clips={}, clip_changed={}, track_changed={}, needs_schedule={}",
        tl.clips.len(), clip_changed, track_pitch_settings_changed, needs_pitch_schedule);

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

    let snap = build_snapshot(&tl, s.sr, s.cache, s.stretch_cache);
    s.duration_frames
        .store(snap.duration_frames, Ordering::Relaxed);
    s.snapshot.store(Arc::new(snap));
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
        debug_eprintln!("[engine] Snapshot stored, handle_clip_pitch_ready done");
    }
}

fn handle_audio_ready(s: &mut EngineWorkerState, _key: super::types::AudioKey) {
    // A decoded/resampled buffer became available.
    // Rebuild the snapshot so missing clips can be attached.
    if let Some(tl) = s.last_timeline.as_ref() {
        let snap = build_snapshot(tl, s.sr, s.cache, s.stretch_cache);
        s.duration_frames
            .store(snap.duration_frames, Ordering::Relaxed);
        s.snapshot.store(Arc::new(snap));
    } else if let Some((path, offset_sec, _target)) = s.last_play_file.as_ref() {
        let snap = build_snapshot_for_file(path.as_path(), s.sr, *offset_sec, s.cache);
        s.duration_frames
            .store(snap.duration_frames, Ordering::Relaxed);
        s.snapshot.store(Arc::new(snap));
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
    let midi_curve = crate::pitch_clip::trim_and_resample_midi(
        &cached.midi,
        frame_period_ms,
        clip.source_start_sec,
        clip.source_end_sec,
        pr,
        clip.length_sec.max(0.0),
    );
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
}
