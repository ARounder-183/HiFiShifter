use crate::state::{Clip, TimelineState};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::Instant;

// ── 全局 clip pitch 分析进度状态 ─────────────────────────────────────────────

/// 当前批次的进度状态（供前端轮询）
#[derive(Debug, Clone, Default)]
pub struct ClipPitchBatchProgress {
    /// 当前正在分析的 clip 名称
    pub current_clip_name: Option<String>,
    /// 已完成的 clip 数量
    pub completed_clips: u32,
    /// 本批次需要分析的 clip 总数
    pub total_clips: u32,
    /// 整体进度 0.0~1.0
    pub progress: f32,
}

struct GlobalBatchState {
    /// 本批次总 clip 数
    total_clips: AtomicU32,
    /// 已完成 clip 数
    completed_clips: AtomicU32,
    /// 当前正在分析的 clip 名称
    current_clip_name: Mutex<Option<String>>,
}

impl GlobalBatchState {
    fn new() -> Self {
        Self {
            total_clips: AtomicU32::new(0),
            completed_clips: AtomicU32::new(0),
            current_clip_name: Mutex::new(None),
        }
    }

    fn reset(&self, total: u32) {
        self.total_clips.store(total, AtomicOrdering::Relaxed);
        self.completed_clips.store(0, AtomicOrdering::Relaxed);
        if let Ok(mut g) = self.current_clip_name.lock() {
            *g = None;
        }
    }

    fn set_current(&self, name: Option<String>) {
        if let Ok(mut g) = self.current_clip_name.lock() {
            *g = name;
        }
    }

    fn complete_one(&self) -> u32 {
        self.completed_clips.fetch_add(1, AtomicOrdering::Relaxed) + 1
    }

    fn snapshot(&self) -> ClipPitchBatchProgress {
        let total = self.total_clips.load(AtomicOrdering::Relaxed);
        let completed = self.completed_clips.load(AtomicOrdering::Relaxed);
        let current = self.current_clip_name.lock().ok().and_then(|g| g.clone());
        let progress = if total == 0 {
            0.0
        } else {
            (completed as f32 / total as f32).clamp(0.0, 1.0)
        };
        ClipPitchBatchProgress {
            current_clip_name: current,
            completed_clips: completed,
            total_clips: total,
            progress,
        }
    }
}

static GLOBAL_BATCH_STATE: OnceLock<GlobalBatchState> = OnceLock::new();

fn global_batch_state() -> &'static GlobalBatchState {
    GLOBAL_BATCH_STATE.get_or_init(GlobalBatchState::new)
}

/// 获取当前 clip pitch 批次分析进度（供 `get_pitch_analysis_progress` 命令调用）
pub fn get_clip_pitch_batch_progress() -> Option<ClipPitchBatchProgress> {
    let s = global_batch_state().snapshot();
    if s.total_clips == 0 {
        None
    } else {
        Some(s)
    }
}

#[derive(Debug, Clone)]
struct ClipPitchKey {
    clip_id: String,
    key: String,
    frame_period_ms: f64,
    sample_rate: u32,
    pre_silence_sec: f64,
    /// true = playback_rate==1，分析源音频全量，cache key 不含 trim
    is_full_source: bool,
}

#[derive(Debug, Clone)]
pub struct CachedClipPitch {
    pub key: String,
    pub midi: Vec<f32>, // timeline frames (frame_period_ms)
}

static GLOBAL_CLIP_PITCH_CACHE: OnceLock<Mutex<HashMap<String, CachedClipPitch>>> = OnceLock::new();
static GLOBAL_CLIP_PITCH_INFLIGHT: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
static CLIP_PITCH_CACHE_MAX_ENTRIES: OnceLock<Option<usize>> = OnceLock::new();

const DEFAULT_CLIP_PITCH_CACHE_MAX_ENTRIES: usize = 4096;

fn clip_pitch_cache_max_entries() -> Option<usize> {
    *CLIP_PITCH_CACHE_MAX_ENTRIES.get_or_init(|| {
        let parsed = std::env::var("HIFISHIFTER_CLIP_PITCH_CACHE_MAX_ENTRIES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok());
        match parsed {
            // 0 = disable hard limit (unbounded)
            Some(0) => None,
            Some(v) => Some(v.max(1)),
            None => Some(DEFAULT_CLIP_PITCH_CACHE_MAX_ENTRIES),
        }
    })
}

pub(crate) fn global_cache() -> &'static Mutex<HashMap<String, CachedClipPitch>> {
    GLOBAL_CLIP_PITCH_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn global_inflight() -> &'static Mutex<HashSet<String>> {
    GLOBAL_CLIP_PITCH_INFLIGHT.get_or_init(|| Mutex::new(HashSet::new()))
}

fn hz_to_midi(hz: f64) -> f32 {
    if !(hz.is_finite() && hz > 1e-6) {
        return 0.0;
    }
    let midi = 69.0 + 12.0 * (hz / 440.0).log2();
    if midi.is_finite() {
        midi as f32
    } else {
        0.0
    }
}

fn quantize_i64(x: f64, scale: f64) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    (x * scale).round() as i64
}

fn quantize_u32(x: f64, scale: f64) -> u32 {
    if !x.is_finite() {
        return 0;
    }
    let v = (x * scale).round();
    if v <= 0.0 {
        0
    } else if v > (u32::MAX as f64) {
        u32::MAX
    } else {
        v as u32
    }
}

// ── file_sig 缓存（TTL 10 秒，避免每次 UpdateTimeline 都做文件系统 I/O）────────

struct FileSigEntry {
    sig: (u64, u64),
    fetched_at: Instant,
}

static GLOBAL_FILE_SIG_CACHE: OnceLock<Mutex<HashMap<PathBuf, FileSigEntry>>> = OnceLock::new();

fn global_file_sig_cache() -> &'static Mutex<HashMap<PathBuf, FileSigEntry>> {
    GLOBAL_FILE_SIG_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 获取文件签名 (len_bytes, modified_ms)，结果缓存 10 秒，避免频繁文件系统 I/O。
fn file_sig(path: &Path) -> (u64, u64) {
    const TTL_SECS: u64 = 10;
    let path_buf = path.to_path_buf();

    // 先查缓存
    {
        let cache = global_file_sig_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = cache.get(&path_buf) {
            if entry.fetched_at.elapsed().as_secs() < TTL_SECS {
                return entry.sig;
            }
        }
    }

    // 缓存未命中或已过期，做真实 I/O
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => {
            // 文件不存在，缓存 (0,0) 并返回
            let mut cache = global_file_sig_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.insert(
                path_buf,
                FileSigEntry {
                    sig: (0, 0),
                    fetched_at: Instant::now(),
                },
            );
            return (0, 0);
        }
    };
    let len = meta.len();
    let mtime_ms = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let sig = (len, mtime_ms);

    let mut cache = global_file_sig_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache.insert(
        path_buf,
        FileSigEntry {
            sig,
            fetched_at: Instant::now(),
        },
    );
    sig
}

/// 检查文件是否存在，复用 file_sig 缓存（len > 0 表示文件存在）。
fn file_exists_cached(path: &Path) -> bool {
    let (len, _) = file_sig(path);
    len > 0
}

fn resample_curve_linear(values: &[f32], out_len: usize) -> Vec<f32> {
    if out_len == 0 {
        return vec![];
    }
    if values.is_empty() {
        return vec![0.0; out_len];
    }
    if values.len() == out_len {
        return values.to_vec();
    }
    if values.len() == 1 {
        return vec![values[0]; out_len];
    }
    if out_len == 1 {
        return vec![values[0]];
    }

    let in_len = values.len();
    let scale = (in_len - 1) as f64 / (out_len - 1) as f64;
    let mut out = vec![0.0f32; out_len];
    for (of, out_v) in out.iter_mut().enumerate() {
        let t_in = (of as f64) * scale;
        let i0 = t_in.floor() as usize;
        let i1 = (i0 + 1).min(in_len - 1);
        let frac = (t_in - (i0 as f64)) as f32;
        let a = values[i0];
        let b = values[i1];
        *out_v = a + (b - a) * frac;
    }
    out
}

fn beat_sec(bpm: f64) -> f64 {
    60.0 / bpm.max(1e-6)
}

fn build_clip_pitch_key(
    _tl: &TimelineState,
    clip: &Clip,
    _root_track_id: &str,
    frame_period_ms: f64,
) -> Option<ClipPitchKey> {
    let source_path = clip.source_path.as_deref()?;

    let clip_timeline_len_sec = clip.length_sec.max(0.0);
    if !(clip_timeline_len_sec.is_finite() && clip_timeline_len_sec > 0.0) {
        return None;
    }

    let playback_rate = clip.playback_rate as f64;
    let playback_rate = if playback_rate.is_finite() && playback_rate > 0.0 {
        playback_rate
    } else {
        1.0
    };

    let pre_silence_sec = (-clip.source_start_sec).max(0.0) / playback_rate.max(1e-6);

    let fp = frame_period_ms.max(0.1);

    // 缓存 key 只包含影响原始音频分析结果的字段：source_path（文件内容签名）+ frame_period。
    // clip_id、root_track_id、bpm 均不参与 hash——相同源文件的多个 clip 共享同一缓存条目，
    // trim/rate 变化在推送/组装阶段按需截取+resample，无需重新分析。
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"clip_pitch_v3_source_midi");
    hasher.update(source_path.as_bytes());
    let (len, mtime) = file_sig(Path::new(source_path));
    hasher.update(&len.to_le_bytes());
    hasher.update(&mtime.to_le_bytes());
    hasher.update(&quantize_u32(fp, 1000.0).to_le_bytes());

    let is_full_source = (playback_rate - 1.0).abs() <= 1e-6;

    let key = hasher.finalize().to_hex().to_string();

    Some(ClipPitchKey {
        clip_id: clip.id.clone(),
        key,
        frame_period_ms: fp,
        sample_rate: 44100,
        pre_silence_sec,
        is_full_source,
    })
}

/// 查询 clip pitch MIDI 缓存。
/// - 缓存命中：直接返回 `Some`。
/// - 缓存未命中：**不再同步计算**，直接返回 `None`。
///   调用方应提前通过 `schedule_clip_pitch_jobs` 触发异步预计算。
pub fn get_or_compute_clip_pitch_midi_global(
    tl: &TimelineState,
    clip: &Clip,
    root_track_id: &str,
    frame_period_ms: f64,
) -> Option<CachedClipPitch> {
    let ck = build_clip_pitch_key(tl, clip, root_track_id, frame_period_ms)?;

    let cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    // 以内容哈希为 key 查找——相同源文件的多个 clip 共享同一缓存条目。
    if let Some(found) = cache.get(&ck.key) {
        return Some(found.clone());
    }
    // 缓存未命中，返回 None，等待异步预计算完成后由 ClipPitchReady 触发 snapshot rebuild。
    None
}

/// 将计算结果写入全局缓存（供异步 worker 调用）。
/// 以内容哈希（`cached.key`）为 key，相同源文件的多个 clip 共享同一条目。
fn store_clip_pitch_cache(cached: CachedClipPitch) {
    let content_key = cached.key.clone();
    let mut cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    cache.insert(content_key, cached);

    if let Some(max_entries) = clip_pitch_cache_max_entries() {
        if cache.len() > max_entries {
            let keys: Vec<String> = cache
                .keys()
                .take(cache.len().saturating_sub(max_entries))
                .cloned()
                .collect();
            for k in keys {
                cache.remove(&k);
            }
        }
    }
}

/// 遍历 timeline 中所有可见 clip，对缓存未命中的 clip 异步提交 pitch MIDI 计算任务。
/// 任务完成后通过 `engine_tx` 发送 `EngineCommand::ClipPitchReady`，触发 snapshot rebuild。
///
/// 利用 `GLOBAL_CLIP_PITCH_INFLIGHT` 去重，同一 clip 不会重复提交。
///
/// `stretch_cache`：若 clip 已有拉伸后 PCM，优先使用它作为音高检测输入。
pub fn schedule_clip_pitch_jobs(
    tl: &TimelineState,
    engine_tx: &mpsc::Sender<crate::audio_engine::types::EngineCommand>,
    app_handle: Option<&tauri::AppHandle>,
    out_rate: u32,
) {
    eprintln!(
        "[pitch_clip] schedule_clip_pitch_jobs called, clips={}, app_handle={}",
        tl.clips.len(),
        app_handle.is_some()
    );

    if !crate::world::is_available() {
        eprintln!("[pitch_clip] WORLD not available, skipping");
        return;
    }

    use crate::pitch_analysis::PitchOrigAnalysisProgressEvent;
    use tauri::Emitter;

    // 收集需要计算的 clip 快照（避免持锁期间做耗时操作）
    let frame_period_ms = 5.0f64;

    // ── 阶段1：收集所有需要分析的 clip ──────────────────────────────────────
    struct PendingJob {
        clip: Clip,
        ck: ClipPitchKey,
        root_track_id: String,
        inflight_key: String,
    }

    let mut pending_jobs: Vec<PendingJob> = Vec::new();

    for clip in &tl.clips {
        // 跳过无效 clip
        let source_path = match clip.source_path.as_deref() {
            Some(p) if !p.is_empty() => p,
            _ => {
                eprintln!("[pitch_clip] clip '{}' skipped: no source_path", clip.id);
                continue;
            }
        };
        if !file_exists_cached(Path::new(source_path)) {
            eprintln!(
                "[pitch_clip] clip '{}' skipped: file not found: {}",
                clip.id, source_path
            );
            continue;
        }

        // 仅对 compose_enabled 的根轨道进行音高分析
        {
            let root_id = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();
            let compose_enabled = tl
                .tracks
                .iter()
                .find(|t| t.id == root_id)
                .map(|t| t.compose_enabled)
                .unwrap_or(false);
            if !compose_enabled {
                eprintln!(
                    "[pitch_clip] clip '{}' skipped: compose_enabled=false for root '{}'",
                    clip.id, root_id
                );
                continue;
            }
        }

        // 尝试构建 key
        let ck = match build_clip_pitch_key(
            tl,
            clip,
            &tl.resolve_root_track_id(&clip.track_id).unwrap_or_default(),
            frame_period_ms,
        ) {
            Some(k) => k,
            None => continue,
        };

        // 缓存命中则跳过（以内容哈希为 key，相同源文件的 clip 共享缓存）
        {
            let cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
            if cache.contains_key(&ck.key) {
                eprintln!(
                    "[pitch_clip] clip '{}' ({}) cache HIT (shared key), skipping",
                    clip.name, clip.id
                );
                continue;
            }
            eprintln!(
                "[pitch_clip] clip '{}' ({}) cache MISS, will analyze",
                clip.name, clip.id
            );
        }

        // inflight 去重：以内容哈希为 key，相同源文件只允许一个分析任务
        let inflight_key = ck.key.clone();
        let should_spawn = {
            let mut set = global_inflight().lock().unwrap_or_else(|e| e.into_inner());
            if set.contains(&inflight_key) {
                eprintln!(
                    "[pitch_clip] clip '{}' ({}) already inflight, skipping",
                    clip.name, clip.id
                );
                false
            } else {
                set.insert(inflight_key.clone());
                true
            }
        };
        if !should_spawn {
            continue;
        }

        // 改动后：始终分析原始源 PCM，不再需要 stretch_cache。
        // trim/rate 变化时在推送/组装阶段按需截取+resample。

        let root_track_id = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();
        pending_jobs.push(PendingJob {
            clip: clip.clone(),
            ck,
            root_track_id,
            inflight_key,
        });
    }

    if pending_jobs.is_empty() {
        eprintln!("[pitch_clip] no pending jobs (all cached or inflight), nothing to do");
        return;
    }
    eprintln!("[pitch_clip] {} clip(s) need analysis", pending_jobs.len());

    // ── 阶段2：重置全局进度状态，批量提交分析任务 ────────────────────────────
    let total = pending_jobs.len() as u32;
    global_batch_state().reset(total);

    // 发送分析开始事件
    if let Some(app) = app_handle {
        let root_track_id = pending_jobs
            .first()
            .map(|j| j.root_track_id.clone())
            .unwrap_or_default();
        eprintln!(
            "[pitch_clip] emitting pitch_orig_analysis_started for root_track_id='{}'",
            root_track_id
        );
        let r1 = app.emit(
            "pitch_orig_analysis_started",
            crate::pitch_analysis::PitchOrigAnalysisStartedEvent {
                root_track_id: root_track_id.clone(),
                key: String::new(),
            },
        );
        eprintln!(
            "[pitch_clip] pitch_orig_analysis_started emit result: {:?}",
            r1
        );
        // 发送初始进度（0%，显示第一个 clip 名称）
        let first_clip_name = pending_jobs.first().map(|j| j.clip.name.clone());
        eprintln!(
            "[pitch_clip] emitting initial progress 0/{}, first_clip={:?}",
            total, first_clip_name
        );
        let r2 = app.emit(
            "pitch_orig_analysis_progress",
            PitchOrigAnalysisProgressEvent {
                root_track_id,
                progress: 0.0,
                current_clip_name: first_clip_name,
                completed_clips: 0,
                total_clips: total,
            },
        );
        eprintln!("[pitch_clip] initial progress emit result: {:?}", r2);
    } else {
        eprintln!("[pitch_clip] WARNING: app_handle is None, cannot emit events!");
    }

    for job in pending_jobs {
        let tx = engine_tx.clone();
        let app_handle_clone = app_handle.cloned();
        let tl_clone = tl.clone();

        std::thread::spawn(move || {
            // 通知进度：开始分析此 clip
            eprintln!(
                "[pitch_clip] thread: starting analysis for clip '{}'",
                job.clip.name
            );
            global_batch_state().set_current(Some(job.clip.name.clone()));

            let midi =
                compute_clip_pitch_midi(&tl_clone, &job.clip, &job.root_track_id, frame_period_ms);

            // 完成一个 clip，更新进度
            let completed = global_batch_state().complete_one();
            global_batch_state().set_current(None);
            eprintln!(
                "[pitch_clip] thread: clip '{}' analysis done, midi={}, completed={}/{}",
                job.clip.name,
                midi.is_some(),
                completed,
                total
            );

            // 发送进度事件
            if let Some(app) = &app_handle_clone {
                let progress = if total == 0 {
                    1.0
                } else {
                    completed as f32 / total as f32
                };
                let r = app.emit(
                    "pitch_orig_analysis_progress",
                    PitchOrigAnalysisProgressEvent {
                        root_track_id: job.root_track_id.clone(),
                        progress: progress.clamp(0.0, 1.0),
                        current_clip_name: None,
                        completed_clips: completed,
                        total_clips: total,
                    },
                );
                eprintln!(
                    "[pitch_clip] thread: progress emit {}/{} result: {:?}",
                    completed, total, r
                );
            } else {
                eprintln!("[pitch_clip] thread: WARNING app_handle is None, cannot emit progress!");
            }

            // 无论成功与否，先清除 inflight 标记
            {
                let mut set = global_inflight().lock().unwrap_or_else(|e| e.into_inner());
                set.remove(&job.inflight_key);
            }

            if let Some(midi_data) = midi {
                let cached = CachedClipPitch {
                    key: job.ck.key.clone(),
                    midi: midi_data,
                };
                store_clip_pitch_cache(cached);
                // 通知引擎缓存已就绪，触发 snapshot rebuild。
                // 以内容哈希查找所有共享该源文件的 clip，逐一发送通知。
                let sharing_clip_ids: Vec<String> = tl_clone
                    .clips
                    .iter()
                    .filter_map(|c| {
                        let root = tl_clone
                            .resolve_root_track_id(&c.track_id)
                            .unwrap_or_default();
                        build_clip_pitch_key(&tl_clone, c, &root, frame_period_ms)
                            .filter(|other_ck| other_ck.key == job.ck.key)
                            .map(|_| c.id.clone())
                    })
                    .collect();
                eprintln!(
                    "[pitch_clip] thread: notifying {} clip(s) sharing content key",
                    sharing_clip_ids.len()
                );
                for cid in sharing_clip_ids {
                    let _ = tx.send(crate::audio_engine::types::EngineCommand::ClipPitchReady {
                        clip_id: cid,
                    });
                }
            }

            // 所有 clip 完成后，重置全局进度状态
            if completed >= total {
                eprintln!(
                    "[pitch_clip] thread: all {} clips done, resetting batch state",
                    total
                );
                global_batch_state().reset(0);
            }
        });
    }
}

pub fn compute_clip_pitch_midi(
    tl: &TimelineState,
    clip: &Clip,
    root_track_id: &str,
    frame_period_ms: f64,
) -> Option<Vec<f32>> {
    if !crate::world::is_available() {
        return None;
    }

    let ck = build_clip_pitch_key(tl, clip, root_track_id, frame_period_ms)?;

    // ── 始终解码源音频全量 PCM 进行分析 ─────────────────────────────────────
    // 缓存中存的是全量源音频的 MIDI 曲线，不含 trim/rate 处理。
    // trim 截取 + rate 拉伸在推送/组装阶段按需执行。
    let source_path = clip.source_path.as_deref()?;
    let (in_rate, in_channels, pcm) =
        crate::audio_utils::decode_audio_f32_interleaved(Path::new(source_path)).ok()?;
    let in_channels_usize = (in_channels as usize).max(1);
    let in_frames = pcm.len() / in_channels_usize;
    if in_frames < 2 {
        return None;
    }
    let analysis_pcm = crate::mixdown::linear_resample_interleaved(
        &pcm,
        in_channels_usize,
        in_rate,
        ck.sample_rate,
    );
    let analysis_rate = ck.sample_rate;
    let analysis_channels = in_channels_usize;

    let analysis_frames = analysis_pcm.len() / analysis_channels;
    if analysis_frames < 2 {
        return None;
    }

    // ── 转 mono + 归一化 ──────────────────────────────────────────────────
    let mut mono_raw: Vec<f64> = Vec::with_capacity(analysis_frames);
    for f in 0..analysis_frames {
        let base = f * analysis_channels;
        let mut sum = 0.0f64;
        for c in 0..analysis_channels {
            sum += analysis_pcm[base + c] as f64;
        }
        mono_raw.push(sum / analysis_channels as f64);
    }

    // remove DC + clamp like other WORLD callers
    let mut mean = 0.0f64;
    for &v in &mono_raw {
        mean += v;
    }
    mean /= mono_raw.len().max(1) as f64;

    let mut max_abs = 0.0f64;
    for &v in &mono_raw {
        let vv = v - mean;
        let a = vv.abs();
        if a.is_finite() && a > max_abs {
            max_abs = a;
        }
    }
    let scale = if max_abs.is_finite() && max_abs > 1.0 {
        (1.0 / max_abs).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let mut mono: Vec<f64> = Vec::with_capacity(mono_raw.len());
    for &v in &mono_raw {
        let vv = (v - mean) * scale;
        mono.push(vv.clamp(-1.0, 1.0));
    }

    // f0
    let prefer = std::env::var("HIFISHIFTER_WORLD_F0")
        .ok()
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "harvest".to_string());

    let frame_period_tl_ms = ck.frame_period_ms.max(0.1);
    let f0_floor = 40.0;
    let f0_ceil = 1600.0;

    let fs_i32 = analysis_rate as i32;
    let f0_hz: Vec<f64> = {
        let try_harvest = || {
            crate::world::compute_f0_hz_harvest(
                &mono,
                fs_i32,
                frame_period_tl_ms,
                f0_floor,
                f0_ceil,
            )
        };
        let try_dio = || {
            crate::world::compute_f0_hz_dio_stonemask(
                &mono,
                fs_i32,
                frame_period_tl_ms,
                f0_floor,
                f0_ceil,
            )
        };

        let res = if prefer == "dio" {
            try_dio().or_else(|_| try_harvest())
        } else {
            try_harvest().or_else(|_| try_dio())
        };

        res.unwrap_or_default()
    };

    if f0_hz.len() < 2 {
        return None;
    }

    let mut midi: Vec<f32> = Vec::with_capacity(f0_hz.len());
    for hz in f0_hz {
        midi.push(hz_to_midi(hz));
    }

    // ── 全量曲线直接返回 ──────────────────────────────────────────────
    // 缓存中始终存全量源音频的 MIDI 曲线。
    // trim 截取 + rate resample 在推送（handle_clip_pitch_ready）
    // 和组装（assemble_pitch_orig_from_cache）阶段按需执行。

    // Small gap fill.
    let gap_ms = std::env::var("HIFISHIFTER_WORLD_F0_GAP_MS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 200.0);
    if gap_ms > 0.0 {
        let gap_frames = ((gap_ms / frame_period_tl_ms).round() as isize).max(1) as usize;
        let mut last = 0.0f32;
        let mut zeros = 0usize;
        for v in midi.iter_mut() {
            if *v > 0.0 {
                last = *v;
                zeros = 0;
            } else {
                zeros += 1;
                if zeros <= gap_frames && last > 0.0 {
                    *v = last;
                }
            }
        }
    }

    Some(midi)
}

/// 从全量 MIDI 曲线中截取 source range 区间并按 playback_rate 重采样。
/// 返回对应 clip 在时间线上可见区间的 MIDI 曲线。
///
/// - `full_midi`：全量源音频的 MIDI 曲线（WORLD 输出，每帧间隔 `frame_period_ms`）
/// - `source_start_sec`：clip 的 source_start_sec（源音频有效区间起点）
/// - `source_end_sec`：clip 的 source_end_sec（源音频有效区间终点）
/// - `playback_rate`：clip 的 playback_rate（>1 加速，<1 减速）
/// - `clip_timeline_len_sec`：clip 在时间线上的可见长度（秒）
pub fn trim_and_resample_midi(
    full_midi: &[f32],
    frame_period_ms: f64,
    source_start_sec: f64,
    source_end_sec: f64,
    playback_rate: f64,
    clip_timeline_len_sec: f64,
) -> Vec<f32> {
    let fp = frame_period_ms.max(0.1);
    let src_start = source_start_sec.max(0.0);

    // 从全量曲线中截取 source range 区间
    let src_start_frame = ((src_start * 1000.0) / fp).round().max(0.0) as usize;

    // 根据 source_end_sec 计算结束帧
    let src_end_frame = ((source_end_sec * 1000.0) / fp).round().max(0.0) as usize;
    let src_end_frame = src_end_frame.min(full_midi.len());
    if src_start_frame >= src_end_frame {
        eprintln!(
            "[pitch:trim] EMPTY: src_start={:.3}s src_end={:.3}s full_midi_len={} \
             start_frame={} end_frame={} → empty",
            source_start_sec,
            source_end_sec,
            full_midi.len(),
            src_start_frame,
            src_end_frame,
        );
        return Vec::new();
    }

    let trimmed = &full_midi[src_start_frame..src_end_frame];

    // 按 1/playback_rate 重采样到 clip timeline 长度
    let target_frames = ((clip_timeline_len_sec * 1000.0) / fp).round().max(1.0) as usize;

    // 防御性 clamp：当 playback_rate ≈ 1.0 时，target_frames 不应超过 trimmed 长度，
    // 避免前端 sourceEndSec 超出源文件实际时长导致曲线被不合理拉伸。
    let rate_near_one = (playback_rate - 1.0).abs() <= 0.01;
    let target_frames = if rate_near_one && target_frames > trimmed.len() && !trimmed.is_empty() {
        eprintln!(
            "[pitch:trim] CLAMP: target_frames {} > trimmed {} (rate≈1), clamping to trimmed.len()",
            target_frames,
            trimmed.len(),
        );
        trimmed.len()
    } else {
        target_frames
    };

    eprintln!(
        "[pitch:trim] src_start={:.3}s src_end={:.3}s rate={:.2} tl_len={:.3}s \
         full_midi_len={} trimmed=[{}..{}]={} → target_frames={}",
        source_start_sec,
        source_end_sec,
        playback_rate,
        clip_timeline_len_sec,
        full_midi.len(),
        src_start_frame,
        src_end_frame,
        trimmed.len(),
        target_frames,
    );

    resample_curve_linear(trimmed, target_frames)
}

/// 使指定 clip 的 pitch MIDI 缓存失效（例如源文件变化后调用）。
/// 以 clip 所对应的 content_hash 为 key 删除缓存，影响所有共享该源文件的 clip。
/// 下次 `schedule_clip_pitch_jobs` 时会重新提交检测任务。
pub fn invalidate_clip_pitch_cache(tl: &TimelineState, clip: &Clip) {
    let root = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();
    let Some(ck) = build_clip_pitch_key(tl, clip, &root, 5.0) else {
        return;
    };
    let mut cache = global_cache().lock().unwrap_or_else(|e| e.into_inner());
    cache.remove(&ck.key);
}

/// 清除指定 clip 对应源文件的 inflight 标记。
/// 当拉伸完成（handle_stretch_ready）后调用，确保后续的
/// schedule_clip_pitch_jobs 不会因为残留的 inflight 标记而跳过该 clip。
pub fn clear_clip_inflight(tl: &TimelineState, clip: &Clip) {
    let root = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();
    let Some(ck) = build_clip_pitch_key(tl, clip, &root, 5.0) else {
        return;
    };
    let mut set = global_inflight().lock().unwrap_or_else(|e| e.into_inner());
    set.remove(&ck.key);
}

#[allow(dead_code)]
pub fn get_clips_for_root<'a>(tl: &'a TimelineState, root_track_id: &str) -> Vec<&'a Clip> {
    let mut out: Vec<&Clip> = tl
        .clips
        .iter()
        .filter(|c| tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id))
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}
