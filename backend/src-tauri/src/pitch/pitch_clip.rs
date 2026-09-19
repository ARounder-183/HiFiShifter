use crate::state::{Clip, TimelineState};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::{mpsc, Mutex, OnceLock};
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
    #[allow(dead_code)]
    clip_id: String,
    key: String,
    frame_period_ms: f64,
    sample_rate: u32,
    #[allow(dead_code)]
    pre_silence_sec: f64,
    /// true = playback_rate==1，分析源音频全量，cache key 不含 trim
    #[allow(dead_code)]
    is_full_source: bool,
}

#[derive(Debug, Clone)]
pub struct CachedClipPitch {
    pub key: String,
    /// 全量源音频的逐帧 MIDI 音高（0 = 无声帧）。
    ///
    /// **可能为空**：FCPE 不可用时仍会缓存电平（见 `level`），此时该字段为空
    /// 向量；消费方应把空向量视同「无音高数据」。
    pub midi: Vec<f32>,
    /// 全量源音频的逐帧电平（f32 linear，1.0 = 数字满量程），与 `midi` 同帧率。
    ///
    /// 与 `midi` 同一次解码 / 重采样产出（零额外 I/O），供 DYN 参数的原声电平
    /// 基线使用。响度分析不依赖声码器可用性，因此 FCPE 缺席时它照样产出。
    pub level: Vec<f32>,
}

/// 单次解码产出的分析结果（音高 + 电平）。
pub struct ClipPitchAnalysis {
    /// 逐帧 MIDI 音高（0 = 无声帧）。FCPE 不可用或推理失败时为空。
    pub midi: Vec<f32>,
    /// 逐帧电平（linear）。与 `midi` 同帧率。
    pub level: Vec<f32>,
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

/// 逐帧原声电平分析（DYN 的 `dyn_orig` 来源）。
///
/// 与静音检测（`audio/silence_detect.rs`）共用同一套已验证的口径：
/// `HOP = 5 ms`（与参数线 `frame_period_ms` 对齐）、`窗口 = 20 ms`、RMS。
/// 窗口以帧中心为基准、向两侧各半窗展开并在片段端点处收缩，因此
/// 输出帧数 = `ceil(frames * 1000 / (sample_rate * fp_ms))`。
///
/// 返回值是**线性幅度**（1.0 = 数字满量程），不是 dB：DYN 曲线本身就是电平
/// 倍率，直接持有线性值可省去渲染路径上的 pow/log。
pub(crate) fn compute_frame_levels(mono: &[f32], sample_rate: u32, frame_period_ms: f64) -> Vec<f32> {
    const WINDOW_MS: f64 = 20.0;
    let fp = frame_period_ms.max(0.1);
    let sr = sample_rate.max(1) as f64;
    if mono.is_empty() {
        return Vec::new();
    }
    let hop_samples = ((fp / 1000.0) * sr).round().max(1.0) as usize;
    let window_samples = (((WINDOW_MS / 1000.0) * sr).round() as usize).max(1) as usize;
    let total_hops = mono.len().div_ceil(hop_samples).max(1);

    let mut out = Vec::with_capacity(total_hops);
    let mut prefix = 0.0f64;
    // 前缀和加速：每个 hop 的窗内平方和 = O(1)。
    let mut prefix_sums = Vec::with_capacity(mono.len() + 1);
    prefix_sums.push(0.0f64);
    for &v in mono {
        let sq = (v as f64) * (v as f64);
        prefix += sq;
        prefix_sums.push(prefix);
    }

    for hop in 0..total_hops {
        let center = (hop * hop_samples) as f64 + hop_samples as f64 * 0.5;
        let start = (center - window_samples as f64 * 0.5).max(0.0) as usize;
        let end = ((center + window_samples as f64 * 0.5) as usize).min(mono.len());
        if end <= start {
            out.push(0.0);
            continue;
        }
        let sum_sq = prefix_sums[end] - prefix_sums[start];
        let rms = (sum_sq / (end - start) as f64).sqrt();
        out.push(if rms.is_finite() { rms as f32 } else { 0.0 });
    }
    out
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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
    hasher.update(b"clip_pitch_v4_fcpe_source_midi");
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

/// 查询 clip 分析缓存（音高 + 电平，同一条目）。
/// - 缓存命中：直接返回 `Some`。
/// - 缓存未命中：**不再同步计算**，直接返回 `None`。
///   调用方应提前通过 `schedule_clip_pitch_jobs` 触发异步预计算。
pub fn get_clip_analysis_global(
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

/// 查询 clip pitch MIDI 缓存（`get_clip_analysis_global` 的音高专用视图：
/// 缓存命中但音高缺失时返回 None）。
pub fn get_or_compute_clip_pitch_midi_global(
    tl: &TimelineState,
    clip: &Clip,
    root_track_id: &str,
    frame_period_ms: f64,
) -> Option<CachedClipPitch> {
    get_clip_analysis_global(tl, clip, root_track_id, frame_period_ms)
        .filter(|cached| !cached.midi.is_empty())
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

/// 遍历 timeline 中所有可见 clip，对缓存未命中的 clip 异步提交分析任务
/// （音高 + 逐帧电平，同一次解码产出）。
/// 任务完成后通过 `engine_tx` 发送 `EngineCommand::ClipPitchReady`，触发 snapshot rebuild。
///
/// 利用 `GLOBAL_CLIP_PITCH_INFLIGHT` 去重，同一 clip 不会重复提交。
///
/// 入选条件（任一满足）：
/// - 所在根轨道开启了 Compose（音高曲线需要它）；
/// - 所在根轨道**正在使用动态（DYN）**，即 dyn 曲线上存在真实目标值 ——
///   动态是混音级参数，未开 Compose 也要生效，因此它的原声基线必须照常分析。
///
/// `stretch_cache`：若 clip 已有拉伸后 PCM，优先使用它作为音高检测输入。
pub fn schedule_clip_pitch_jobs(
    tl: &TimelineState,
    engine_tx: &mpsc::Sender<crate::audio_engine::types::EngineCommand>,
    app_handle: Option<&tauri::AppHandle>,
    _out_rate: u32,
) {
    debug_eprintln!(
        "[pitch_clip] schedule_clip_pitch_jobs called, clips={}, app_handle={}",
        tl.clips.len(),
        app_handle.is_some()
    );

    // 注意：这里**不再**因 FCPE 不可用而整体早退 —— 分析同时产出 DYN 所需的
    // 原声电平，而电平不依赖声码器（见 `analyze_clip_pitch_and_level`）。
    let fcpe_available = crate::fcpe_onnx::is_available();

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
                debug_eprintln!("[pitch_clip] clip '{}' skipped: no source_path", clip.id);
                continue;
            }
        };
        if !file_exists_cached(Path::new(source_path)) {
            debug_eprintln!(
                "[pitch_clip] clip '{}' skipped: file not found: {}",
                clip.id, source_path
            );
            continue;
        }

        // 入选条件（任一满足）：
        // - 根轨道开启 Compose（音高曲线需要它）；
        // - 该根轨道需要动态（DYN）的原声电平。
        //
        // 第二条**刻意不看 compose_enabled**：动态是混音级参数，与"是否合成"
        // 无关，未开启合成的原始音频轨道同样支持。
        {
            let root_id = tl.resolve_root_track_id(&clip.track_id).unwrap_or_default();
            let root_track = tl.tracks.iter().find(|t| t.id == root_id);
            let compose_enabled = root_track.map(|t| t.compose_enabled).unwrap_or(false);
            let dyn_needed = dyn_needs_level_analysis(tl, &root_id);
            if !compose_enabled && !dyn_needed {
                debug_eprintln!(
                    "[pitch_clip] clip '{}' skipped: compose_enabled=false and dyn unused for root '{}'",
                    clip.id, root_id
                );
                continue;
            }
            if !fcpe_available && !dyn_needed {
                // 音高需要 FCPE；没有 FCPE 又不需要电平时没有可产出的数据。
                debug_eprintln!(
                    "[pitch_clip] clip '{}' skipped: FCPE unavailable and dyn unused",
                    clip.id
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
                debug_eprintln!(
                    "[pitch_clip] clip '{}' ({}) cache HIT (shared key), skipping",
                    clip.name, clip.id
                );
                continue;
            }
            debug_eprintln!(
                "[pitch_clip] clip '{}' ({}) cache MISS, will analyze",
                clip.name, clip.id
            );
        }

        // inflight 去重：以内容哈希为 key，相同源文件只允许一个分析任务
        let inflight_key = ck.key.clone();
        let should_spawn = {
            let mut set = global_inflight().lock().unwrap_or_else(|e| e.into_inner());
            if set.contains(&inflight_key) {
                debug_eprintln!(
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
        debug_eprintln!("[pitch_clip] no pending jobs (all cached or inflight), nothing to do");
        return;
    }
    debug_eprintln!("[pitch_clip] {} clip(s) need analysis", pending_jobs.len());

    // ── 阶段2：重置全局进度状态，批量提交分析任务 ────────────────────────────
    let total = pending_jobs.len() as u32;
    global_batch_state().reset(total);

    // 发送分析开始事件
    if let Some(app) = app_handle {
        let root_track_id = pending_jobs
            .first()
            .map(|j| j.root_track_id.clone())
            .unwrap_or_default();
        debug_eprintln!(
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
        log::warn!(
            "[pitch_clip] pitch_orig_analysis_started emit result: {:?}",
            r1
        );
        // 发送初始进度（0%，显示第一个 clip 名称）
        let first_clip_name = pending_jobs.first().map(|j| j.clip.name.clone());
        log::warn!(
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
        log::info!("[pitch_clip] initial progress emit result: {:?}", r2);
    } else {
        log::error!("[pitch_clip] WARNING: app_handle is None, cannot emit events!");
    }

    for job in pending_jobs {
        let tx = engine_tx.clone();
        let app_handle_clone = app_handle.cloned();
        let tl_clone = tl.clone();

        std::thread::spawn(move || {
            // 通知进度：开始分析此 clip
            log::warn!(
                "[pitch_clip] thread: starting analysis for clip '{}'",
                job.clip.name
            );
            global_batch_state().set_current(Some(job.clip.name.clone()));

            let analysis =
                analyze_clip_pitch_and_level(&tl_clone, &job.clip, &job.root_track_id, frame_period_ms);
            // 只要拿到电平（或音高）就值得写缓存：DYN 的原声基线独立于声码器。
            let has_data = analysis
                .as_ref()
                .map(|a| !a.midi.is_empty() || !a.level.is_empty())
                .unwrap_or(false);

            // 完成一个 clip，更新进度
            let completed = global_batch_state().complete_one();
            global_batch_state().set_current(None);
            log::warn!(
                "[pitch_clip] thread: clip '{}' analysis done, has_data={}, completed={}/{}",
                job.clip.name,
                has_data,
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
                debug_eprintln!(
                    "[pitch_clip] thread: progress emit {}/{} result: {:?}",
                    completed, total, r
                );
            } else {
                log::warn!("[pitch_clip] thread: WARNING app_handle is None, cannot emit progress!");
            }

            // 无论成功与否，先清除 inflight 标记
            {
                let mut set = global_inflight().lock().unwrap_or_else(|e| e.into_inner());
                set.remove(&job.inflight_key);
            }

            if let Some(analysis) = analysis.filter(|_| has_data) {
                let cached = CachedClipPitch {
                    key: job.ck.key.clone(),
                    midi: analysis.midi,
                    level: analysis.level,
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
                log::warn!(
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
                log::warn!(
                    "[pitch_clip] thread: all {} clips done, resetting batch state",
                    total
                );
                global_batch_state().reset(0);
            }
        });
    }
}

/// 分析单个 clip 的源音频，产出**音高曲线 + 逐帧电平**。
///
/// 一次解码同时服务两个用途（零额外 I/O）：FCPE 推理得到 MIDI 音高，同一份
/// mono PCM 上再做逐帧 RMS 得到电平。音高需要 FCPE（不可用时 `midi` 为空），
/// 电平不需要 —— DYN 参数的原声基线在任何环境下都要能算出来。
pub fn analyze_clip_pitch_and_level(
    tl: &TimelineState,
    clip: &Clip,
    root_track_id: &str,
    frame_period_ms: f64,
) -> Option<ClipPitchAnalysis> {
    let ck = build_clip_pitch_key(tl, clip, root_track_id, frame_period_ms)?;

    // ── 始终解码源音频全量 PCM 进行分析 ─────────────────────────────────────
    // 缓存中存的是全量源音频的曲线，不含 trim/rate 处理。
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

    // ── 逐帧电平（DYN 的原声基线）───────────────────────────────────────
    // 用**去直流但未归一化**的信号：归一化（scale）是为 FCPE 准备的动态范围
    // 拉伸，若把它算进电平，响度就会随"这一片段有多响"被反复改写，
    // DYN 的目标电平也就失去了绝对意义。
    let level: Vec<f32> = {
        let dc_removed: Vec<f32> = mono_raw.iter().map(|&v| (v - mean) as f32).collect();
        compute_frame_levels(&dc_removed, analysis_rate, ck.frame_period_ms)
    };

    // ── 音高（FCPE）──────────────────────────────────────────────────
    // 声码器不可用时不再整体失败：电平是有独立价值的产出（DYN 不需要声码器），
    // 因此这里只让 midi 为空。
    let mut midi: Vec<f32> = Vec::new();
    if crate::fcpe_onnx::is_available() {
        let frame_period_tl_ms = ck.frame_period_ms.max(0.1);
        let f0_floor = crate::fcpe_onnx::FCPE_F0_MIN_HZ;
        let f0_ceil = crate::fcpe_onnx::FCPE_F0_MAX_HZ;

        match crate::fcpe_onnx::infer_f0_hz(
            &mono,
            analysis_rate,
            frame_period_tl_ms,
            f0_floor,
            f0_ceil,
        ) {
            Ok(f0_hz) if f0_hz.len() >= 2 => {
                midi = Vec::with_capacity(f0_hz.len());
                for hz in f0_hz {
                    midi.push(hz_to_midi(hz));
                }
            }
            Ok(_) => {
                log::warn!(
                    "[pitch_clip] FCPE returned too few frames for clip '{}'",
                    clip.name
                );
            }
            Err(e) => {
                log::error!(
                    "[pitch_clip] FCPE inference failed for clip '{}' ({}): {}",
                    clip.name, clip.id, e
                );
            }
        }
    }
    // ── 全量曲线直接返回 ──────────────────────────────────────────────
    // 缓存中始终存全量源音频的曲线。
    // trim 截取 + rate resample 在推送（handle_clip_pitch_ready）
    // 和组装（assemble_pitch_orig_from_cache）阶段按需执行。

    // Small gap fill.
    let gap_ms = std::env::var("HIFISHIFTER_FCPE_F0_GAP_MS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 200.0);
    if gap_ms > 0.0 {
        let gap_frames = ((gap_ms / ck.frame_period_ms.max(0.1)).round() as isize).max(1) as usize;
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

    Some(ClipPitchAnalysis { midi, level })
}

/// 兼容包装：只要音高曲线。FCPE 不可用或推理失败时返回 None
/// （与旧行为一致 —— 调用方据此判定"音高不可用"）。
pub fn compute_clip_pitch_midi(
    tl: &TimelineState,
    clip: &Clip,
    root_track_id: &str,
    frame_period_ms: f64,
) -> Option<Vec<f32>> {
    let analysis = analyze_clip_pitch_and_level(tl, clip, root_track_id, frame_period_ms)?;
    if analysis.midi.is_empty() {
        return None;
    }
    Some(analysis.midi)
}

/// 该根轨道组是否需要**原声电平（DYN 基线）**分析。
///
/// 判定条件（任一）：
/// - `dyn` 曲线上已有真实目标值（用户画过线）；
/// - 参数面板正在编辑动态（前端显式登记，见 `dyn_panel_open_roots`）——
///   用户刚切到动态面板、尚未落笔时也必须先把基线算出来，否则面板里的
///   虚线（原声电平）与按 dB 缩放的波形都是空的。
///
/// 该函数**刻意不检查 `compose_enabled`**：动态是混音级参数，与"合成"无关，
/// 未开启合成的原始音频轨道同样支持。
pub(crate) fn dyn_needs_level_analysis(tl: &TimelineState, root_track_id: &str) -> bool {
    if let Some(entry) = tl.params_by_root_track.get(root_track_id) {
        let drawn = entry
            .extra_curves
            .get(crate::renderer::common_params::DYN_PARAM_ID)
            .map(|curve| curve.iter().any(|&v| v >= 0.0))
            .unwrap_or(false);
        if drawn {
            return true;
        }
    }
    // 面板打开集合用"读取即释放"的方式查询：绝不能在持锁时再调用任何会
    // 加同一把锁的代码（std::sync::Mutex 不可重入）。
    let panel_open = match dyn_panel_open_roots().lock() {
        Ok(roots) => roots.contains(root_track_id),
        Err(e) => e.into_inner().contains(root_track_id),
    };
    if panel_open {
        return true;
    }
    false
}

/// 前端正在编辑「动态」面板的根轨道集合（面板打开期间登记）。
///
/// 作用：让"打开了动态面板但还没画任何线"的状态也能触发原声电平分析，
/// 这是虚线基线与 dB 波形能立刻显示出来的前提。面板离开时注销，
/// 避免为一个没人看的参数持续付出分析开销。
pub fn dyn_panel_open_roots() -> &'static std::sync::Mutex<std::collections::HashSet<String>> {
    static ROOTS: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    ROOTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

/// 逐帧分析曲线（电平/音高皆可）从**全量源音频域**映射到 clip 可见区间。
///
/// `trim_and_resample_midi` 的数学与参数种类无关（截取窗口、Loop 回绕、倒放锚定、
/// 前导静音、按 rate 重采样），因此这里直接委托，只提供一个语义正确的名字，
/// 避免在电平分析处出现「midi」字样的误导。**单一实现**，不要在别处复制。
pub fn trim_and_resample_curve(
    full_curve: &[f32],
    frame_period_ms: f64,
    source_start_sec: f64,
    source_end_sec: f64,
    playback_rate: f64,
    clip_timeline_len_sec: f64,
    loop_enabled: bool,
    media_total_sec: Option<f64>,
    reversed: bool,
) -> Vec<f32> {
    trim_and_resample_midi(
        full_curve,
        frame_period_ms,
        source_start_sec,
        source_end_sec,
        playback_rate,
        clip_timeline_len_sec,
        loop_enabled,
        media_total_sec,
        reversed,
    )
}

/// 从全量 MIDI 曲线中截取 source range 区间并按 playback_rate 重采样。
/// 返回对应 clip 在时间线上可见区间的 MIDI 曲线。
///
/// - `full_midi`：全量源音频的 MIDI 曲线（FCPE 输出，每帧间隔 `frame_period_ms`）
/// - `source_start_sec`：clip 的 source_start_sec（源音频有效区间起点）
/// - `source_end_sec`：clip 的 source_end_sec（源音频有效区间终点）
/// - `playback_rate`：clip 的 playback_rate（>1 加速，<1 减速）
/// - `clip_timeline_len_sec`：clip 在时间线上的可见长度（秒）
/// - `loop_enabled`：Loop（循环源）属性
/// - `media_total_sec`：源媒体总时长（Loop 模式的回绕周期 D）；非 Loop 或未知传 None
/// - `reversed`：是否倒放。Loop 模式下倒放从 `source_end` 向下遍历并回绕，
///   与音频渲染的环绕方向一致；非 Loop 时函数内部按升序窗口处理，
///   调用方须传入重定向窗口 `[se−len·r, se]`（见
///   `clip_pitch_trim_window_sec`）并在输出后整体翻转 —— 翻转使前导/
///   尾部静音自动落到正确一侧。
#[allow(clippy::too_many_arguments)]
pub fn trim_and_resample_midi(
    full_midi: &[f32],
    frame_period_ms: f64,
    source_start_sec: f64,
    source_end_sec: f64,
    playback_rate: f64,
    clip_timeline_len_sec: f64,
    loop_enabled: bool,
    media_total_sec: Option<f64>,
    reversed: bool,
) -> Vec<f32> {
    let fp = frame_period_ms.max(0.1);
    let src_start = source_start_sec.max(0.0);

    // 速率净化：非有限 / 过小的速率按 1.0 处理（两条 Loop 路径共用）。
    let rate = if playback_rate.is_finite() && playback_rate > 1e-6 {
        playback_rate
    } else {
        1.0
    };

    // 派生窗口（非 Loop 正放）：终点 = 起点 + 时间线长度×速率，与音频渲染、
    // 前端编辑模型一致。循环开关反复切换或历史数据可能留下与长度脱钩的
    // 陈旧 source_end —— 在此统一派生，避免曲线窗口被冻结在错误位置。
    // 注意用**原始** source_start（可为负，前导静音场景），不是上面的
    // clamp 值；Loop / 倒放保持调用方传入的锚点字段。
    let source_end_sec = if !loop_enabled && !reversed {
        source_start_sec + clip_timeline_len_sec.max(0.0) * rate
    } else {
        source_end_sec
    };

    // 从全量曲线中截取 source range 区间
    let src_start_frame = ((src_start * 1000.0) / fp).round().max(0.0) as usize;

    // 根据 source_end_sec 计算结束帧
    let src_end_frame = ((source_end_sec * 1000.0) / fp).round().max(0.0) as usize;
    let src_end_frame = src_end_frame.min(full_midi.len());

    // ── Loop（循环源）：对整个媒体文件做模运算回绕 ────────────────────────────
    // idx(i) = floor_mod(anchor ± round(i·rate), N)，N 覆盖完整媒体时长；
    // 正放锚点 = 原始 source_start（可为负，floor_mod 正确环绕），
    // 倒放锚点 = min(source_end, D)（向下遍历）。
    // 该路径不依赖源窗口的先后次序 —— Loop 下 split 等编辑会产生
    // start > end 的"环绕窗口"（音频只由锚点与 D 决定），必须先于
    // 窗口空判执行，否则这类 clip 的音高曲线会被误判为空。
    if loop_enabled {
        if let Some(total_sec) = media_total_sec.filter(|v| v.is_finite() && *v > 0.0) {
            if full_midi.is_empty() {
                return Vec::new();
            }
            let n = (((total_sec * 1000.0) / fp).round() as usize)
                .min(full_midi.len())
                .max(1);
            let anchor_f = ((source_start_sec * 1000.0) / fp).round() as i64;
            // 倒放锚点与音频路径同约定：min(source_end, D) 后不做 max(0)，
            // 负 source_end 由 rem_euclid 统一环绕。
            let anchor_r = ((source_end_sec.min(total_sec)) * 1000.0 / fp).round() as i64;
            let target_frames = ((clip_timeline_len_sec * 1000.0) / fp).round().max(1.0) as usize;
            let mut out = Vec::with_capacity(target_frames);
            for i in 0..target_frames {
                let consumed = (i as f64 * rate).round() as i64;
                let idx_i = if reversed {
                    anchor_r - 1 - consumed
                } else {
                    anchor_f + consumed
                };
                let idx = idx_i.rem_euclid(n as i64) as usize;
                out.push(full_midi[idx]);
            }
            return out;
        }
    }

    // Loop（循环源）+ 媒体时长未知 + 环绕窗口（split 产生 start > end）：
    // 若落到下方 `src_start_frame >= src_end_frame` 的空判会把曲线整体清空。
    // 退化为"整条缓存曲线即回绕周期"（与 assemble_pitch_orig_from_cache 的
    // 回退一致），保证仍有内容可显示/编辑。
    if loop_enabled && src_start_frame >= src_end_frame {
        if full_midi.is_empty() {
            return Vec::new();
        }
        let n = full_midi.len();
        let anchor_f = ((source_start_sec * 1000.0) / fp).round() as i64;
        let anchor_r = src_end_frame as i64;
        let target_frames = ((clip_timeline_len_sec * 1000.0) / fp).round().max(1.0) as usize;
        let mut out = Vec::with_capacity(target_frames);
        for i in 0..target_frames {
            let consumed = (i as f64 * rate).round() as i64;
            let idx_i = if reversed {
                anchor_r - 1 - consumed
            } else {
                anchor_f + consumed
            };
            let idx = idx_i.rem_euclid(n as i64) as usize;
            out.push(full_midi[idx]);
        }
        return out;
    }

    // 非 Loop：窗口越出媒体域（负起点的前导静音 / 终点越过缓存末端甚至
    // 整窗在媒体外的尾部静音 —— 编辑器无界延伸可达）时，走**带静音的
    // 窗口映射**：输出帧 i ↔ 源坐标 win_start + (i−lead)·rate，域外为静音。
    // 完全在媒体域内的窗口保持既有"截取 + 线性重采样"路径不变（含下方
    // Loop 回退与 rate≈1 防御性 clamp）。
    if !loop_enabled {
        let raw_end_frame_i64 = ((source_end_sec * 1000.0) / fp).round() as i64;
        let window_crosses_media = source_start_sec < 0.0
            || source_end_sec < 0.0
            || raw_end_frame_i64 > full_midi.len() as i64;
        if window_crosses_media {
            let target_frames = ((clip_timeline_len_sec.max(0.0) * 1000.0) / fp)
                .round()
                .max(1.0) as usize;
            return assemble_nonloop_pitch_from_window(
                full_midi,
                fp,
                source_start_sec,
                source_end_sec,
                rate,
                target_frames,
            );
        }
    }

    if src_start_frame >= src_end_frame {
        log::warn!(
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

    // 媒体时长未知：退化为窗口回绕，保证仍有内容可显示。
    // 倒放从窗口末端向下遍历（与音频方向一致），不能忽略 reversed。
    if loop_enabled && target_frames > trimmed.len() {
        let window_frames = trimmed.len();
        let mut out = Vec::with_capacity(target_frames);
        for i in 0..target_frames {
            let u = i as f64 * rate;
            let wrapped = (u % (window_frames as f64)).round() as usize;
            let idx = if reversed {
                (src_end_frame - 1).saturating_sub(wrapped.min(window_frames - 1))
            } else {
                src_start_frame + wrapped.min(window_frames - 1)
            };
            out.push(full_midi[idx]);
        }
        return out;
    }

    // 防御性 clamp：当 playback_rate ≈ 1.0 时，target_frames 不应超过 trimmed 长度，
    // 避免前端 sourceEndSec 超出源文件实际时长导致曲线被不合理拉伸。
    let rate_near_one = (playback_rate - 1.0).abs() <= 0.01;
    let target_frames =
        if !loop_enabled && rate_near_one && target_frames > trimmed.len() && !trimmed.is_empty() {
            log::warn!(
            "[pitch:trim] CLAMP: target_frames {} > trimmed {} (rate≈1), clamping to trimmed.len()",
            target_frames,
            trimmed.len(),
        );
            trimmed.len()
        } else {
            target_frames
        };

    log::warn!(
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

/// 非 Loop **消费窗口 → clip 时间线帧曲线**（带静音的窗口映射）。
///
/// 输出长度恒为 `target_frames`；输出帧 `i` 对应源坐标
/// `win_start_sec + i·rate`（消费自窗口起点开始），源坐标落在
/// `[0, win_end_frame)` 之外（媒体前导/尾部越界区）输出 0 —— 与音频
/// 渲染的静音表达逐帧一致。
///
/// 窗口完全在媒体域内时无需走此路径（保留既有线性重采样管线）；
/// 倒放由调用方先以重定向窗口 `[se−len·r, se]` 调用、再整体翻转输出
/// （翻转后前导/尾部静音自动互换到正确一侧）。
pub(crate) fn assemble_nonloop_pitch_from_window(
    full_midi: &[f32],
    fp: f64,
    win_start_sec: f64,
    win_end_sec: f64,
    rate: f64,
    target_frames: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; target_frames];
    if target_frames == 0 {
        return out;
    }
    let win_start_frame_f = (win_start_sec * 1000.0) / fp;
    // 【必须同时钳到曲线长度】窗口终点可以合法地越过缓存末端（编辑器允许把
    // Clip 无界延伸出媒体之外，那段应当是静音）。此前的判定只比较
    // `idx >= win_end_frame`，于是在"窗口越界 + 索引落在 win_end 与缓存末端
    // 之间"时直接越界索引 —— 曲线上限必须参与判定，否则这是一个必然 panic
    // （release 下 panic=abort，表现为整个应用崩溃）。
    //
    // 历史：音高路径靠调用方的 `window_crosses_media` 预判掩盖了大部分场景，
    // 但只要窗口终点超过缓存长度且曲线比 win_end 短就会触发；DYN 的电平曲线
    // 长度与音高帧数未必逐帧一致，因此这条路径被高频命中。
    let win_end_frame = ((win_end_sec * 1000.0) / fp)
        .round()
        .clamp(0.0, full_midi.len() as f64) as i64;
    // 消费方向：时间线帧 i 直接对应源坐标 win_start + i·rate（消费自窗口
    // 起点开始）；窗口起点在媒体起点之前的部分自然落为前导静音。
    for (i, slot) in out.iter_mut().enumerate() {
        let src_f = win_start_frame_f + i as f64 * rate;
        let idx = src_f.round() as i64;
        if idx < 0 || idx >= win_end_frame {
            continue; // 媒体域外 → 静音
        }
        *slot = full_midi[idx as usize];
    }
    out
}

/// 使指定 clip 的 pitch MIDI 缓存失效（例如源文件变化后调用）。
/// 以 clip 所对应的 content_hash 为 key 删除缓存，影响所有共享该源文件的 clip。
/// 下次 `schedule_clip_pitch_jobs` 时会重新提交检测任务。
#[allow(dead_code)]
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
#[allow(dead_code)]
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
    let mut out: Vec<&'a Clip> = tl
        .clips
        .iter()
        .filter(|c| tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id))
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

#[cfg(test)]
mod tests {
    use super::{
        assemble_nonloop_pitch_from_window, dyn_needs_level_analysis, dyn_panel_open_roots,
        trim_and_resample_midi,
    };

    /// Loop + 媒体时长未知 + 环绕窗口（start > end，split 产生）：
    /// 不得落入"空窗口"提前返回 —— 退化为整条缓存曲线回绕，输出非空
    /// 且相位与逐帧 floor_mod 参考一致。
    #[test]
    fn loop_wrapped_window_without_media_duration_does_not_collapse_to_empty() {
        // 环绕窗口 [3.5, 3.0)（start > end），缓存曲线 100 帧（1s @10ms）。
        let full: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = trim_and_resample_midi(
            &full, 10.0, 3.5,  // source_start_sec（> end）
            3.0,  // source_end_sec
            1.0,  // playback_rate
            2.0,  // clip_timeline_len_sec → target = 200 帧
            true, // loop_enabled
            None, // 媒体时长未知
            false,
        );
        assert_eq!(out.len(), 200, "curve must cover the clip length");
        assert!(out.iter().any(|&v| v > 0.0), "curve must not be empty");

        // 与逐帧 floor_mod 参考对拍：idx = floor_mod(anchor + i, N)。
        let anchor = ((3.5f64 * 1000.0) / 10.0).round() as i64; // 350
        for (i, v) in out.iter().enumerate() {
            let expect = full[(anchor + i as i64).rem_euclid(100) as usize];
            assert!((v - expect).abs() < 1e-6, "frame {i}: {v} != {expect}");
        }
    }

    /// Loop + 已知媒体时长：逐帧 floor_mod(anchor ± round(i·rate), N)
    /// 映射正确（正放），且长度等于 clip 时间线帧数。
    #[test]
    fn loop_with_media_duration_maps_per_frame_floor_mod() {
        // 媒体 1s（100 帧 @10ms），锚点 -0.25s（负值环绕到尾部一侧）。
        let full: Vec<f32> = (0..100).map(|i| (i * 7) as f32 % 13.0).collect();
        let out = trim_and_resample_midi(&full, 10.0, -0.25, 1.0, 1.0, 1.5, true, Some(1.0), false);
        assert_eq!(out.len(), 150);
        let anchor = ((-0.25f64 * 1000.0) / 10.0).round() as i64; // -25
        for (i, v) in out.iter().enumerate() {
            let idx = (anchor + i as i64).rem_euclid(100);
            assert!((v - full[idx as usize]).abs() < 1e-6, "frame {i}");
        }
    }

    /// 窗口终点越过缓存末端（编辑器允许把 Clip 延伸出媒体之外）时不得越界。
    ///
    /// 这是 DYN 原声电平曲线路径上真实发生过的崩溃：`win_end_frame` 只按窗口
    /// 秒数计算，一旦它大于曲线长度，`full[idx]` 就越界 panic（release 下
    /// `panic = "abort"`，表现为整个应用崩溃）。越界部分必须是静音（0）。
    #[test]
    fn window_end_beyond_curve_length_is_silence_not_panic() {
        // 曲线 400 帧（4s @10ms），窗口取到 5.2s → 越界 120 帧。
        let full: Vec<f32> = vec![1.0f32; 400];
        let out = assemble_nonloop_pitch_from_window(&full, 10.0, 0.0, 5.2, 1.0, 600);
        assert_eq!(out.len(), 600);
        for (i, v) in out.iter().enumerate() {
            if i < 400 {
                assert!((v - 1.0).abs() < 1e-6, "帧 {i} 应在媒体域内");
            } else {
                assert_eq!(*v, 0.0, "帧 {i} 越界应为静音");
            }
        }
    }

    /// 窗口终点甚至早于曲线末端时，也不得读取到窗口之外的部分。
    #[test]
    fn window_end_clamps_to_curve_length_when_window_is_shorter() {
        let full: Vec<f32> = vec![2.0f32; 400];
        let out = assemble_nonloop_pitch_from_window(&full, 10.0, 0.0, 1.0, 1.0, 200);
        assert_eq!(out.len(), 200);
        // 窗口 1s = 100 帧，其后应为静音。
        for (i, v) in out.iter().enumerate() {
            let expected = if i < 100 { 2.0 } else { 0.0 };
            assert!((v - expected).abs() < 1e-6, "帧 {i}");
        }
    }

    /// 动态（DYN）的电平分析**不得**依赖 `compose_enabled`。
    ///
    /// 回归护栏：动态是混音级参数，与"是否启用合成"无关。此前把分析挂在
    /// pitch 的调度条件上（受 compose 门控），导致未开启合成的轨道永远拿不到
    /// 原声电平 —— 虚线基线与 dB 波形都是空的。
    #[test]
    fn dyn_level_analysis_is_independent_of_compose_enabled() {
        use crate::renderer::common_params::DYN_PARAM_ID;

        let mut tl = crate::state::TimelineState::default();
        let root_id = tl.add_track(Some("root".to_string()), None, None);
        // 明确关闭合成：这条断言就是本测试的全部意义。
        if let Some(t) = tl.tracks.iter_mut().find(|t| t.id == root_id) {
            t.compose_enabled = false;
        }

        let mut entry = crate::state::TrackParamsState::default();

        // 1) 还没画任何 dyn 线、面板也没打开 → 不需要分析。
        tl.params_by_root_track
            .insert("root".to_string(), entry.clone());
        assert!(
            !dyn_needs_level_analysis(&tl, "root"),
            "无曲线且面板未打开时不应触发分析"
        );

        // 2) 用户画了真实目标值 → 需要分析（即便 compose 关闭）。
        entry
            .extra_curves
            .insert(DYN_PARAM_ID.to_string(), vec![1.0f32, 0.5, 2.0]);
        tl.params_by_root_track
            .insert("root".to_string(), entry.clone());
        assert!(
            dyn_needs_level_analysis(&tl, "root"),
            "compose 关闭但用户画了 dyn 时仍必须分析"
        );

        // 3) 只有哨兵帧（−1 = 沿用原声）且面板没打开 → 不需要分析
        //    （用户既没画过线，也没在看这个面板）。
        let sentinel_only = crate::renderer::common_params::DYN_FOLLOW_ORIG;
        entry
            .extra_curves
            .insert(DYN_PARAM_ID.to_string(), vec![sentinel_only; 8]);
        tl.params_by_root_track
            .insert("root".to_string(), entry.clone());
        assert!(
            !dyn_needs_level_analysis(&tl, "root"),
            "只有哨兵帧且面板未打开时不必分析"
        );

        // 4) 面板打开（前端登记）→ 必须分析，哪怕一个点都还没画。
        //    这是虚线基线与 dB 波形能在切到动态面板后立刻显示的前提。
        {
            let mut roots = dyn_panel_open_roots().lock().unwrap_or_else(|e| e.into_inner());
            roots.insert("root".to_string());
        }
        assert!(
            dyn_needs_level_analysis(&tl, "root"),
            "面板已打开时必须分析，否则基线/波形永远是空的"
        );
        // 清理，避免污染同进程内的其它测试。
        {
            let mut roots = dyn_panel_open_roots().lock().unwrap_or_else(|e| e.into_inner());
            roots.remove("root");
        }
        assert!(
            !dyn_needs_level_analysis(&tl, "root"),
            "面板关闭后应停止分析"
        );
    }
}
