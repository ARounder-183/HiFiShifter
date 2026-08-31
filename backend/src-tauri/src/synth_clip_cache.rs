//! 通用 per-clip 合成结果缓存（WORLD / ONNX 共享）。
//!
//! 以 `(clip_id, param_hash)` 为 key，缓存合成结果（stereo interleaved PCM）。
//! 参数不变时直接复用缓存，避免重复合成；参数变化时自动失效并重新合成。
//!
//! # 设计
//! - 进程级全局 `Mutex<SynthClipCache>`，实时路径与离线路径共享
//! - LRU 淘汰，容量上限 64 个 clip
//! - `param_hash` 使用 FNV-1a 64-bit，覆盖 clip 时间参数 + pitch_edit 曲线片段
//!
//! # 浮点参数量化（2026-06-30 修复）
//! 在 `compute_rendered_clip_hash` 中, formant_morph 的 f1 / f2 / strength
//! 不再直接按 raw bits 哈希, 而是与 `formant_cache.rs::make_formant_cache_key`
//! 保持一致的量化粒度后再混入哈希:
//! - target_f1_hz / target_f2_hz: 0.1 Hz 步长
//! - strength:                    0.001 步长
//!
//! 原因: 前后端 JSON 往返 / serde 反序列化 / React 状态重建等场景容易让 f64
//! 的 raw bits 在用户感知不到的精度下抖动, 若直接按 bits 哈希会让用户"没改
//! 共振峰参数"的情况下 RenderedClipCache 反复 miss, 表现为"共振峰参数
//! 频繁出现异常的缓存失效"。

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use crate::pitch_editing::PitchCurvesSnapshot;

// 导入 clip 渲染状态管理器
use crate::audio_engine::byte_budget_cache::ByteBudgetCache;
use crate::clip_rendering_state::{global_clip_rendering_state, ClipRenderingState};

// ─── 缓存容量 ──────────────────────────────────────────────────────────────────

const DEFAULT_CAPACITY: usize = 64;

// ─── Key / Entry ───────────────────────────────────────────────────────────────

/// 缓存 key：clip 唯一标识 + 参数哈希。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SynthClipCacheKey {
    pub clip_id: String,
    pub param_hash: u64,
}

/// 缓存 entry：合成结果（stereo interleaved PCM）。
#[derive(Debug, Clone)]
pub struct SynthClipCacheEntry {
    /// Stereo interleaved PCM，长度 = `frames * 2`。
    pub pcm_stereo: Arc<Vec<f32>>,
    /// 有效帧数。
    pub frames: u64,
    /// 采样率（Hz）。
    pub sample_rate: u32,
}

// ─── Cache ─────────────────────────────────────────────────────────────────────

/// Byte-budgeted LRU cache for per-clip synthesis results (WORLD and ONNX share).
pub struct SynthClipCache {
    inner: ByteBudgetCache<SynthClipCacheKey, SynthClipCacheEntry>,
}

impl SynthClipCache {
    /// 创建指定容量和字节预算的缓存。
    pub fn new(capacity: usize, budget_bytes: u64) -> Self {
        Self {
            inner: ByteBudgetCache::new(capacity, budget_bytes),
        }
    }

    /// 查询缓存。命中时将 key 移到 front（最近使用）。
    pub fn get(&mut self, key: &SynthClipCacheKey) -> Option<&SynthClipCacheEntry> {
        self.inner.get(key)
    }

    /// 插入缓存。字节预算自动管理淘汰。
    pub fn insert(&mut self, key: SynthClipCacheKey, entry: SynthClipCacheEntry) {
        let weight = entry.pcm_stereo.len() as u64 * 4; // f32 = 4 bytes
        self.inner.insert(key, entry, weight);
    }

    /// 使指定 clip_id 的所有缓存失效（不论 param_hash）。
    pub fn invalidate(&mut self, clip_id: &str) {
        self.inner.invalidate_where(|k| k.clip_id == clip_id);
    }

    /// 清空所有缓存。
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// 清空所有缓存并返回估算释放的字节数（仅 PCM 数据部分）。
    pub fn clear_and_estimate_bytes(&mut self) -> u64 {
        let bytes = self.inner.total_bytes();
        self.inner.clear();
        bytes
    }

    /// 当前缓存条目数。
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 当前缓存总字节数。
    pub fn total_bytes(&self) -> u64 {
        self.inner.total_bytes()
    }
}

// ─── 全局实例 ──────────────────────────────────────────────────────────────────

static GLOBAL_SYNTH_CLIP_CACHE: OnceLock<Mutex<SynthClipCache>> = OnceLock::new();

/// 获取进程级全局合成 clip 缓存。
///
/// 首次调用时初始化，容量为 [`DEFAULT_CAPACITY`]（64）。
/// WORLD 和 ONNX 共享同一个缓存实例。
pub fn global_synth_clip_cache() -> &'static Mutex<SynthClipCache> {
    GLOBAL_SYNTH_CLIP_CACHE.get_or_init(|| {
        let budget = crate::audio_engine::byte_budget_cache::env_cache_budget_bytes() / 4; // 1/4 of total budget
        Mutex::new(SynthClipCache::new(DEFAULT_CAPACITY, budget))
    })
}

// ─── Clip 渲染状态集成 ──────────────────────────────────────────────────────────

/// 检查 clip 是否已渲染完成（缓存命中）
pub fn is_clip_rendered(clip_id: &str, param_hash: u64) -> bool {
    let key = SynthClipCacheKey {
        clip_id: clip_id.to_string(),
        param_hash,
    };

    let mut cache = global_synth_clip_cache()
        .lock()
        .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());

    cache.get(&key).is_some()
}

/// 设置 clip 渲染状态
pub fn set_clip_rendering_state(
    clip_id: &str,
    state: ClipRenderingState,
    progress: f32,
    error: Option<String>,
) {
    let mut state_manager = global_clip_rendering_state()
        .lock()
        .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());

    state_manager.set_state(clip_id, state, progress, error);
}

/// 获取 clip 渲染状态
pub fn get_clip_rendering_state(clip_id: &str) -> Option<ClipRenderingState> {
    let state_manager = global_clip_rendering_state()
        .lock()
        .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());

    state_manager.get_state(clip_id).map(|info| info.state)
}

/// 检查 clip 是否就绪（缓存命中且状态为 Ready）
pub fn is_clip_ready(clip_id: &str, param_hash: u64) -> bool {
    let state_manager = global_clip_rendering_state()
        .lock()
        .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());

    state_manager.is_ready(clip_id) && is_clip_rendered(clip_id, param_hash)
}

/// 标记 clip 渲染开始
pub fn mark_clip_rendering_start(clip_id: &str) {
    set_clip_rendering_state(clip_id, ClipRenderingState::Rendering, 0.0, None);
}

/// 标记 clip 渲染完成
pub fn mark_clip_rendering_complete(clip_id: &str, param_hash: u64, entry: SynthClipCacheEntry) {
    // 插入缓存
    let key = SynthClipCacheKey {
        clip_id: clip_id.to_string(),
        param_hash,
    };

    let mut cache = global_synth_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    cache.insert(key, entry);

    // 更新状态
    set_clip_rendering_state(clip_id, ClipRenderingState::Ready, 1.0, None);
}

/// 标记 clip 渲染失败
pub fn mark_clip_rendering_failed(clip_id: &str, error: String) {
    set_clip_rendering_state(clip_id, ClipRenderingState::Failed, 0.0, Some(error));
}

/// 清理超时的渲染任务
pub fn cleanup_timeout_rendering_tasks() -> Vec<String> {
    let mut state_manager = global_clip_rendering_state()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    state_manager.cleanup_timeouts()
}

// ─── param_hash 计算 ───────────────────────────────────────────────────────────

/// 计算 clip 的参数哈希（FNV-1a 64-bit）。
///
/// 输入覆盖：
/// - `clip_id`：clip 唯一标识
/// - `start_frame` / `end_frame`：clip 在时间轴上的帧范围
/// - `sr`：采样率
/// - `pitch_edit` 曲线中与 clip 时间范围重叠的片段
/// - `extra_curves`：声码器专属自动化曲线（AutomationCurve 类型）
/// - `extra_params`：声码器专属静态参数（StaticEnum 类型）
///
/// 任意参数变化 → hash 变化 → 缓存失效 → 重新合成。
pub fn compute_param_hash<K, V, I>(
    clip_id: &str,
    start_frame: u64,
    end_frame: u64,
    sr: u32,
    renderer_id: &str,
    curves: &PitchCurvesSnapshot<'_>,
    extra_curves: I,
    extra_params: &std::collections::HashMap<String, f64>,
) -> u64
where
    K: AsRef<str>,
    V: AsRef<[f32]>,
    I: IntoIterator<Item = (K, V)>,
{
    // FNV-1a 64-bit 初始值
    let mut h: u64 = 14695981039346656037u64;

    macro_rules! mix_bytes {
        ($bytes:expr) => {
            for &b in $bytes {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211u64);
            }
        };
    }

    mix_bytes!(clip_id.as_bytes());
    mix_bytes!(renderer_id.as_bytes());
    mix_bytes!(&start_frame.to_le_bytes());
    mix_bytes!(&end_frame.to_le_bytes());
    mix_bytes!(&sr.to_le_bytes());

    // 混入与 clip 时间范围重叠的 pitch_edit 曲线片段
    let fp = curves.frame_period_ms.max(0.1);
    let start_sec = start_frame as f64 / sr.max(1) as f64;
    let end_sec = end_frame as f64 / sr.max(1) as f64;
    let start_idx = ((start_sec * 1000.0) / fp).floor().max(0.0) as usize;
    let end_idx = ((end_sec * 1000.0) / fp).ceil().max(0.0) as usize;

    let edit = &curves.pitch_edit;
    let lo = start_idx.min(edit.len());
    let hi = end_idx.min(edit.len());
    for &v in &edit[lo..hi] {
        mix_bytes!(&v.to_bits().to_le_bytes());
    }

    // 混入 extra_curves（AutomationCurve 类型参数），按 key 排序保证确定性
    let mut sorted_curves: Vec<(K, V)> = extra_curves.into_iter().collect();
    sorted_curves.sort_by(|(k1, _), (k2, _)| k1.as_ref().cmp(k2.as_ref()));
    for (k, v) in sorted_curves {
        mix_bytes!(k.as_ref().as_bytes());
        for &val in v.as_ref().iter() {
            mix_bytes!(&val.to_bits().to_le_bytes());
        }
    }

    // 混入 extra_params（StaticEnum 类型参数），按 key 排序保证确定性
    let mut sorted_params: Vec<(&String, &f64)> = extra_params.iter().collect();
    sorted_params.sort_by_key(|(k, _)| k.as_str());
    for (k, v) in sorted_params {
        mix_bytes!(k.as_bytes());
        mix_bytes!(&v.to_le_bytes());
    }

    h
}

// ─── 整 Clip 渲染缓存（Phase 2: Clip 级预渲染 + 实时混音）────────────────────

/// 整 Clip 渲染缓存默认容量。
const DEFAULT_RENDERED_CLIP_CAPACITY: usize = 1024;

static RENDERED_CLIP_CAPACITY: OnceLock<usize> = OnceLock::new();

fn rendered_clip_capacity() -> usize {
    *RENDERED_CLIP_CAPACITY.get_or_init(|| {
        std::env::var("HIFISHIFTER_RENDERED_CLIP_CACHE_CAPACITY")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_RENDERED_CLIP_CAPACITY)
    })
}

/// 整 Clip 渲染缓存的 key。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderedClipCacheKey {
    pub clip_id: String,
    /// 综合参数哈希（覆盖 pitch_edit + source_path + trim + playback_rate）。
    pub param_hash: u64,
}

/// 整 Clip 渲染缓存的 entry：预渲染后的完整 clip stereo PCM。
#[derive(Debug, Clone)]
pub struct RenderedClipCacheEntry {
    /// Stereo interleaved PCM（从 clip local frame 0 开始），长度 = clip_frames * 2。
    pub pcm_stereo: Arc<Vec<f32>>,
    /// 可选的独立气声 stem；存在时在播放回调中按当前 breath_gain 曲线实时混入。
    pub breath_noise_stereo: Option<Arc<Vec<f32>>>,
    /// clip 帧数。
    pub frames: u64,
    /// 采样率（Hz）。
    pub sample_rate: u32,
    /// 渲染时该 Clip 的 active take id。用于垫音（fallback）查找时识别跨
    /// Take 的旧渲染：undo 回退等场景下同 clip_id 换了 take，旧条目的内容
    /// 与当前可听内容无关，不得作为垫音（None = 旧条目/未知，宽松放行）。
    pub rendered_take_id: Option<String>,
}

/// 整 Clip 渲染结果的 byte-budgeted LRU 缓存。
///
/// 与 [`SynthClipCache`]（per-segment）共存，用于 Clip 级预渲染缓存。
/// audio callback 中通过 `EngineClip.rendered_pcm` 直接读取，不经过此缓存。
/// 此缓存主要在 `build_snapshot` 阶段查询并填充 `rendered_pcm`。
pub struct RenderedClipCache {
    inner: ByteBudgetCache<RenderedClipCacheKey, RenderedClipCacheEntry>,
}

impl RenderedClipCache {
    /// 创建指定容量和字节预算的缓存。
    pub fn new(capacity: usize, budget_bytes: u64) -> Self {
        Self {
            inner: ByteBudgetCache::new(capacity, budget_bytes),
        }
    }

    /// 查询缓存。命中时将 key 移到 front（最近使用）。
    pub fn get(&mut self, key: &RenderedClipCacheKey) -> Option<&RenderedClipCacheEntry> {
        self.inner.get(key)
    }

    /// 插入缓存。字节预算自动管理淘汰。
    pub fn insert(&mut self, key: RenderedClipCacheKey, entry: RenderedClipCacheEntry) {
        let pcm_bytes = entry.pcm_stereo.len() as u64 * 4;
        let noise_bytes = entry
            .breath_noise_stereo
            .as_ref()
            .map(|n| n.len() as u64 * 4)
            .unwrap_or(0);
        let weight = pcm_bytes + noise_bytes;
        self.inner.insert(key, entry, weight);
    }

    /// 使指定 clip_id 的所有缓存失效（不论 param_hash）。
    pub fn invalidate(&mut self, clip_id: &str) {
        self.inner.invalidate_where(|k| k.clip_id == clip_id);
    }

    /// 清空所有缓存。
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// 当前缓存条目数。
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 当前缓存总字节数。
    pub fn total_bytes(&self) -> u64 {
        self.inner.total_bytes()
    }

    /// 确保缓存容量不小于给定值（仅增不减）。
    pub fn ensure_capacity(&mut self, min_capacity: usize) {
        self.inner.ensure_capacity(min_capacity);
    }
}

// ─── 整 Clip 渲染缓存全局实例 ─────────────────────────────────────────────────

static GLOBAL_RENDERED_CLIP_CACHE: OnceLock<Mutex<RenderedClipCache>> = OnceLock::new();

/// 获取进程级全局整 Clip 渲染缓存。
///
/// 首次调用时初始化，容量为 `rendered_clip_capacity()`。
pub fn global_rendered_clip_cache() -> &'static Mutex<RenderedClipCache> {
    GLOBAL_RENDERED_CLIP_CACHE.get_or_init(|| {
        let budget = crate::audio_engine::byte_budget_cache::env_cache_budget_bytes() / 2; // 1/2 of total budget
        Mutex::new(RenderedClipCache::new(rendered_clip_capacity(), budget))
    })
}

// ─── Pending Rendered Keys（渲染线程 → snapshot 的 cache_key 传递）──────────

static PENDING_RENDERED_KEYS: OnceLock<Mutex<HashMap<String, RenderedClipCacheKey>>> =
    OnceLock::new();

/// 获取进程级全局 pending_rendered_keys。
///
/// 渲染线程成功渲染 clip 后将 `(clip_id, cache_key)` 写入此 map，
/// `build_snapshot` 优先从此 map 查找 cache_key（避免双重 hash 计算的不一致问题）。
pub fn global_pending_rendered_keys() -> &'static Mutex<HashMap<String, RenderedClipCacheKey>> {
    PENDING_RENDERED_KEYS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 渲染线程调用：注册一个 clip 的 cache_key。
pub fn register_pending_rendered_key(clip_id: &str, key: RenderedClipCacheKey) {
    let mut map = global_pending_rendered_keys()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    map.insert(clip_id.to_string(), key);
}

/// `build_snapshot` 调用：查找某个 clip 的渲染线程 cache_key。
///
/// 若找到，则使用此 key 查询 `rendered_clip_cache`，避免自行重新计算 hash。
pub fn lookup_pending_rendered_key(clip_id: &str) -> Option<RenderedClipCacheKey> {
    let map = global_pending_rendered_keys()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    map.get(clip_id).cloned()
}

/// 清除单个 clip 的 pending rendered key。
pub fn remove_pending_rendered_key(clip_id: &str) {
    let mut map = global_pending_rendered_keys()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    map.remove(clip_id);
}

/// 清空所有 pending rendered keys（播放停止或新一轮渲染开始时调用）。
pub fn clear_pending_rendered_keys() {
    let mut map = global_pending_rendered_keys()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    map.clear();
}

/// 计算整 Clip 渲染的参数哈希。
///
/// 输入覆盖：
/// - `clip_id`：clip 唯一标识
/// - `source_path`：源文件路径
/// - `start_frame` / `end_frame`：clip 在时间轴上的帧范围
/// - `sr`：采样率
/// - `pitch_edit` 曲线中与 clip 时间范围重叠的片段
/// - `playback_rate`：播放速率
/// - `extra_curves`：声码器专属自动化曲线（AutomationCurve 类型）
/// - `extra_params`：声码器专属静态参数（StaticEnum 类型）
pub fn compute_rendered_clip_hash(
    clip_id: &str,
    source_path: &str,
    start_frame: u64,
    end_frame: u64,
    sr: u32,
    renderer_id: &str,
    pitch_edit: &[f32],
    frame_period_ms: f64,
    playback_rate: f64,
    extra_curves: &std::collections::HashMap<String, Vec<f32>>,
    extra_params: &std::collections::HashMap<String, f64>,
    formant_morph: Option<&crate::state::ClipFormantMorph>,
    input_pitch_curve: Option<&[f32]>,
    // 源文件的 mtime（Unix 秒），用于区分同路径不同内容的文件版本。
    // 当文件被外部替换后，此值变化 → hash 变化 → 旧渲染缓存自动失效。
    source_file_mtime: Option<u64>,
    // Loop（循环源）属性：Loop 与非 Loop 的渲染输出完全不同（整文件平铺 vs
    // 窗口切片），必须参与哈希 —— 否则后台预渲染与 Loop 开关切换之间的竞态
    // 会把旧域的渲染结果当作新域的缓存命中。
    loop_enabled: bool,
    // 量化的源窗口 `(source_start_sec·1000, source_end_sec·1000)`：
    // 渲染输入随 trim/split 的锚点推进而变化，同样必须参与哈希。
    source_range_q: (i64, i64),
) -> u64 {
    let mut h: u64 = 14695981039346656037u64;

    fn include_rendered_extra_curve(renderer_id: &str, param_id: &str) -> bool {
        // vslib 把全部曲线烘焙进合成输出（含共通 volume/pan），
        // 因此这些曲线必须参与渲染缓存 key。
        if renderer_id == "vslib" {
            return true;
        }
        // 共通 volume/pan 在 mix 阶段实时应用，改变它们不应触发底层重渲染。
        if crate::renderer::common_params::is_common_mix_param(param_id) {
            return false;
        }
        // nsf-hifigan 的气声与张力属于渲染后处理，有独立缓存 key。
        !(renderer_id == "nsf_hifigan_onnx"
            && matches!(param_id, "breath_gain" | "hifigan_tension"))
    }

    macro_rules! mix_bytes {
        ($bytes:expr) => {
            for &b in $bytes {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211u64);
            }
        };
    }

    mix_bytes!(clip_id.as_bytes());
    mix_bytes!(source_path.as_bytes());
    // 混入 source_file_mtime：同路径不同文件内容 → 不同 mtime → 不同 hash → 缓存自动失效
    if let Some(mtime) = source_file_mtime {
        mix_bytes!(&mtime.to_le_bytes());
    }
    mix_bytes!(renderer_id.as_bytes());
    mix_bytes!(&start_frame.to_le_bytes());
    mix_bytes!(&end_frame.to_le_bytes());
    mix_bytes!(&sr.to_le_bytes());
    mix_bytes!(&playback_rate.to_bits().to_le_bytes());
    mix_bytes!(&[u8::from(loop_enabled)]);
    mix_bytes!(&source_range_q.0.to_le_bytes());
    mix_bytes!(&source_range_q.1.to_le_bytes());

    // 混入与 clip 时间范围重叠的 pitch_edit 曲线片段
    let fp = frame_period_ms.max(0.1);
    let start_sec = start_frame as f64 / sr.max(1) as f64;
    let end_sec = end_frame as f64 / sr.max(1) as f64;
    let start_idx = ((start_sec * 1000.0) / fp).floor().max(0.0) as usize;
    let end_idx = ((end_sec * 1000.0) / fp).ceil().max(0.0) as usize;

    let lo = start_idx.min(pitch_edit.len());
    let hi = end_idx.min(pitch_edit.len());
    for &v in &pitch_edit[lo..hi] {
        mix_bytes!(&v.to_bits().to_le_bytes());
    }

    // 混入“渲染输入 pitch curve”（clip 局部时间轴），
    // 以便缓存键直接跟随渲染输入变化，而不依赖额外 offset salt。
    if let Some(curve) = input_pitch_curve {
        mix_bytes!(b"input_pitch_curve");
        for &v in curve {
            mix_bytes!(&v.to_bits().to_le_bytes());
        }
    }

    // 混入 extra_curves，并且【只 Hash 当前时间切片的片段】，避免性能问题与错误缓存失效
    let mut sorted_curves: Vec<(&String, &[f32])> = extra_curves
        .iter()
        .map(|(k, v)| (k, v.as_slice()))
        .collect();
    sorted_curves.sort_by_key(|(k, _)| k.as_str());
    for (k, v) in sorted_curves {
        // 调用已定义好的过滤函数，防止后处理参数改变引发灾难级的底层重渲染
        if !include_rendered_extra_curve(renderer_id, k) {
            continue;
        }
        mix_bytes!(k.as_bytes());
        let curve_lo = start_idx.min(v.len());
        let curve_hi = end_idx.min(v.len());
        for &val in &v[curve_lo..curve_hi] {
            mix_bytes!(&val.to_bits().to_le_bytes());
        }
    }

    // 混入 extra_params（StaticEnum 类型参数），按 key 排序保证确定性
    let mut sorted_params: Vec<(&String, &f64)> = extra_params.iter().collect();
    sorted_params.sort_by_key(|(k, _)| k.as_str());
    for (k, v) in sorted_params {
        mix_bytes!(k.as_bytes());
        mix_bytes!(&v.to_le_bytes());
    }

    if let Some(formant) = formant_morph {
        mix_bytes!(b"clip_formant_morph");
        mix_bytes!(&[u8::from(formant.enabled)]);
        // 量化后再哈希, 避免浮点 raw bits 的微小抖动 (前后端 round-trip / 状态
        // 重建等场景) 触发 RenderedClipCache 误失效, 与 formant_cache.rs 中
        // make_formant_cache_key 的量化粒度保持一致 (Bug 修复, 2026-06-30):
        //   - target_f1_hz / target_f2_hz: 0.1 Hz 步长
        //   - strength:                    0.001 步长 (千分位)
        let f1_q: i64 = (formant.target_f1_hz * 10.0).round() as i64;
        let f2_q: i64 = (formant.target_f2_hz * 10.0).round() as i64;
        let st_q: i64 = (formant.strength * 1000.0).round() as i64;
        mix_bytes!(&f1_q.to_le_bytes());
        mix_bytes!(&f2_q.to_le_bytes());
        mix_bytes!(&st_q.to_le_bytes());
    }

    h
}

pub fn compute_breath_noise_hash(
    clip_id: &str,
    source_path: &str,
    start_frame: u64,
    end_frame: u64,
    sr: u32,
    renderer_id: &str,
    pitch_edit: &[f32],
    frame_period_ms: f64,
    playback_rate: f64,
    extra_curves: &std::collections::HashMap<String, Vec<f32>>,
    extra_params: &std::collections::HashMap<String, f64>,
    formant_morph: Option<&crate::state::ClipFormantMorph>,
    source_file_mtime: Option<u64>,
    loop_enabled: bool,
    source_range_q: (i64, i64),
) -> u64 {
    let filtered_curves: std::collections::HashMap<String, Vec<f32>> = extra_curves
        .iter()
        .filter(|(k, _)| k.as_str() != "formant_shift_cents")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    compute_rendered_clip_hash(
        clip_id,
        source_path,
        start_frame,
        end_frame,
        sr,
        renderer_id,
        pitch_edit,
        frame_period_ms,
        playback_rate,
        &filtered_curves,
        extra_params,
        formant_morph,
        None,
        source_file_mtime,
        loop_enabled,
        source_range_q,
    )
}

fn curve_slice_bounds(
    start_frame: u64,
    end_frame: u64,
    sr: u32,
    frame_period_ms: f64,
    len: usize,
) -> (usize, usize) {
    let fp = frame_period_ms.max(0.1);
    let start_sec = start_frame as f64 / sr.max(1) as f64;
    let end_sec = end_frame as f64 / sr.max(1) as f64;
    let start_idx = ((start_sec * 1000.0) / fp).floor().max(0.0) as usize;
    let end_idx = ((end_sec * 1000.0) / fp).ceil().max(0.0) as usize;
    (start_idx.min(len), end_idx.min(len))
}

pub fn compute_hifigan_tension_hash(
    clip_id: &str,
    base_param_hash: u64,
    start_frame: u64,
    end_frame: u64,
    sr: u32,
    frame_period_ms: f64,
    pitch_orig: &[f32],
    tension_curve: Option<&[f32]>,
) -> u64 {
    let mut h: u64 = 14695981039346656037u64;

    macro_rules! mix_bytes {
        ($bytes:expr) => {
            for &b in $bytes {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211u64);
            }
        };
    }

    mix_bytes!(clip_id.as_bytes());
    mix_bytes!(b"hifigan_tension");
    mix_bytes!(&base_param_hash.to_le_bytes());
    mix_bytes!(&start_frame.to_le_bytes());
    mix_bytes!(&end_frame.to_le_bytes());
    mix_bytes!(&sr.to_le_bytes());

    let (pitch_lo, pitch_hi) = curve_slice_bounds(
        start_frame,
        end_frame,
        sr,
        frame_period_ms,
        pitch_orig.len(),
    );
    for &value in &pitch_orig[pitch_lo..pitch_hi] {
        mix_bytes!(&value.to_bits().to_le_bytes());
    }

    if let Some(curve) = tension_curve {
        let (curve_lo, curve_hi) =
            curve_slice_bounds(start_frame, end_frame, sr, frame_period_ms, curve.len());
        for &value in &curve[curve_lo..curve_hi] {
            mix_bytes!(&value.to_bits().to_le_bytes());
        }
    }

    h
}

/// HiFiGAN tension 后处理缓存 key。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensionRenderedClipCacheKey {
    pub clip_id: String,
    pub base_param_hash: u64,
    pub tension_hash: u64,
}

/// HiFiGAN tension 后处理缓存 entry。
#[derive(Debug, Clone)]
pub struct TensionRenderedClipCacheEntry {
    pub pcm_stereo: Arc<Vec<f32>>,
    pub frames: u64,
    pub sample_rate: u32,
    /// 渲染时该 Clip 的 active take id；语义同
    /// [`RenderedClipCacheEntry::rendered_take_id`]（垫音防跨 Take 复用）。
    pub rendered_take_id: Option<String>,
}

pub struct TensionRenderedClipCache {
    inner: ByteBudgetCache<TensionRenderedClipCacheKey, TensionRenderedClipCacheEntry>,
}

impl TensionRenderedClipCache {
    pub fn new(capacity: usize, budget_bytes: u64) -> Self {
        Self {
            inner: ByteBudgetCache::new(capacity, budget_bytes),
        }
    }

    pub fn get(
        &mut self,
        key: &TensionRenderedClipCacheKey,
    ) -> Option<&TensionRenderedClipCacheEntry> {
        self.inner.get(key)
    }

    pub fn insert(
        &mut self,
        key: TensionRenderedClipCacheKey,
        entry: TensionRenderedClipCacheEntry,
    ) {
        let weight = entry.pcm_stereo.len() as u64 * 4;
        self.inner.insert(key, entry, weight);
    }

    pub fn invalidate(&mut self, clip_id: &str) {
        self.inner.invalidate_where(|k| k.clip_id == clip_id);
    }

    /// 当前缓存条目数。
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 确保缓存容量不小于给定值（仅增不减）。
    pub fn ensure_capacity(&mut self, min_capacity: usize) {
        self.inner.ensure_capacity(min_capacity);
    }

    /// 当前缓存总字节数。
    pub fn total_bytes(&self) -> u64 {
        self.inner.total_bytes()
    }
}

static GLOBAL_TENSION_RENDERED_CLIP_CACHE: OnceLock<Mutex<TensionRenderedClipCache>> =
    OnceLock::new();

pub fn global_tension_rendered_clip_cache() -> &'static Mutex<TensionRenderedClipCache> {
    GLOBAL_TENSION_RENDERED_CLIP_CACHE.get_or_init(|| {
        let budget = crate::audio_engine::byte_budget_cache::env_cache_budget_bytes() / 4;
        Mutex::new(TensionRenderedClipCache::new(
            rendered_clip_capacity(),
            budget,
        ))
    })
}

// ─── Breath Noise 独立缓存（formant 变化时可复用，避免重复 HNSEP 分离）─────────

/// Breath Noise 缓存的 key：使用不含 formant_shift_cents 的 base hash。
///
/// formant 变化时 RenderedClipCache 的 hash 不变（因为 formant 已排除），
/// 但如果其他参数（pitch_edit、playback_rate 等）变化，此 key 也会变化。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BreathNoiseCacheKey {
    pub clip_id: String,
    /// 与 RenderedClipCacheKey.param_hash 相同（不含 formant_shift_cents）。
    pub param_hash: u64,
}

/// Breath Noise 缓存的 entry：HNSEP 分离后的 noise stem（stereo interleaved）。
#[derive(Debug, Clone)]
pub struct BreathNoiseCacheEntry {
    pub noise_stereo: Arc<Vec<f32>>,
    pub frames: u64,
    pub sample_rate: u32,
}

/// Breath Noise 独立 byte-budgeted LRU 缓存。
///
/// 在 Breath 路径中，`breath_noise_stereo`（= unity_mix - harmonic_only）不受 formant 影响。
/// 当仅 formant 变化时，可直接复用此缓存中的 noise stem，跳过第二次 render_variant 调用，
/// 从而避免每个 clip 的两次 HNSEP 推理变为一次。
pub struct BreathNoiseCache {
    inner: ByteBudgetCache<BreathNoiseCacheKey, BreathNoiseCacheEntry>,
}

impl BreathNoiseCache {
    pub fn new(capacity: usize, budget_bytes: u64) -> Self {
        Self {
            inner: ByteBudgetCache::new(capacity, budget_bytes),
        }
    }

    pub fn get(&mut self, key: &BreathNoiseCacheKey) -> Option<&BreathNoiseCacheEntry> {
        self.inner.get(key)
    }

    pub fn insert(&mut self, key: BreathNoiseCacheKey, entry: BreathNoiseCacheEntry) {
        let weight = entry.noise_stereo.len() as u64 * 4;
        self.inner.insert(key, entry, weight);
    }

    pub fn invalidate(&mut self, clip_id: &str) {
        self.inner.invalidate_where(|k| k.clip_id == clip_id);
    }

    /// 当前缓存条目数。
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 确保缓存容量不小于给定值（仅增不减）。
    pub fn ensure_capacity(&mut self, min_capacity: usize) {
        self.inner.ensure_capacity(min_capacity);
    }

    /// 当前缓存总字节数。
    pub fn total_bytes(&self) -> u64 {
        self.inner.total_bytes()
    }
}

static GLOBAL_BREATH_NOISE_CACHE: OnceLock<Mutex<BreathNoiseCache>> = OnceLock::new();

/// 获取进程级全局 Breath Noise 缓存。
pub fn global_breath_noise_cache() -> &'static Mutex<BreathNoiseCache> {
    GLOBAL_BREATH_NOISE_CACHE.get_or_init(|| {
        let budget = crate::audio_engine::byte_budget_cache::env_cache_budget_bytes() / 8;
        Mutex::new(BreathNoiseCache::new(rendered_clip_capacity(), budget))
    })
}

/// 使指定 clip 的所有渲染缓存失效（SynthClipCache + RenderedClipCache + TensionRenderedClipCache + BreathNoiseCache）。
///
/// 此函数应在 pitch_edit 或其他影响合成的参数发生变化时调用，
/// 确保旧的预渲染结果不会被错误复用。
///
/// # 诊断
/// 会打印诊断日志帮助调试缓存失效相关问题。
pub fn invalidate_clip_all_caches(clip_id: &str) {
    // 1. SynthClipCache 失效（per-segment 合成缓存）
    {
        let mut cache = global_synth_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let before = cache.len();
        cache.invalidate(clip_id);
        if cache.len() < before {
            log::warn!(
                "[cache:invalidate] clip_id={} SynthClipCache invalidated",
                clip_id
            );
        }
    }

    // 2. RenderedClipCache 失效（整 Clip 预渲染缓存）
    {
        let mut cache = global_rendered_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let before = cache.len();
        cache.invalidate(clip_id);
        if cache.len() < before {
            log::warn!(
                "[cache:invalidate] clip_id={} RenderedClipCache invalidated (had {} entries)",
                clip_id, before
            );
        }
    }

    // 3. TensionRenderedClipCache 失效（HiFiGAN tension 专用缓存）
    {
        let mut cache = global_tension_rendered_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let before = cache.len();
        cache.invalidate(clip_id);
        if cache.len() < before {
            log::warn!(
                "[cache:invalidate] clip_id={} TensionRenderedClipCache invalidated",
                clip_id
            );
        }
    }

    // 4. BreathNoiseCache 失效（Breath Noise 独立缓存）
    {
        let mut cache = global_breath_noise_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let before = cache.len();
        cache.invalidate(clip_id);
        if cache.len() < before {
            log::warn!(
                "[cache:invalidate] clip_id={} BreathNoiseCache invalidated",
                clip_id
            );
        }
    }

    // 5. pending_rendered_keys 清除（渲染线程正在处理的 clip）
    {
        let mut map = global_pending_rendered_keys()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if map.remove(clip_id).is_some() {
            log::warn!(
                "[cache:invalidate] clip_id={} pending_rendered_key removed",
                clip_id
            );
        }
    }

    // 6. HiFiGAN 推理 chunk 缓存失效（按 clip_id 索引）
    //    当源文件被替换后，旧的推理输出不再有效，必须使 chunk 缓存失效。
    crate::renderer::hifigan::invalidate_chunk_cache_for_clip(clip_id);

    log::warn!(
        "[cache:invalidate] clip_id={} all caches invalidated",
        clip_id
    );
}

/// 专门为音高编辑提供的“柔性”缓存失效策略：仅失效片段级合成缓存，并解除旧的
/// `pending_rendered_keys` 绑定。
///
/// 必须保留 `RenderedClipCache`：
///
/// 1. 该缓存的 key 已经包含完整渲染参数（pitch_edit、renderer、curves、params、
///    source/trim/rate/formant 等），真实参数变化会自动产生新的 hash，旧条目
///    自然不会再被精确命中，不会造成错误复用。
/// 2. 保留最近一次渲染结果，可在新渲染完成前无缝垫音，避免播放瞬间出现静音。
/// 3. 引擎 `last_timeline` 有时晚于 AppState 时间线更新（例如音高分析完成后
///    异步组装 pitch_orig/pitch_edit 的场景），播放时据此产生的“假失效”不应摧毁
///    已经按当前参数渲染好的缓存；否则首次播放会整段静音、气声缺失，第二次播放才恢复。
pub fn invalidate_clip_for_pitch_edit(clip_id: &str) {
    log::warn!(
        "[cache:invalidate] clip_id={clip_id} pitch_edit invalidated (synth + pending keys cleared, rendered cache kept)"
    );
    // 1. SynthClipCache 失效
    {
        let mut cache = global_synth_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cache.invalidate(clip_id);
    }
    // 2. pending_rendered_keys 清除
    {
        let mut map = global_pending_rendered_keys()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.remove(clip_id);
    }
    // 注意：不要失效 RenderedClipCache，原因见上方文档注释。
}

/// 垫音身份校验：条目与当前 Clip 的 active take 都已知时必须一致。
/// 任一方未知（旧条目 / 无 take 工程）保持既有宽松行为，避免回归。
fn take_identity_matches(entry_take: Option<&str>, active_take: Option<&str>) -> bool {
    match (entry_take, active_take) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

/// 获取指定 clip 最近一次成功的整 clip 渲染结果（用作平滑过渡的垫音）。
///
/// `active_take_id` 为当前活跃 take：同 clip_id 换了 take 的旧渲染（undo
/// 回退等场景）与当前可听内容无关，不得作为垫音。
pub fn get_latest_rendered_pcm(
    clip_id: &str,
    active_take_id: Option<&str>,
) -> Option<(Arc<Vec<f32>>, Option<Arc<Vec<f32>>>)> {
    let cache = global_rendered_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let entry = cache
        .inner
        .iter()
        .find(|(k, v)| {
            k.clip_id == clip_id
                && take_identity_matches(v.rendered_take_id.as_deref(), active_take_id)
        })
        .map(|(_, v)| v)?;
    Some((entry.pcm_stereo.clone(), entry.breath_noise_stereo.clone()))
}

/// 获取指定 clip 最近一次成功的 Tension 渲染结果（用作平滑过渡的垫音）
pub fn get_latest_tension_rendered_pcm(
    clip_id: &str,
    active_take_id: Option<&str>,
) -> Option<Arc<Vec<f32>>> {
    let cache = global_tension_rendered_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let entry = cache
        .inner
        .iter()
        .find(|(k, v)| {
            k.clip_id == clip_id
                && take_identity_matches(v.rendered_take_id.as_deref(), active_take_id)
        })
        .map(|(_, v)| v)?;
    Some(entry.pcm_stereo.clone())
}

#[cfg(test)]
mod tests {
    use super::compute_rendered_clip_hash;

    #[test]
    fn rendered_clip_hash_changes_when_formant_morph_changes() {
        let formant_a = crate::state::ClipFormantMorph {
            enabled: true,
            target_f1_hz: 700.0,
            target_f2_hz: 1_400.0,
            strength: 0.55,
        };
        let formant_b = crate::state::ClipFormantMorph {
            target_f1_hz: 900.0,
            ..formant_a.clone()
        };

        let hash_a = compute_rendered_clip_hash(
            "clip-1",
            "demo.wav",
            0,
            48_000,
            48_000,
            "nsf_hifigan_onnx",
            &[60.0, 61.0, 62.0],
            5.0,
            1.0,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            Some(&formant_a),
            None,
            None, // source_file_mtime
            false,
            (0, 1_000),
        );
        let hash_b = compute_rendered_clip_hash(
            "clip-1",
            "demo.wav",
            0,
            48_000,
            48_000,
            "nsf_hifigan_onnx",
            &[60.0, 61.0, 62.0],
            5.0,
            1.0,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            Some(&formant_b),
            None,
            None, // source_file_mtime
            false,
            (0, 1_000),
        );

        assert_ne!(hash_a, hash_b);
    }

    #[test]
    fn rendered_clip_hash_changes_when_loop_or_source_range_changes() {
        // Loop（循环源）与源窗口（trim/split 锚点推进）都会改变渲染输入，
        // 任一变化必须使渲染缓存失效。
        let base = |loop_enabled: bool, range: (i64, i64)| {
            compute_rendered_clip_hash(
                "clip-1",
                "demo.wav",
                0,
                48_000,
                48_000,
                "nsf_hifigan_onnx",
                &[60.0, 61.0, 62.0],
                5.0,
                1.0,
                &std::collections::HashMap::new(),
                &std::collections::HashMap::new(),
                None,
                None,
                None, // source_file_mtime
                loop_enabled,
                range,
            )
        };
        assert_ne!(base(false, (0, 1_000)), base(true, (0, 1_000)));
        assert_ne!(base(false, (0, 1_000)), base(false, (500, 1_000)));
    }

    #[test]
    fn rendered_clip_hash_is_stable_under_subquantum_formant_jitter() {
        // 量化粒度: f1/f2 步长 0.1 Hz, strength 步长 0.001。
        // 在该粒度以下的浮点抖动 (前后端 round-trip / serde 反序列化 / 状态重建)
        // 不应改变 hash, 否则会触发 RenderedClipCache 的"假性失效"。
        let formant_a = crate::state::ClipFormantMorph {
            enabled: true,
            target_f1_hz: 700.0,
            target_f2_hz: 1_400.0,
            strength: 0.55,
        };
        let formant_b = crate::state::ClipFormantMorph {
            // 抖动远小于量化步长 (0.001 Hz << 0.1 Hz, 0.0001 << 0.001)
            target_f1_hz: 700.0 + 1e-6,
            target_f2_hz: 1_400.0 - 5e-6,
            strength: 0.55 + 1e-7,
            ..formant_a.clone()
        };

        let make_hash = |formant: &crate::state::ClipFormantMorph| {
            compute_rendered_clip_hash(
                "clip-1",
                "demo.wav",
                0,
                48_000,
                48_000,
                "nsf_hifigan_onnx",
                &[60.0, 61.0, 62.0],
                5.0,
                1.0,
                &std::collections::HashMap::new(),
                &std::collections::HashMap::new(),
                Some(formant),
                None,
                None, // source_file_mtime
                false,
                (0, 1_000),
            )
        };

        assert_eq!(make_hash(&formant_a), make_hash(&formant_b));
    }
}
