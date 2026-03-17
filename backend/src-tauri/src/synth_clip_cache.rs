//! 通用 per-clip 合成结果缓存（WORLD / ONNX 共享）。
//!
//! 以 `(clip_id, param_hash)` 为 key，缓存合成结果（stereo interleaved PCM）。
//! 参数不变时直接复用缓存，避免重复合成；参数变化时自动失效并重新合成。
//!
//! # 设计
//! - 进程级全局 `Mutex<SynthClipCache>`，实时路径与离线路径共享
//! - LRU 淘汰，容量上限 64 个 clip
//! - `param_hash` 使用 FNV-1a 64-bit，覆盖 clip 时间参数 + pitch_edit 曲线片段

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use crate::pitch_editing::PitchCurvesSnapshot;

// 导入 clip 渲染状态管理器
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

/// LRU 缓存，存储 per-clip 合成结果（WORLD 和 ONNX 共享）。
pub struct SynthClipCache {
    inner: HashMap<SynthClipCacheKey, SynthClipCacheEntry>,
    /// 按访问顺序排列的 key 列表（front = 最近使用，back = 最久未使用）。
    order: VecDeque<SynthClipCacheKey>,
    capacity: usize,
}

impl SynthClipCache {
    /// 创建指定容量的缓存。
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// 查询缓存。命中时将 key 移到 front（最近使用）。
    pub fn get(&mut self, key: &SynthClipCacheKey) -> Option<&SynthClipCacheEntry> {
        if !self.inner.contains_key(key) {
            return None;
        }
        // 将命中的 key 移到 front
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos).unwrap();
            self.order.push_front(k);
        }
        self.inner.get(key)
    }

    /// 插入缓存。若已满则淘汰最久未使用的 entry。
    pub fn insert(&mut self, key: SynthClipCacheKey, entry: SynthClipCacheEntry) {
        if self.inner.contains_key(&key) {
            // 更新已有 entry，移到 front
            self.inner.insert(key.clone(), entry);
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                let k = self.order.remove(pos).unwrap();
                self.order.push_front(k);
            }
            return;
        }

        // 容量已满时淘汰 back（最久未使用）
        while self.inner.len() >= self.capacity {
            if let Some(evict_key) = self.order.pop_back() {
                self.inner.remove(&evict_key);
            } else {
                break;
            }
        }

        self.order.push_front(key.clone());
        self.inner.insert(key, entry);
    }

    /// 使指定 clip_id 的所有缓存失效（不论 param_hash）。
    pub fn invalidate(&mut self, clip_id: &str) {
        let keys_to_remove: Vec<SynthClipCacheKey> = self
            .inner
            .keys()
            .filter(|k| k.clip_id == clip_id)
            .cloned()
            .collect();
        for k in &keys_to_remove {
            self.inner.remove(k);
            if let Some(pos) = self.order.iter().position(|o| o == k) {
                self.order.remove(pos);
            }
        }
    }

    /// 清空所有缓存。
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.inner.clear();
        self.order.clear();
    }

    /// 清空所有缓存并返回估算释放的字节数（仅 PCM 数据部分）。
    pub fn clear_and_estimate_bytes(&mut self) -> u64 {
        let bytes: u64 = self
            .inner
            .values()
            .map(|e| e.pcm_stereo.len() as u64 * 4) // f32 = 4 字节
            .sum();
        self.inner.clear();
        self.order.clear();
        bytes
    }

    /// 当前缓存条目数。
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

// ─── 全局实例 ──────────────────────────────────────────────────────────────────

static GLOBAL_SYNTH_CLIP_CACHE: OnceLock<Mutex<SynthClipCache>> = OnceLock::new();

/// 获取进程级全局合成 clip 缓存。
///
/// 首次调用时初始化，容量为 [`DEFAULT_CAPACITY`]（64）。
/// WORLD 和 ONNX 共享同一个缓存实例。
pub fn global_synth_clip_cache() -> &'static Mutex<SynthClipCache> {
    GLOBAL_SYNTH_CLIP_CACHE.get_or_init(|| Mutex::new(SynthClipCache::new(DEFAULT_CAPACITY)))
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
pub fn compute_param_hash(
    clip_id: &str,
    start_frame: u64,
    end_frame: u64,
    sr: u32,
    renderer_id: &str,
    curves: &PitchCurvesSnapshot,
    extra_curves: &std::collections::HashMap<String, Vec<f32>>,
    extra_params: &std::collections::HashMap<String, f64>,
) -> u64 {
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
    let mut sorted_curves: Vec<(&String, &Vec<f32>)> = extra_curves.iter().collect();
    sorted_curves.sort_by_key(|(k, _)| k.as_str());
    for (k, v) in sorted_curves {
        mix_bytes!(k.as_bytes());
        for &val in v.iter() {
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
}

/// 整 Clip 渲染结果的 LRU 缓存。
///
/// 与 [`SynthClipCache`]（per-segment）共存，用于 Clip 级预渲染缓存。
/// audio callback 中通过 `EngineClip.rendered_pcm` 直接读取，不经过此缓存。
/// 此缓存主要在 `build_snapshot` 阶段查询并填充 `rendered_pcm`。
pub struct RenderedClipCache {
    inner: HashMap<RenderedClipCacheKey, RenderedClipCacheEntry>,
    order: VecDeque<RenderedClipCacheKey>,
    capacity: usize,
}

impl RenderedClipCache {
    /// 创建指定容量的缓存。
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// 查询缓存。命中时将 key 移到 front（最近使用）。
    pub fn get(&mut self, key: &RenderedClipCacheKey) -> Option<&RenderedClipCacheEntry> {
        if !self.inner.contains_key(key) {
            return None;
        }
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos).unwrap();
            self.order.push_front(k);
        }
        self.inner.get(key)
    }

    /// 插入缓存。若已满则淘汰最久未使用的 entry。
    pub fn insert(&mut self, key: RenderedClipCacheKey, entry: RenderedClipCacheEntry) {
        if self.inner.contains_key(&key) {
            self.inner.insert(key.clone(), entry);
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                let k = self.order.remove(pos).unwrap();
                self.order.push_front(k);
            }
            return;
        }

        while self.inner.len() >= self.capacity {
            if let Some(evict_key) = self.order.pop_back() {
                self.inner.remove(&evict_key);
            } else {
                break;
            }
        }

        self.order.push_front(key.clone());
        self.inner.insert(key, entry);
    }

    /// 使指定 clip_id 的所有缓存失效（不论 param_hash）。
    pub fn invalidate(&mut self, clip_id: &str) {
        let keys_to_remove: Vec<RenderedClipCacheKey> = self
            .inner
            .keys()
            .filter(|k| k.clip_id == clip_id)
            .cloned()
            .collect();
        for k in &keys_to_remove {
            self.inner.remove(k);
            if let Some(pos) = self.order.iter().position(|o| o == k) {
                self.order.remove(pos);
            }
        }
    }

    /// 清空所有缓存。
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.inner.clear();
        self.order.clear();
    }

    /// 当前缓存条目数。
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 确保缓存容量不小于给定值（仅增不减）。
    pub fn ensure_capacity(&mut self, min_capacity: usize) {
        let next = min_capacity.max(1);
        if next > self.capacity {
            self.capacity = next;
            self.inner
                .reserve(self.capacity.saturating_sub(self.inner.len()));
            self.order
                .reserve(self.capacity.saturating_sub(self.order.len()));
        }
    }
}

// ─── 整 Clip 渲染缓存全局实例 ─────────────────────────────────────────────────

static GLOBAL_RENDERED_CLIP_CACHE: OnceLock<Mutex<RenderedClipCache>> = OnceLock::new();

/// 获取进程级全局整 Clip 渲染缓存。
///
/// 首次调用时初始化，容量为 `rendered_clip_capacity()`。
pub fn global_rendered_clip_cache() -> &'static Mutex<RenderedClipCache> {
    GLOBAL_RENDERED_CLIP_CACHE
        .get_or_init(|| Mutex::new(RenderedClipCache::new(rendered_clip_capacity())))
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
) -> u64 {
    let mut h: u64 = 14695981039346656037u64;

    fn include_rendered_extra_curve(renderer_id: &str, param_id: &str) -> bool {
        !(renderer_id == "nsf_hifigan_onnx"
            && matches!(
                param_id,
                "breath_gain" | "hifigan_tension" | "hifigan_volume"
            ))
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
    mix_bytes!(renderer_id.as_bytes());
    mix_bytes!(&start_frame.to_le_bytes());
    mix_bytes!(&end_frame.to_le_bytes());
    mix_bytes!(&sr.to_le_bytes());
    mix_bytes!(&playback_rate.to_bits().to_le_bytes());

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

    // 混入 extra_curves（AutomationCurve 类型参数），按 key 排序保证确定性
    let mut sorted_curves: Vec<(&String, &Vec<f32>)> = extra_curves.iter().collect();
    sorted_curves.sort_by_key(|(k, _)| k.as_str());
    for (k, v) in sorted_curves {
        if !include_rendered_extra_curve(renderer_id, k.as_str()) {
            continue;
        }
        mix_bytes!(k.as_bytes());
        for &val in v.iter() {
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
    tension_curve: Option<&Vec<f32>>,
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
}

pub struct TensionRenderedClipCache {
    inner: HashMap<TensionRenderedClipCacheKey, TensionRenderedClipCacheEntry>,
    order: VecDeque<TensionRenderedClipCacheKey>,
    capacity: usize,
}

impl TensionRenderedClipCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    pub fn get(
        &mut self,
        key: &TensionRenderedClipCacheKey,
    ) -> Option<&TensionRenderedClipCacheEntry> {
        if !self.inner.contains_key(key) {
            return None;
        }
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let hit = self.order.remove(pos).unwrap();
            self.order.push_front(hit);
        }
        self.inner.get(key)
    }

    pub fn insert(
        &mut self,
        key: TensionRenderedClipCacheKey,
        entry: TensionRenderedClipCacheEntry,
    ) {
        if self.inner.contains_key(&key) {
            self.inner.insert(key.clone(), entry);
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                let hit = self.order.remove(pos).unwrap();
                self.order.push_front(hit);
            }
            return;
        }

        while self.inner.len() >= self.capacity {
            if let Some(evict_key) = self.order.pop_back() {
                self.inner.remove(&evict_key);
            } else {
                break;
            }
        }

        self.order.push_front(key.clone());
        self.inner.insert(key, entry);
    }

    pub fn invalidate(&mut self, clip_id: &str) {
        let keys_to_remove: Vec<TensionRenderedClipCacheKey> = self
            .inner
            .keys()
            .filter(|key| key.clip_id == clip_id)
            .cloned()
            .collect();
        for key in &keys_to_remove {
            self.inner.remove(key);
            if let Some(pos) = self.order.iter().position(|item| item == key) {
                self.order.remove(pos);
            }
        }
    }

    /// 确保缓存容量不小于给定值（仅增不减）。
    pub fn ensure_capacity(&mut self, min_capacity: usize) {
        let next = min_capacity.max(1);
        if next > self.capacity {
            self.capacity = next;
            self.inner
                .reserve(self.capacity.saturating_sub(self.inner.len()));
            self.order
                .reserve(self.capacity.saturating_sub(self.order.len()));
        }
    }
}

static GLOBAL_TENSION_RENDERED_CLIP_CACHE: OnceLock<Mutex<TensionRenderedClipCache>> =
    OnceLock::new();

pub fn global_tension_rendered_clip_cache() -> &'static Mutex<TensionRenderedClipCache> {
    GLOBAL_TENSION_RENDERED_CLIP_CACHE
        .get_or_init(|| Mutex::new(TensionRenderedClipCache::new(rendered_clip_capacity())))
}

#[cfg(test)]
mod tests {
    use super::{compute_hifigan_tension_hash, compute_param_hash, compute_rendered_clip_hash};
    use crate::pitch_editing::PitchCurvesSnapshot;
    use std::collections::HashMap;

    #[test]
    fn rendered_clip_hash_ignores_breath_gain_curve() {
        let pitch_edit = vec![60.0f32; 32];
        let mut extra_params = HashMap::new();
        extra_params.insert("breath_enabled".to_string(), 1.0);

        let mut curve_a = HashMap::new();
        curve_a.insert("breath_gain".to_string(), vec![1.0f32; 32]);

        let mut curve_b = HashMap::new();
        curve_b.insert("breath_gain".to_string(), vec![0.25f32; 32]);
        curve_b.insert("hifigan_tension".to_string(), vec![18.0f32; 32]);

        let hash_a = compute_rendered_clip_hash(
            "clip-1",
            "input.wav",
            0,
            44_100,
            44_100,
            "nsf_hifigan_onnx",
            &pitch_edit,
            5.0,
            1.0,
            &curve_a,
            &extra_params,
        );
        let hash_b = compute_rendered_clip_hash(
            "clip-1",
            "input.wav",
            0,
            44_100,
            44_100,
            "nsf_hifigan_onnx",
            &pitch_edit,
            5.0,
            1.0,
            &curve_b,
            &extra_params,
        );

        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn hifigan_tension_hash_changes_with_curve() {
        let pitch_orig = vec![60.0f32; 64];
        let curve_a = vec![0.0f32; 64];
        let curve_b = vec![35.0f32; 64];

        let hash_a = compute_hifigan_tension_hash(
            "clip-1",
            11,
            0,
            44_100,
            44_100,
            5.0,
            &pitch_orig,
            Some(&curve_a),
        );
        let hash_b = compute_hifigan_tension_hash(
            "clip-1",
            11,
            0,
            44_100,
            44_100,
            5.0,
            &pitch_orig,
            Some(&curve_b),
        );

        assert_ne!(hash_a, hash_b);
    }

    #[test]
    fn segment_param_hash_changes_with_formant_curve() {
        let curves = PitchCurvesSnapshot {
            frame_period_ms: 5.0,
            pitch_orig: vec![],
            pitch_edit: vec![60.0f32; 64],
        };
        let extra_params = HashMap::new();

        let mut curve_a = HashMap::new();
        curve_a.insert("formant_shift_cents".to_string(), vec![0.0f32; 64]);

        let mut curve_b = HashMap::new();
        curve_b.insert("formant_shift_cents".to_string(), vec![240.0f32; 64]);

        let hash_a = compute_param_hash(
            "clip-1",
            0,
            44_100,
            44_100,
            "nsf_hifigan_onnx",
            &curves,
            &curve_a,
            &extra_params,
        );
        let hash_b = compute_param_hash(
            "clip-1",
            0,
            44_100,
            44_100,
            "nsf_hifigan_onnx",
            &curves,
            &curve_b,
            &extra_params,
        );

        assert_ne!(hash_a, hash_b);
    }
}
