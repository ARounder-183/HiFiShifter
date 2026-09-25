//! 假立体声（L/R 实质一致）判定与判定结果记忆。
//!
//! 背景：真立体声源在渲染时会把整条处理器链按声道跑两遍
//! （`pitch_editing.rs` 的声道扇出），耗时翻倍。大量"人力"素材实际上是
//! 单声道内容被混流成双声道（"假立体声"）——两声道逐样本相同，折叠为
//! 单声道后听感完全不变、耗时减半。
//!
//! 本模块是全工程**唯一**定义"L/R 是否一致"语义的地方；调用方只消费
//! [`ChannelVerdict`]，不得在别处复刻比较逻辑。

use std::collections::HashMap;
use std::path::Path;
use std::sync::{OnceLock, RwLock};

// ─── 判定结果 ────────────────────────────────────────────────────────────────

/// 单个媒体文件的声道一致性判定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelVerdict {
    /// 单声道源（`channels < 2`）—— 本就单声道，无需转换，也不算假立体声。
    Mono,
    /// L/R 存在实质差异（真立体声）。
    TrueStereo,
    /// L/R 在容差内一致（假立体声）—— 折叠为单声道不损失内容。
    FakeStereo,
    /// 无法判定（解码失败 / 无有效采样）—— 调用方按"保持原状"处理。
    Unknown,
}

// ─── 判定参数 ────────────────────────────────────────────────────────────────

/// 抽样判定参数。
///
/// 字段与持久化策略 `config::ChannelImportPolicy` 一一对应；本结构只承载
/// 判定所需的三个数值，不含模式（模式属于调用方的决策）。
#[derive(Debug, Clone, PartialEq)]
pub struct DetectOptions {
    /// 每个抽样窗口的时长（秒）。
    pub window_sec: f64,
    /// 抽样窗口数；`0` = 不抽样、扫描全部给定区间。
    pub window_count: usize,
    /// 逐样本绝对差容差。
    ///
    /// 无需再单独设 RMS 阈值：只要每个样本都落在容差内，RMS 差必然也落在
    /// 容差内（`|Σd²/n|^0.5 ≤ max|d|`），多一个阈值只是同一约束的重复表达。
    pub tolerance: f32,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            window_sec: 0.25,
            window_count: 12,
            tolerance: 1e-3,
        }
    }
}

impl DetectOptions {
    /// 规范化：钳制到有意义的范围（与 `ChannelImportPolicy::normalized` 同口径）。
    pub fn normalized(&self) -> Self {
        Self {
            window_sec: if self.window_sec.is_finite() {
                self.window_sec.clamp(0.05, 5.0)
            } else {
                0.25
            },
            window_count: self.window_count.min(256),
            tolerance: if self.tolerance.is_finite() {
                // 与 `ChannelImportPolicy::normalized` 同口径：[0, 1]（满幅）。
                self.tolerance.clamp(0.0, 1.0)
            } else {
                1e-6
            },
        }
    }

    /// 抽样窗口覆盖的总时长上限（秒）；`window_count == 0` 时为 `None`（不限）。
    pub fn scan_budget_sec(&self) -> Option<f64> {
        if self.window_count == 0 {
            None
        } else {
            Some(self.window_sec * self.window_count as f64)
        }
    }

    /// 策略签名：任一参数变化都必须产出不同值，使旧判定缓存失效。
    pub fn signature(&self) -> u64 {
        let n = self.normalized();
        let mut h: u64 = 14695981039346656037u64;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211u64);
            }
        };
        mix(&n.window_sec.to_bits().to_le_bytes());
        mix(&(n.window_count as u64).to_le_bytes());
        mix(&n.tolerance.to_bits().to_le_bytes());
        h
    }
}

// ─── 纯判定核心 ──────────────────────────────────────────────────────────────

/// 逐帧差累计器。
///
/// 判定规则是**逐帧绝对差 ≤ 容差**（任一帧超差即否决），不需要 RMS 累计 ——
/// 早期版本曾累计平方差算 RMS，阈值移除后成了纯开销（热路径上每帧一次
/// 浮点乘加），已随阈值一并删除。
#[derive(Debug, Default, Clone, Copy)]
struct DiffAcc {
    frames: u64,
}

impl DiffAcc {
    /// 累计一帧；返回该帧是否仍在容差内（`false` = 已确定不是假立体声）。
    #[inline]
    fn push(&mut self, l: f32, r: f32, tolerance: f32) -> bool {
        self.frames += 1;
        (l - r).abs() <= tolerance
    }
}

/// 判定交错 PCM 的 L/R 是否在容差内一致。
///
/// `pcm` 应是**实际消费区间**的交错采样（而非整个文件）：判定结果只对该区间
/// 负责，这与渲染/听感实际使用的区间一致。
///
/// 窗口沿区间均匀铺开且**首尾都落在区间内**，因此"开头一致、结尾分叉"的素材
/// 不会漏判。任一窗口出现超容差样本即短路返回 [`ChannelVerdict::TrueStereo`]。
///
/// 这是全模块唯一的比较入口：WAV 逐窗口解码路径与容器前缀解码路径最终都汇到
/// [`compare_frames`]，不得在别处复刻比较逻辑。
pub fn analyze_interleaved(
    pcm: &[f32],
    channels: u16,
    sample_rate: u32,
    opts: &DetectOptions,
) -> ChannelVerdict {
    if channels < 2 {
        return ChannelVerdict::Mono;
    }
    let ch = channels as usize;
    let frames = pcm.len() / ch;
    if frames == 0 {
        return ChannelVerdict::Unknown;
    }

    let opts = opts.normalized();
    let mut acc = DiffAcc::default();

    for (start, len) in sample_windows(frames, sample_rate, &opts) {
        let last = (start + len).min(frames);
        if !compare_frames(pcm, ch, start, last, opts.tolerance, &mut acc) {
            return ChannelVerdict::TrueStereo;
        }
    }
    finalize(acc)
}

/// 比较 `[first, last)` 帧的 L/R；返回 `false` 表示已发现超容差样本
/// （调用方应短路，不必继续扫）。
fn compare_frames(
    pcm: &[f32],
    channels: usize,
    first: usize,
    last: usize,
    tolerance: f32,
    acc: &mut DiffAcc,
) -> bool {
    for f in first..last {
        let base = f * channels;
        if base + 1 >= pcm.len() {
            break;
        }
        if !acc.push(pcm[base], pcm[base + 1], tolerance) {
            return false;
        }
    }
    true
}

/// 在 `frames` 帧内计算抽样窗口（帧下标区间）。
///
/// - `window_count == 0`：单窗口覆盖全部帧（全量扫描）。
/// - 窗口总长 ≥ 区间长：单窗口覆盖全部帧（没有抽样余地）。
/// - 否则沿区间均匀取 `window_count` 个窗口，最后一个窗口右对齐到区间末尾，
///   保证尾部也被检查。
fn sample_windows(frames: usize, sample_rate: u32, opts: &DetectOptions) -> Vec<(usize, usize)> {
    if frames == 0 {
        return Vec::new();
    }
    if opts.window_count == 0 {
        return vec![(0, frames)];
    }

    let sr = if sample_rate == 0 { 44_100 } else { sample_rate } as f64;
    let win = ((opts.window_sec * sr).round() as usize).max(1);
    let count = opts.window_count.max(1);

    if win >= frames || count == 1 {
        return vec![(0, frames)];
    }

    let span = frames - win;
    let mut out = Vec::with_capacity(count);
    let mut last_start = None;
    for i in 0..count {
        // 用整数插值避免逐次浮点累加的漂移。
        let start = span.saturating_mul(i) / (count - 1);
        if last_start == Some(start) {
            continue;
        }
        last_start = Some(start);
        out.push((start, win));
    }
    out
}

// ─── 判定结果记忆 ────────────────────────────────────────────────────────────

/// 判定缓存键。
///
/// **必须含内容指纹**：`mtime` 只有整秒精度，同秒内被替换为同大小文件会产生
/// 假命中（`synth_clip_cache` 的渲染哈希已就此加过一层防护，此处同理）。
/// **必须含策略签名**：用户调整窗口/容差后旧判定必须失效。
/// **必须含区间**：判定只对给定区间负责——"前 1 秒一致、之后分叉"的文件在
/// 窄区间下是假立体声、在全文件下是真立体声，两种结论不可互相顶替。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VerdictKey {
    /// 小写规范化后的绝对路径。
    pub path: String,
    /// 源文件内容指纹（head+tail+size 的 FNV-1a）。
    pub fingerprint: Option<u64>,
    pub channels: u16,
    pub sample_rate: u32,
    /// 源域区间的毫秒量化 `(start, end)`；`None` = 整个文件。
    /// 量化避免浮点尾差造成永不命中（与 `synth_clip_cache::source_range_q` 同约定）。
    pub region_q: Option<(i64, i64)>,
    /// 策略签名（[`DetectOptions::signature`]）。
    pub policy_sig: u64,
}

impl VerdictKey {
    pub fn new(
        path: &Path,
        fingerprint: Option<u64>,
        channels: u16,
        sample_rate: u32,
        region: Option<(f64, f64)>,
        opts: &DetectOptions,
    ) -> Self {
        Self {
            path: normalize_path_key(path),
            fingerprint,
            channels,
            sample_rate,
            region_q: quantize_region(region),
            policy_sig: opts.signature(),
        }
    }
}

/// 源域秒区间 → 毫秒量化（与 `synth_clip_cache` 的 `source_range_q` 同口径）。
pub fn quantize_region(region: Option<(f64, f64)>) -> Option<(i64, i64)> {
    region.map(|(s, e)| {
        let q = |v: f64| {
            if v.is_finite() {
                (v * 1000.0).round() as i64
            } else {
                0
            }
        };
        (q(s), q(e))
    })
}

fn normalize_path_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/").to_lowercase()
}

/// 缓存容量上限：超出即整体清空（判定成本远低于维护 LRU 的复杂度，
/// 且条目本身很小）。避免长会话无界增长。
const VERDICT_CACHE_CAPACITY: usize = 4096;

fn verdict_cache() -> &'static RwLock<HashMap<VerdictKey, ChannelVerdict>> {
    static CACHE: OnceLock<RwLock<HashMap<VerdictKey, ChannelVerdict>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn verdict_cache_get(key: &VerdictKey) -> Option<ChannelVerdict> {
    verdict_cache()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(key)
        .copied()
}

pub fn verdict_cache_put(key: VerdictKey, verdict: ChannelVerdict) {
    let mut cache = verdict_cache().write().unwrap_or_else(|e| e.into_inner());
    if cache.len() >= VERDICT_CACHE_CAPACITY && !cache.contains_key(&key) {
        cache.clear();
    }
    cache.insert(key, verdict);
}

#[cfg(test)]
pub fn verdict_cache_clear() {
    verdict_cache()
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}

/// 测试专用：串行化所有触碰全局判定缓存的测试。
///
/// 缓存是进程级单例，而 verdict 缓存测试互相之间以"清空 → 塞入 → 查询"的
/// 方式断言；并行运行时彼此的 clear 会吃掉对方刚写入的条目（偶发
/// `原键仍命中` 失败）。持同一把进程级互斥锁即可，不影响并行度敏感的
/// 解码类测试。
#[cfg(test)]
fn verdict_cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

// ─── 媒体文件判定（抽样解码） ────────────────────────────────────────────────

/// 判定一个媒体文件在给定源域区间内的 L/R 一致性（先查缓存，未命中才解码；
/// 导入 / 扫描热路径统一走本函数）。`region` 为源域秒区间（`None` = 整个文件）。
///
/// **会解码音频**，调用方必须保证不在持有 timeline 全局锁时调用
///（慢盘/网络盘上可能耗时数百毫秒）。
pub fn verdict_for_file(
    path: &Path,
    region: Option<(f64, f64)>,
    opts: &DetectOptions,
) -> ChannelVerdict {
    let opts = opts.normalized();
    verdict_for_regions(path, &[region], &opts)
        .into_iter()
        .next()
        .unwrap_or(ChannelVerdict::Unknown)
}

/// 同一文件的**多个**消费区间批量判定（扫描 / 迁移热路径的 I/O 去重入口）。
///
/// 判定语义与逐区间调用 [`verdict_for_file`] 完全一致：同样的窗口抽样、容差
/// 与预算上限，结果照样进出进程级判定缓存（命中条目直接采用）。差别只在
/// **I/O 组织**：
/// - WAV：打开一次，逐区间 seek（原来每区间各开一次文件）；
/// - 非 WAV：Symphonia 前缀解码只做**一次**，解码上界取"各区间覆盖需求的
///   max"（原来每个区间各自从文件头解到自己的终点）；
/// - 头部探测与内容指纹（缓存键的原料）每文件只做一次（原来逐区间各做一遍，
///   指纹要读 head+tail，批量下是纯重复 I/O）。
///
/// 【指纹缺位不缓存】指纹读不出（文件被占用 / 读取中途出错）时**不能**拿
/// `fingerprint: None` 的键去缓存：键的其余部分（path/region/policy）相同而
/// 文件已被替换时，旧判定会被误命中 —— 指纹字段正是防"同路径文件替换"的。
/// 此时本批照样完整解码判定，只是不读写缓存。
///
/// 返回值与 `regions` 一一对应。
pub fn verdict_for_regions(
    path: &Path,
    regions: &[Option<(f64, f64)>],
    opts: &DetectOptions,
) -> Vec<ChannelVerdict> {
    let opts = opts.normalized();
    let mut out = vec![ChannelVerdict::Unknown; regions.len()];
    if regions.is_empty() || !path.exists() {
        return out;
    }

    // 头部 + 指纹每文件探一次；读不出则本批全部照常解码判定，只是不读写缓存
    //（"指纹缺位不缓存"契约见函数文档）。
    let key_base: Option<(u16, u32, u64)> = match (
        crate::audio_utils::try_read_audio_header_only(path),
        crate::audio_utils::compute_file_fingerprint(path),
    ) {
        (Some(header), Some(fingerprint)) => Some((header.channels, header.sample_rate, fingerprint)),
        _ => None,
    };

    // 缓存命中的区间直接采用；未命中的收集起来走一次共享 I/O。
    let mut pending_indices: Vec<usize> = Vec::with_capacity(regions.len());
    let mut pending_regions: Vec<Option<(f64, f64)>> = Vec::with_capacity(regions.len());
    for (i, region) in regions.iter().enumerate() {
        let mut hit = None;
        if let Some((channels, sample_rate, fingerprint)) = key_base {
            let key = VerdictKey::new(path, Some(fingerprint), channels, sample_rate, *region, &opts);
            hit = verdict_cache_get(&key);
        }
        match hit {
            Some(verdict) => out[i] = verdict,
            None => {
                pending_indices.push(i);
                pending_regions.push(*region);
            }
        }
    }
    if pending_indices.is_empty() {
        return out;
    }

    let verdicts = if is_wav(path) {
        analyze_wav_regions(path, &pending_regions, &opts)
    } else {
        analyze_other_container_regions(path, &pending_regions, &opts)
    };

    for (k, &i) in pending_indices.iter().enumerate() {
        let verdict = verdicts
            .get(k)
            .copied()
            .unwrap_or(ChannelVerdict::Unknown);
        out[i] = verdict;
        // 只有确定性的结论才值得记忆：Unknown 可能只是文件暂时不可读
        //（被占用 / 正在写入），缓存它会让后续导入永久失去判定机会。
        if verdict != ChannelVerdict::Unknown {
            if let Some((channels, sample_rate, fingerprint)) = key_base {
                verdict_cache_put(
                    VerdictKey::new(path, Some(fingerprint), channels, sample_rate, pending_regions[k], &opts),
                    verdict,
                );
            }
        }
    }
    out
}

/// 为文件构造缓存键（头部探测或内容指纹读不出时返回 `None`，退化为不缓存）。
fn is_wav(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
}

/// WAV：按各 `region` 均匀取窗口，逐窗口 seek 后比较。
///
/// 批量入口（同一文件只打开一次）；单区间调用经由 [`verdict_for_regions`]。
fn analyze_wav_regions(
    path: &Path,
    regions: &[Option<(f64, f64)>],
    opts: &DetectOptions,
) -> Vec<ChannelVerdict> {
    use hound::WavReader;

    let mut out = vec![ChannelVerdict::Unknown; regions.len()];
    let mut reader = match WavReader::open(path) {
        Ok(r) => r,
        Err(_) => return out,
    };
    let spec = reader.spec();
    if spec.channels < 2 {
        return vec![ChannelVerdict::Mono; regions.len()];
    }

    let total_frames = reader.duration() as u64;
    if total_frames == 0 {
        return out;
    }
    let sample_rate = spec.sample_rate;

    for (i, region) in regions.iter().enumerate() {
        out[i] = analyze_wav_region(&mut reader, &spec, total_frames, sample_rate, *region, opts);
    }
    out
}

/// WAV 单区间判定（reader 由批量入口打开并复用；`seek` 是绝对定位，区间间共享安全）。
fn analyze_wav_region(
    reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>,
    spec: &hound::WavSpec,
    total_frames: u64,
    sample_rate: u32,
    region: Option<(f64, f64)>,
    opts: &DetectOptions,
) -> ChannelVerdict {
    use hound::SampleFormat;

    let (region_start, region_end) = resolve_region(region, total_frames, sample_rate);
    let region_frames = region_end.saturating_sub(region_start);
    if region_frames == 0 {
        return ChannelVerdict::Unknown;
    }

    let mut acc = DiffAcc::default();
    for (offset, len) in sample_windows(region_frames as usize, sample_rate, opts) {
        let start = region_start + offset as u64;
        let start = start.min(total_frames.saturating_sub(1));
        let len = len.min((total_frames - start) as usize);
        if len == 0 {
            continue;
        }
        if reader.seek(start as u32).is_err() {
            return ChannelVerdict::Unknown;
        }

        // 读入本窗口后交给统一的比较函数 —— 采样格式差异只影响"怎么读"，
        // 不影响"怎么比"。
        let mut scratch: Vec<f32> = Vec::with_capacity(len * 2);
        let read_ok = match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Int, 16) => {
                let scale = 1.0 / (i16::MAX as f32);
                read_window(reader, len, &mut scratch, |s: i16| s as f32 * scale)
            }
            (SampleFormat::Int, 24) => {
                // hound 把 24-bit 作为符号扩展的 i32 返回，按 i32::MAX 归一化会
                // 缩小约 256 倍（波形与音频近乎无声）。
                let scale = 1.0 / ((1u32 << 23) as f32);
                read_window(reader, len, &mut scratch, |s: i32| s as f32 * scale)
            }
            (SampleFormat::Int, 32) => {
                let scale = 1.0 / (i32::MAX as f32);
                read_window(reader, len, &mut scratch, |s: i32| s as f32 * scale)
            }
            (SampleFormat::Float, 32) => {
                read_window(reader, len, &mut scratch, |s: f32| s)
            }
            _ => return ChannelVerdict::Unknown,
        };
        if !read_ok {
            return ChannelVerdict::Unknown;
        }
        let frames_read = scratch.len() / 2;
        if !compare_frames(&scratch, 2, 0, frames_read, opts.tolerance, &mut acc) {
            return ChannelVerdict::TrueStereo;
        }
    }

    finalize(acc)
}

/// 非 WAV：从区间起点限量解码后比较（批量入口：同一文件只解码一次）。
///
/// 与 WAV 路径的差异（已知限制，出于成本考虑）：Symphonia 的逐包解码不保证
/// 窗口级随机访问，因此只解码"从文件头到各区间所需终点"的**连续前缀**并逐帧
/// 比较，而不是像 WAV 那样在区间内分散取窗口。
///
/// 【解码总量上界】批量路径取"各区间覆盖需求的 max"，且整体封顶到抽样预算
///（未设预算时按区间全量，但封顶 5 分钟，避免"不抽样"在大文件上退化成整轨
/// 解码）。区间起点本身超过预算 ⇒ 非随机访问容器无法在预算内到达消费区间，
/// 该区间不做结论（Unknown 不进缓存，后续扫描会再试）。
///
/// 【为什么按文件分组】同一源文件被多个 Take 以不同区间引用是常态（人声切片、
/// 多轨引用同一条伴奏）：逐 Take 独立解码会让一次整工程扫描的解码量随引用数
/// 线性放大；分组后解码量只与**文件数**成正比。
fn analyze_other_container_regions(
    path: &Path,
    regions: &[Option<(f64, f64)>],
    opts: &DetectOptions,
) -> Vec<ChannelVerdict> {
    let mut out = vec![ChannelVerdict::Unknown; regions.len()];
    let Some(header) = crate::audio_utils::try_read_audio_header_only(path) else {
        return out;
    };
    if header.channels < 2 {
        return vec![ChannelVerdict::Mono; regions.len()];
    }

    let sample_rate = if header.sample_rate == 0 {
        44_100
    } else {
        header.sample_rate
    };
    let total_frames = header.total_frames;
    if total_frames == 0 {
        return out;
    }

    const HARD_CAP_SEC: f64 = 300.0;
    let budget_sec = opts
        .scan_budget_sec()
        .unwrap_or(HARD_CAP_SEC)
        .min(HARD_CAP_SEC);
    let budget_frames = ((budget_sec * sample_rate as f64).round() as usize).max(1);

    // 逐区间规划覆盖需求；全部越界 / 空区间则无需解码。
    struct RegionPlan {
        start: u64,
        window: usize,
    }
    let mut plans: Vec<Option<RegionPlan>> = Vec::with_capacity(regions.len());
    let mut shared_decode_limit = 0usize;
    for region in regions {
        let (region_start, region_end) = resolve_region(*region, total_frames, sample_rate);
        let region_frames = region_end.saturating_sub(region_start);
        if region_frames == 0 || region_start as usize >= budget_frames {
            plans.push(None);
            continue;
        }
        let window = budget_frames.min(region_frames as usize);
        shared_decode_limit = shared_decode_limit
            .max((region_start as usize).saturating_add(window).min(budget_frames));
        plans.push(Some(RegionPlan {
            start: region_start,
            window,
        }));
    }
    if shared_decode_limit == 0 {
        return out;
    }

    let mut pcm: Vec<f32> = Vec::new();
    if crate::media::decode_media_audio_prefix_f32(path, None, shared_decode_limit)
        .map(|(_sr, _ch, data)| pcm = data)
        .is_err()
    {
        return out;
    }
    if pcm.is_empty() {
        return out;
    }

    let ch = header.channels as usize;
    let decoded_frames = pcm.len() / ch;
    for (i, plan) in plans.iter().enumerate() {
        let Some(plan) = plan else { continue };
        let Some((first, last)) =
            container_analysis_range(plan.start, plan.window, decoded_frames)
        else {
            // 解码没到达区间起点（文件比 header 声明的短）→ 该区间不做结论。
            continue;
        };
        // 交给统一比较入口（并在该窗口内再按 window_count 抽样）。
        out[i] = analyze_interleaved(
            &pcm[first * ch..last * ch],
            header.channels,
            sample_rate,
            opts,
        );
    }
    out
}

/// 从当前读位置读 `frames` 帧（每帧 2 声道）到 `out`；返回是否完整读完。
///
/// 短读（文件实际比 header 短）不算错误：已读到的部分仍会被比较。
fn read_window<R, S, F>(
    reader: &mut hound::WavReader<R>,
    frames: usize,
    out: &mut Vec<f32>,
    convert: F,
) -> bool
where
    R: std::io::Read + std::io::Seek,
    S: hound::Sample,
    F: Fn(S) -> f32,
{
    out.clear();
    out.reserve(frames * 2);
    let mut samples = reader.samples::<S>();
    for _ in 0..frames {
        let (Some(l), Some(r)) = (samples.next(), samples.next()) else {
            break;
        };
        let (Ok(l), Ok(r)) = (l, r) else {
            break;
        };
        out.push(convert(l));
        out.push(convert(r));
    }
    !out.is_empty()
}

/// 非 WAV 路径在**已解码缓冲**中应当比较的帧下标区间 `[first, last)`。
///
/// 缓冲是从文件起点开始解码的，所以消费区间起点 `region_start` 之前的帧
/// 属于无关音频，必须跳过 —— 否则区间起点非 0 时会拿文件头的音频下结论。
fn container_analysis_range(
    region_start: u64,
    window_frames: usize,
    decoded_frames: usize,
) -> Option<(usize, usize)> {
    let first = (region_start as usize).min(decoded_frames);
    let last = first.saturating_add(window_frames).min(decoded_frames);
    if last <= first {
        None
    } else {
        Some((first, last))
    }
}

/// 把源域秒区间解析为帧下标区间；越界部分钳制到 `[0, total_frames)`。
fn resolve_region(
    region: Option<(f64, f64)>,
    total_frames: u64,
    sample_rate: u32,
) -> (u64, u64) {
    let sr = sample_rate.max(1) as f64;
    let (start_sec, end_sec) = region.unwrap_or((0.0, total_frames as f64 / sr));
    let to_frame = |sec: f64| -> u64 {
        if sec.is_finite() && sec > 0.0 {
            (sec * sr).round().max(0.0) as u64
        } else {
            0
        }
    };
    let start = to_frame(start_sec).min(total_frames);
    let end = to_frame(end_sec).min(total_frames).max(start);
    (start, end)
}

fn finalize(acc: DiffAcc) -> ChannelVerdict {
    if acc.frames == 0 {
        return ChannelVerdict::Unknown;
    }
    // 走到这里说明所有被检查的样本都在容差内。
    ChannelVerdict::FakeStereo
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    fn interleave(pairs: &[[f32; 2]]) -> Vec<f32> {
        let mut v = Vec::with_capacity(pairs.len() * 2);
        for p in pairs {
            v.push(p[0]);
            v.push(p[1]);
        }
        v
    }

    fn opts() -> DetectOptions {
        DetectOptions::default()
    }

    #[test]
    fn mono_source_is_not_fake_stereo() {
        let pcm = vec![0.1f32, 0.2, 0.3];
        assert_eq!(
            analyze_interleaved(&pcm, 1, SR, &opts()),
            ChannelVerdict::Mono
        );
    }

    #[test]
    fn identical_planes_are_fake_stereo() {
        let pcm = interleave(&[[0.5, 0.5], [-0.25, -0.25], [0.0, 0.0]]);
        assert_eq!(
            analyze_interleaved(&pcm, 2, SR, &opts()),
            ChannelVerdict::FakeStereo
        );
    }

    #[test]
    fn differing_planes_are_true_stereo() {
        let pcm = interleave(&[[0.5, 0.5], [0.5, -0.5]]);
        assert_eq!(
            analyze_interleaved(&pcm, 2, SR, &opts()),
            ChannelVerdict::TrueStereo
        );
    }

    #[test]
    fn tolerance_absorbs_codec_noise() {
        // 有损编码后左右声道可能有 1e-5 级差异。
        let pcm = interleave(&[[0.5, 0.5 + 2e-6], [0.25, 0.25 - 2e-6]]);
        let loose = DetectOptions {
            tolerance: 1e-5,
            ..Default::default()
        };
        assert_eq!(
            analyze_interleaved(&pcm, 2, SR, &loose),
            ChannelVerdict::FakeStereo
        );

        // 同一份数据在更严容差下判为真立体声。
        let strict = DetectOptions {
            tolerance: 1e-7,
            ..Default::default()
        };
        assert_eq!(
            analyze_interleaved(&pcm, 2, SR, &strict),
            ChannelVerdict::TrueStereo
        );
    }

    #[test]
    fn empty_and_degenerate_inputs_are_unknown() {
        assert_eq!(
            analyze_interleaved(&[], 2, SR, &opts()),
            ChannelVerdict::Unknown
        );
        // 尾部残缺（不足一帧）不该 panic。
        assert_eq!(
            analyze_interleaved(&[0.5f32], 2, SR, &opts()),
            ChannelVerdict::Unknown
        );
    }

    #[test]
    fn sampling_checks_the_tail_not_only_the_head() {
        // 4 秒素材，前 3.9 秒完全一致、最后 0.1 秒分叉。
        // 采样必须覆盖尾部，否则会误判为假立体声。
        let frames = SR as usize * 4;
        let mut pcm = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let v = 0.3f32;
            pcm.push(v);
            pcm.push(if i > frames - SR as usize / 10 { -v } else { v });
        }
        assert_eq!(
            analyze_interleaved(&pcm, 2, SR, &opts()),
            ChannelVerdict::TrueStereo
        );
    }

    #[test]
    fn window_count_zero_scans_everything() {
        // 单窗口全量扫描时，中段分叉也必须被发现。
        let frames = SR as usize * 2;
        let mut pcm = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let v = 0.2f32;
            pcm.push(v);
            pcm.push(if i == frames / 2 { -v } else { v });
        }
        let full = DetectOptions {
            window_count: 0,
            ..Default::default()
        };
        assert_eq!(
            analyze_interleaved(&pcm, 2, SR, &full),
            ChannelVerdict::TrueStereo
        );
    }

    #[test]
    fn sample_windows_cover_head_and_tail() {
        let o = DetectOptions {
            window_sec: 0.1,
            window_count: 4,
            tolerance: 1e-6,
        };
        let wins = sample_windows(100_000, SR, &o);
        assert_eq!(wins.len(), 4);
        assert_eq!(wins.first().copied().unwrap().0, 0, "首个窗口必须在起点");
        let (last_start, last_len) = *wins.last().unwrap();
        assert_eq!(last_start + last_len, 100_000, "末个窗口必须右对齐到末尾");
    }

    #[test]
    fn sample_windows_degenerate_cases() {
        // 区间短于一个窗口 → 单窗口全覆盖。
        let o = DetectOptions {
            window_sec: 1.0,
            window_count: 12,
            tolerance: 1e-6,
        };
        assert_eq!(sample_windows(100, SR, &o), vec![(0, 100)]);
        assert!(sample_windows(0, SR, &o).is_empty());
    }

    #[test]
    fn verdict_cache_hit_and_miss() {
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let base = VerdictKey {
            path: "c:/a/x.wav".into(),
            fingerprint: Some(1234),
            channels: 2,
            sample_rate: SR,
            region_q: None,
            policy_sig: 0xABCD,
        };
        assert!(verdict_cache_get(&base).is_none());
        verdict_cache_put(base.clone(), ChannelVerdict::FakeStereo);
        assert_eq!(
            verdict_cache_get(&base),
            Some(ChannelVerdict::FakeStereo)
        );
    }

    #[test]
    fn verdict_cache_key_separates_fingerprint_policy_and_region() {
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let base = VerdictKey {
            path: "c:/a/x.wav".into(),
            fingerprint: Some(1234),
            channels: 2,
            sample_rate: SR,
            region_q: None,
            policy_sig: 1,
        };
        verdict_cache_put(base.clone(), ChannelVerdict::TrueStereo);

        // 内容指纹变了（文件被替换）→ 必须失效。
        let replaced = VerdictKey {
            fingerprint: Some(999),
            ..base.clone()
        };
        assert!(verdict_cache_get(&replaced).is_none());

        // 策略签名变了（用户改了容差）→ 必须失效。
        let repolicy = VerdictKey {
            policy_sig: 2,
            ..base.clone()
        };
        assert!(verdict_cache_get(&repolicy).is_none());

        // 区间变了 → 必须失效（判定只对给定区间负责）。
        let reregion = VerdictKey {
            region_q: Some((0, 1000)),
            ..base.clone()
        };
        assert!(verdict_cache_get(&reregion).is_none());

        // 原键仍命中。
        assert_eq!(verdict_cache_get(&base), Some(ChannelVerdict::TrueStereo));
    }

    #[test]
    fn policy_signature_tracks_parameters() {
        let a = DetectOptions::default();
        assert_eq!(a.signature(), a.signature());

        let b = DetectOptions {
            tolerance: 1e-4,
            ..Default::default()
        };
        assert_ne!(a.signature(), b.signature());

        let c = DetectOptions {
            window_count: 3,
            ..Default::default()
        };
        assert_ne!(a.signature(), c.signature());

        let d = DetectOptions {
            window_sec: 0.5,
            ..Default::default()
        };
        assert_ne!(a.signature(), d.signature());
    }

    #[test]
    fn normalized_clamps_out_of_range_values() {
        let wild = DetectOptions {
            window_sec: 999.0,
            window_count: 99_999,
            tolerance: 5.0,
        };
        let n = wild.normalized();
        assert!(n.window_sec <= 5.0 && n.window_sec >= 0.05);
        assert_eq!(n.window_count, 256);
        assert!(n.tolerance <= 1.0);

        let nan = DetectOptions {
            window_sec: f64::NAN,
            window_count: 12,
            tolerance: f32::NAN,
        };
        let n = nan.normalized();
        assert_eq!(n.window_sec, 0.25);
        assert_eq!(n.tolerance, 1e-6);
    }

    #[test]
    fn unknown_verdict_is_not_cached() {
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let missing = Path::new("C:/definitely/not/here.wav");
        assert_eq!(
            verdict_for_file(missing, None, &opts()),
            ChannelVerdict::Unknown
        );
        assert!(verdict_cache_get(&VerdictKey {
            path: normalize_path_key(missing),
            fingerprint: None,
            channels: 0,
            sample_rate: 0,
            region_q: None,
            policy_sig: opts().signature(),
        })
        .is_none());
    }

    #[test]
    fn quantize_region_is_millisecond_and_none_preserving() {
        assert_eq!(quantize_region(None), None);
        assert_eq!(
            quantize_region(Some((1.0004, 2.9996))),
            Some((1000, 3000))
        );
        // 非有限值不 panic。
        assert_eq!(
            quantize_region(Some((f64::NAN, f64::INFINITY))),
            Some((0, 0))
        );
    }

    #[test]
    fn container_range_skips_frames_before_the_region_start() {
        // 回归：非 WAV 路径从文件头解码，若从缓冲开头就比较，区间起点非 0
        // 时会拿与消费区间无关的音频下结论。
        let (first, last) = container_analysis_range(1000, 500, 10_000).unwrap();
        assert_eq!(first, 1000, "必须跳过区间起点之前的帧");
        assert_eq!(last, 1500);
    }

    #[test]
    fn container_range_clamps_to_decoded_length() {
        // 文件比 header 声明的短：解码没到区间起点 → 不做结论。
        assert_eq!(container_analysis_range(5000, 500, 1000), None);
        // 部分覆盖：截到已解码末尾。
        let (first, last) = container_analysis_range(800, 500, 1000).unwrap();
        assert_eq!((first, last), (800, 1000));
    }

    #[test]
    fn container_range_from_zero_covers_the_window() {
        let (first, last) = container_analysis_range(0, 300, 10_000).unwrap();
        assert_eq!((first, last), (0, 300));
        assert_eq!(container_analysis_range(0, 300, 0), None);
    }

    /// 写入一个临时 WAV（32f，指定声道内容）并返回路径。
    fn write_temp_wav(name: &str, sample_rate: u32, frames: &[[f32; 2]]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("hifishifter_stereo_detect_test");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(name);
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("create wav");
        for f in frames {
            w.write_sample(f[0]).expect("write l");
            w.write_sample(f[1]).expect("write r");
        }
        w.finalize().expect("finalize wav");
        path
    }

    #[test]
    fn wav_file_fake_stereo_is_detected_via_seek_windows() {
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        // 2 秒，远超 12 × 0.25s 的抽样预算 → 走 seek 窗口路径。
        let frames: Vec<[f32; 2]> = (0..SR as usize * 2)
            .map(|i| {
                let v = ((i as f32) * 0.001).sin() * 0.5;
                [v, v]
            })
            .collect();
        let path = write_temp_wav("fake_stereo.wav", SR, &frames);

        assert_eq!(
            verdict_for_file(&path, None, &opts()),
            ChannelVerdict::FakeStereo
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wav_file_true_stereo_is_detected() {
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let frames: Vec<[f32; 2]> = (0..SR as usize * 2)
            .map(|i| {
                let v = ((i as f32) * 0.001).sin() * 0.5;
                [v, -v]
            })
            .collect();
        let path = write_temp_wav("true_stereo.wav", SR, &frames);

        assert_eq!(
            verdict_for_file(&path, None, &opts()),
            ChannelVerdict::TrueStereo
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wav_verdict_is_memoized_and_region_scoped() {
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        // 前 1 秒一致、后 1 秒分叉。
        let half = SR as usize;
        let frames: Vec<[f32; 2]> = (0..half * 2)
            .map(|i| {
                let v = 0.4f32;
                if i < half {
                    [v, v]
                } else {
                    [v, -v]
                }
            })
            .collect();
        let path = write_temp_wav("region_scoped.wav", SR, &frames);

        // 只看前 1 秒 → 假立体声。
        assert_eq!(
            verdict_for_file(&path, Some((0.0, 1.0)), &opts()),
            ChannelVerdict::FakeStereo
        );
        // 整个文件 → 真立体声（尾部窗口覆盖到分叉段）。
        assert_eq!(
            verdict_for_file(&path, None, &opts()),
            ChannelVerdict::TrueStereo
        );

        // 记忆生效：同一区间重复查询仍返回首次结论（此处两次调用都命中缓存，
        // 断言的是"结果稳定"而非"重新解码"——后者由 verdict_cache 单测覆盖）。
        assert_eq!(
            verdict_for_file(&path, Some((0.0, 1.0)), &opts()),
            ChannelVerdict::FakeStereo
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn resolve_region_clamps_and_handles_none() {
        // None → 整文件。
        assert_eq!(resolve_region(None, 44_100, SR), (0, 44_100));
        // 越界钳制。
        assert_eq!(resolve_region(Some((-5.0, 1e9)), 44_100, SR), (0, 44_100));
        // 正常区间。
        assert_eq!(resolve_region(Some((0.5, 1.0)), 44_100, SR), (22_050, 44_100));
        // 反向区间 → 空区间（起点与终点相同），不 panic。
        let (s, e) = resolve_region(Some((2.0, 1.0)), 44_100, SR);
        assert_eq!(s, e);
    }
}
