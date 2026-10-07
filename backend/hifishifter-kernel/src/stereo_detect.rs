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
/// 判定所需的数值，不含模式（模式属于调用方的决策）。
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
    /// 非 WAV 容器的**解码总量预算**（秒）。
    ///
    /// WAV 可以按窗口 seek，代价与窗口数成正比。容器则分两条取数路径，本预算在
    /// 两条上都是"我们愿意为此付出多少解码"的上界，但**衡量对象不同**：
    ///
    /// - **seek 采样**（首选，见 [`analyze_container_regions_by_seek`]）：代价 ≈
    ///   窗口数 × (窗口长 + 预热)，与文件多长无关。预算限制的是**总解码量**，
    ///   因此半小时素材也能在预算内覆盖全长；
    /// - **顺序单遍解码**（回落）：Symphonia 的逐包解码不保证随机访问，要看到
    ///   "第 N 秒"就必须从文件头解到第 N 秒。此时预算限制的是**可达位置**，
    ///   超出它的窗口收不到。
    ///
    /// 无论哪条路径，覆盖不完整都只会落到 `Unknown`（不缓存、不权威），留给
    /// 下一次更大预算的扫描。
    ///
    /// **必须计入 [`Self::signature`]**：预算不同 ⇒ 能覆盖的区间不同 ⇒ 结论
    /// 不可互相顶替。这也让"导入时的短预算结论"天然不会顶掉"扫描时的长预算
    /// 结论"—— 两者签名不同，缓存与判定档案都不会误命中。
    pub container_budget_sec: f64,
}

/// 容器解码预算的默认值（后台扫描 / 整工程迁移用）。
///
/// 覆盖到 30 分钟：人声/伴奏素材极少更长。seek 采样下 12 个默认窗口只需几秒
/// 音频的解码量（与文件多长无关），即便回落到顺序路径，30 分钟音频的解码也
/// 只在后台线程上进行。
pub const DEFAULT_CONTAINER_BUDGET_SEC: f64 = 1800.0;

/// 容器解码预算：导入等**用户正在等待**的路径用。
///
/// 导入是命令线程上的同步路径，不能为一个"锦上添花"的折叠把用户卡住。
/// 该值同时足够让 seek 采样覆盖全长（默认 12 窗口仅需数秒解码量），因此长素材
/// 通常也能当场判定；只有配置了极多窗口、或容器不支持 seek 时才会返回"待定"，
/// 由导入后触发的后台扫描带完整预算补判 —— 用户看到的结果不变，只是晚一点。
pub const IMPORT_CONTAINER_BUDGET_SEC: f64 = 30.0;

/// 判定容差的默认值：满幅的 1%（≈ -40 dBFS）。
///
/// 判定问的是"折叠会不会改变听感"，而不是"两个声道是否逐比特相同"。有损编码
/// ——尤其是 mp3 joint stereo 的 M/S 量化残留——解码后左右声道本就带着微小差异，
/// 量级常在 0.1%~1%。容差取得过严会把大量"内容其实一致"的素材判成真立体声，
/// 表现为"该折叠的没折叠"。1% 是覆盖这类残留的保守值；真立体声的差异比它大
/// 好几个数量级，不会因此被误折叠。
pub const DEFAULT_TOLERANCE: f32 = 0.01;

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            window_sec: 0.25,
            window_count: 12,
            tolerance: DEFAULT_TOLERANCE,
            container_budget_sec: DEFAULT_CONTAINER_BUDGET_SEC,
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
                // 非有限值（NaN/Inf）回退到默认容差 —— 与策略层的回退保持同一
                // 口径，避免"同一个非法输入在两处得到不同容差"。
                DEFAULT_TOLERANCE
            },
            container_budget_sec: if self.container_budget_sec.is_finite() {
                self.container_budget_sec
                    .clamp(0.0, DEFAULT_CONTAINER_BUDGET_SEC)
            } else {
                DEFAULT_CONTAINER_BUDGET_SEC
            },
        }
    }

    /// 换一个容器解码预算（导入路径用它换取同步判定的低延迟）。
    pub fn with_container_budget_sec(mut self, sec: f64) -> Self {
        self.container_budget_sec = sec;
        self.normalized()
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
        mix(&n.container_budget_sec.to_bits().to_le_bytes());
        h
    }
}

// ─── 纯判定核心 ──────────────────────────────────────────────────────────────

/// 逐帧差累计器。
///
/// 判定规则是**逐帧绝对差 ≤ 容差**（任一帧超差即否决，安全侧：宁可不优化，
/// 也不误折叠真立体声）。除了帧数，还累计超差帧数与最大绝对差 —— 它们构成
/// [`VerdictDetail`] 的诊断证据，是用户决定要不要放宽容差的唯一依据。
#[derive(Debug, Default, Clone, Copy)]
struct DiffAcc {
    frames: u64,
    violating: u64,
    max_abs_diff: f32,
}

impl DiffAcc {
    /// 累计一帧；返回该帧是否仍在容差内（`false` = 已确定不是假立体声）。
    #[inline]
    fn push(&mut self, l: f32, r: f32, tolerance: f32) {
        self.frames += 1;
        let diff = (l - r).abs();
        if diff > tolerance {
            self.violating += 1;
        }
        if diff > self.max_abs_diff {
            self.max_abs_diff = diff;
        }
    }
}

/// 一次判定的完整结果：结论 + 支撑它的证据。
///
/// 证据（超差样本数 / 最大绝对差）在 `TrueStereo` 时是用户调容差的唯一依据 ——
/// 「差多少、差多少个样本」远比一句「不是假立体声」有用。诊断信息随结论一起
/// 进判定缓存，因此命中缓存时同样可得。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerdictDetail {
    pub verdict: ChannelVerdict,
    /// 实际参与比较的帧数。
    pub frames_compared: u64,
    /// 超差帧数（`|L-R| > tolerance`）。
    pub violating_frames: u64,
    /// 观测到的最大绝对差。
    pub max_abs_diff: f32,
}

impl VerdictDetail {
    /// 非判定结论（单声道 / 无法判定）的零证据占位。
    pub fn bare(verdict: ChannelVerdict) -> Self {
        Self {
            verdict,
            frames_compared: 0,
            violating_frames: 0,
            max_abs_diff: 0.0,
        }
    }

    /// 超差样本占比（`0.0..=1.0`）；无样本时为 0。
    pub fn violating_ratio(self) -> f64 {
        if self.frames_compared == 0 {
            0.0
        } else {
            self.violating_frames as f64 / self.frames_compared as f64
        }
    }
}

/// 判定交错 PCM 的 L/R 是否在容差内一致。
///
/// `pcm` 应是**实际消费区间**的交错采样（而非整个文件）：判定结果只对该区间
/// 负责，这与渲染/听感实际使用的区间一致。
///
/// 窗口沿区间均匀铺开且**首尾都落在区间内**，因此"开头一致、结尾分叉"的素材
/// 不会漏判。
///
/// 不短路：所有抽样窗口都会被扫完。判定要的是"差了多少个样本、差到什么量级"
/// 这种可展示的证据，而不是一个布尔值；抽样总量（默认 3 秒音频）下的比较开销
/// 与解码相比可以忽略，而短路会让诊断信息只剩"第一个超差样本"。
///
/// 【为什么是 test-only】生产的两条路径都需要**跨多个解码缓冲累计证据**
///（WAV 逐窗口 seek、容器流式收割），因此它们直接调用 [`compare_frames`] +
/// [`finalize`]。本函数是这两步在**单个缓冲**上的组合，供测试断言"整段判定"
/// 的语义使用 —— 保留它而不是在测试里各写一份，是为了让被测组合与生产一致。
#[cfg(test)]
pub fn analyze_interleaved(
    pcm: &[f32],
    channels: u16,
    sample_rate: u32,
    opts: &DetectOptions,
) -> ChannelVerdict {
    analyze_interleaved_detailed(pcm, channels, sample_rate, opts).verdict
}

/// [`analyze_interleaved`] 的带证据版本（同样 test-only）。
#[cfg(test)]
pub fn analyze_interleaved_detailed(
    pcm: &[f32],
    channels: u16,
    sample_rate: u32,
    opts: &DetectOptions,
) -> VerdictDetail {
    if channels < 2 {
        return VerdictDetail::bare(ChannelVerdict::Mono);
    }
    let ch = channels as usize;
    let frames = pcm.len() / ch;
    if frames == 0 {
        return VerdictDetail::bare(ChannelVerdict::Unknown);
    }

    let opts = opts.normalized();
    let mut acc = DiffAcc::default();

    for (start, len) in sample_windows(frames, sample_rate, &opts) {
        let last = (start + len).min(frames);
        compare_frames(pcm, ch, start, last, opts.tolerance, &mut acc);
    }
    finalize(acc)
}

/// 比较 `[first, last)` 帧的 L/R，把证据累计进 `acc`。
fn compare_frames(
    pcm: &[f32],
    channels: usize,
    first: usize,
    last: usize,
    tolerance: f32,
    acc: &mut DiffAcc,
) {
    for f in first..last {
        let base = f * channels;
        if base + 1 >= pcm.len() {
            break;
        }
        acc.push(pcm[base], pcm[base + 1], tolerance);
    }
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

    let sr = if sample_rate == 0 {
        44_100
    } else {
        sample_rate
    } as f64;
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

fn verdict_cache() -> &'static RwLock<HashMap<VerdictKey, VerdictDetail>> {
    static CACHE: OnceLock<RwLock<HashMap<VerdictKey, VerdictDetail>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn verdict_cache_get(key: &VerdictKey) -> Option<VerdictDetail> {
    verdict_cache()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(key)
        .copied()
}

pub fn verdict_cache_put(key: VerdictKey, detail: VerdictDetail) {
    let mut cache = verdict_cache().write().unwrap_or_else(|e| e.into_inner());
    if cache.len() >= VERDICT_CACHE_CAPACITY && !cache.contains_key(&key) {
        cache.clear();
    }
    cache.insert(key, detail);
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
    verdict_for_file_detailed(path, region, opts).verdict
}

/// [`verdict_for_file`] 的带证据版本。
pub fn verdict_for_file_detailed(
    path: &Path,
    region: Option<(f64, f64)>,
    opts: &DetectOptions,
) -> VerdictDetail {
    verdict_for_regions_detailed(path, &[region], opts)
        .into_iter()
        .next()
        .unwrap_or(VerdictDetail::bare(ChannelVerdict::Unknown))
}

/// 同一文件的**多个**消费区间批量判定（扫描 / 迁移热路径的 I/O 去重入口）。
///
/// 判定语义与逐区间调用 [`verdict_for_file`] 完全一致：同样的窗口抽样、容差
/// 与预算上限，结果照样进出进程级判定缓存（命中条目直接采用）。差别只在
/// **I/O 组织**：
/// - WAV：打开一次，逐区间 seek（原来每区间各开一次文件）；
/// - 非 WAV：Symphonia 顺序解码只做**一次**，沿途按窗口收割（原来只能看
///   文件头 3 秒，见 [`analyze_other_container_regions`]）；
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
    verdict_for_regions_detailed(path, regions, opts)
        .into_iter()
        .map(|detail| detail.verdict)
        .collect()
}

/// [`verdict_for_regions`] 的带证据版本（诊断信息随结论一起进缓存）。
pub fn verdict_for_regions_detailed(
    path: &Path,
    regions: &[Option<(f64, f64)>],
    opts: &DetectOptions,
) -> Vec<VerdictDetail> {
    let opts = opts.normalized();
    let mut out = vec![VerdictDetail::bare(ChannelVerdict::Unknown); regions.len()];
    if regions.is_empty() || !path.exists() {
        return out;
    }

    // 头部 + 指纹每文件探一次；读不出则本批全部照常解码判定，只是不读写缓存
    //（"指纹缺位不缓存"契约见函数文档）。
    let key_base: Option<(u16, u32, u64)> = match (
        crate::audio_utils::try_read_audio_header_only(path),
        crate::audio_utils::compute_file_fingerprint(path),
    ) {
        (Some(header), Some(fingerprint)) => {
            Some((header.channels, header.sample_rate, fingerprint))
        }
        _ => None,
    };

    // 缓存命中的区间直接采用；未命中的收集起来走一次共享 I/O。
    let mut pending_indices: Vec<usize> = Vec::with_capacity(regions.len());
    let mut pending_regions: Vec<Option<(f64, f64)>> = Vec::with_capacity(regions.len());
    for (i, region) in regions.iter().enumerate() {
        let mut hit = None;
        if let Some((channels, sample_rate, fingerprint)) = key_base {
            let key = VerdictKey::new(
                path,
                Some(fingerprint),
                channels,
                sample_rate,
                *region,
                &opts,
            );
            hit = verdict_cache_get(&key);
        }
        match hit {
            Some(detail) => out[i] = detail,
            None => {
                pending_indices.push(i);
                pending_regions.push(*region);
            }
        }
    }
    if pending_indices.is_empty() {
        return out;
    }

    // WAV 走 hound 快路径（可 seek，代价只与窗口数成正比）；hound 读不了的
    // WAV（8-bit / 64-bit float / 扩展名骗人）返回 `None`，改走 Symphonia。
    let details = if is_wav(path) {
        match analyze_wav_regions(path, &pending_regions, &opts) {
            Some(details) => details,
            None => analyze_other_container_regions(path, &pending_regions, &opts),
        }
    } else {
        analyze_other_container_regions(path, &pending_regions, &opts)
    };

    for (k, &i) in pending_indices.iter().enumerate() {
        let detail = details
            .get(k)
            .copied()
            .unwrap_or(VerdictDetail::bare(ChannelVerdict::Unknown));
        out[i] = detail;
        // 只有确定性的结论才值得记忆：Unknown 可能只是文件暂时不可读
        //（被占用 / 正在写入），缓存它会让后续导入永久失去判定机会。
        if detail.verdict != ChannelVerdict::Unknown {
            if let Some((channels, sample_rate, fingerprint)) = key_base {
                verdict_cache_put(
                    VerdictKey::new(
                        path,
                        Some(fingerprint),
                        channels,
                        sample_rate,
                        pending_regions[k],
                        &opts,
                    ),
                    detail,
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
///
/// 返回 `None` 表示**这个 WAV 不是 hound 能读的采样格式**（8-bit int、
/// 64-bit float、或扩展名是 .wav 而内容是别的容器）。调用方据此改走 Symphonia
/// 路径 —— 若在这里直接返回 `Unknown`，这些素材会被静默判成"读不了"并永远
/// 重试，而 Symphonia 其实完全能读它们。
fn analyze_wav_regions(
    path: &Path,
    regions: &[Option<(f64, f64)>],
    opts: &DetectOptions,
) -> Option<Vec<VerdictDetail>> {
    use hound::SampleFormat;
    use hound::WavReader;

    let mut reader = WavReader::open(path).ok()?;
    let spec = reader.spec();
    // hound 只为 i16 / i32 / f32 实现采样读取；其余位深交给 Symphonia。
    let readable = matches!(
        (spec.sample_format, spec.bits_per_sample),
        (SampleFormat::Int, 16)
            | (SampleFormat::Int, 24)
            | (SampleFormat::Int, 32)
            | (SampleFormat::Float, 32)
    );
    if !readable {
        return None;
    }
    if spec.channels < 2 {
        return Some(vec![
            VerdictDetail::bare(ChannelVerdict::Mono);
            regions.len()
        ]);
    }

    let total_frames = reader.duration() as u64;
    if total_frames == 0 {
        return Some(vec![
            VerdictDetail::bare(ChannelVerdict::Unknown);
            regions.len()
        ]);
    }
    let sample_rate = spec.sample_rate;

    let mut out = vec![VerdictDetail::bare(ChannelVerdict::Unknown); regions.len()];
    for (i, region) in regions.iter().enumerate() {
        out[i] = analyze_wav_region(&mut reader, &spec, total_frames, sample_rate, *region, opts);
    }
    Some(out)
}

/// WAV 单区间判定（reader 由批量入口打开并复用；`seek` 是绝对定位，区间间共享安全）。
///
/// 【读失败的方向性】中途 seek/读取失败时**已比过的窗口不作废**，但覆盖不完整
/// 只允许下发 `TrueStereo`（见 [`degrade_incomplete`]）：折叠是不可逆的听感
/// 损失，"没看全"时的保守方向永远是"不折叠"。
fn analyze_wav_region(
    reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>,
    spec: &hound::WavSpec,
    total_frames: u64,
    sample_rate: u32,
    region: Option<(f64, f64)>,
    opts: &DetectOptions,
) -> VerdictDetail {
    use hound::SampleFormat;

    let (region_start, region_end) = resolve_region(region, total_frames, sample_rate);
    let region_frames = region_end.saturating_sub(region_start);
    if region_frames == 0 {
        return VerdictDetail::bare(ChannelVerdict::Unknown);
    }

    let mut acc = DiffAcc::default();
    let mut complete = true;
    for (offset, len) in sample_windows(region_frames as usize, sample_rate, opts) {
        let start = region_start + offset as u64;
        let start = start.min(total_frames.saturating_sub(1));
        let len = len.min((total_frames - start) as usize);
        if len == 0 {
            continue;
        }
        if reader.seek(start as u32).is_err() {
            complete = false;
            break;
        }

        // 读入本窗口后交给统一的比较函数 —— 采样格式差异只影响"怎么读"，
        // 不影响"怎么比"。
        let mut scratch: Vec<f32> = Vec::with_capacity(len * 2);
        let read_ok = match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Int, 16) => {
                let scale = 1.0 / (i16::MAX as f32);
                read_window(reader, len, spec.channels, &mut scratch, |s: i16| {
                    s as f32 * scale
                })
            }
            (SampleFormat::Int, 24) => {
                // hound 把 24-bit 作为符号扩展的 i32 返回，按 i32::MAX 归一化会
                // 缩小约 256 倍（波形与音频近乎无声）。
                let scale = 1.0 / ((1u32 << 23) as f32);
                read_window(reader, len, spec.channels, &mut scratch, |s: i32| {
                    s as f32 * scale
                })
            }
            (SampleFormat::Int, 32) => {
                let scale = 1.0 / (i32::MAX as f32);
                read_window(reader, len, spec.channels, &mut scratch, |s: i32| {
                    s as f32 * scale
                })
            }
            (SampleFormat::Float, 32) => {
                read_window(reader, len, spec.channels, &mut scratch, |s: f32| s)
            }
            _ => {
                complete = false;
                break;
            }
        };
        if !read_ok {
            complete = false;
            break;
        }
        let frames_read = scratch.len() / 2;
        compare_frames(&scratch, 2, 0, frames_read, opts.tolerance, &mut acc);
    }

    let detail = finalize(acc);
    if complete {
        detail
    } else {
        degrade_incomplete(detail)
    }
}

/// 非 WAV 容器：**两级取数，同一套判定**。
///
/// # 为什么不是"前缀解码"
///
/// 旧实现只解码"从文件头到各区间需求"的连续前缀，且上界被 `window_sec ×
/// window_count`（默认 3 秒）封死。两条硬伤，正是"该判为假立体声却没判"的主因：
///
/// 1. 区间起点晚于 3 秒 ⇒ 直接不做结论。任何从中段切出来的 Take（trim / 拆分 /
///    跨 Clip 粘贴 —— 人声切片的常态）在非 WAV 源上**永远判不出来**；
/// 2. 起点早于 3 秒也只看到头 3 秒。而"前几秒有立体声 intro、主体是单声道"
///    恰恰是假立体声最典型的形态 —— 只看开头必然判成真立体声。
///
/// # 现在怎么做（两级）
///
/// 两级都从**同一份窗口规划**出发（每个区间用与 WAV 路径相同的 [`sample_windows`]，
/// 首尾都锚定在区间内），区别只在"怎么把窗口覆盖的音频取回来"：
///
/// 1. **seek 采样**（[`analyze_container_regions_by_seek`]，首选）：逐个窗口 seek
///    到附近再短解码。代价与**窗口数**成正比、与文件长度无关，因此长素材也能
///    覆盖全长。落点错位或容器不支持 seek 时整条路径作废；
/// 2. **顺序单遍解码**（回落）：解到最后一个窗口的终点，沿途收割，其余帧解完即弃。
///    代价与**文件长度**成正比，受 [`DetectOptions::container_budget_sec`] 限制。
///
/// 两条路径的收获结果都交给同一个 [`aggregate_region_windows`]，所以"怎么取数"
/// 不会渗进"怎么判"。
///
/// # 覆盖不足时的方向性
///
/// 预算不够（顺序路径）或文件比 header 声明的短 ⇒ 有窗口收不到 ⇒ 覆盖不完整 ⇒
/// 只允许下发 `TrueStereo`（见 [`degrade_incomplete`]）。折叠是不可逆的听感损失，
/// "没看全"时永远不折叠。
///
/// 【为什么按文件分组】同一源文件被多个 Take 以不同区间引用是常态（人声切片、
/// 多轨引用同一条伴奏）：逐 Take 独立解码会让一次整工程扫描的解码量随引用数
/// 线性放大；分组后解码量只与**文件数**成正比。
/// 一次容器判定的窗口规划：区间 → 窗口下标 → 绝对帧位窗口。
struct ContainerWindowPlan {
    /// 全部窗口（按区间顺序拼接，绝对帧位）。
    windows: Vec<(u64, usize)>,
    /// 第 i 个区间用到哪些窗口（`windows` 的下标）。
    region_windows: Vec<Vec<usize>>,
    /// 第 i 个区间是否有窗口被丢弃。
    region_truncated: Vec<bool>,
}

/// 按区间规划窗口；`budget_frames` 是**可达上界**（不是总预算）。
///
/// 顺序路径拿它当"值得解到多远"，因此传真实的解码预算；seek 路径不受位置限制，
/// 传 `u64::MAX` 即可（真正的总预算由调用方另行核算）。
fn plan_container_regions(
    regions: &[Option<(f64, f64)>],
    total_frames: u64,
    budget_frames: u64,
    sample_rate: u32,
    opts: &DetectOptions,
) -> ContainerWindowPlan {
    // 【截断必须逐区间记账】`was_truncated` 只描述**该区间自己**有没有被削掉
    // 窗口。绝不能汇总成一个文件级标志再回流到所有区间 —— 那会让"文件里存在
    // 某个读不到的区间"把**同文件的其他区间全部降级**，表现为同一文件里有的
    // Take 折叠了、有的没有（判定结果取决于它和谁被放在同一批里分析）。
    let mut windows: Vec<(u64, usize)> = Vec::new();
    let mut region_windows: Vec<Vec<usize>> = Vec::with_capacity(regions.len());
    let mut region_truncated: Vec<bool> = Vec::with_capacity(regions.len());
    for region in regions {
        let (region_start, region_end) = resolve_region(*region, total_frames, sample_rate);
        let (planned, was_truncated) = plan_region_windows(
            region_start,
            region_end,
            total_frames,
            budget_frames,
            sample_rate,
            opts,
        );
        let mut indices = Vec::with_capacity(planned.len());
        for window in planned {
            indices.push(windows.len());
            windows.push(window);
        }
        region_windows.push(indices);
        region_truncated.push(was_truncated);
    }
    ContainerWindowPlan {
        windows,
        region_windows,
        region_truncated,
    }
}

/// 逐区间汇总：把自己的窗口拼起来比较，覆盖不完整时按 [`degrade_incomplete`]
/// 收敛。
///
/// 顺序路径与 seek 路径共用本函数，保证"怎么取数"的差异不会渗进"怎么判"，
/// 两条路径对同一份收获结果必然给出同一结论。
fn aggregate_region_windows(
    plan: &ContainerWindowPlan,
    harvested: &[Option<(Vec<f32>, u16)>],
    region_count: usize,
    tolerance: f32,
) -> Vec<VerdictDetail> {
    let mut out = vec![VerdictDetail::bare(ChannelVerdict::Unknown); region_count];
    for (region_index, indices) in plan.region_windows.iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        let mut acc = DiffAcc::default();
        let mut complete = true;
        for &window_index in indices {
            match harvested.get(window_index).and_then(|slot| slot.as_ref()) {
                Some((pcm, channels)) => {
                    let ch = (*channels).max(1) as usize;
                    let frames = pcm.len() / ch;
                    compare_frames(pcm, ch, 0, frames, tolerance, &mut acc);
                }
                // 没收到 ⇒ 解码没走到 / 中途失败 ⇒ 这一段没看。
                None => complete = false,
            }
        }
        let detail = finalize(acc);
        // 逐区间收敛：只有**本区间**覆盖不完整时才降级。同文件其它区间的截断
        // 与本区间无关 —— 判定结果不能取决于它和谁被放在同一批里分析。
        out[region_index] = if complete && !plan.region_truncated[region_index] {
            detail
        } else {
            degrade_incomplete(detail)
        };
    }
    out
}

/// seek 采样的预热余量（秒）。
///
/// 有损编解码器 seek 之后要解若干帧才能输出正确样本（mp3 的比特池、AAC 的
/// overlap 都要求如此）。留一段余量让"落在目标之前在别处"的窗口不被误判为
/// 覆盖完整。
const SEEK_PRIMING_SEC: f64 = 0.30;

/// 落点是否可接受：允许偏差最多一个窗口长（请求起点 + `align_tolerance`）。
/// 独立成纯函数以便测试钉住边界语义（恰好一个窗口长 ⇒ 接受；再多 ⇒ 作废）。
fn seek_landing_within_tolerance(landed: u64, start: u64, align_tolerance: u64) -> bool {
    landed <= start.saturating_add(align_tolerance)
}

/// 优先用 seek 采样判定；不适用时返回 `None` 交由调用方回落到顺序路径。
///
/// # 为什么 seek 能同时更快、更准
///
/// 顺序路径必须解到"最后一个窗口的终点"，所以代价与**文件长度**成正比，且
/// 预算一紧就只能丢窗口（丢窗口 ⇒ 覆盖不完整 ⇒ 只敢下发 `TrueStereo`，
/// 长素材于是永远折叠不了）。seek 让每个窗口的代价固定在"窗口长 + 预热"，
/// 与文件多长无关 —— 于是半小时素材的**总解码量**降到几秒音频，
/// `container_budget_sec` 也就从"能看多远"回归成它该有的语义："**愿意付多少
/// 总解码量**"。长素材因此第一次能拿到覆盖完整的结论。
///
/// # 全有或全无的正确性闸门
///
/// 只要有一个窗口没拿到，或落点比请求起点晚了超过一个窗口长，整条路径作废、
/// 回落顺序路径。这不是保守过度：若某个容器的 seek 静默返回文件头，十几个窗口
/// 会全部采到同一段音频，那正是本模块最要防的事（拿错位置的音频冒充目标位置，
/// 从而把真立体声误判成可折叠）。宁可退回慢但确定的路径。
fn analyze_container_regions_by_seek(
    path: &Path,
    regions: &[Option<(f64, f64)>],
    opts: &DetectOptions,
    total_frames: u64,
    sample_rate: u32,
    budget_frames: u64,
) -> Option<Vec<VerdictDetail>> {
    // 全可达规划：窗口不再受"位置"限制（这正是 seek 的全部意义）。
    let plan = plan_container_regions(regions, total_frames, u64::MAX, sample_rate, opts);
    if plan.windows.is_empty() {
        return None;
    }

    let window_frames = ((opts.window_sec * sample_rate as f64).round() as u64).max(1);
    let priming_frames = ((SEEK_PRIMING_SEC * sample_rate as f64).round() as u64).max(1);

    // 代价核算：每个窗口 ≈ 窗口长 + 预热。预算不够就不启动这条路径 —— 否则
    // "预算"就形同虚设，一个 window_count 拉满的配置能把单次判定变成几百次 seek。
    let per_window = window_frames.saturating_add(priming_frames);
    if per_window.saturating_mul(plan.windows.len() as u64) > budget_frames {
        return None;
    }

    // 落点容差 = 一个窗口长：Coarse seek 允许落在附近，采到邻居位置对"左右是否
    // 一致"的统计判定无实质影响；再远就是在采别处的音频了。
    let align_tolerance = window_frames;
    let decode_cap = (align_tolerance + window_frames + priming_frames * 2) as usize;

    let mut harvested: Vec<Option<(Vec<f32>, u16)>> = vec![None; plan.windows.len()];
    let starts = crate::media::visit_media_audio_windows_by_seek(
        path,
        None,
        &plan.windows,
        sample_rate,
        decode_cap,
        &mut |index, pcm, channels, _rate| {
            if let Some(slot) = harvested.get_mut(index) {
                *slot = Some((pcm.to_vec(), channels));
            }
            Ok(())
        },
    )
    .ok()?;

    for (index, (start, _len)) in plan.windows.iter().enumerate() {
        let landed = starts.get(index).copied().flatten()?;
        if !seek_landing_within_tolerance(landed, *start, align_tolerance) {
            return None;
        }
    }

    Some(aggregate_region_windows(
        &plan,
        &harvested,
        regions.len(),
        opts.tolerance,
    ))
}

fn analyze_other_container_regions(
    path: &Path,
    regions: &[Option<(f64, f64)>],
    opts: &DetectOptions,
) -> Vec<VerdictDetail> {
    let unknown = || vec![VerdictDetail::bare(ChannelVerdict::Unknown); regions.len()];
    let Some(header) = crate::audio_utils::try_read_audio_header_only(path) else {
        return unknown();
    };
    if header.channels < 2 {
        return vec![VerdictDetail::bare(ChannelVerdict::Mono); regions.len()];
    }

    let sample_rate = if header.sample_rate == 0 {
        44_100
    } else {
        header.sample_rate
    };
    // total_frames 缺失（部分容器探测不出精确帧数）时用时长估算，而不是整批
    // 放弃 —— 判定的区间解析只需要一个足够准的总长。
    let total_frames = if header.total_frames > 0 {
        header.total_frames
    } else if header.duration_sec.is_finite() && header.duration_sec > 0.0 {
        (header.duration_sec * sample_rate as f64).round() as u64
    } else {
        0
    };
    if total_frames == 0 {
        return unknown();
    }

    let budget_frames = ((opts.container_budget_sec * sample_rate as f64).round() as u64).max(1);

    // ① 先试 seek 采样：代价与文件长度无关，且能覆盖全长。不适用时返回 None。
    if let Some(details) = analyze_container_regions_by_seek(
        path,
        regions,
        opts,
        total_frames,
        sample_rate,
        budget_frames,
    ) {
        return details;
    }

    // ② 回落：单遍顺序解码，沿途收割（代价与文件长度成正比，受预算限制）。
    let plan = plan_container_regions(regions, total_frames, budget_frames, sample_rate, opts);
    if plan.windows.is_empty() {
        return unknown();
    }

    // 解码上界 = 最后一个窗口的终点（窗口已按区间顺序生成，取 max 稳妥）。
    let decode_limit = plan
        .windows
        .iter()
        .map(|(start, len)| start.saturating_add(*len as u64))
        .max()
        .unwrap_or(0)
        .min(budget_frames) as usize;

    let mut harvested: Vec<Option<(Vec<f32>, u16)>> = vec![None; plan.windows.len()];
    let _ = crate::media::visit_media_audio_windows(
        path,
        None,
        &plan.windows,
        decode_limit,
        &mut |index, pcm, channels, _sample_rate| {
            if let Some(slot) = harvested.get_mut(index) {
                *slot = Some((pcm.to_vec(), channels));
            }
            Ok(())
        },
    );

    aggregate_region_windows(&plan, &harvested, regions.len(), opts.tolerance)
}

/// 覆盖不完整时的结论收敛：只有 `TrueStereo` 是**可以安全下发**的结论。
///
/// 折叠（L/R 视为一致 → 单声道）是**不可逆的听感损失**：真立体声一旦被折叠就
/// 回不来了。所以"没看全就下结论"只允许朝"不折叠"的方向 —— 宁可漏掉一次优化
/// （`Unknown` 不进缓存，后续扫描会带着完整预算重试），也不能折叠一段没检查过
/// 的音频。
fn degrade_incomplete(detail: VerdictDetail) -> VerdictDetail {
    if detail.verdict == ChannelVerdict::FakeStereo {
        VerdictDetail::bare(ChannelVerdict::Unknown)
    } else {
        detail
    }
}

/// 为一个源域帧区间规划**绝对帧位**的收割窗口；返回 `(窗口表, 是否被截断)`。
///
/// 窗口由与 WAV 路径同一个 [`sample_windows`] 生成（首尾都锚定在区间内），
/// 再平移到绝对帧位。三条收敛规则：
///
/// - 越过 `total_frames` 的窗口丢弃（文件比 header 声明的短）；
/// - 终点超出 `budget_frames` 的窗口丢弃 —— Symphonia 不保证随机访问，看
///   "第 N 秒"必须真的解到第 N 秒，预算是我们愿意付出的解码量上界；
/// - 任何丢弃都置 `truncated`，调用方据此只允许下发 `TrueStereo`。
///
/// 抽成纯函数是为了可测：窗口位置算错会**静默地比较错误的音频位置**（正是
/// 上一轮 `container_analysis_range` 那个 bug 的形状），必须能被单元测试钉住。
fn plan_region_windows(
    region_start: u64,
    region_end: u64,
    total_frames: u64,
    budget_frames: u64,
    sample_rate: u32,
    opts: &DetectOptions,
) -> (Vec<(u64, usize)>, bool) {
    let region_frames = region_end.saturating_sub(region_start);
    if region_frames == 0 {
        return (Vec::new(), false);
    }
    let mut windows = Vec::new();
    let mut truncated = false;
    for (offset, len) in sample_windows(region_frames as usize, sample_rate, opts) {
        let start = region_start.saturating_add(offset as u64);
        if start >= total_frames {
            truncated = true;
            continue;
        }
        let len = len.min((total_frames - start) as usize);
        if len == 0 {
            continue;
        }
        if start.saturating_add(len as u64) > budget_frames {
            truncated = true;
            continue;
        }
        windows.push((start, len));
    }
    (windows, truncated)
}

/// 从当前读位置读 `frames` 帧到 `out`（每帧只保留**前两个声道**）；返回是否
/// 读到了内容。
///
/// 【为什么必须按 `channels` 步进】WAV 的交错采样是 `frame-major`：每帧
/// `channels` 个采样。旧实现逐对顺序读，对 >2 声道的文件会拿 `(ch0,ch1)`、
/// `(ch2,ch3)`、`(ch4,ch0)` … 去比 —— 比的根本不是"左右声道"。本工程下游
/// （`channel_mode::effective_channels` / `condition_take_channels`）只消费前
/// 两个声道，所以这里也只看前两个，语义与渲染一致。
///
/// 短读（文件实际比 header 短）不算错误：已读到的部分仍会被比较。
fn read_window<R, S, F>(
    reader: &mut hound::WavReader<R>,
    frames: usize,
    channels: u16,
    out: &mut Vec<f32>,
    convert: F,
) -> bool
where
    R: std::io::Read + std::io::Seek,
    S: hound::Sample,
    F: Fn(S) -> f32,
{
    let channels = channels.max(1) as usize;
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
        // 跳过本帧剩余的声道，保持帧对齐。
        for _ in 2..channels {
            if samples.next().is_none() {
                break;
            }
        }
        out.push(convert(l));
        out.push(convert(r));
    }
    !out.is_empty()
}

/// 把源域秒区间解析为帧下标区间；越界部分钳制到 `[0, total_frames)`。
fn resolve_region(region: Option<(f64, f64)>, total_frames: u64, sample_rate: u32) -> (u64, u64) {
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

fn finalize(acc: DiffAcc) -> VerdictDetail {
    let verdict = if acc.frames == 0 {
        // 一个样本都没比到（空区间 / 解码失败）：不做结论。
        ChannelVerdict::Unknown
    } else if acc.violating == 0 {
        // 所有被检查的样本都在容差内。
        ChannelVerdict::FakeStereo
    } else {
        ChannelVerdict::TrueStereo
    };
    VerdictDetail {
        verdict,
        frames_compared: acc.frames,
        violating_frames: acc.violating,
        max_abs_diff: acc.max_abs_diff,
    }
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
            ..Default::default()
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
            ..Default::default()
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
        let fake = VerdictDetail {
            verdict: ChannelVerdict::FakeStereo,
            frames_compared: 100,
            violating_frames: 0,
            max_abs_diff: 0.0,
        };
        verdict_cache_put(base.clone(), fake);
        assert_eq!(verdict_cache_get(&base), Some(fake));
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
        let real = VerdictDetail {
            verdict: ChannelVerdict::TrueStereo,
            frames_compared: 100,
            violating_frames: 9,
            max_abs_diff: 0.7,
        };
        verdict_cache_put(base.clone(), real);

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
        assert_eq!(verdict_cache_get(&base), Some(real));
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
            ..Default::default()
        };
        let n = wild.normalized();
        assert!(n.window_sec <= 5.0 && n.window_sec >= 0.05);
        assert_eq!(n.window_count, 256);
        assert!(n.tolerance <= 1.0);

        let nan = DetectOptions {
            window_sec: f64::NAN,
            window_count: 12,
            tolerance: f32::NAN,
            ..Default::default()
        };
        let n = nan.normalized();
        assert_eq!(n.window_sec, 0.25);
        assert_eq!(
            n.tolerance, DEFAULT_TOLERANCE,
            "非法容差回退到默认值，与策略层同口径"
        );
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
        assert_eq!(quantize_region(Some((1.0004, 2.9996))), Some((1000, 3000)));
        // 非有限值不 panic。
        assert_eq!(
            quantize_region(Some((f64::NAN, f64::INFINITY))),
            Some((0, 0))
        );
    }

    #[test]
    fn plan_windows_lands_inside_a_late_region() {
        // 回归（旧实现的核心缺陷）：区间起点远晚于抽样预算时，非 WAV 路径曾
        // 直接放弃（Unknown）。现在窗口必须落在区间**内部**，一个都不能越界。
        let opts = DetectOptions {
            window_sec: 0.25,
            window_count: 12,
            tolerance: 1e-3,
            container_budget_sec: DEFAULT_CONTAINER_BUDGET_SEC,
        };
        let sr = 44_100u32;
        let total = sr as u64 * 300; // 5 分钟
        let start = sr as u64 * 60; // 从第 60 秒开始消费
        let end = sr as u64 * 70;
        let (windows, truncated) = plan_region_windows(start, end, total, total, sr, &opts);

        assert!(!truncated, "预算充足时不得截断");
        assert_eq!(windows.len(), 12, "12 个抽样窗口一个都不能少");
        for (window_start, len) in &windows {
            assert!(
                *window_start >= start,
                "窗口起点 {window_start} 落在区间起点 {start} 之前"
            );
            assert!(
                window_start + *len as u64 <= end,
                "窗口终点 {} 越出区间终点 {end}",
                window_start + *len as u64
            );
        }
        // 首尾都要锚定在区间内：只看开头正是"该判没判"的成因。
        assert_eq!(
            windows.first().unwrap().0,
            start,
            "首个窗口必须锚定区间起点"
        );
        assert_eq!(
            windows.last().unwrap().0 + windows.last().unwrap().1 as u64,
            end,
            "末个窗口必须右对齐到区间终点"
        );
    }

    #[test]
    fn plan_windows_truncates_beyond_the_decode_budget() {
        let opts = DetectOptions {
            window_sec: 0.25,
            window_count: 12,
            tolerance: 1e-3,
            container_budget_sec: 30.0,
        };
        let sr = 44_100u32;
        let total = sr as u64 * 300;
        let budget = sr as u64 * 30;
        // 区间整个落在预算之外：没有窗口可收，且必须报告截断。
        let (windows, truncated) =
            plan_region_windows(sr as u64 * 60, sr as u64 * 70, total, budget, sr, &opts);
        assert!(windows.is_empty(), "预算外的区间不该产出窗口");
        assert!(truncated, "预算外必须报告截断（上层据此不下折叠结论）");

        // 区间横跨预算边界：只收预算内的窗口，仍然报告截断。
        let (windows, truncated) = plan_region_windows(0, total, total, budget, sr, &opts);
        assert!(truncated, "横跨预算边界必须报告截断");
        assert!(!windows.is_empty(), "预算内的窗口仍应被收割");
        for (window_start, len) in &windows {
            assert!(window_start + *len as u64 <= budget, "窗口越出解码预算");
        }
    }

    #[test]
    fn plan_windows_drops_what_the_file_cannot_hold() {
        let opts = DetectOptions::default();
        let sr = 44_100u32;
        // header 声明 100 秒，但只规划到 50 秒：区间超出文件的部分要丢弃。
        let total = sr as u64 * 100;
        let (windows, truncated) =
            plan_region_windows(sr as u64 * 50, sr as u64 * 120, total, u64::MAX, sr, &opts);
        assert!(truncated, "越出文件末尾必须报告截断");
        for (window_start, len) in &windows {
            assert!(
                window_start + *len as u64 <= total,
                "窗口越出文件末尾 {total}"
            );
        }
    }

    #[test]
    fn plan_windows_is_empty_for_a_degenerate_region() {
        let opts = DetectOptions::default();
        let sr = 44_100u32;
        let total = sr as u64 * 10;
        // 起止相同（trim 到零长）→ 没有窗口，且不算"截断"（区间本身是空的）。
        let (windows, truncated) = plan_region_windows(1000, 1000, total, total, sr, &opts);
        assert!(windows.is_empty());
        assert!(!truncated);
    }

    #[test]
    fn incomplete_coverage_never_folds() {
        // 折叠是不可逆的听感损失：覆盖不全时"假立体声"必须降级为"不做结论"。
        let fake = VerdictDetail {
            verdict: ChannelVerdict::FakeStereo,
            frames_compared: 100,
            violating_frames: 0,
            max_abs_diff: 0.0,
        };
        assert_eq!(degrade_incomplete(fake).verdict, ChannelVerdict::Unknown);

        // "真立体声"是安全结论（不折叠），照常下发。
        let real = VerdictDetail {
            verdict: ChannelVerdict::TrueStereo,
            frames_compared: 100,
            violating_frames: 7,
            max_abs_diff: 0.5,
        };
        assert_eq!(degrade_incomplete(real).verdict, ChannelVerdict::TrueStereo);
        // 证据一并保留，便于用户判断要不要放宽容差。
        assert_eq!(degrade_incomplete(real).violating_frames, 7);
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

    /// 仓库自带的非 WAV 夹具（mp3）。相对路径取决于工作目录，因此从
    /// `CARGO_MANIFEST_DIR` 反推仓库根，避免夹具"永远找不到"而让测试空跑。
    fn demo_mp3() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("third_party/signalsmith-stretch/signalsmith-stretch/web/demo/loop.mp3");
        path.is_file().then_some(path)
    }

    /// 走**顺序**路径判定（与本模块的生产入口不同：它绕开"优先 seek"的选择逻辑）。
    ///
    /// 存在的意义是给 seek 路径当对照物 —— 只有把两条取数路径摆在一起比，才能
    /// 证明"换了取数方式"没有顺带改掉"怎么判"。
    fn verdicts_via_sequential_path(
        path: &Path,
        regions: &[Option<(f64, f64)>],
        opts: &DetectOptions,
    ) -> Vec<VerdictDetail> {
        let header = crate::audio_utils::try_read_audio_header_only(path).expect("probe");
        let sr = header.sample_rate.max(1);
        let total = header.total_frames;
        let budget_frames = ((opts.container_budget_sec * sr as f64).round() as u64).max(1);
        let plan = plan_container_regions(regions, total, budget_frames, sr, opts);
        let decode_limit = plan
            .windows
            .iter()
            .map(|(start, len)| start.saturating_add(*len as u64))
            .max()
            .unwrap_or(0)
            .min(budget_frames) as usize;
        let mut harvested: Vec<Option<(Vec<f32>, u16)>> = vec![None; plan.windows.len()];
        let _ = crate::media::visit_media_audio_windows(
            path,
            None,
            &plan.windows,
            decode_limit,
            &mut |index, pcm, channels, _rate| {
                if let Some(slot) = harvested.get_mut(index) {
                    *slot = Some((pcm.to_vec(), channels));
                }
                Ok(())
            },
        );
        aggregate_region_windows(&plan, &harvested, regions.len(), opts.tolerance)
    }

    #[test]
    fn seek_sampling_engages_and_agrees_with_the_sequential_path() {
        // seek 采样是本模块唯一的"换一种取数方式"的优化。它必须能真的跑起来
        //（否则这段代码是死码，长素材的覆盖问题一点没解决），且对同一份窗口
        // 给出与顺序收割**相同的结论**（否则就是把优化变成了行为变更）。
        let Some(path) = demo_mp3() else {
            return;
        };
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let header = crate::audio_utils::try_read_audio_header_only(&path).expect("probe");
        let sr = header.sample_rate.max(1);
        let total = header.total_frames;
        let total_sec = total as f64 / sr as f64;

        let regions = [None, Some((total_sec * 0.5, total_sec))];
        let opts = DetectOptions::default();
        let budget_frames = ((opts.container_budget_sec * sr as f64).round() as u64).max(1);

        let sequential = verdicts_via_sequential_path(&path, &regions, &opts);
        let by_seek =
            analyze_container_regions_by_seek(&path, &regions, &opts, total, sr, budget_frames);

        let by_seek = by_seek.expect("默认预算下 seek 采样应当可用（mp3 demuxer 支持 seek）");
        for (index, (lhs, rhs)) in by_seek.iter().zip(sequential.iter()).enumerate() {
            assert_eq!(
                lhs.verdict, rhs.verdict,
                "第 {index} 个区间：seek 与顺序路径的结论必须一致"
            );
        }
    }

    #[test]
    fn seek_gives_full_coverage_where_a_tight_sequential_budget_cannot() {
        // 这条钉的是本次改动的**收益**：`container_budget_sec` 的语义从"能看多远"
        // 变成"愿意付多少总解码量"。顺序路径的代价与文件长度成正比，预算一紧就
        // 只能丢窗口 ⇒ 覆盖不完整 ⇒ FakeStereo 被降级成 Unknown（长素材永远折叠
        // 不了）；seek 路径的代价只与窗口数有关，同样的预算下能覆盖到文件结尾。
        let Some(path) = demo_mp3() else {
            return;
        };
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let header = crate::audio_utils::try_read_audio_header_only(&path).expect("probe");
        let sr = header.sample_rate.max(1);
        let total = header.total_frames;
        let total_sec = total as f64 / sr as f64;

        // 容差拉满 ⇒ 只要覆盖完整就必然判 FakeStereo，于是结论直接反映覆盖率。
        let opts = DetectOptions {
            tolerance: 1.0,
            ..Default::default()
        };
        let window_budget_sec = 12.0 * (opts.window_sec + SEEK_PRIMING_SEC);
        // 预算必须"够 seek 抽完所有窗口、却不够顺序解到文件结尾"。
        let budget_sec = window_budget_sec + 1.0;
        if total_sec <= budget_sec + 2.0 {
            // 素材太短，顺序路径同样能覆盖到结尾：测不出差别，跳过。
            return;
        }
        let opts = DetectOptions {
            container_budget_sec: budget_sec,
            ..opts
        };

        let regions = [None];
        let sequential = verdicts_via_sequential_path(&path, &regions, &opts);
        let by_seek = analyze_container_regions_by_seek(&path, &regions, &opts, total, sr, {
            ((budget_sec * sr as f64).round() as u64).max(1)
        })
        .expect("seek 应当可用");

        assert_eq!(
            by_seek[0].verdict,
            ChannelVerdict::FakeStereo,
            "seek 路径应当在预算内覆盖整文件"
        );
        assert_eq!(
            sequential[0].verdict,
            ChannelVerdict::Unknown,
            "同样的预算下顺序路径覆盖不全，只允许保守结论（这正是改动要解决的问题）"
        );
    }

    /// 生成一个**长**素材，供"长文件"用例使用。
    ///
    /// # 为什么要自己造，而且特意用非 `.wav` 扩展名
    ///
    /// - 仓库里没有长素材，而拼接 mp3 行不通：每段都自带声明时长的头，探出来的
    ///   仍是第一段的时长（那样"长"是假的，测试会悄悄退化成空转）。
    /// - 造 WAV 内容但**存成 `.bin`**：`.wav` 会走 hound 的按窗口 seek 快路径，
    ///   那条路径本来就不受预算的位置限制，测不出我们要测的东西；换个扩展名后
    ///   才会走 Symphonia 的容器路径，也就是真实长素材（mp3/flac/m4a…）走的路。
    /// - RIFF 头里的 data 长度是**真实的**，所以探测出的总时长是准的。
    ///
    /// 返回 `(路径, 秒数, 采样率)`。
    fn write_long_fake_stereo_wav(secs: usize, sample_rate: u32) -> Option<std::path::PathBuf> {
        let dir = std::env::temp_dir().join("hifishifter_stereo_detect");
        std::fs::create_dir_all(&dir).ok()?;
        let path = dir.join(format!("long_fake_stereo_{secs}s_{sample_rate}.bin"));
        if path.is_file() {
            return Some(path);
        }

        let frames = secs * sample_rate as usize;
        let mut data = Vec::with_capacity(frames * 4);
        for i in 0..frames {
            // 两个声道**逐样本相同** ⇒ 真正的假立体声，默认容差下必然判 FakeStereo。
            let phase = (i as f32 / sample_rate as f32) * 2.0 * std::f32::consts::PI * 220.0;
            let sample = (phase.sin() * 0.5 * i16::MAX as f32) as i16;
            data.extend_from_slice(&sample.to_le_bytes());
            data.extend_from_slice(&sample.to_le_bytes());
        }

        let mut out = Vec::with_capacity(44 + data.len());
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&2u16.to_le_bytes()); // 双声道
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
        out.extend_from_slice(&4u16.to_le_bytes()); // block align
        out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        std::fs::write(&path, out).ok()?;
        Some(path)
    }

    #[test]
    fn a_long_container_folds_under_the_import_budget() {
        // 用户报告的场景：拖入一个**很长的**素材，然后等了好几秒。
        //
        // 改动前，导入路径只有两条烂路：解到文件结尾（数秒等待），或在预算内丢
        // 窗口 ⇒ 覆盖不完整 ⇒ 只敢下发 TrueStereo ⇒ 长素材永远折叠不了。
        // 改动后 seek 采样让代价只随**窗口数**增长，于是同一份导入预算就够覆盖
        // 全长 —— 这条测试把"更慢"与"漏判"两个方向一起钉住。
        const SECS: usize = 90;
        const SR: u32 = 22_050;
        let Some(long) = write_long_fake_stereo_wav(SECS, SR) else {
            eprintln!("[skip] 无法生成长素材 fixture");
            return;
        };
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();

        let Some(header) = crate::audio_utils::try_read_audio_header_only(&long) else {
            eprintln!("[skip] 生成的长容器探测失败：{}", long.display());
            let _ = std::fs::remove_file(&long);
            return;
        };
        assert_eq!(header.channels, 2, "生成的素材必须是双声道");
        let total_sec = header.total_frames as f64 / header.sample_rate.max(1) as f64;
        assert!(
            (total_sec - SECS as f64).abs() < 1.0,
            "RIFF 头声明的时长必须真实（{total_sec}s ≈ {SECS}s），否则'长'是假的"
        );

        // 默认容差：逐样本相同的两声道 ⇒ 覆盖完整就必然判 FakeStereo。
        let opts = DetectOptions {
            container_budget_sec: IMPORT_CONTAINER_BUDGET_SEC,
            ..Default::default()
        };
        assert!(
            total_sec > opts.container_budget_sec * 2.0,
            "素材必须显著长于导入预算，否则测不出'顺序路径够不着'"
        );

        let regions = [None];
        let budget_frames =
            ((opts.container_budget_sec * header.sample_rate as f64).round() as u64).max(1);

        // ① seek 采样：代价只随窗口数增长 ⇒ 导入预算内覆盖 90 秒全长。
        let by_seek = analyze_container_regions_by_seek(
            &long,
            &regions,
            &opts,
            header.total_frames,
            header.sample_rate,
            budget_frames,
        )
        .expect("长素材也必须能用 seek 采样");
        assert_eq!(
            by_seek[0].verdict,
            ChannelVerdict::FakeStereo,
            "seek 采样必须在导入预算内覆盖长素材全长"
        );

        // ② 对照：同样的预算下顺序路径够不到结尾，只能给出保守结论 —— 这正是
        //    改动前长素材永远折叠不了的原因。
        let sequential = verdicts_via_sequential_path(&long, &regions, &opts);
        assert_eq!(
            sequential[0].verdict,
            ChannelVerdict::Unknown,
            "顺序路径在同样的导入预算下够不到结尾（改动要解决的就是它）"
        );

        // ③ 生产入口（自动选路）必须真的走到 seek，而不是回落。
        let production = analyze_other_container_regions(&long, &regions, &opts);
        assert_eq!(
            production[0].verdict,
            ChannelVerdict::FakeStereo,
            "生产入口必须为长素材选出 seek 路径并给出完整覆盖的结论"
        );

        let _ = std::fs::remove_file(&long);
    }

    #[test]
    fn seek_path_is_abandoned_when_landing_off_target() {
        // 全有或全无闸门的直接体现：落点容差是「请求起点 + 一个窗口长」。这里
        // 不构造真实的错位 seek（那需要伪造容器），只钉住判定落点是否可接受的
        // 那段纯逻辑，防止将来把容差放宽到"总是接受"。
        let window_frames = 11_025u64; // 0.25s @ 44.1kHz
        let start = 441_000u64;
        let align_tolerance = window_frames;

        assert!(
            seek_landing_within_tolerance(start + 1, start, align_tolerance),
            "窗口内落点必须接受"
        );
        assert!(
            seek_landing_within_tolerance(start + align_tolerance, start, align_tolerance),
            "恰好一个窗口长的偏差仍在容差边界内"
        );
        assert!(
            !seek_landing_within_tolerance(start + align_tolerance + 1, start, align_tolerance),
            "超过一个窗口长即视为错位 ⇒ 整条路径作废"
        );
    }

    #[test]
    fn a_truncated_region_does_not_degrade_its_siblings() {
        // 回归：截断标记曾是一个**文件级**标志，回流到该文件所有区间。于是
        // "文件里存在某个读不到的区间"会把同文件的其他区间全部降级成
        // Unknown —— 表现为同一文件里有的 Take 折叠了、有的没有，而判定结果
        // 取决于某个区间和谁被放在同一批里分析（批次边界一变，结论就变）。
        let Some(path) = demo_mp3() else {
            return;
        };
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let header = crate::audio_utils::try_read_audio_header_only(&path).expect("probe");
        let sr = header.sample_rate;
        let total = header.total_frames;
        let total_sec = total as f64 / sr as f64;
        // 容差拉满 ⇒ 任何差异都判"一致" ⇒ 覆盖完整就必然是 FakeStereo。
        let opts = DetectOptions {
            tolerance: 1.0,
            container_budget_sec: 2.0,
            ..Default::default()
        };

        // 区间 0 完全在预算内（0~1s）；区间 1 远超 2 秒预算 ⇒ 只有它被截断。
        let r0 = Some((0.0, 1.0));
        let r1 = Some((total_sec * 0.8, total_sec * 0.8 + 0.5));

        // 冷缓存：两个区间在**同一次**调用里一起分析（污染只在同批时发生）。
        let both = verdict_for_regions_detailed(&path, &[r0, r1], &opts);
        assert_eq!(
            both[0].verdict,
            ChannelVerdict::FakeStereo,
            "覆盖完整的区间不得因同文件另一个区间被截断而降级"
        );
        assert!(
            both[0].frames_compared > 0,
            "结论必须建立在真实比较过的样本上"
        );
        assert_eq!(
            both[1].verdict,
            ChannelVerdict::Unknown,
            "超出解码预算的区间仍应读作'读不到'"
        );
    }

    #[test]
    fn a_region_is_judged_the_same_whatever_it_is_batched_with() {
        // 更强的性质：一个区间的判定**与批次无关** —— 单独判、和好区间一起判、
        // 和坏区间一起判，都必须得到同一个结论。这条直接钉死"同文件有的折叠
        // 有的没折叠"这类不一致。
        let Some(path) = demo_mp3() else {
            return;
        };
        let header = crate::audio_utils::try_read_audio_header_only(&path).expect("probe");
        let total_sec = header.total_frames as f64 / header.sample_rate as f64;
        let opts = DetectOptions {
            tolerance: 1.0,
            container_budget_sec: 2.0,
            ..Default::default()
        };
        let good = Some((0.0, 1.0));
        let bad = Some((total_sec * 0.8, total_sec * 0.8 + 0.5));

        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let alone = verdict_for_regions_detailed(&path, &[good], &opts)[0].verdict;
        // 每次都清缓存：命中缓存会掩盖批次相关的问题（这正是原 bug 难现的原因）。
        verdict_cache_clear();
        let with_bad = verdict_for_regions_detailed(&path, &[good, bad], &opts)[0].verdict;
        verdict_cache_clear();
        let with_good =
            verdict_for_regions_detailed(&path, &[good, Some((2.0, 3.0))], &opts)[0].verdict;

        assert_eq!(alone, ChannelVerdict::FakeStereo);
        assert_eq!(with_bad, alone, "和读不到的区间同批，结论不得改变");
        assert_eq!(with_good, alone, "和另一个好区间同批，结论不得改变");
    }

    #[test]
    fn every_slice_of_one_fake_stereo_source_folds_together() {
        // 直接映射用户报告的场景：同一文件（mp3）被切成多个 Take，消费区间
        // 各不相同，有的甚至越出文件末尾。它们必须**全部**得出同一结论，
        // 而不是有的折叠有的不折叠。
        let Some(path) = demo_mp3() else {
            return;
        };
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let header = crate::audio_utils::try_read_audio_header_only(&path).expect("probe");
        let total_sec = header.total_frames as f64 / header.sample_rate as f64;
        // 容差拉满 ⇒ 该文件处处判"一致"，于是覆盖完整的切片都应折叠。
        let opts = DetectOptions {
            tolerance: 1.0,
            ..Default::default()
        };

        let slices = vec![
            Some((0.0, 1.0)),                               // 开头
            Some((total_sec * 0.5, total_sec * 0.5 + 1.0)), // 中段
            Some((total_sec - 1.0, total_sec)),             // 结尾
        ];
        let details = verdict_for_regions_detailed(&path, &slices, &opts);
        for (index, detail) in details.iter().enumerate() {
            assert_eq!(
                detail.verdict,
                ChannelVerdict::FakeStereo,
                "同一假立体声源的第 {index} 个切片结论必须与其他切片一致"
            );
        }
    }

    #[test]
    fn container_region_far_into_the_file_is_still_judged() {
        // 回归（旧实现的核心缺陷）：非 WAV 路径只能看文件头 3 秒，且区间起点
        // 一旦晚于抽样预算就直接 `Unknown` —— 从中段切出来的 Take 在 mp3/flac
        // 源上**永远判不出来**。现在窗口沿整个消费区间铺开，中段同样有结论。
        verdict_cache_clear();
        let _cache_guard = verdict_cache_test_lock();
        let Some(path) = demo_mp3() else {
            return;
        };
        let header = crate::audio_utils::try_read_audio_header_only(&path).expect("probe");
        assert!(header.sample_rate > 0);
        let sr = header.sample_rate as f64;
        let total_sec = header.total_frames as f64 / sr;
        assert!(total_sec > 10.0, "夹具太短，取不到'3 秒预算之外'的区间");

        let opts = DetectOptions::default();
        // 区间整个落在旧预算（0.25 × 12 = 3 秒）之外。
        let region = Some((total_sec * 0.6, (total_sec * 0.7).min(total_sec)));
        let detail = verdict_for_file_detailed(&path, region, &opts);
        assert_ne!(
            detail.verdict,
            ChannelVerdict::Unknown,
            "中段区间必须能得出结论（旧实现会因起点超出 3 秒预算而放弃）"
        );
        assert!(
            detail.frames_compared > 0,
            "结论必须建立在真实比较过的样本上，而不是空转"
        );
    }

    #[test]
    fn container_windows_cover_the_head_and_tail_of_the_region() {
        // 容器路径现在与 WAV 路径共用同一套窗口规划：首尾都锚定在区间内。
        // 只判开头正是"该判没判"的成因（前几秒有立体声 intro、主体是单声道）。
        let Some(path) = demo_mp3() else {
            return;
        };
        let header = crate::audio_utils::try_read_audio_header_only(&path).expect("probe");
        let sr = header.sample_rate;
        let total = header.total_frames;
        let opts = DetectOptions::default();
        let start = total / 3;
        let end = total / 3 * 2;
        let budget = total;
        let (windows, truncated) = plan_region_windows(start, end, total, budget, sr, &opts);
        assert!(!truncated);
        assert_eq!(windows.len(), opts.window_count);
        assert_eq!(windows.first().unwrap().0, start, "首个窗口锚定区间起点");
        let last = windows.last().unwrap();
        assert_eq!(last.0 + last.1 as u64, end, "末个窗口右对齐到区间终点");
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
        assert_eq!(
            resolve_region(Some((0.5, 1.0)), 44_100, SR),
            (22_050, 44_100)
        );
        // 反向区间 → 空区间（起点与终点相同），不 panic。
        let (s, e) = resolve_region(Some((2.0, 1.0)), 44_100, SR);
        assert_eq!(s, e);
    }
}
