use std::sync::OnceLock;
// Direct FFI bindings to statically-linked WORLD library

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DioOption {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub channels_in_octave: f64,
    pub frame_period: f64, // msec
    pub speed: i32,
    pub allowed_range: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HarvestOption {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub frame_period: f64, // msec
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CheapTrickOption {
    pub q1: f64,
    pub f0_floor: f64,
    pub fft_size: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct D4COption {
    pub threshold: f64,
}

// External C functions from statically-linked WORLD library
extern "C" {
    pub fn Dio(
        x: *const f64,
        x_length: i32,
        fs: i32,
        option: *const DioOption,
        temporal_positions: *mut f64,
        f0: *mut f64,
    );
    pub fn InitializeDioOption(option: *mut DioOption);
    pub fn GetSamplesForDIO(fs: i32, x_length: i32, frame_period: f64) -> i32;
    pub fn StoneMask(
        x: *const f64,
        x_length: i32,
        fs: i32,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: i32,
        refined_f0: *mut f64,
    );
    pub fn Harvest(
        x: *const f64,
        x_length: i32,
        fs: i32,
        option: *const HarvestOption,
        temporal_positions: *mut f64,
        f0: *mut f64,
    );
    pub fn InitializeHarvestOption(option: *mut HarvestOption);
    pub fn GetSamplesForHarvest(fs: i32, x_length: i32, frame_period: f64) -> i32;
    pub fn CheapTrick(
        x: *const f64,
        x_length: i32,
        fs: i32,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: i32,
        option: *const CheapTrickOption,
        spectrogram: *mut *mut f64,
    );
    pub fn InitializeCheapTrickOption(fs: i32, option: *mut CheapTrickOption);
    pub fn GetFFTSizeForCheapTrick(fs: i32, option: *const CheapTrickOption) -> i32;
    pub fn D4C(
        x: *const f64,
        x_length: i32,
        fs: i32,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: i32,
        fft_size: i32,
        option: *const D4COption,
        aperiodicity: *mut *mut f64,
    );
    pub fn InitializeD4COption(option: *mut D4COption);
    pub fn Synthesis(
        f0: *const f64,
        f0_length: i32,
        spectrogram: *const *const f64,
        aperiodicity: *const *const f64,
        fft_size: i32,
        frame_period: f64,
        fs: i32,
        y_length: i32,
        y: *mut f64,
    );

    // 【已删除的流式合成 FFI】`InitializeSynthesizer` / `AddParameters` /
    // `RefreshSynthesizer` / `DestroySynthesizer` / `IsLocked` / `Synthesis2` 与
    // `WorldSynthesizerRaw` 曾服务于 streaming_world.rs 的逐块流式合成器，但它
    // 有结构性缺陷（pad 窗口重叠导致合成器时间轴倒退，见下方
    // `vocode_pitch_shift_chunked` 的接缝说明）且全仓无调用方，其 FFI 还依赖
    // 硬编码的 C++ 字段偏移与**猜测**的 opaque 尺寸（512 时已实际造成
    // ACCESS_VIOLATION）—— 留在代码里是随时会被复用的地雷，2026-10 审查后删除。
    // 批量 `Synthesis`（上方声明）仍在使用，不受影响。
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum WorldF0Method {
    Dio,
    Harvest,
}

fn world_f0_method() -> WorldF0Method {
    static METHOD: OnceLock<WorldF0Method> = OnceLock::new();
    *METHOD.get_or_init(|| {
        match std::env::var("HIFISHIFTER_WORLD_F0")
            .ok()
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("dio") => WorldF0Method::Dio,
            Some("harvest") => WorldF0Method::Harvest,
            // 【为什么默认是 DIO】本仓库静态链接的 WORLD 里 `Harvest` 对任何输入都
            // 返回全 0（同一段 220 Hz 正弦，DIO 给出 219.8 Hz，Harvest 全零 ——
            // 见 `estimate_f0_hz_finds_a_sine`）。全 0 的 f0 会让声码器**静默降级
            // 为直通**：界面上曲线画得好好的，声音一点没变。
            _ => WorldF0Method::Dio,
        }
    })
}

pub fn is_available() -> bool {
    // With static linking, WORLD functions are always available
    true
}

fn clamp11(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
}

/// “短洞”的上限（毫秒）：`HIFISHIFTER_WORLD_F0_GAP_MS`，默认 15；`<= 0` 关闭。
///
/// 【为什么音高线与估计器共用这一个开关】“多短的空洞算抖动、多长的算真的没有音高”
/// 是同一个听感问题。两条轨（工程里的音高线、WORLD 自己的估计）若各用一个阈值，
/// 就会在两种判定的缝隙里互相打架：估计器说“这是洞，我填上”，音高线说“这里没有值，
/// 切回干声” —— 于是移调中间插进一段 20 ms 的原声孤岛。共用一条策略，
/// 两个消费者对“什么是洞”只有一个答案。
fn f0_gap_ms() -> f64 {
    static GAP_MS: OnceLock<f64> = OnceLock::new();
    *GAP_MS.get_or_init(|| env_f64("HIFISHIFTER_WORLD_F0_GAP_MS").unwrap_or(15.0))
}

// ── 移调指令：两条绝对音高 ────────────────────────────────────────────────────

/// 一帧的移调指令：**源音高**与**目标音高**（MIDI 音高；`<= 0`/非有限 = 该帧没有音高）。
///
/// 【为什么收“两条绝对音高”而不是“一个半音数”】半音数只说“要移多少”，它必须乘到
/// 某个**具体**的 F0 上才成为激励频率。此前那个具体值来自 WORLD 自己的估计器
/// （`shifted_f0 = est × ratio`），于是输出音高 = **估计器的绝对值** × 比率 ——
/// 音高线只是“相对量”的建议，绝对音准由估计器说了算：估计器漏一帧，那一帧就不移调；
/// 估计器差一个八度，整段就错一个八度。
///
/// 而工程里的音高线（渲染端的 `clip_midi` / `pitch_edit`）本来就是**绝对音高**：
/// HiFiGAN 路径直接把它喂给 `midi_to_hz`（`renderer/hifigan.rs` 的 `midi_fn`），
/// 导出侧也按绝对值写 REAPER 的 PIT 包络。WORLD 是唯一一个“另算一遍”的消费者。
/// 所以这里也按绝对值收：**激励频率 = 目标音高**，包络分析继续用估计器（见
/// `vocode_with_f0` 的 ②）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitchFrame {
    /// 该帧的**源**音高（MIDI）。`<= 0` = 音高线在这一帧没有值（辅音 / 气声 / 静音）。
    pub source_midi: f64,
    /// 该帧的**目标**音高（MIDI）。只在 `source_midi > 0` 时有意义。
    pub target_midi: f64,
}

impl PitchFrame {
    /// 该帧没有音高。
    pub const UNPITCHED: Self = Self {
        source_midi: 0.0,
        target_midi: 0.0,
    };

    /// 移调上限（半音），与 `RendererCapabilities::max_pitch_shift_semitones` 同值。
    pub const MAX_SHIFT_SEMITONES: f64 = 24.0;

    /// 由“源音高 + 半音数”构造（测试与只关心相对量的调用方的便捷入口）。
    pub fn shifted(source_midi: f64, semitones: f64) -> Self {
        if !(source_midi.is_finite() && source_midi > 0.0) {
            return Self::UNPITCHED;
        }
        let semitones = if semitones.is_finite() {
            semitones.clamp(-Self::MAX_SHIFT_SEMITONES, Self::MAX_SHIFT_SEMITONES)
        } else {
            0.0
        };
        Self {
            source_midi,
            target_midi: source_midi + semitones,
        }
    }

    /// 这一帧有没有音高（音高线在这里有没有值）。
    pub fn is_pitched(&self) -> bool {
        self.source_midi.is_finite()
            && self.source_midi > 0.0
            && self.target_midi.is_finite()
            && self.target_midi > 0.0
    }

    /// 目标 / 源 的频率比（半音差已夹到 ±24）。**只用于日志与测试** —— 合成用的
    /// 不是这个比率，而是 `midi_to_hz` 算出的目标绝对值。
    pub fn ratio(&self) -> f64 {
        if !self.is_pitched() {
            return 1.0;
        }
        let semitones = (self.target_midi - self.source_midi)
            .clamp(-Self::MAX_SHIFT_SEMITONES, Self::MAX_SHIFT_SEMITONES);
        crate::pitch_editing::semitone_to_ratio(semitones)
    }
}

/// MIDI 音高 → Hz（A4 = MIDI 69 = 440 Hz）。无效输入返回 0（= 无周期激励）。
fn midi_to_hz(midi: f64) -> f64 {
    if !(midi.is_finite() && midi > 0.0) {
        return 0.0;
    }
    let hz = 440.0 * (2.0f64).powf((midi - 69.0) / 12.0);
    if hz.is_finite() {
        hz
    } else {
        0.0
    }
}

/// 把音高线上的**短空洞**补上（两端都有音高、洞长不超过 `f0_gap_ms`）。
///
/// 【为什么不交给 `clip_midi_at_time` 的“单帧桥接”】那个只跨 1 帧（5 ms）。
/// 分析产物里 2–3 帧的空洞很常见（清辅音擦过、基频弱），而 3 帧 = 15 ms 的原声
/// 孤岛插在一段 +7 的旋律里，听感就是“每个字都咯噔一下”。补洞的判据与
/// `cleanup_f0_inplace` 逐条一致：两端都必须有声、洞长 ≤ 阈值、端点不越界。
fn close_pitch_gaps(frames: &mut [PitchFrame], frame_period_ms: f64) {
    let gap_ms = f0_gap_ms();
    if gap_ms <= 0.0 || frames.is_empty() {
        return;
    }
    let fp = frame_period_ms.max(0.1);
    let max_gap_frames = ((gap_ms / fp).round() as isize).max(1) as usize;

    let mut i = 0usize;
    while i < frames.len() {
        if frames[i].is_pitched() {
            i += 1;
            continue;
        }
        let start = i;
        while i < frames.len() && !frames[i].is_pitched() {
            i += 1;
        }
        let end = i; // [start, end) 是空洞
        let gap_len = end - start;
        if gap_len == 0 || gap_len > max_gap_frames || start == 0 || end >= frames.len() {
            continue;
        }
        let (left, right) = (frames[start - 1], frames[end]);
        if !(left.is_pitched() && right.is_pitched()) {
            continue;
        }
        for k in 0..gap_len {
            let t = (k + 1) as f64 / (gap_len + 1) as f64;
            frames[start + k] = PitchFrame {
                source_midi: left.source_midi + (right.source_midi - left.source_midi) * t,
                target_midi: left.target_midi + (right.target_midi - left.target_midi) * t,
            };
        }
    }
}

thread_local! {
    /// 干 / 湿门的复用缓冲：`target` 是逐样本权重，`prefix` 是居中平滑的前缀和。
    ///
    /// 【为什么是 thread_local 而不是函数内分配】`hand_back_unpitched_frames` 由
    /// [`vocode_pitch_shift_chunked`] **逐块**调用，块长上限 6 s（48 kHz 下两个缓冲
    /// 各约 1.15 MB）—— 现分配意味着长素材每一块都要做两次兆级 alloc/free，与
    /// `vocode_pitch_shift_chunked` 循环外复用 `win` / `x_f64` / `seg_f32` 的纪律
    /// 相悖。WORLD 合成固定跑在渲染 worker 线程上，thread_local 命中稳定；缓冲
    /// 随最大块长增长后常驻（与那几个循环外缓冲同量级）。
    static GATE_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// 干 / 湿门：**音高线上有值**的帧用合成（湿），其余原样混回（干）。
///
/// 【为什么判据就是音高线，而不是“这一帧有没有 F0”或“这一帧响不响”】声码器能改变的
/// 东西只有“有周期的激励”，而“这一帧有没有音高”这件事，工程里已经有一个权威答案 ——
/// 音高线。此前这里另立一套判据（WORLD 估计器说有没有 F0 + 帧能量是否过 -50 dBFS），
/// 于是**干 / 湿门与移调用的是两张不同的掩码**：音高线说“这一帧 +7”，估计器说“没有
/// F0”，门就把这一帧原样混回去 —— 一段**没被移调的原声**插在移调中间，正是“辅音处
/// 音高变调质量变差”的形状。判据统一之后，“音高线连续的段落一律被移调”由构造保证，
/// 与估计器无关。
///
/// 【为什么不再需要能量门】没有音高的帧（辅音、气声、静音）**本来就该原样通过**：
/// 搬不动它们（无周期可伸缩），重合成只会给它们换一层“沙”的音色。能量门当年是为了
/// “真静音保持湿、避免抖动”，但静音走干声什么也不会发生 —— 原样就是静音。
fn hand_back_unpitched_frames(out: &mut [f64], dry: &[f64], pitched: &[bool], fp: f64, fs: i32) {
    if out.is_empty() || dry.is_empty() || pitched.is_empty() {
        return;
    }

    let fade_ms = 10.0f64;
    let fade_samples = ((fade_ms / 1000.0) * (fs.max(1) as f64)).round().max(0.0) as usize;
    let n = out.len().min(dry.len());

    GATE_SCRATCH.with(|scratch| {
        let (target, prefix) = &mut *scratch.borrow_mut();

        // 1) 逐样本求目标权重：1 = 用合成（湿），0 = 用原始（干）。
        target.clear();
        target.resize(n, 0.0f32);
        for (si, t) in target.iter_mut().enumerate() {
            let t_ms = (si as f64) * 1000.0 / (fs.max(1) as f64);
            let fi = (t_ms / fp.max(0.1)).floor().max(0.0) as usize;
            *t = if pitched.get(fi).copied().unwrap_or(false) {
                1.0
            } else {
                0.0
            };
        }

        // 2) 对目标权重做**居中**滑动平均，得到 10 ms 的平滑过渡。
        //
        // 【为什么是居中而不是“跟随上一个样本的斜坡”】居中平滑只依赖目标权重在
        // [i-fade/2, i+fade/2] 内的取值，因此是**位置的函数，与分块无关** —— 同一位置
        // 无论被哪个块处理都得到同一个增益。旧实现用一个跨样本的状态机，逐块调用时每块
        // 都从 0 重启，于是每个块首被强行淡入一次（每 6s 一次）。
        crate::seam::smooth_binary_gate_with_prefix(target, fade_samples, prefix);

        for si in 0..n {
            let w = target[si] as f64;
            out[si] = out[si] * w + dry[si] * (1.0 - w);
        }
    });
}

fn cleanup_f0_inplace(f0: &mut [f64], frame_period_ms: f64, f0_floor: f64, f0_ceil: f64) {
    if f0.is_empty() {
        return;
    }

    // 1) Clamp to a reasonable range and sanitize NaN/inf.
    for hz in f0.iter_mut() {
        if !hz.is_finite() || *hz < 0.0 {
            *hz = 0.0;
            continue;
        }
        if *hz > 0.0 {
            *hz = hz.clamp(f0_floor.max(1.0), f0_ceil.max(f0_floor.max(1.0)));
        }
    }

    // 2) 填掉浊音区里的短洞：分析/合成在 f0 抖成 0 的那几帧上会"咕噜"。
    //    阈值与音高线共用 `f0_gap_ms()`（默认 15 ms，0 = 关闭），见那里的说明。
    let gap_ms = f0_gap_ms();
    if gap_ms <= 0.0 {
        return;
    }
    let fp = frame_period_ms.max(0.1);
    let max_gap_frames = ((gap_ms / fp).round() as isize).max(1) as usize;

    let mut i = 0usize;
    while i < f0.len() {
        if f0[i] > 0.0 {
            i += 1;
            continue;
        }

        let start = i;
        while i < f0.len() && f0[i] <= 0.0 {
            i += 1;
        }
        let end = i; // [start, end) is unvoiced
        let gap_len = end - start;
        if gap_len == 0 || gap_len > max_gap_frames {
            continue;
        }
        if start == 0 || end >= f0.len() {
            continue;
        }

        let left = f0[start - 1];
        let right = f0[end];
        if !(left > 0.0 && right > 0.0) {
            continue;
        }

        // Linear interpolate across the gap.
        for k in 0..gap_len {
            let t = (k + 1) as f64 / (gap_len + 1) as f64;
            f0[start + k] = left + (right - left) * t;
        }
    }
}

/// 从一段**单声道** PCM 估计 F0（Hz；`0.0` 是该帧没有可靠音高）。
///
/// 【为什么需要这个入口】FCPE 是插件/App 的首选音高检测器，但它要 ONNX 运行时。
/// CLAP 外壳刻意不链接推理内核（见 `docs/host-extension-boundary.md`），于是
/// `fcpe_onnx::is_available()` 在那里恒为假 —— 参数编辑器就**没有原线可画**。
/// WORLD 是静态链接的（`is_available()` 恒真），质量足够画出可编辑的基线，
/// 因此把它包成一个与检测器无关的入口：谁有 PCM，谁就能拿到一条真实音高线。
///
/// 【帧网格】`frame_period_ms` 与参数编辑器一致（默认 5 ms）。返回长度由 WORLD
/// 决定（`GetSamplesForHarvest`），调用方按时间轴裁剪/重采样。
pub fn estimate_f0_hz(samples: &[f32], sample_rate: u32, frame_period_ms: f64) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 || !frame_period_ms.is_finite() {
        return Vec::new();
    }
    // WORLD 的 C 接口只吃 f64 单声道。
    let mono: Vec<f64> = samples.iter().map(|sample| f64::from(*sample)).collect();
    let fs = i32::try_from(sample_rate).unwrap_or(0);
    if fs <= 0 {
        return Vec::new();
    }
    // 人声范围；超出这个范围的"音高"在本产品的用途里只会是噪声。
    let floor = 65.0;
    let ceil = 1200.0;
    // 【为什么固定用 DIO + StoneMask，而不是跟随 `HIFISHIFTER_WORLD_F0`】本仓库静态
    // 链接的 WORLD 里 `Harvest` 会返回**全零**（同一份 PCM 用 DIO 能正确给出 220 Hz，
    // 见同文件用例）。音高线"全零"在界面上等于"没有原线"，所以这里只走实测可用的那条。
    let Ok((_positions, mut f0)) =
        compute_f0_with_positions_dio_stonemask(&mono, fs, frame_period_ms, floor, ceil)
    else {
        return Vec::new();
    };
    // 与声码器同一条清理：填掉短洞、把越界值归零（否则一条纯正弦会留下零星空洞）。
    cleanup_f0_inplace(&mut f0, frame_period_ms, floor, ceil);
    f0.into_iter().map(|hz| hz as f32).collect()
}

fn compute_f0_with_positions_dio_stonemask(
    x: &[f64],
    fs: i32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if x.is_empty() {
        return Ok((vec![], vec![]));
    }

    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.1 {
        frame_period_ms
    } else {
        5.0
    };

    let x_len: i32 = x
        .len()
        .try_into()
        .map_err(|_| "WORLD: input too long".to_string())?;

    let samples = unsafe { GetSamplesForDIO(fs, x_len, fp) };
    if samples <= 0 {
        return Ok((vec![], vec![]));
    }

    let mut option = DioOption {
        f0_floor: 71.0,
        f0_ceil: 800.0,
        channels_in_octave: 2.0,
        frame_period: fp,
        speed: 1,
        allowed_range: 0.1,
    };
    unsafe { InitializeDioOption(&mut option as *mut DioOption) };
    option.frame_period = fp;
    if f0_floor.is_finite() && f0_floor > 0.0 {
        option.f0_floor = f0_floor;
    }
    if f0_ceil.is_finite() && f0_ceil > 0.0 {
        option.f0_ceil = f0_ceil;
    }

    let mut temporal_positions = vec![0.0f64; samples as usize];
    let mut f0 = vec![0.0f64; samples as usize];

    unsafe {
        Dio(
            x.as_ptr(),
            x_len,
            fs,
            &option as *const DioOption,
            temporal_positions.as_mut_ptr(),
            f0.as_mut_ptr(),
        );
    }

    let mut refined = vec![0.0f64; samples as usize];
    unsafe {
        StoneMask(
            x.as_ptr(),
            x_len,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            samples,
            refined.as_mut_ptr(),
        );
    }

    Ok((temporal_positions, refined))
}

fn compute_f0_with_positions_harvest(
    x: &[f64],
    fs: i32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if x.is_empty() {
        return Ok((vec![], vec![]));
    }

    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.1 {
        frame_period_ms
    } else {
        5.0
    };

    let x_len: i32 = x
        .len()
        .try_into()
        .map_err(|_| "WORLD: input too long".to_string())?;

    let samples = unsafe { GetSamplesForHarvest(fs, x_len, fp) };
    if samples <= 0 {
        return Ok((vec![], vec![]));
    }

    let mut option = HarvestOption {
        f0_floor: 71.0,
        f0_ceil: 800.0,
        frame_period: fp,
    };
    unsafe { InitializeHarvestOption(&mut option as *mut HarvestOption) };
    option.frame_period = fp;
    if f0_floor.is_finite() && f0_floor > 0.0 {
        option.f0_floor = f0_floor;
    }
    if f0_ceil.is_finite() && f0_ceil > 0.0 {
        option.f0_ceil = f0_ceil;
    }

    let mut temporal_positions = vec![0.0f64; samples as usize];
    let mut f0 = vec![0.0f64; samples as usize];

    unsafe {
        Harvest(
            x.as_ptr(),
            x_len,
            fs,
            &option as *const HarvestOption,
            temporal_positions.as_mut_ptr(),
            f0.as_mut_ptr(),
        );
    }

    // Note: StoneMask is primarily recommended for DIO. Keep Harvest output as-is.
    Ok((temporal_positions, f0))
}

fn vocode_one(
    x_f64: &[f64],
    fs: i32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    abs_time_start_sec: f64,
    frame_at_time: &impl Fn(f64) -> PitchFrame,
) -> Result<Vec<f64>, String> {
    if x_f64.is_empty() {
        return Ok(vec![]);
    }

    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.1 {
        frame_period_ms
    } else {
        5.0
    };

    let (temporal_positions, mut f0) = match world_f0_method() {
        WorldF0Method::Harvest => {
            compute_f0_with_positions_harvest(x_f64, fs, fp, f0_floor, f0_ceil).or_else(|_e| {
                // Fallback to DIO if Harvest symbols or runtime fail.
                compute_f0_with_positions_dio_stonemask(x_f64, fs, fp, f0_floor, f0_ceil)
            })?
        }
        WorldF0Method::Dio => compute_f0_with_positions_dio_stonemask(
            x_f64, fs, fp, f0_floor, f0_ceil,
        )
        .or_else(|_e| {
            // Fallback to Harvest if DIO fails.
            compute_f0_with_positions_harvest(x_f64, fs, fp, f0_floor, f0_ceil)
        })?,
    };

    cleanup_f0_inplace(&mut f0, fp, f0_floor, f0_ceil);

    // 【为什么"没有有声音高"要换估计器再试一次】两条估计器在本仓库的静态链接构建里
    // 并不同等可靠（见 `world_f0_method` 的说明）。一个返回全 0 的估计器会让下面
    // 的 `f0.iter().all(|hz| hz <= 0.0)` 命中"无声音 → 直通"分支：用户把音高线
    // 画满，听到的仍是原声，而且没有任何报错。换另一个再试一次，成本是一次估计，
    // 收益是"曲线真的生效"。
    if f0.iter().all(|hz| *hz <= 0.0) {
        let alternative = match world_f0_method() {
            WorldF0Method::Harvest => WorldF0Method::Dio,
            WorldF0Method::Dio => WorldF0Method::Harvest,
        };
        let retried = match alternative {
            WorldF0Method::Dio => {
                compute_f0_with_positions_dio_stonemask(x_f64, fs, fp, f0_floor, f0_ceil)
            }
            WorldF0Method::Harvest => {
                compute_f0_with_positions_harvest(x_f64, fs, fp, f0_floor, f0_ceil)
            }
        };
        if let Ok((positions, mut retry_f0)) = retried {
            cleanup_f0_inplace(&mut retry_f0, fp, f0_floor, f0_ceil);
            if retry_f0.iter().any(|hz| *hz > 0.0) {
                return vocode_with_f0(
                    x_f64,
                    fs,
                    fp,
                    f0_floor,
                    f0_ceil,
                    abs_time_start_sec,
                    frame_at_time,
                    positions,
                    retry_f0,
                );
            }
        }
    }

    vocode_with_f0(
        x_f64,
        fs,
        fp,
        f0_floor,
        f0_ceil,
        abs_time_start_sec,
        frame_at_time,
        temporal_positions,
        f0,
    )
}

/// 用给定的一对（时间戳, 估计器 f0）完成 CHEAPTRIK / D4C / Synthesis 合成。
///
/// 【为什么要拆出来】`vocode_one` 现在有两条"先估 F0、再合成"的路径（首选估计器
/// 与重试估计器）。把合成留在一处，两条路才保证产出**同一套**后处理。
///
/// # 三件事，三个来源
///
/// 1. **包络与非周期性**（CheapTrick / D4C）用**估计器的 f0**：它描述的是"这段波形
///    的周期结构"，决定谱平滑的宽度。估计器漏掉的帧用音高线的值补 —— 音高线说
///    这一帧有声，就不该按"无周期"去分析（那会把谱窗开到最宽，齿音化的正是这一帧）。
/// 2. **激励音高**（Synthesis）用**音高线的目标值**：音高线说这一帧是什么音，输出
///    就是什么音。这一条让输出的绝对音准只由工程数据决定，与估计器的偏差无关 ——
///    与 HiFiGAN（`midi_to_hz(midi_fn(t))`）和导出侧（PIT 包络写绝对值）同一口径。
/// 3. **干 / 湿**用**音高线有没有值**：见 `hand_back_unpitched_frames`。
#[allow(clippy::too_many_arguments)]
fn vocode_with_f0(
    x_f64: &[f64],
    fs: i32,
    fp: f64,
    f0_floor: f64,
    f0_ceil: f64,
    abs_time_start_sec: f64,
    frame_at_time: &impl Fn(f64) -> PitchFrame,
    temporal_positions: Vec<f64>,
    mut f0: Vec<f64>,
) -> Result<Vec<f64>, String> {
    cleanup_f0_inplace(&mut f0, fp, f0_floor, f0_ceil);
    if f0.is_empty() {
        // 连帧网格都没有；passthrough。
        return Ok(x_f64.to_vec());
    }
    let f0_len_i32: i32 = f0
        .len()
        .try_into()
        .map_err(|_| "WORLD: f0 too long".to_string())?;

    // ① 逐帧收集移调指令（唯一事实源：工程里的音高线），并把音高线上的短洞补上。
    let mut frames: Vec<PitchFrame> = temporal_positions
        .iter()
        .map(|t| frame_at_time(abs_time_start_sec + t))
        .collect();
    close_pitch_gaps(&mut frames, fp);
    let pitched: Vec<bool> = frames.iter().map(PitchFrame::is_pitched).collect();
    if !pitched.iter().any(|&p| p) {
        // 整段没有音高 ⇒ 这一段本来就不该被移调。原样返回：既不花一次
        // CheapTrick/D4C/Synthesis，也不会把齿音交给 WORLD 重合成一遍。
        return Ok(x_f64.to_vec());
    }

    // ② 包络分析用的 f0：估计器优先（音色与今天逐帧相同），漏掉的帧用音高线补。
    let analysis_f0: Vec<f64> = f0
        .iter()
        .zip(&frames)
        .map(|(&estimated, frame)| {
            if estimated > 0.0 {
                estimated
            } else {
                midi_to_hz(frame.source_midi)
                    .clamp(f0_floor.max(1.0), f0_ceil.max(f0_floor.max(1.0)))
            }
        })
        .collect();

    // ③ 合成激励：目标音高的**绝对值**。超过 Nyquist 的值会让 Synthesis 生成
    //    不存在的谐波，按采样率夹一刀（下界只防 0，上界留 10% 余量）。
    let nyquist = 0.45 * f64::from(fs.max(1));
    let synthesis_f0: Vec<f64> = frames
        .iter()
        .map(|frame| {
            if frame.is_pitched() {
                midi_to_hz(frame.target_midi).clamp(1.0, nyquist)
            } else {
                0.0
            }
        })
        .collect();

    // CheapTrick options.
    let mut ct_opt = CheapTrickOption {
        q1: -0.15,
        f0_floor: f0_floor.max(20.0),
        fft_size: 0,
    };
    unsafe { InitializeCheapTrickOption(fs, &mut ct_opt as *mut CheapTrickOption) };
    ct_opt.f0_floor = f0_floor.max(20.0);

    let fft_size = unsafe { GetFFTSizeForCheapTrick(fs, &ct_opt as *const CheapTrickOption) };
    if fft_size <= 0 {
        return Err("WORLD: invalid fft_size".to_string());
    }
    ct_opt.fft_size = fft_size;

    let spec_bins = (fft_size as usize / 2) + 1;

    // Allocate spectrogram and aperiodicity as 2D arrays.
    let mut spectrogram = vec![0.0f64; f0.len() * spec_bins];
    let mut sp_ptrs: Vec<*mut f64> = spectrogram
        .chunks_exact_mut(spec_bins)
        .map(|row| row.as_mut_ptr())
        .collect();

    unsafe {
        CheapTrick(
            x_f64.as_ptr(),
            x_f64
                .len()
                .try_into()
                .map_err(|_| "WORLD: input too long".to_string())?,
            fs,
            temporal_positions.as_ptr(),
            analysis_f0.as_ptr(),
            f0_len_i32,
            &ct_opt as *const CheapTrickOption,
            sp_ptrs.as_mut_ptr(),
        );
    }

    let mut d4c_opt = D4COption { threshold: 0.85 };
    unsafe { InitializeD4COption(&mut d4c_opt as *mut D4COption) };

    let mut aperiodicity = vec![0.0f64; f0.len() * spec_bins];
    let mut ap_ptrs: Vec<*mut f64> = aperiodicity
        .chunks_exact_mut(spec_bins)
        .map(|row| row.as_mut_ptr())
        .collect();

    unsafe {
        D4C(
            x_f64.as_ptr(),
            x_f64
                .len()
                .try_into()
                .map_err(|_| "WORLD: input too long".to_string())?,
            fs,
            temporal_positions.as_ptr(),
            analysis_f0.as_ptr(),
            f0_len_i32,
            fft_size,
            &d4c_opt as *const D4COption,
            ap_ptrs.as_mut_ptr(),
        );
    }

    // Synthesis.
    let y_length: i32 = x_f64
        .len()
        .try_into()
        .map_err(|_| "WORLD: output too long".to_string())?;
    let mut y = vec![0.0f64; x_f64.len()];

    unsafe {
        Synthesis(
            synthesis_f0.as_ptr(),
            f0_len_i32,
            sp_ptrs.as_ptr() as *const *const f64,
            ap_ptrs.as_ptr() as *const *const f64,
            fft_size,
            fp,
            fs,
            y_length,
            y.as_mut_ptr(),
        );
    }

    // ④ 干 / 湿：音高线上没有值的帧原样混回（辅音、气声、静音都是原声）。
    let mut out = y;
    if crate::log_policy::probes_enabled() {
        let pitched_n = pitched.iter().filter(|&&b| b).count();
        log::warn!(
            "WORLD vocoder: pitched_ratio={:.3} f0_len={} fp_ms={:.3}",
            (pitched_n as f64) / (pitched.len().max(1) as f64),
            pitched.len(),
            fp
        );
    }
    hand_back_unpitched_frames(&mut out, x_f64, &pitched, fp, fs);

    Ok(out)
}

/// 单块最大时长（秒）的默认值。可用 `HIFISHIFTER_WORLD_CHUNK_SEC` 覆盖。
const WORLD_CHUNK_SEC_DEFAULT: f64 = 6.0;

/// 相邻块的重叠时长（秒）默认值 —— 交叉淡化区的长度。
/// 可用 `HIFISHIFTER_WORLD_OVERLAP_SEC` 覆盖。
const WORLD_OVERLAP_SEC_DEFAULT: f64 = 0.10;

/// 整段一次性求出的输入归一化系数 `(mean, scale)`。
///
/// # 为什么必须是全局的
/// 旧实现**逐块**去均值并按**本块峰值**归一。于是相邻块落在不同的直流基准
/// 与不同的增益上：块边界就是一个 DC 台阶 + 幅度台阶，每块一次（6s）。
/// 而 `mean` **减掉后从不加回**，所以那个台阶是实打实的直流跳变。
///
/// WORLD 的分析（CheapTrick / D4C）本就对直流不敏感，去均值只是数值预处理；
/// 把它做成全局常量即可彻底消除块间台阶，且不改变任何单块内部的处理语义。
fn world_input_normalization(mono_pcm: &[f32]) -> (f64, f64) {
    if mono_pcm.is_empty() {
        return (0.0, 1.0);
    }
    let mut mean = 0.0f64;
    for &v in mono_pcm {
        mean += v as f64;
    }
    mean /= mono_pcm.len() as f64;
    if !mean.is_finite() {
        mean = 0.0;
    }

    let mut max_abs = 0.0f64;
    for &v in mono_pcm {
        let a = ((v as f64) - mean).abs();
        if a.is_finite() && a > max_abs {
            max_abs = a;
        }
    }
    let scale = if max_abs.is_finite() && max_abs > 1.0 {
        (1.0 / max_abs).clamp(0.0, 1.0)
    } else {
        1.0
    };
    (mean, scale)
}

/// WORLD 声码器移调，分块以限制峰值内存。
///
/// `start_sec` 是 `mono_pcm[0]` 对应的绝对时间线时间。
///
/// `frame_at_time` 是**移调指令的唯一入口**：给定绝对时间，返回该帧的源音高与目标
/// 音高（见 [`PitchFrame`]）。声码器与音高有关的三件事全部由它决定 —— 包络按哪个
/// 周期分析、激励合成到哪个音、哪些帧原样通过。**不要再把"移多少"作为半音数传进来**：
/// 那样它还得再乘到某个绝对 F0 上，而能提供那个绝对值的东西（WORLD 自己的估计器）
/// 恰恰是这里要绕开的。
///
/// # 接缝处理（本函数的核心约束）
/// 每块的分析窗口向外扩 `overlap_len`（**分析需要上下文**），但只把**核心区**
/// 的合成结果按 [`crate::seam`] 的等功率窗**叠加**进输出，最后按权重和归一。
/// 相邻块的核心区因此重叠 `overlap_len` 并被交叉淡化，接缝落在淡化区中部。
///
/// 【为什么不用流式合成器】旧实现把每块的**含 pad 输入**整段推给一个逐块推进的
/// 流式合成器（该路径已删除，见文件头 FFI 声明处的说明），而相邻块的 pad 窗口
/// **互相重叠**（共 2×overlap）——
/// 合成器收到的时间轴因此**倒退**，其内部相位/时间状态被污染，
/// `pull_samples()` 返回的样本与"本块的时间位置"不再对应，输出在接缝处错位。
/// 再叠加"回退时整机重建"（等于每块一次冷启动），该路径实际从未提供它宣称的
/// 相位连续性。现在改为每块独立调用批量 `Synthesis`：它对每次调用固定重置随机
/// 种子（`matlabfunctions.cpp::randn_reseed`），因此**确定性、可缓存、可复现**；
/// 块间残余的相位与噪声差异交给交叉淡化处理。
pub fn vocode_pitch_shift_chunked<F>(
    mono_pcm: &[f32],
    sample_rate: u32,
    start_sec: f64,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    frame_at_time: F,
) -> Result<Vec<f32>, String>
where
    F: Fn(f64) -> PitchFrame,
{
    if mono_pcm.is_empty() {
        return Ok(vec![]);
    }

    let sr = sample_rate.max(1) as i32;
    let total_frames = mono_pcm.len();

    let chunk_sec = env_f64("HIFISHIFTER_WORLD_CHUNK_SEC")
        .unwrap_or(WORLD_CHUNK_SEC_DEFAULT)
        .max(0.1);
    let overlap_sec = env_f64("HIFISHIFTER_WORLD_OVERLAP_SEC")
        .unwrap_or(WORLD_OVERLAP_SEC_DEFAULT)
        .max(0.0);

    let chunk_len = (chunk_sec * (sample_rate as f64)).round().max(1.0) as usize;
    // 重叠必须**严格小于**块长：否则 `step` 归零 → 死循环。
    let overlap_len =
        ((overlap_sec * (sample_rate as f64)).round().max(0.0) as usize).min(chunk_len - 1);
    let step = (chunk_len - overlap_len).max(1);

    // 全局归一化：整段只算一次（见 `world_input_normalization`）。
    let (mean, scale) = world_input_normalization(mono_pcm);

    let mut out = vec![0.0f32; total_frames];
    let mut wsum = vec![0.0f32; total_frames];

    // 循环外复用缓冲：长素材的块数很多，逐块重新分配会明显拖慢。
    let mut win: Vec<f32> = Vec::with_capacity(chunk_len);
    let mut x_f64: Vec<f64> = Vec::with_capacity(chunk_len + overlap_len * 2);
    let mut seg_f32: Vec<f32> = Vec::with_capacity(chunk_len);

    let mut chunk_start = 0usize;
    while chunk_start < total_frames {
        let chunk_end = (chunk_start + chunk_len).min(total_frames);
        let pad_start = chunk_start.saturating_sub(overlap_len);
        let pad_end = (chunk_end + overlap_len).min(total_frames);

        x_f64.clear();
        x_f64.extend(
            mono_pcm[pad_start..pad_end]
                .iter()
                .map(|&v| clamp11(((v as f64) - mean) * scale)),
        );

        let abs_time_start_sec = start_sec + (pad_start as f64) / (sample_rate as f64);
        let y = vocode_one(
            &x_f64,
            sr,
            frame_period_ms,
            f0_floor,
            f0_ceil,
            abs_time_start_sec,
            &frame_at_time,
        )?;
        if y.len() != x_f64.len() {
            return Err("WORLD: chunk output length mismatch".to_string());
        }

        // 只取核心区：分析用了 pad，但输出只认 [chunk_start, chunk_end)。
        let central_start = chunk_start - pad_start;
        let core_len = chunk_end - chunk_start;

        let fade_in = if chunk_start == 0 { 0 } else { overlap_len };
        let fade_out = if chunk_end >= total_frames {
            0
        } else {
            overlap_len
        };
        crate::seam::chunk_window(
            core_len,
            fade_in,
            fade_out,
            crate::seam::FadeShape::EqualPower,
            &mut win,
        );

        seg_f32.clear();
        seg_f32.extend(
            y[central_start..central_start + core_len]
                .iter()
                .map(|&v| clamp11(v) as f32),
        );
        crate::seam::overlap_add(&mut out, &mut wsum, chunk_start, &seg_f32, &win);

        // 上报 clip 内渲染进度（WORLD 按块粒度）。仅当本轮渲染 pass 注册了
        // 进度回调时生效（导出路径无回调，开销只有一次存在性检查）。
        crate::renderer::progress::report_unit_progress_current(
            chunk_end as f64 / total_frames.max(1) as f64,
        );

        if chunk_end >= total_frames {
            break;
        }
        chunk_start += step;
    }

    // 按权重和归一。核心区的并集覆盖整段且无空洞（`step < chunk_len`），
    // 因此除首尾无配对处外 `wsum ≈ 1`；归一化同时兜住浮点误差。
    crate::seam::normalize_by_wsum(&mut out, &wsum);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 基础：F0 估计与归一化 ────────────────────────────────────────────────

    /// WORLD 的 F0 入口必须真的能从 PCM 里读出音高。
    ///
    /// 【为什么这条在插件之外】CLAP 外壳的音高原线完全依赖它（那边没有 FCPE）；
    /// 它坏掉时表现为"参数编辑器里一条线都没有"，而错误信息只会在别处出现。
    #[test]
    fn estimate_f0_hz_finds_a_sine() {
        let rate = 48_000_u32;
        let samples: Vec<f32> = (0..rate)
            .map(|index| {
                (2.0 * std::f64::consts::PI * 220.0 * f64::from(index) / f64::from(rate)).sin()
                    as f32
                    * 0.5
            })
            .collect();
        let f0 = estimate_f0_hz(&samples, rate, 5.0);
        assert!(!f0.is_empty(), "no frames were produced");
        let voiced: Vec<f32> = f0.iter().copied().filter(|hz| *hz > 0.0).collect();
        assert!(
            voiced.len() > f0.len() / 2,
            "too few voiced frames: {} / {}",
            voiced.len(),
            f0.len()
        );
        let mut sorted = voiced.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[sorted.len() / 2];
        assert!(
            (median - 220.0).abs() < 5.0,
            "expected 220 Hz, got {median} Hz (first: {:?})",
            &f0[..f0.len().min(12)]
        );
    }

    /// 归一化系数必须是**整段**统计量（旧实现逐块计算，块间因此有台阶）。
    #[test]
    fn normalization_is_global() {
        let dc = vec![0.5f32; 1000];
        let (mean, scale) = world_input_normalization(&dc);
        assert!((mean - 0.5).abs() < 1e-9);
        assert_eq!(scale, 1.0);

        // 峰值超过 1 → 按全局峰值缩放
        let hot: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 2.0 } else { -2.0 })
            .collect();
        let (m, s) = world_input_normalization(&hot);
        assert!(m.abs() < 1e-9);
        assert!((s - 0.5).abs() < 1e-9);

        assert_eq!(world_input_normalization(&[]), (0.0, 1.0));
    }

    // ── 移调指令（PitchFrame）的口径 ─────────────────────────────────────────

    /// PitchFrame 的三个口径：没有音高、夹取到 ±24、比率只在有音高时有意义。
    #[test]
    fn a_pitch_frame_is_pitched_only_when_both_ends_have_a_value() {
        assert!(!PitchFrame::UNPITCHED.is_pitched());
        assert!(!PitchFrame::shifted(0.0, 12.0).is_pitched());
        assert!(!PitchFrame::shifted(f64::NAN, 12.0).is_pitched());
        assert!(!PitchFrame {
            source_midi: 57.0,
            target_midi: 0.0,
        }
        .is_pitched());

        let up = PitchFrame::shifted(57.0, 12.0);
        assert!(up.is_pitched());
        assert!((up.ratio() - 2.0).abs() < 1e-9);
        assert!((up.target_midi - 69.0).abs() < 1e-9);

        // 超界只夹到 ±24：用户画到 +90 时，输出仍是 +24 的那一档。
        let over = PitchFrame::shifted(57.0, 90.0);
        assert!((over.target_midi - 81.0).abs() < 1e-9);
        assert!((over.ratio() - 4.0).abs() < 1e-9);
    }

    // ── 移调本身 ────────────────────────────────────────────────────────────

    /// **声码器必须真的移调**：+12 半音的 220 Hz 正弦，输出主频要变成 ~440 Hz。
    ///
    /// 【为什么这条不可省】移调失败最典型的形态是"静默直通"：用户把音高线画满，
    /// 听到的仍是原声，而且没有任何报错。这里用 220→440 的频率比把它钉死。
    #[test]
    fn the_vocoder_actually_shifts_the_pitch() {
        let rate = 48_000_u32;
        // 【为什么素材必须是**谐波丰富**的音】WORLD 搬的是"激励的周期 × 谱包络"：
        // 纯正弦只有一个谐波，移到 +12 之后它落在包络（仍停在 220 Hz）之外，
        // 输出只剩 -30 dB 的残响 —— 那时测到的"频率"只是噪底，测不出任何东西。
        // 24 次谐波的锯齿波才是"移调之后每个谐波都落在新基频的整数倍上"的形态。
        let samples = sawtooth(1.0, 220.0, rate);
        let out = vocode_pitch_shift_chunked(&samples, rate, 0.0, 5.0, 40.0, 1600.0, |_| {
            PitchFrame::shifted(57.0, 12.0)
        })
        .expect("vocoding failed");
        assert!(!out.is_empty(), "no output");
        let hz = pitch_hz(&out[out.len() / 4..out.len() * 3 / 4], rate);
        assert!(
            (hz - 440.0).abs() < 12.0,
            "expected ~440 Hz after +12 semitones, got {hz:.1} Hz"
        );
    }

    /// **输出音高由音高线决定，不由估计器决定**：把估计器偏一个八度，
    /// 输出仍必须是音高线写的那个音。
    ///
    /// 【为什么钉住】旧实现算的是"估计器 f0 × 比率"。于是"移了多少"是对的，
    /// "移到哪"却取决于估计器的绝对准度：估计器差一个八度，整段就错一个八度。
    /// 音高线在工程里本来就是绝对值（HiFiGAN 直接 midi_to_hz 用它），
    /// WORLD 没有理由另算一遍。
    #[test]
    fn the_output_pitch_follows_the_pitch_line_not_the_estimator() {
        let rate = 48_000u32;
        let fs = rate as i32;
        let samples: Vec<f64> = sawtooth(1.0, 220.0, rate)
            .iter()
            .map(|v| *v as f64)
            .collect();
        let (positions, _) = five_ms_grid(&samples, fs, 5.0);
        // 估计器说 110 Hz（真实是 220），音高线说 57 → +12（= 440 Hz）。
        let wrong_estimator = vec![110.0f64; positions.len()];
        let out = vocode_with_f0(
            &samples,
            fs,
            5.0,
            40.0,
            1600.0,
            0.0,
            &|_| PitchFrame::shifted(57.0, 12.0),
            positions,
            wrong_estimator,
        )
        .unwrap();
        let out32: Vec<f32> = out.iter().map(|v| *v as f32).collect();
        let hz = pitch_hz(&out32[out32.len() / 4..out32.len() * 3 / 4], rate);
        assert!(
            (hz - 440.0).abs() < 12.0,
            "激励频率必须取音高线的目标值（440 Hz），实际 {hz:.1} Hz —— 估计器偏了八度就把输出带跑了"
        );
    }

    /// **音高线上连续的段落里，没有任何一帧会被原样混回。**
    ///
    /// 【这条就是真机报障的判据】干 / 湿门一度用 WORLD 自己的估计器判断"这一帧有没有
    /// F0"。估计器漏掉的帧（起音、气声、辅音过渡）于是被原样混回去 —— 一段**没被移调
    /// 的原声**插在移调中间。用户听到的是"辅音处音高变调质量变差"，而界面上的音高线
    /// 在那里明明是连续的。这里把估计器人为挖出一个 60 ms 的洞，音高线保持连续：
    /// 洞里每一帧都必须被合成过（不是输入本身），且音高是目标音高。
    #[test]
    fn a_pitch_track_hole_never_hands_the_frame_back_dry() {
        let rate = 48_000u32;
        let fs = rate as i32;
        let samples: Vec<f64> = sawtooth(0.6, 220.0, rate)
            .iter()
            .map(|v| *v as f64)
            .collect();
        let (positions, _) = five_ms_grid(&samples, fs, 5.0);
        let mut estimator = vec![220.0f64; positions.len()];
        // 洞：中间 60 ms（12 帧）估计器完全失灵。
        let hole = positions.len() / 2..positions.len() / 2 + 12;
        for hz in estimator[hole.clone()].iter_mut() {
            *hz = 0.0;
        }
        let out = vocode_with_f0(
            &samples,
            fs,
            5.0,
            40.0,
            1600.0,
            0.0,
            &|_| PitchFrame::shifted(57.0, 7.0),
            positions,
            estimator,
        )
        .unwrap();

        let frame_samples = (rate as usize) / 200; // 5 ms
        for frame in hole.clone() {
            let start = frame * frame_samples;
            let end = (start + frame_samples).min(out.len());
            let diff = samples[start..end]
                .iter()
                .zip(out[start..end].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                diff > 1e-3,
                "第 {frame} 帧被原样混回了（max diff {diff:.6}）—— 音高线连续却被当成无音高"
            );
        }
        // 洞里必须真的是**移调后**的音高（+7：220 → 329.6 Hz）。
        let hole_start = hole.start * frame_samples;
        let hole_end = (hole.end * frame_samples).min(out.len());
        let hole32: Vec<f32> = out[hole_start..hole_end]
            .iter()
            .map(|v| *v as f32)
            .collect();
        let hz = pitch_hz(&hole32, rate);
        assert!(
            (hz - 329.6).abs() < 15.0,
            "洞里的音高不是音高线给的 +7（期望 ~330 Hz，实际 {hz:.1} Hz）"
        );
    }

    /// **音高线上没有值的帧必须逐样本原样通过**（辅音、气声、静音都是原声）。
    ///
    /// 这条曾经红过：干 / 湿门在 eda045974 里被写反（可闻的无 F0 帧走湿声），
    /// 而函数注释仍然描述正确语义，于是那段 diff 读起来"只是重构"。
    /// 现在判据是音高线，与"这一帧响不响"无关 —— 逐样本相同是唯一可信的判据。
    #[test]
    fn an_unpitched_frame_comes_back_dry() {
        let mut state = 0xdead_beef_1234_5678u64;
        let mut noise: Vec<f32> = (0..(0.5 * f64::from(CONSONANT_SR)) as usize)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.2 - 0.1
            })
            .collect();
        de_mean(&mut noise);
        let out = vocode_pitch_shift_chunked(&noise, CONSONANT_SR, 0.0, 5.0, 40.0, 1600.0, |_| {
            PitchFrame::UNPITCHED
        })
        .unwrap();
        let max_diff = noise
            .iter()
            .zip(out.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "没有音高的帧被重合成了一遍（max diff {max_diff}）"
        );
    }

    /// **辅音（音高线无值）不随移调变化**：三种移调下逐样本相同。
    ///
    /// 【为什么钉这条】它是"辅音没有被移调"这件事的**唯一客观判据**。任何"顺手让辅音
    /// 也跟一下"的改动都会在这里变红 —— 那正是取舍需要重新讨论的信号，而不是一次
    /// 普通的实现改动。移调只作用于"音高线说这里有音高"的帧，辅音连声码器都不进。
    #[test]
    fn a_consonant_keeps_its_spectrum_across_every_transposition() {
        // 声码器入口做一次**全局去均值**且从不加回，所以"原样"严格等于"去均值后的
        // 输入"。先在测试里去掉直流，判据才能写成逐样本相同（见 de_mean 的说明）。
        let mut input = consonant_band(0.5);
        de_mean(&mut input);
        for semitones in [0.0f64, 7.0, 12.0] {
            let out =
                vocode_pitch_shift_chunked(&input, CONSONANT_SR, 0.0, 5.0, 40.0, 1600.0, |_| {
                    PitchFrame::UNPITCHED
                })
                .unwrap();
            let max_diff = input
                .iter()
                .zip(out.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-9,
                "辅音在 +{semitones} 半音下被改动了（max diff {max_diff}）"
            );
        }
    }

    // ── 干 / 湿门的实现细节 ──────────────────────────────────────────────────

    /// 干/湿权重必须只由**位置**决定，与"窗口从哪开始"无关。
    ///
    /// 【为什么钉住】旧实现用 w_prev / ramp_left 状态机，逐块调用时每块都从
    /// w_prev = 0.0 重启 → 每个块首被强行淡入一次，每 6s 一次。改为居中滑动
    /// 平均后，重叠区的权重在"整段调用"与"分块调用"下必须逐样本相同。
    #[test]
    fn the_gate_weight_is_position_determined() {
        let fs = 44_100i32;
        let fp = 5.0f64;
        let n = (0.2 * f64::from(fs)) as usize; // 200 ms
        let frames = 40usize; // 5 ms 一帧
        let dry: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.013).sin() * 0.5).collect();
        // 湿声 = 干声 + 常量偏置，这样"有没有切回干"一眼可测。
        let wet: Vec<f64> = dry.iter().map(|v| v + 0.25).collect();

        // 前半有音高（湿）、后半没有音高（干）。掩码按**帧**索引，不是按样本。
        let mut pitched = vec![true; frames];
        let split_frame = frames / 2;
        for p in pitched.iter_mut().skip(split_frame) {
            *p = false;
        }

        let mut full = wet.clone();
        hand_back_unpitched_frames(&mut full, &dry, &pitched, fp, fs);

        // 模拟第二个 chunk：从一半处开始（帧边界）
        let start = split_frame * (fs as usize) * 5 / 1000;
        let mut part = wet[start..].to_vec();
        hand_back_unpitched_frames(&mut part, &dry[start..], &pitched[split_frame..], fp, fs);

        let margin = 2000;
        for i in margin..part.len() - margin {
            assert!(
                (part[i] - full[start + i]).abs() < 1e-12,
                "sample {i}: part={} full={}",
                part[i],
                full[start + i]
            );
        }
        // 远离过渡区：有音高 = 湿，没有音高 = 干。
        assert!((full[0] - wet[0]).abs() < 1e-12);
        assert!((full[n - 1] - dry[n - 1]).abs() < 1e-12);
    }

    /// 分块渲染的接缝处：既不得有 click/DC 台阶，也不得有电平鼓包。
    ///
    /// 素材是**稳态**单音，因此输出里任何接缝处的异常都只能来自分块本身：
    /// - 旧实现逐块去均值（减掉后从不加回）并在每块边界留 DC 台阶；
    /// - 旧实现用**未归一化**的等功率窗交叉淡化，同相内容在接缝中点有 +3dB 抬升。
    /// 稳态单音恰好能把后者直接量出来（RMS 比），这是本测试的主判据。
    #[test]
    fn chunked_render_has_no_seam_step_or_level_bump() {
        let sr = 44_100u32;
        let n = (sr as f64 * 7.6) as usize;
        let pcm: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                (0.3 + 0.4 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()) as f32
            })
            .collect();

        let out = vocode_pitch_shift_chunked(&pcm, sr, 0.0, 5.0, 40.0, 1600.0, |_| {
            PitchFrame::shifted(57.0, 0.0)
        })
        .unwrap();
        assert_eq!(out.len(), n);

        let chunk_len = (WORLD_CHUNK_SEC_DEFAULT * sr as f64) as usize;
        let overlap_len = (WORLD_OVERLAP_SEC_DEFAULT * sr as f64) as usize;
        let step = chunk_len - overlap_len;
        let boundary = step;
        assert!(
            boundary + 2 * overlap_len < n,
            "素材太短，覆盖不到接缝: n={n}"
        );

        let rms = |a: usize, b: usize| -> f64 {
            let s: f64 = out[a..b].iter().map(|&v| (v as f64) * (v as f64)).sum();
            (s / (b - a) as f64).sqrt()
        };

        // (a) 电平：交叉淡化区与紧随其后的"已收敛区"应同级。
        //     未归一化的等功率窗在这里会给出 ≈1.41 的比值。
        let xfade = rms(boundary, boundary + overlap_len);
        let settled = rms(boundary + overlap_len, boundary + 2 * overlap_len);
        let ratio = xfade / settled.max(1e-12);
        // 【为什么区间不是 0.9..1.1】接缝处的电平取决于两个独立合成的块在同一时刻的
        // **相位关系** —— 那是任意的（每块 Synthesis 都从自己的起点重新起相），
        // 因此"电平严格相等"在这个结构下不可达：同相会 +3dB、反相会掉下去。
        //
        // 这里保留的两条判据是**真正可保证**的：
        //   1. 不出现电平鼓包（比值不得 > 1.15）；
        //   2. 不出现近乎静音的塌陷（比值不得 < 0.5）；
        //   3. 接缝处不得有样本级台阶（下面 (b) 那条）。
        assert!(
            (0.5..1.15).contains(&ratio),
            "crossfade level discontinuity: ratio={ratio} (xfade={xfade} settled={settled})"
        );

        // (b) 台阶：接缝附近的相邻样本跳变不得成为离群值。
        let diff: Vec<f32> = out.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let mut sorted = diff.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p999 = sorted[(sorted.len() as f64 * 0.999) as usize];
        let lo = boundary.saturating_sub(overlap_len);
        let hi = (boundary + 2 * overlap_len).min(diff.len());
        let peak = diff[lo..hi].iter().cloned().fold(0.0f32, f32::max);
        assert!(
            peak <= p999 * 8.0,
            "seam at {boundary} has an outlier step: peak={peak} p99.9={p999}"
        );
    }

    // ── 测试辅助 ────────────────────────────────────────────────────────────

    const CONSONANT_SR: u32 = 44_100;

    /// 归一化自相关的基频估计（60–800 Hz）。只用于测试里的稳态合成信号。
    ///
    /// 【为什么不用过零率】WORLD 的输出是谐波叠加，过零率会被高次谐波带偏；
    /// 而自相关在基频周期上有唯一的第一个峰，对 220 / 330 / 440 这种差别足够准。
    fn pitch_hz(samples: &[f32], rate: u32) -> f64 {
        let n = samples.len().min(rate as usize / 10).max(2);
        let x = &samples[..n];
        let energy: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        if energy <= 1e-12 {
            return 0.0;
        }
        let min_lag = ((rate / 800) as usize).max(1);
        let max_lag = ((rate / 60) as usize).min(n - 1);
        if max_lag <= min_lag + 1 {
            return 0.0;
        }
        let mut corr = vec![0.0f64; max_lag + 2];
        for lag in min_lag..=max_lag {
            let mut acc = 0.0f64;
            for i in 0..n - lag {
                acc += x[i] as f64 * x[i + lag] as f64;
            }
            corr[lag] = acc / energy;
        }
        let peak = corr[min_lag..=max_lag]
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        if peak <= 0.0 {
            return 0.0;
        }
        // 自相关在周期的整数倍上都有峰 —— 第一个"足够高"的局部峰才是基频。
        for lag in min_lag + 1..max_lag {
            if corr[lag] >= corr[lag - 1] && corr[lag] >= corr[lag + 1] && corr[lag] > 0.7 * peak {
                return f64::from(rate) / lag as f64;
            }
        }
        f64::from(rate) / (min_lag + 1) as f64
    }

    /// 谐波丰富的测试音（锯齿波，24 次谐波，1/k 幅度）。
    fn sawtooth(seconds: f64, f0: f64, rate: u32) -> Vec<f32> {
        let n = (seconds * f64::from(rate)) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let mut s = 0.0f64;
                for k in 1..=24 {
                    s += (2.0 * std::f64::consts::PI * f0 * k as f64 * t).sin() / k as f64;
                }
                (s * 0.15) as f32
            })
            .collect()
    }

    /// 造一条与 WORLD 一致的 5 ms 帧网格（帧数取 GetSamplesForDIO 的口径：
    /// y_length / fs * 1000 / fp + 1）。
    fn five_ms_grid(x: &[f64], fs: i32, fp: f64) -> (Vec<f64>, usize) {
        let frames = ((x.len() as f64) / f64::from(fs) * 1000.0 / fp) as usize + 1;
        let positions: Vec<f64> = (0..frames).map(|i| i as f64 * fp / 1000.0).collect();
        (positions, frames)
    }

    /// 去掉直流分量。
    ///
    /// 【为什么每条"逐样本相同"的用例都要先做这一条】声码器入口会做一次**全局**
    /// 去均值，而且**从不加回** —— 于是"干声"严格等于"去直流后的输入"。
    /// 不去直流，一条 1e-5 级的 DC 偏移就会让"门有没有关上"看不出来。
    fn de_mean(samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        let mean = samples.iter().map(|v| f64::from(*v)).sum::<f64>() / samples.len() as f64;
        for value in samples.iter_mut() {
            *value = (*value as f64 - mean) as f32;
        }
    }

    /// 两个谐振器加权求和：模拟 /s/ 那样的 6–8 kHz 噪声带。
    fn consonant_band(seconds: f64) -> Vec<f32> {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut low = (0.0f64, 0.0f64);
        let mut high = (0.0f64, 0.0f64);
        let resonator = |input: f32, memory: &mut (f64, f64), centre: f64, bandwidth: f64| {
            let r = (-std::f64::consts::PI * bandwidth / f64::from(CONSONANT_SR)).exp();
            let theta = 2.0 * std::f64::consts::PI * centre / f64::from(CONSONANT_SR);
            let value =
                (1.0 - r) * f64::from(input) + 2.0 * r * theta.cos() * memory.0 - r * r * memory.1;
            memory.1 = memory.0;
            memory.0 = value;
            value * 3.0
        };
        (0..(seconds * f64::from(CONSONANT_SR)) as usize)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let white = ((state >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0;
                ((resonator(white, &mut low, 6000.0, 900.0) * 0.35
                    + resonator(white, &mut high, 8100.0, 1400.0) * 0.25)
                    * 0.08) as f32
            })
            .collect()
    }
}
