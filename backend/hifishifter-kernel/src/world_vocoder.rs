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

    // ── 流式合成（WorldSynthesizer / synthesisrealtime.h）──────────────────
    pub fn InitializeSynthesizer(
        fs: i32,
        frame_period: f64,
        fft_size: i32,
        buffer_size: i32,
        number_of_pointers: i32,
        synth: *mut WorldSynthesizerRaw,
    );
    pub fn AddParameters(
        f0: *mut f64,
        f0_length: i32,
        spectrogram: *mut *mut f64,
        aperiodicity: *mut *mut f64,
        synth: *mut WorldSynthesizerRaw,
    ) -> i32;
    #[allow(dead_code)]
    pub fn RefreshSynthesizer(synth: *mut WorldSynthesizerRaw);
    pub fn DestroySynthesizer(synth: *mut WorldSynthesizerRaw);
    pub fn IsLocked(synth: *mut WorldSynthesizerRaw) -> i32;
    pub fn Synthesis2(synth: *mut WorldSynthesizerRaw) -> i32;
}

// ── WorldSynthesizer 不透明句柄 ────────────────────────────────────────────────
//
// WorldSynthesizer 结构体内部含有大量裸指针和 FFT 状态，Rust 侧不需要直接访问字段，
// 只需保证内存布局足够大即可。我们用一个足够大的字节数组作为不透明存储，
// 实际大小由 C++ 侧的 sizeof(WorldSynthesizer) 决定。
//
// 精确计算（x86_64，MSVC ABI）：
//   基本字段（fs/frame_period/buffer_size/...到 randn_state 结束）：约 192 字节
//   RandnState（4×uint32）：16 字节
//   MinimumPhaseAnalysis（含 2 个 fft_plan，每个 72 字节）：176 字节
//   InverseRealFFT（含 1 个 fft_plan）：96 字节
//   ForwardRealFFT（含 1 个 fft_plan）：96 字节
//   合计：约 560 字节
//
// 原来预留 512 字节不足（差 ~48 字节），导致 InitializeSynthesizer 写入越界，
// 引发 STATUS_ACCESS_VIOLATION。现扩大到 1024 字节，留足安全余量。
//
// 注意：该结构体**只能**通过 Box::new(zeroed()) 分配在堆上，
// 绝不能在栈上创建（避免栈溢出和移动后指针失效）。
#[repr(C)]
pub struct WorldSynthesizerRaw {
    _opaque: [u8; 1024],
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
            _ => WorldF0Method::Harvest,
        }
    })
}

pub fn is_available() -> bool {
    // With static linking, WORLD functions are always available
    true
}

fn ratio_from_semitones(semitones: f64) -> f64 {
    crate::pitch_editing::semitone_to_ratio(semitones)
}

fn clamp11(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
}

const WORLD_DRY_SILENCE_RMS: f64 = 0.003;

fn blend_unvoiced_regions_with_silence_gate(
    out: &mut [f64],
    dry: &[f64],
    voiced: &[bool],
    fp: f64,
    fs: i32,
) {
    static SILENCE_RMS: OnceLock<f64> = OnceLock::new();
    let silence_rms = *SILENCE_RMS.get_or_init(|| {
        env_f64("HIFISHIFTER_WORLD_DRY_SILENCE_RMS")
            .unwrap_or(WORLD_DRY_SILENCE_RMS)
            .max(0.0)
    });
    blend_unvoiced_regions_with_silence_gate_impl(out, dry, voiced, fp, fs, silence_rms);
}

fn blend_unvoiced_regions_with_silence_gate_impl(
    out: &mut [f64],
    dry: &[f64],
    voiced: &[bool],
    fp: f64,
    fs: i32,
    silence_rms: f64,
) {
    if out.is_empty() || dry.is_empty() || voiced.is_empty() {
        return;
    }

    let fade_ms = 10.0f64;
    let fade_samples = ((fade_ms / 1000.0) * (fs.max(1) as f64)).round().max(0.0) as usize;
    let frame_samples = ((fp.max(0.1) / 1000.0) * (fs.max(1) as f64))
        .round()
        .max(1.0) as usize;

    let n = out.len().min(dry.len());

    // 1) 逐样本求"目标权重"：1 = 用合成（湿），0 = 用原始（干）。
    //
    // 浊音帧恒为湿；非浊音帧若原始信号本身有能量（齿音/气声）则切回干，
    // 否则（真静音）保留合成 —— 合成的静音也是静音，切不切听感相同，
    // 保持湿可避免在"静音↔气声"的判定抖动处反复切换。
    let mut target = vec![0.0f32; n];
    for (si, t) in target.iter_mut().enumerate() {
        let t_ms = (si as f64) * 1000.0 / (fs.max(1) as f64);
        let fi = (t_ms / fp.max(0.1)).floor().max(0.0) as usize;
        if fi >= voiced.len() {
            *t = 1.0;
            continue;
        }
        if voiced[fi] {
            *t = 1.0;
            continue;
        }
        // 非浊音帧：原始信号是否有能量？
        let start = fi.saturating_mul(frame_samples).min(dry.len());
        let end = ((fi + 1).saturating_mul(frame_samples)).min(dry.len());
        let allowed = if start >= end {
            false
        } else {
            let mut energy = 0.0f64;
            for &sample in &dry[start..end] {
                energy += sample * sample;
            }
            (energy / (end - start) as f64).sqrt() >= silence_rms
        };
        *t = if allowed { 0.0 } else { 1.0 };
    }

    // 2) 对目标权重做**居中**滑动平均，得到 10ms 的平滑过渡。
    //
    // 【为什么是居中而不是"跟随上一个样本的斜坡"】旧实现用一个跨越整段的
    // `w_prev` / `ramp_left` 状态机：它把"上一次的权重"当作历史，于是
    // **权重依赖窗口从哪里开始**。本函数在分块渲染里被**逐块**调用，
    // 每块都从 `w_prev = 0.0` 重启，结果是**每个块的开头都被强行淡入一次**
    // （10ms 从 0 爬到 1），每 6s 一次。
    //
    // 居中滑动平均只依赖目标权重在 `[i-fade/2, i+fade/2]` 内的取值 ——
    // 而目标权重由 `voiced` 与 `dry` 决定，这两者在相邻块的**重叠区**
    // 是同一份数据。因此平滑结果是**位置的函数，与分块无关**，
    // 逐块调用与整段调用给出同一个权重。实现见 `seam::smooth_binary_gate`。
    crate::seam::smooth_binary_gate(&mut target, fade_samples);

    for si in 0..n {
        let w = target[si] as f64;
        let wet = out[si];
        let dry_sample = dry[si];
        out[si] = wet * w + dry_sample * (1.0 - w);
    }
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

    // 2) Fill short unvoiced gaps inside voiced regions.
    // This reduces analysis/synthesis instability ("gargling") when f0 flickers to 0 for a few frames.
    // Default: 15ms; set HIFISHIFTER_WORLD_F0_GAP_MS=0 to disable.
    static GAP_MS: OnceLock<f64> = OnceLock::new();
    let gap_ms = *GAP_MS.get_or_init(|| env_f64("HIFISHIFTER_WORLD_F0_GAP_MS").unwrap_or(15.0));
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
    semitone_at_time: &impl Fn(f64) -> f64,
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

    let f0_len_i32: i32 = f0
        .len()
        .try_into()
        .map_err(|_| "WORLD: f0 too long".to_string())?;

    if f0.is_empty() {
        // Nothing voiced; passthrough.
        return Ok(x_f64.to_vec());
    }

    // Precompute voiced flags. WORLD uses 0 Hz for unvoiced.
    let voiced: Vec<bool> = f0.iter().map(|&hz| hz > 0.0).collect();

    // Create shifted f0.
    let mut shifted_f0 = vec![0.0f64; f0.len()];
    for i in 0..f0.len() {
        let hz = f0[i];
        if hz > 0.0 {
            let t = temporal_positions.get(i).copied().unwrap_or(0.0);
            let abs_t = abs_time_start_sec + t;
            let semitones = semitone_at_time(abs_t);
            let r = ratio_from_semitones(semitones);
            shifted_f0[i] = hz * r;
        }
    }

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
            f0.as_ptr(),
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
            f0.as_ptr(),
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
            shifted_f0.as_ptr(),
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

    // Blend vocoded output with original for unvoiced / aperiodic regions.
    // This significantly reduces the typical "sand/noise" artifacts after pitch edits,
    // especially on fricatives/breath sounds where WORLD vocoding is brittle.
    // NOTE: This is not a low-pass on the control curve; it is voiced/unvoiced gating.
    let mut out = y;
    if !voiced.is_empty() {
        let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
        if debug {
            let voiced_n = voiced.iter().filter(|&&b| b).count();
            let ratio = (voiced_n as f64) / (voiced.len().max(1) as f64);
            log::warn!(
                "WORLD vocoder: voiced_ratio={:.3} f0_len={} fp_ms={:.3}",
                ratio,
                voiced.len(),
                fp
            );
        }
        blend_unvoiced_regions_with_silence_gate(&mut out, x_f64, &voiced, fp, fs);
    }

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
/// # 接缝处理（本函数的核心约束）
/// 每块的分析窗口向外扩 `overlap_len`（**分析需要上下文**），但只把**核心区**
/// 的合成结果按 [`crate::seam`] 的等功率窗**叠加**进输出，最后按权重和归一。
/// 相邻块的核心区因此重叠 `overlap_len` 并被交叉淡化，接缝落在淡化区中部。
///
/// 【为什么不再用流式合成器】旧实现把每块的**含 pad 输入**整段推给
/// `StreamingWorldSynthesizer`，而相邻块的 pad 窗口**互相重叠**（共 2×overlap）——
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
    semitone_at_time: F,
) -> Result<Vec<f32>, String>
where
    F: Fn(f64) -> f64,
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
            &semitone_at_time,
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

    /// 干/湿混合的权重必须只由**位置**决定，与"窗口从哪开始"无关。
    ///
    /// 【为什么钉住】旧实现用 `w_prev` / `ramp_left` 状态机，逐块调用时每块都从
    /// `w_prev = 0.0` 重启 → 每个块首被强行淡入一次，每 6s 一次。改为居中滑动
    /// 平均后，重叠区的权重在"整段调用"与"分块调用"下必须逐样本相同。
    #[test]
    fn unvoiced_blend_weight_is_position_determined() {
        let fs = 44_100i32;
        let fp = 5.0f64;
        let frame_samples = ((fp / 1000.0) * fs as f64) as usize; // 220
        let n_frames = 40usize;
        let n = n_frames * frame_samples;
        let dry: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.013).sin() * 0.5).collect();
        // 前半浊音、后半非浊音（干声有能量 → 应切回干）
        let mut voiced = vec![true; n_frames];
        for v in voiced.iter_mut().skip(20) {
            *v = false;
        }

        let mut full = dry.clone();
        blend_unvoiced_regions_with_silence_gate_impl(&mut full, &dry, &voiced, fp, fs, 0.001);

        // 模拟第二个 chunk：从第 20 帧开始（对齐到帧边界）
        let start = 20 * frame_samples;
        let mut part = dry[start..].to_vec();
        blend_unvoiced_regions_with_silence_gate_impl(
            &mut part,
            &dry[start..],
            &voiced[20..],
            fp,
            fs,
            0.001,
        );

        // 窗口中部（远离首尾边界效应）必须与整段结果一致
        let margin = 2000;
        for i in margin..part.len() - margin {
            assert!(
                (part[i] - full[start + i]).abs() < 1e-12,
                "sample {i}: part={} full={}",
                part[i],
                full[start + i]
            );
        }
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

        let out = vocode_pitch_shift_chunked(&pcm, sr, 0.0, 5.0, 40.0, 1600.0, |_| 0.0).unwrap();
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
        assert!(
            (0.9..1.1).contains(&ratio),
            "crossfade level bump: ratio={ratio} (xfade={xfade} settled={settled})"
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
}
